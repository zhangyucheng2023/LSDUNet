from utils import *
import math
from math import log10
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext as _null_ctx

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import laplace as _laplacian
from torch.amp import autocast
from tqdm import tqdm
from metrics import compute_roi_mask, compute_ece, compute_brier_score, _get_lpips


class EMA:
    """Exponential Moving Average for model weights.

    DDP-aware: when dist is initialized, all-reduce shadow parameters across ranks
    so every rank keeps an identical EMA copy. This prevents EMA divergence when
    each rank sees a different data shard.

    Performance: uses a single flattened buffer for all-reduce (1 NCCL call per
    sync instead of N per-parameter calls). For a model with ~500 params this
    cuts sync overhead by ~500x.
    """
    def __init__(self, model, decay=0.999, ddp_sync=True, warm_decay=0.9):
        self.decay = decay
        self.warm_decay = warm_decay
        self.ddp_sync = ddp_sync and dist.is_available() and dist.is_initialized()
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        # Build flattened buffer for efficient all-reduce (1 NCCL call vs N)
        self._build_flat_buffer()
        # Ensure initial shadow is identical across ranks
        if self.ddp_sync:
            self._sync_shadow()

    def _build_flat_buffer(self):
        """Build a single flattened buffer from all shadow params."""
        self._flat_meta = []  # (name, offset, numel, shape)
        offset = 0
        for name, t in self.shadow.items():
            numel = t.numel()
            self._flat_meta.append((name, offset, numel, t.shape))
            offset += numel
        # Use device/dtype from first param (all params typically on same device)
        sample = next(iter(self.shadow.values()))
        self._flat_buffer = torch.zeros(offset, dtype=sample.dtype, device=sample.device)

    @torch.no_grad()
    def _sync_shadow(self):
        """All-reduce shadow parameters to keep them identical across ranks.

        Uses a single flattened all-reduce for efficiency: O(1) NCCL calls
        instead of O(N_params).
        """
        if not self.ddp_sync:
            return
        # Copy shadow params into flat buffer
        for name, offset, numel, _ in self._flat_meta:
            self._flat_buffer[offset:offset + numel] = self.shadow[name].detach().reshape(-1)
        # Single all-reduce
        dist.all_reduce(self._flat_buffer, op=dist.ReduceOp.AVG)
        # Copy back to shadow params
        for name, offset, numel, shape in self._flat_meta:
            self.shadow[name].copy_(self._flat_buffer[offset:offset + numel].view(shape))

    @torch.no_grad()
    def update(self, model, epoch=0, warm_epochs=5):
        # Warmup decay: 从 warm_decay 线性增长到 decay, 初期跟踪更快
        if warm_epochs > 0 and epoch < warm_epochs:
            cur_decay = self.warm_decay + (self.decay - self.warm_decay) * (epoch + 1) / warm_epochs
        else:
            cur_decay = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                # p.data is already synchronized across ranks by DDP backward,
                # so local EMA update produces identical shadow on every rank.
                self.shadow[name].mul_(cur_decay).add_(p.data, alpha=1 - cur_decay)
        # Defensive periodic sync (cheap insurance against any drift)
        if self.ddp_sync:
            self._sync_shadow()

    @torch.no_grad()
    def apply(self, model):
        """Apply EMA weights to model. Call before validation."""
        self.backup = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        """Restore original weights after validation."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = None


# LPIPS model: use shared instance from metrics.py to avoid duplicate init
# (metrics.py handles lazy loading + SSL workaround + GPU placement)
_lpips_model = None  # lazily fetched on first use via _get_lpips()
HAS_LPIPS = True  # will be set to False if _get_lpips() returns False


def _ensure_lpips():
    """Lazily initialize LPIPS model from metrics.py (shared singleton)."""
    global _lpips_model, HAS_LPIPS
    if _lpips_model is None and HAS_LPIPS:
        _lpips_model = _get_lpips()
        if _lpips_model is False:
            _lpips_model = None
            HAS_LPIPS = False
    return _lpips_model


def compute_lpips_batch(pred, target):
    """Compute LPIPS for a batch of 2D images.
    pred, target: [N, H, W] grayscale or [N, C, H, W] multi-channel."""
    model = _ensure_lpips()
    if model is None:
        return None
    if pred.dim() == 3:
        pred_t = pred.unsqueeze(1).repeat(1, 3, 1, 1)
        target_t = target.unsqueeze(1).repeat(1, 3, 1, 1)
    else:
        pred_t = pred
        target_t = target
    if pred_t.max() > 1.5:
        pred_t = pred_t / 255.0
        target_t = target_t / 255.0
    device = next(model.parameters()).device
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    with torch.no_grad():
        return model(pred_t, target_t).mean().item()


# ─── 可微损失函数 (DWT 小波 + SSIM) ───

def dwt_haar_2d(x, levels=3):
    """多级 Haar 小波分解 (可微, 纯 PyTorch).

    Args:
        x: [B, C, H, W], H/W 必须能被 2^levels 整除
        levels: 分解级数

    Returns:
        coeffs: list of (LL, (HL, LH, HH)) per level, LL 是低频, HL/LH/HH 是高频
    """
    coeffs = []
    current = x
    for _ in range(levels):
        a = current[:, :, 0::2, 0::2]
        b = current[:, :, 1::2, 0::2]
        c = current[:, :, 0::2, 1::2]
        d = current[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 2
        hl = (a - b + c - d) / 2  # 水平高频
        lh = (a + b - c - d) / 2  # 垂直高频
        hh = (a - b - c + d) / 2  # 对角高频
        coeffs.append((ll, (hl, lh, hh)))
        current = ll
    return coeffs


def wavelet_loss(pred, target, levels=3):
    """多级 Haar 小波 L1 损失.

    替代 FFT 频率损失:
    - 多尺度局部化 (FFT 是全局频率, 小波是局部时频)
    - 边缘保持 (高频系数直接对应边缘)
    - O(N) 复杂度 (FFT 是 O(N log N))
    - 支持 bf16 (fft2 不支持)

    Args:
        pred, target: [B, C, H, W]
    """
    pred_coeffs = dwt_haar_2d(pred, levels)
    tgt_coeffs = dwt_haar_2d(target, levels)
    loss = 0.0
    for (p_ll, (p_hl, p_lh, p_hh)), (t_ll, (t_hl, t_lh, t_hh)) in \
            zip(pred_coeffs, tgt_coeffs):
        loss = loss + F.l1_loss(p_ll, t_ll)
        loss = loss + F.l1_loss(p_hl, t_hl)
        loss = loss + F.l1_loss(p_lh, t_lh)
        loss = loss + F.l1_loss(p_hh, t_hh)
    return loss / levels


def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """可微 SSIM 损失 (1 - SSIM).

    用 avg_pool 作为窗口, 无需外部依赖.

    Args:
        pred, target: [B, C, H, W] in [0, 1]
    """
    pad = window_size // 2
    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_t = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_p_sq
    sigma_t_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_t_sq
    sigma_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return 1.0 - ssim_map.mean()


def train_3d(train_loader, model, optimizer, device, grad_clip=1.0,
             grad_accum=1,
             ddp_model=None, ema=None,
             w_edge=0.1, w_freq=0.01, w_ortho=0.01, w_nll=0.01, w_ssim=0.1,
             w_lowrank=0.01, w_sparse=0.01,
             epoch=0, warm_epochs=5):
    """Simplified trainer for LSDUNet-Lite.

    No gradient checkpointing needed (lite model is memory-efficient).
    No complex AMP dtype tracking (just bfloat16 on Ampere+).

    Loss weights (parameterized for grid-search / ablation):
      w_edge: weight for Sobel edge loss
      w_freq: weight for DWT wavelet loss (replaces FFT)
      w_ortho: weight for sampling-matrix orthogonality loss
      w_nll: weight for uncertainty NLL loss
      w_ssim: weight for differentiable SSIM loss
      w_lowrank: weight for L+S low-rank regularization
      w_sparse: weight for L+S sparsity regularization

    Warmup: auxiliary losses linearly scale from 0 to full weight during
    warm_epochs to avoid early-training instability.
    """
    model.train()
    sum_loss = 0
    accum_count = 0

    # AMP: bfloat16 on Ampere+ (no GradScaler needed), fp16 with GradScaler on older
    use_amp = (device.type == 'cuda')
    amp_dtype = None
    scaler = None
    if use_amp:
        major = torch.cuda.get_device_capability(device)[0]
        if major >= 8:
            amp_dtype = torch.bfloat16
        else:
            from torch.amp import GradScaler
            scaler = GradScaler(device.type)

    pbar = tqdm(train_loader, desc='train', dynamic_ncols=True, mininterval=0.5) if is_main_process() else train_loader
    for step_i, (inputs, _) in enumerate(pbar):
        inputs = inputs.to(device)
        # Assert expected 5D input: [B, T, C, H, W]
        assert inputs.dim() == 5, f"Expected 5D input [B,T,C,H,W], got {inputs.dim()}D shape {inputs.shape}"
        target = inputs

        # Forward pass: use DDP-wrapped model (not base_model) so gradient sync works
        fwd_model = ddp_model if ddp_model is not None else model
        ctx = autocast(device.type, dtype=amp_dtype) if amp_dtype else (
            autocast(device.type) if scaler else _null_ctx())
        with ctx:
            outputs, uncertainty = fwd_model(inputs, return_uncertainty=True)

            # Reconstruction loss (MSE)
            loss_rec = F.mse_loss(outputs, target)

            # Edge loss (Sobel filter - preserves tactile edges)
            # outputs: [B, T, C, H, W] -> grayscale [B*T, 1, H, W]
            pred_gray = outputs.mean(dim=2).reshape(-1, 1, *outputs.shape[-2:])
            tgt_gray = target.mean(dim=2).reshape(-1, 1, *target.shape[-2:])
            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=outputs.dtype, device=outputs.device).view(1,1,3,3)
            sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=outputs.dtype, device=outputs.device).view(1,1,3,3)
            pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
            pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
            tgt_edge_x = F.conv2d(tgt_gray, sobel_x, padding=1)
            tgt_edge_y = F.conv2d(tgt_gray, sobel_y, padding=1)
            loss_edge = F.mse_loss(pred_edge_x, tgt_edge_x) + F.mse_loss(pred_edge_y, tgt_edge_y)

            # Wavelet loss (DWT - multi-scale local frequency, replaces FFT)
            # Haar 小波: 多尺度边缘保持, O(N) 复杂度, 支持 bf16 (fft2 不支持)
            pred_2d = outputs.flatten(0, 1)  # [B*T, C, H, W]
            tgt_2d = target.flatten(0, 1)
            loss_freq = wavelet_loss(pred_2d, tgt_2d, levels=3)

            # SSIM loss (differentiable structural similarity)
            loss_ssim = ssim_loss(pred_2d, tgt_2d)

            # Uncertainty-weighted NLL loss: 0.5 * (log(σ²) + (x-μ)²/σ²)
            # Small weight to avoid destabilizing training
            var = uncertainty.float()  # [B, T, 1, H, W]
            sq_err = (outputs.float() - target.float()).pow(2).mean(dim=2, keepdim=True)  # [B, T, 1, H, W]
            loss_nll = 0.5 * (torch.log(var) + sq_err / var).mean()

            # Total loss (parameterized weights for grid-search)
            # 辅助损失 warmup: warmup 阶段线性增长, 避免训练初期不稳定
            aux_scale = min(1.0, float(epoch + 1) / max(warm_epochs, 1))
            # NOTE: adaptive_s.ortho_loss() is on base_model; access via model.module if DDP
            ortho_model = model.module if hasattr(model, 'module') else model
            loss = loss_rec  # 主损失不受 warmup 影响
            loss = loss + aux_scale * (w_edge * loss_edge + w_freq * loss_freq
                                       + w_ssim * loss_ssim + w_nll * loss_nll)
            loss = loss + aux_scale * w_ortho * ortho_model.adaptive_s.ortho_loss()

            # L+S 低秩稀疏正则化损失 (Schatten-0.5 范数 + L1)
            # Schatten-p (p=0.5) 比核范数更精确促进低秩: Σ σ^p
            # 用空间池化后小矩阵 SVD, 开销可忽略
            if hasattr(ortho_model, 'get_ls_regularization'):
                Ls, Ss = ortho_model.get_ls_regularization()
                if Ls:  # 训练时才有缓存
                    loss_lowrank = 0.0
                    for L in Ls:
                        # 双重 L+S 形状不一致, 按维度分支:
                        #   ① 显式 LSDecomposition: [B, C, T, H, W] (5D)
                        #   ② 隐式 Kalman L/S:      [B, T, C]      (3D, 已是序列级矩阵)
                        if L.dim() == 5:
                            B_l, _, T_l, _, _ = L.shape
                            # 池化 [B, C, T, 4, 4] → [B, T, C*16]
                            L_pool = F.adaptive_avg_pool3d(L, (T_l, 4, 4))
                            L_mat = L_pool.reshape(B_l, T_l, -1)
                        else:
                            # Kalman 隐式 L: [B, T, C] 直接当作矩阵做 SVD
                            L_mat = L
                            B_l = L.shape[0]
                        # svdvals: 只算奇异值, 比 svd 更快更稳定 (无需 U/V)
                        # clamp 防止 σ→0 时 σ^0.5 梯度爆炸 (0.5/√σ→∞)
                        # L/S 已 detach, 用 FP32 计算 SVD (BF16 svdvals 不支持)
                        try:
                            S_sval = torch.linalg.svdvals(L_mat.float())
                            loss_lowrank = loss_lowrank + S_sval.clamp(min=1e-6).pow(0.5).sum() / B_l
                        except RuntimeError:
                            # SVD 不收敛时退回 Frobenius (数值安全)
                            loss_lowrank = loss_lowrank + L.float().pow(2).mean()
                    loss_lowrank = loss_lowrank / len(Ls)
                    # S 稀疏项: abs().mean() 对任意维度都成立, 无需分支
                    loss_sparse = sum(S.abs().mean() for S in Ss) / len(Ss)
                    loss = loss + aux_scale * (w_lowrank * loss_lowrank + w_sparse * loss_sparse)

            loss = loss / grad_accum

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[rank {rank}] Non-finite loss at step {step_i}: {loss_val}, stopping.")
            raise RuntimeError(f"Training diverged: loss={loss_val} at step {step_i}")

        # Backward pass
        is_last_micro = ((accum_count + 1) % grad_accum == 0)
        if scaler is not None:
            if ddp_model is not None and not is_last_micro:
                with ddp_model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
        else:
            if ddp_model is not None and not is_last_micro:
                with ddp_model.no_sync():
                    loss.backward()
            else:
                loss.backward()

        accum_count += 1
        if accum_count % grad_accum == 0:
            if scaler is not None:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            # EMA update
            if ema is not None:
                ema.update(model, epoch=epoch, warm_epochs=warm_epochs)

        sum_loss += loss_val * grad_accum
        if is_main_process() and step_i % 10 == 0:
            pbar.set_postfix({'loss': f'{loss_val * grad_accum:.6f}'})

    # Flush leftover gradients
    if accum_count % grad_accum != 0:
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()
        # EMA update
        if ema is not None:
            ema.update(model, epoch=epoch, warm_epochs=warm_epochs)

    return sum_loss / len(train_loader)


def valid_3d(val_loader, model, device, ddp=False, ema=None, collect_uncertainty=False):

    sum_ssim = 0
    sum_lpips = 0
    sum_edge_psnr = 0
    sum_roi_psnr = 0
    sum_roi_ssim = 0
    lpips_count = 0
    n_samples = 0
    total_mse = 0.0
    total_pixels = 0
    # For uncertainty calibration (ECE / Brier)
    all_uncertainty = [] if collect_uncertainty else None
    all_abs_error = [] if collect_uncertainty else None

    # Inference: use bf16 on Ampere+, fp16 on older GPUs
    use_amp = (device.type == 'cuda')
    amp_dtype = None
    if use_amp:
        major = torch.cuda.get_device_capability(device)[0]
        if major >= 8:
            amp_dtype = torch.bfloat16

    model.eval()
    # DDP: move LPIPS model to this rank's GPU (lazy init via shared singleton)
    _lpips = _ensure_lpips()
    if _lpips is not None and device.type == 'cuda':
        lpips_dev = next(_lpips.parameters()).device
        if lpips_dev != device:
            _lpips.to(device)

    # Apply EMA weights for validation
    if ema is not None:
        ema.apply(model)
    try:
        pbar = tqdm(val_loader, desc='valid', dynamic_ncols=True, mininterval=0.5) if is_main_process() else val_loader
        with torch.inference_mode():
            for step_i, (inputs, _) in enumerate(pbar):
                inputs = inputs.to(device)
                # Assert expected 5D input: [B, T, C, H, W]
                assert inputs.dim() == 5, f"Expected 5D input [B,T,C,H,W], got {inputs.dim()}D shape {inputs.shape}"
                B, T, C, H, W = inputs.shape
                y_ch = inputs

                if amp_dtype is not None:
                    with autocast(device.type, dtype=amp_dtype):
                        if collect_uncertainty:
                            outputs, uncertainty = model(y_ch, return_uncertainty=True)
                        else:
                            outputs = model(y_ch)
                elif use_amp:
                    with autocast(device.type):
                        if collect_uncertainty:
                            outputs, uncertainty = model(y_ch, return_uncertainty=True)
                        else:
                            outputs = model(y_ch)
                else:
                    if collect_uncertainty:
                        outputs, uncertainty = model(y_ch, return_uncertainty=True)
                    else:
                        outputs = model(y_ch)

                pred = outputs.float()  # convert back to fp32 for metric computation
                target = y_ch
                mse = F.mse_loss(pred, target)
                total_mse += mse.item() * pred.numel()
                total_pixels += pred.numel()
                running_psnr = 10 * log10(1 / (total_mse / total_pixels)) if total_mse > 0 else 100.0

                # Uncertainty calibration collection (subsample for memory)
                if collect_uncertainty and all_uncertainty is not None:
                    unc = uncertainty.float().cpu().numpy().reshape(-1)
                    err = (pred - target).abs().float().cpu().numpy().reshape(-1)
                    # Subsample to avoid OOM on large val sets
                    if len(unc) > 50000:
                        idx = np.random.choice(len(unc), 50000, replace=False)
                        unc = unc[idx]
                        err = err[idx]
                    all_uncertainty.append(unc)
                    all_abs_error.append(err)

                # Batch LPIPS: process all B*T frames in one forward pass
                if HAS_LPIPS:
                    bt = B * T
                    bt_preds = pred.reshape(bt, C, *pred.shape[-2:]).cpu()
                    bt_targets = target.reshape(bt, C, *target.shape[-2:]).cpu()
                    lpips_mean = compute_lpips_batch(bt_preds, bt_targets)
                    if lpips_mean is not None:
                        sum_lpips += lpips_mean * bt
                        lpips_count += bt

                for b in range(B):
                    for t in range(T):
                        n_samples += 1
                        pred_np = pred[b, t].mean(dim=0).cpu().numpy().astype(np.float64)
                        tgt_np = target[b, t].mean(dim=0).cpu().numpy().astype(np.float64)

                        # SSIM
                        sum_ssim += ssim(tgt_np, pred_np, data_range=1)

                        # Edge PSNR (Laplacian domain)
                        pred_edge = _laplacian(pred_np)
                        tgt_edge = _laplacian(tgt_np)
                        edge_range = max(float(pred_edge.max() - pred_edge.min()),
                                         float(tgt_edge.max() - tgt_edge.min()), 1e-8)
                        edge_mse = np.mean((pred_edge - tgt_edge) ** 2)
                        if edge_mse > 0:
                            sum_edge_psnr += 10 * np.log10(edge_range ** 2 / edge_mse)
                        else:
                            sum_edge_psnr += 100.0

                        # ROI PSNR / SSIM (统一使用 metrics.compute_roi_mask)
                        mask = compute_roi_mask(tgt_np)
                        pred_roi = pred_np[mask > 0.5]
                        tgt_roi = tgt_np[mask > 0.5]
                        if len(pred_roi) > 0:
                            roi_mse = np.mean((pred_roi - tgt_roi) ** 2)
                            if roi_mse > 0:
                                sum_roi_psnr += 10 * np.log10(1.0 / roi_mse)
                            else:
                                sum_roi_psnr += 100.0
                            pred_masked = pred_np * mask + tgt_np * (1 - mask)
                            tgt_masked = tgt_np * mask + tgt_np * (1 - mask)
                            sum_roi_ssim += ssim(tgt_masked, pred_masked, data_range=1)

                if is_main_process() and step_i % 10 == 0:
                    postfix = {
                        'psnr': f'{running_psnr:.2f}',
                        'ssim': f'{sum_ssim / n_samples:.4f}',
                        'edge': f'{sum_edge_psnr / n_samples:.2f}',
                        'roi': f'{sum_roi_psnr / n_samples:.2f}',
                    }
                    if lpips_count > 0:
                        postfix['lpips'] = f'{sum_lpips / lpips_count:.4f}'
                    pbar.set_postfix(postfix)
    finally:
        if ema is not None:
            ema.restore(model)

    n = max(n_samples, 1)

    # DDP: all-reduce metrics across GPUs
    if ddp and dist.is_available() and dist.is_initialized():
        metrics_t = torch.tensor([total_mse, float(total_pixels), float(sum_ssim),
                                  float(n_samples), float(sum_lpips), float(lpips_count),
                                  float(sum_edge_psnr), float(sum_roi_psnr),
                                  float(sum_roi_ssim)], device=device)
        dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
        total_mse = metrics_t[0].item()
        total_pixels = int(metrics_t[1].item())
        sum_ssim = metrics_t[2].item()
        n_samples = int(metrics_t[3].item())
        sum_lpips = metrics_t[4].item()
        lpips_count = int(metrics_t[5].item())
        sum_edge_psnr = metrics_t[6].item()
        sum_roi_psnr = metrics_t[7].item()
        sum_roi_ssim = metrics_t[8].item()
        n = max(n_samples, 1)

    final_psnr = 10 * log10(1 / (total_mse / total_pixels)) if total_mse > 0 else 100.0
    ret = (
        final_psnr,
        sum_ssim / n,
        sum_lpips / lpips_count if lpips_count > 0 else None,
        sum_edge_psnr / n,
        sum_roi_psnr / n,
        sum_roi_ssim / n,
    )

    # Uncertainty calibration metrics (ECE / Brier Score)
    if collect_uncertainty and all_uncertainty:
        unc_arr = np.concatenate(all_uncertainty)
        err_arr = np.concatenate(all_abs_error)
        # uncertainty is variance, convert to std for calibration
        unc_std = np.sqrt(np.clip(unc_arr, 1e-6, None))
        ece = compute_ece(unc_std, err_arr)
        brier = compute_brier_score(unc_std, err_arr)
        ret = ret + (ece, brier)
    elif collect_uncertainty:
        ret = ret + (0.0, 0.0)

    return ret


