from utils import *
import math
from math import log10
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import laplace as _laplacian
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint as _checkpoint
from tqdm import tqdm
from metrics import compute_roi_mask

try:
    import ssl
    import warnings
    _orig_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lpips as _lpips_lib
        _lpips_model = _lpips_lib.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
    ssl._create_default_https_context = _orig_context
    HAS_LPIPS = True
except Exception:
    _lpips_model = None
    HAS_LPIPS = False


def compute_lpips_batch(pred, target):
    """Compute LPIPS for a batch of 2D images.
    pred, target: [N, H, W] grayscale or [N, C, H, W] multi-channel."""
    if not HAS_LPIPS or _lpips_model is None:
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
    device = next(_lpips_model.parameters()).device
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    with torch.no_grad():
        return _lpips_model(pred_t, target_t).mean().item()


def train_3d(train_loader, model, optimizer, device, grad_clip=1.0,
             grad_accum=1, use_ckpt=True,
             ddp_model=None):
    model.train()
    sum_loss = 0
    accum_count = 0

    # Auto mixed precision: Ampere+ (sm_80) → bf16, older → fp16 with GradScaler
    use_amp = (device.type == 'cuda')
    amp_dtype = None
    scaler = None
    if use_amp:
        major = torch.cuda.get_device_capability(device)[0]
        if major >= 8:
            amp_dtype = torch.bfloat16  # bf16: no GradScaler needed (Ampere/Ada/Blackwell)
        else:
            from torch.amp import GradScaler
            scaler = GradScaler(device.type)  # fp16: needs GradScaler

    use_ckpt = use_ckpt and (device.type == 'cuda')

    def _ckpt_iter(gdb, dst, x, y, adaptive_s, R, s_dyn, x_as, _amp_dtype, _device):
        """Checkpointed iteration block. Inputs are already cast to target dtype
        before entering checkpoint, so checkpoint saves bf16 tensors (not fp32)."""
        if _amp_dtype is not None:
            with autocast(_device.type, dtype=_amp_dtype):
                x, x_as = gdb(x, y, adaptive_s, R, s_dyn, x_as)
                x = dst(x)
        else:
            x, x_as = gdb(x, y, adaptive_s, R, s_dyn, x_as)
            x = dst(x)
        return x, x_as

    def _forward(inputs):
        y, s_dyn = model.adaptive_s(inputs.view(-1, 1, *inputs.shape[-2:]))
        x_r = model.R(y, s_dyn)
        x_r = x_r.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], *inputs.shape[-2:])
        x_r = x_r.permute(0, 2, 1, 3, 4)
        x = model.tokenizer(x_r)
        x_as = None
        for i in range(model.iter_num):
            if use_ckpt and x.requires_grad:
                # Cast to bf16 BEFORE checkpoint so checkpoint saves bf16 (288MB)
                # instead of fp32 (576MB), cutting saved memory by ~50%.
                if amp_dtype is not None:
                    x_ckpt = x.to(dtype=amp_dtype)
                    y_ckpt = y.to(dtype=amp_dtype)
                    s_ckpt = s_dyn.to(dtype=amp_dtype)
                    xa_ckpt = [a.to(dtype=amp_dtype) for a in x_as] if x_as is not None else None
                else:
                    x_ckpt, y_ckpt, s_ckpt, xa_ckpt = x, y, s_dyn, x_as
                x, x_as = _checkpoint(
                    _ckpt_iter,
                    model.gdb[i], model.dst[i],
                    x_ckpt, y_ckpt, model.adaptive_s, model.R, s_ckpt, xa_ckpt,
                    amp_dtype, device,
                    use_reentrant=False)
            else:
                if amp_dtype is not None:
                    with autocast(device.type, dtype=amp_dtype):
                        x, x_as = model.gdb[i](x, y, model.adaptive_s, model.R, s_dyn, x_as)
                        x = model.dst[i](x)
                elif scaler is not None:
                    with autocast(device.type):
                        x, x_as = model.gdb[i](x, y, model.adaptive_s, model.R, s_dyn, x_as)
                        x = model.dst[i](x)
                else:
                    x, x_as = model.gdb[i](x, y, model.adaptive_s, model.R, s_dyn, x_as)
                    x = model.dst[i](x)
        x_mean = model.proj_out(x).permute(0, 2, 1, 3, 4)
        loss = F.mse_loss(x_mean, y_ch)
        loss = loss + 0.01 * model.adaptive_s.ortho_loss()
        return loss / grad_accum

    pbar = tqdm(train_loader, desc='train', dynamic_ncols=True, mininterval=0.5) if is_main_process() else train_loader
    for step_i, (inputs, _) in enumerate(pbar):
        inputs = inputs.to(device)
        y_ch = inputs

        if amp_dtype is not None:
            with autocast(device.type, dtype=amp_dtype):
                loss = _forward(inputs)
        elif scaler is not None:
            with autocast(device.type):
                loss = _forward(inputs)
        else:
            loss = _forward(inputs)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] "
                  f"Non-finite loss at step {step_i}: {loss_val}, stopping.")
            raise RuntimeError(f"Training diverged: loss={loss_val} at step {step_i}")

        if scaler is not None:
            is_last_micro_batch = ((accum_count + 1) % grad_accum == 0)
            if ddp_model is not None and not is_last_micro_batch:
                with ddp_model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
        else:
            # DDP: skip gradient sync for all but the last micro-batch
            is_last_micro_batch = ((accum_count + 1) % grad_accum == 0)
            if ddp_model is not None and not is_last_micro_batch:
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

        sum_loss += loss_val * grad_accum
        if is_main_process() and step_i % 10 == 0:
            pbar.set_postfix({'loss': f'{loss_val * grad_accum:.6f}'})

    # Flush leftover gradients from incomplete last accumulation group
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

    return sum_loss / len(train_loader)


def valid_3d(val_loader, model, device, ddp=False):

    sum_ssim = 0
    sum_lpips = 0
    sum_edge_psnr = 0
    sum_roi_psnr = 0
    sum_roi_ssim = 0
    lpips_count = 0
    n_samples = 0
    total_mse = 0.0
    total_pixels = 0

    # Inference: use bf16 on Ampere+, fp16 on older GPUs
    use_amp = (device.type == 'cuda')
    amp_dtype = None
    if use_amp:
        major = torch.cuda.get_device_capability(device)[0]
        if major >= 8:
            amp_dtype = torch.bfloat16

    model.eval()
    # DDP: move LPIPS model to this rank's GPU (initialized at import time on cuda:0)
    if HAS_LPIPS and _lpips_model is not None and device.type == 'cuda':
        lpips_dev = next(_lpips_model.parameters()).device
        if lpips_dev != device:
            _lpips_model.to(device)
    pbar = tqdm(val_loader, desc='valid', dynamic_ncols=True, mininterval=0.5) if is_main_process() else val_loader
    with torch.inference_mode():
        for step_i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            B, T, C, H, W = inputs.shape
            y_ch = inputs

            if amp_dtype is not None:
                with autocast(device.type, dtype=amp_dtype):
                    outputs = model(y_ch)
            elif use_amp:
                with autocast(device.type):
                    outputs = model(y_ch)
            else:
                outputs = model(y_ch)

            pred = outputs.float()  # convert back to fp32 for metric computation
            target = y_ch
            mse = F.mse_loss(pred, target)
            total_mse += mse.item() * pred.numel()
            total_pixels += pred.numel()
            running_psnr = 10 * log10(1 / (total_mse / total_pixels)) if total_mse > 0 else 100.0

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
    return ret


