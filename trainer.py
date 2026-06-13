from utils import *

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import laplace as _laplacian
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from metrics import compute_roi_mask

# LPIPS — 首次加载时会从 torch hub 下载权重，内网环境可能需要关闭 SSL 验证
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
    """Compute LPIPS for a batch of 2D images. pred, target: [N, H, W]"""
    if not HAS_LPIPS or _lpips_model is None:
        return None
    pred_t = pred.unsqueeze(1).repeat(1, 3, 1, 1)  # [N, 3, H, W]
    target_t = target.unsqueeze(1).repeat(1, 3, 1, 1)
    if pred_t.max() > 1.5:
        pred_t = pred_t / 255.0
        target_t = target_t / 255.0
    device = next(_lpips_model.parameters()).device
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    with torch.no_grad():
        return _lpips_model(pred_t, target_t).mean().item()


def train_3d(train_loader, model, criterion, optimizer, device):
    model.train()
    sum_loss = 0
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(device.type) if use_amp else None
    pbar = tqdm(train_loader, desc='train', dynamic_ncols=True)

    for inputs, _ in pbar:
        inputs = inputs.to(device)
        B, T, C, H, W = inputs.shape  # [B, T, 1, H, W], already [0,1] from ToTensor
        y_ch = inputs
        optimizer.zero_grad()

        if use_amp:
            with autocast(device.type):
                outputs = model(y_ch)
                loss = criterion(outputs, y_ch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(y_ch)
            loss = criterion(outputs, y_ch)
            loss.backward()
            optimizer.step()

        sum_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return sum_loss / len(train_loader)


def valid_3d(val_loader, model, device):

    sum_ssim = 0
    sum_lpips = 0
    sum_edge_psnr = 0
    sum_roi_psnr = 0
    sum_roi_ssim = 0
    lpips_count = 0
    n_samples = 0
    total_mse = 0.0
    total_pixels = 0

    model.eval()
    pbar = tqdm(val_loader, desc='valid', dynamic_ncols=True)
    with torch.no_grad():
        for _, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            B, T, C, H, W = inputs.shape  # [B, T, 1, H, W], already [0,1] from ToTensor
            y_ch = inputs
            outputs = model(y_ch)
            pred = outputs[0]
            target = y_ch
            mse = F.mse_loss(pred, target)
            total_mse += mse.item() * pred.numel()
            total_pixels += pred.numel()
            running_psnr = 10 * log10(1 / (total_mse / total_pixels)) if total_mse > 0 else 100.0
            for b in range(B):
                for t in range(T):
                    n_samples += 1
                    pred_np = pred[b, t, 0].cpu().numpy().astype(np.float64)
                    tgt_np = target[b, t, 0].cpu().numpy().astype(np.float64)

                    # SSIM
                    sum_ssim += ssim(tgt_np, pred_np, data_range=1)

                    # LPIPS
                    if HAS_LPIPS:
                        lpips_val = compute_lpips_batch(
                            pred[b, t, 0].cpu().unsqueeze(0),
                            target[b, t, 0].cpu().unsqueeze(0))
                        if lpips_val is not None:
                            sum_lpips += lpips_val
                            lpips_count += 1

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


