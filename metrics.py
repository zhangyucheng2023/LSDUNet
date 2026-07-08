"""Metrics: PSNR, SSIM, LPIPS, Edge_PSNR, ROI-PSNR, ROI-SSIM, Temporal_PSNR, efficiency."""
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.filters import threshold_otsu
from scipy import ndimage

# ─── LPIPS (optional) ───
_lpips_fn = None


def _get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import ssl
            import lpips
            import torch
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ssl._create_default_https_context = ssl._create_unverified_context
                _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
                if torch.cuda.is_available():
                    _lpips_fn = _lpips_fn.cuda()
        except Exception:
            _lpips_fn = False
    return _lpips_fn


# ═══════════════════════════════════════════════════════════
# 1. 全局指标
# ═══════════════════════════════════════════════════════════

def compute_psnr(pred, target, data_range=1.0):
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)
    return _psnr(target, pred, data_range=data_range)


def compute_ssim(pred, target, data_range=1.0):
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)
    if pred.ndim == 3:
        return _ssim(target, pred, data_range=data_range, channel_axis=-1)
    return _ssim(target, pred, data_range=data_range)


# ═══════════════════════════════════════════════════════════
# 2. 感知质量 — LPIPS
# ═══════════════════════════════════════════════════════════

def compute_lpips(pred, target):
    """LPIPS perceptual similarity. Requires: pip install lpips.
    pred/target: [H, W] grayscale or [H, W, 3] RGB."""
    lpips_fn = _get_lpips()
    if lpips_fn is False:
        return None
    import torch
    if pred.ndim == 3:
        pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0)
        target_t = torch.from_numpy(target).float().permute(2, 0, 1).unsqueeze(0)
    else:
        pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    if pred_t.max() > 1:
        pred_t = pred_t / 255.0
        target_t = target_t / 255.0
    if pred_t.shape[1] == 1:
        pred_t = pred_t.repeat(1, 3, 1, 1)
        target_t = target_t.repeat(1, 3, 1, 1)
    pred_t = pred_t.to(next(lpips_fn.parameters()).device)
    target_t = target_t.to(next(lpips_fn.parameters()).device)
    with torch.no_grad():
        score = lpips_fn(pred_t, target_t).item()
    return score


# ═══════════════════════════════════════════════════════════
# 3. 接触区域 (ROI) — 触觉先验定制指标
# ═══════════════════════════════════════════════════════════

def compute_roi_mask(img, method='otsu', threshold=None):
    """Binary contact region mask via Otsu thresholding.
    Accepts [H, W] grayscale or [H, W, C] multi-channel (uses luminance)."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img_clipped = np.clip(img, 0, 1).astype(np.float64)

    if method == 'otsu':
        img_uint8 = (img_clipped * 255).astype(np.uint8)
        try:
            thresh = threshold_otsu(img_uint8)
            mask = (img_uint8 > thresh).astype(np.float32)
            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.float32)
            mask = ndimage.binary_closing(mask, structure=np.ones((5, 5))).astype(np.float32)
        except ValueError:
            # All pixel values are identical (e.g., no-contact frame) — fallback to threshold
            threshold = float(img_clipped.mean())
            mask = (img_clipped > threshold).astype(np.float32)
    elif method == 'threshold':
        if threshold is None:
            threshold = np.mean(img_clipped) + 0.3 * np.std(img_clipped)
        mask = (img_clipped > threshold).astype(np.float32)
    else:
        raise ValueError(f"Unknown ROI method: {method}")

    if mask.sum() == 0:
        h, w = img.shape
        y, x = np.ogrid[:h, :w]
        center = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2) < min(h, w) / 3
        mask = center.astype(np.float32)

    return mask


def compute_roi_psnr(pred, target, mask=None, data_range=1.0):
    """PSNR over contact region only. Handles [H, W] or [H, W, C]."""
    if mask is None:
        mask = compute_roi_mask(target)
    if mask.sum() < 1:
        return 0.0
    if pred.ndim == 3:
        mask_b = mask[..., None]
    else:
        mask_b = mask
    pred_roi = pred[mask_b > 0.5]
    target_roi = target[mask_b > 0.5]
    if len(pred_roi) == 0:
        return 0.0
    mse = np.mean((pred_roi - target_roi) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(data_range ** 2 / mse)


def compute_roi_ssim(pred, target, mask=None, data_range=1.0):
    """SSIM over contact region only. Handles [H, W] or [H, W, C]."""
    if mask is None:
        mask = compute_roi_mask(target)
    if mask.sum() < 1:
        return 0.0
    if pred.ndim == 3:
        mask_b = mask[..., None]
    else:
        mask_b = mask
    pred_masked = pred * mask_b + target * (1 - mask_b)
    target_masked = target * mask_b + target * (1 - mask_b)
    if pred.ndim == 3:
        return _ssim(target_masked, pred_masked, data_range=data_range, channel_axis=-1)
    return _ssim(target_masked, pred_masked, data_range=data_range)


# ═══════════════════════════════════════════════════════════
# 4. 综合评估接口
# ═══════════════════════════════════════════════════════════

def evaluate_all(pred, target):
    """Compute all reconstruction quality metrics for a single frame."""
    if pred.max() > 1.5:
        pred = pred / 255.0
    if target.max() > 1.5:
        target = target / 255.0

    # Dimension mismatch (e.g., RGB pred vs grayscale heightmap GT):
    # convert both to luminance so all metric functions receive matching shapes.
    if pred.ndim != target.ndim:
        if pred.ndim == 3:
            pred = pred.mean(axis=-1)
        if target.ndim == 3:
            target = target.mean(axis=-1)

    metrics = {
        'PSNR': compute_psnr(pred, target),
        'SSIM': compute_ssim(pred, target),
        'LPIPS': compute_lpips(pred, target),
        'Edge_PSNR': compute_edge_psnr(pred, target),
    }

    roi_mask = compute_roi_mask(target)
    metrics['ROI_PSNR'] = compute_roi_psnr(pred, target, roi_mask)
    metrics['ROI_SSIM'] = compute_roi_ssim(pred, target, roi_mask)

    return metrics


# ═══════════════════════════════════════════════════════════
# 5. 计算效率指标 (Params / FLOPs / FPS)
# ═══════════════════════════════════════════════════════════

def count_parameters(model):
    """Trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_flops(model, input_shape=(1, 8, 3, 224, 224), device='cpu', verbose=False):
    """Measure FLOPs (G) using fvcore. Returns None if not installed or fails."""
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch
        model.eval()
        x = torch.randn(*input_shape).to(device)
        flops = FlopCountAnalysis(model, x)
        return flops.total() / 1e9
    except Exception as e:
        if verbose:
            print(f"  [info] FLOPs measurement failed: {e}")
        return None


def measure_fps(model, input_shape=(1, 8, 3, 224, 224), device='cpu', warmup=10, repeats=100):
    """Measure inference FPS (clips/sec). Returns 0.0 on failure."""
    import torch
    import time
    try:
        model.eval()
        x = torch.randn(*input_shape).to(device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            with torch.no_grad():
                _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return repeats / elapsed
    except Exception as e:
        print(f"  [warn] FPS measurement failed: {e}")
        return 0.0


def compute_edge_psnr(pred, target, data_range=1.0):
    """PSNR in Laplacian gradient domain for edge preservation.
    Handles [H, W] or [H, W, C] (per-channel averaged)."""
    from scipy.ndimage import laplace as _laplacian
    if pred.ndim == 3:
        ch_psnrs = [compute_edge_psnr(pred[..., c], target[..., c], data_range)
                    for c in range(pred.shape[-1])]
        return float(np.mean(ch_psnrs))
    pred_edge = _laplacian(pred.astype(np.float64))
    target_edge = _laplacian(target.astype(np.float64))
    edge_range = max(pred_edge.max() - pred_edge.min(),
                     target_edge.max() - target_edge.min(), 1e-8)
    mse = np.mean((pred_edge - target_edge) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(edge_range ** 2 / mse)


def compute_temporal_psnr(preds, targets, data_range=1.0):
    """PSNR of frame-to-frame differences (temporal consistency).
    Handles [H, W] or [H, W, C] arrays."""
    if len(preds) < 2:
        return None
    pair_psnrs = []
    for i in range(1, len(preds)):
        pred_diff = np.abs(np.clip(preds[i], 0, data_range) -
                           np.clip(preds[i - 1], 0, data_range))
        target_diff = np.abs(np.clip(targets[i], 0, data_range) -
                             np.clip(targets[i - 1], 0, data_range))
        diff_range = max(pred_diff.max(), target_diff.max(), 1e-8)
        mse = np.mean((pred_diff - target_diff) ** 2)
        if mse == 0:
            pair_psnrs.append(100.0)
        else:
            pair_psnrs.append(10 * np.log10(diff_range ** 2 / mse))
    return float(np.mean(pair_psnrs))


def get_efficiency_metrics(model, input_shape=(1, 8, 3, 224, 224), device='cpu',
                           verbose=True):
    """Get Params (M), FLOPs (G), FPS."""
    params = count_parameters(model)
    flops = measure_flops(model, input_shape, device, verbose=verbose)
    fps = measure_fps(model, input_shape, device)

    if verbose:
        print(f"  [效率] Params: {params:.3f} M", end='')
        if flops is not None:
            print(f"  |  FLOPs: {flops:.3f} G", end='')
        else:
            print(f"  |  FLOPs: N/A", end='')
        print(f"  |  FPS: {fps:.1f} clip/s")

    return {
        'Params': params,
        'FLOPs': flops,
        'FPS': fps,
    }


# ═══════════════════════════════════════════════════════════
# 6. 不确定性校准指标 (ECE / Brier Score)
# ═══════════════════════════════════════════════════════════

def compute_ece(uncertainty, error, n_bins=10):
    """Expected Calibration Error for regression uncertainty.

    For each uncertainty bin, compare predicted std vs observed RMSE.
    Lower ECE = better calibrated uncertainty.

    Args:
        uncertainty: [N] array of predicted std (sqrt(variance))
        error: [N] array of |pred - target|
        n_bins: number of bins for uncertainty

    Returns:
        ece: scalar calibration error
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    if len(uncertainty) == 0:
        return 0.0
    # Clip uncertainty to avoid log issues
    uncertainty = np.clip(uncertainty, 1e-6, None)
    bin_edges = np.linspace(uncertainty.min(), uncertainty.max(), n_bins + 1)
    ece = 0.0
    n_total = len(uncertainty)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (uncertainty >= bin_edges[i]) & (uncertainty <= bin_edges[i + 1])
        else:
            mask = (uncertainty >= bin_edges[i]) & (uncertainty < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        pred_mean = uncertainty[mask].mean()
        obs_rmse = np.sqrt((error[mask] ** 2).mean())
        ece += (mask.sum() / n_total) * abs(pred_mean - obs_rmse)
    return float(ece)


def compute_brier_score(uncertainty, error):
    """Brier score for regression uncertainty.

    BS = mean((uncertainty^2 - error^2)^2)
    Lower BS = better calibrated.

    Args:
        uncertainty: [N] array of predicted std
        error: [N] array of |pred - target|

    Returns:
        brier: scalar
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    if len(uncertainty) == 0:
        return 0.0
    uncertainty = np.clip(uncertainty, 1e-6, None)
    return float(np.mean((uncertainty ** 2 - error ** 2) ** 2))
