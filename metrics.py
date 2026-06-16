"""
Metrics module for LSDUNet evaluation.
- Global: PSNR, SSIM
- Perceptual & local: LPIPS, ROI-PSNR, ROI-SSIM
- Edge preservation: Edge_PSNR (Laplacian domain)
- Temporal consistency: Temporal_PSNR (frame-difference domain)
- Efficiency: Params, FLOPs, FPS
"""
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
    return _ssim(target, pred, data_range=data_range)


# ═══════════════════════════════════════════════════════════
# 2. 感知质量 — LPIPS
# ═══════════════════════════════════════════════════════════

def compute_lpips(pred, target):
    """
    LPIPS (Learned Perceptual Image Patch Similarity), lower is better.
    Requires: pip install lpips
    """
    lpips_fn = _get_lpips()
    if lpips_fn is False:
        return None
    import torch
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    if pred_t.max() > 1:
        pred_t = pred_t / 255.0
        target_t = target_t / 255.0
    pred_t = pred_t.repeat(1, 3, 1, 1).to(next(lpips_fn.parameters()).device)
    target_t = target_t.repeat(1, 3, 1, 1).to(next(lpips_fn.parameters()).device)
    with torch.no_grad():
        score = lpips_fn(pred_t, target_t).item()
    return score


# ═══════════════════════════════════════════════════════════
# 3. 接触区域 (ROI) — 触觉先验定制指标
# ═══════════════════════════════════════════════════════════

def compute_roi_mask(img, method='otsu', threshold=None):
    """
    触觉图像接触区域二值掩码。

    Args:
        img: 2D numpy array (H, W), normalized to [0, 1]
        method: 'otsu' or 'threshold'
        threshold: manual threshold for method='threshold'

    Returns:
        mask: binary 2D array, 1 = contact region
    """
    img_clipped = np.clip(img, 0, 1).astype(np.float64)

    if method == 'otsu':
        img_uint8 = (img_clipped * 255).astype(np.uint8)
        thresh = threshold_otsu(img_uint8)
        mask = (img_uint8 > thresh).astype(np.float32)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.float32)
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5))).astype(np.float32)
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
    """PSNR 仅计算接触区域 (ROI) 内像素。"""
    if mask is None:
        mask = compute_roi_mask(target)
    if mask.sum() < 1:
        return 0.0
    pred_roi = pred[mask > 0.5]
    target_roi = target[mask > 0.5]
    if len(pred_roi) == 0:
        return 0.0
    mse = np.mean((pred_roi - target_roi) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(data_range ** 2 / mse)


def compute_roi_ssim(pred, target, mask=None, data_range=1.0):
    """SSIM 仅计算接触区域 (ROI) 内像素。非 ROI 区域用 target 填充避免拉低局部窗口统计。"""
    if mask is None:
        mask = compute_roi_mask(target)
    if mask.sum() < 1:
        return 0.0
    pred_masked = pred * mask + target * (1 - mask)
    target_masked = target * mask + target * (1 - mask)
    return _ssim(target_masked, pred_masked, data_range=data_range)


# ═══════════════════════════════════════════════════════════
# 4. 综合评估接口
# ═══════════════════════════════════════════════════════════

def evaluate_all(pred, target):
    """
    计算单帧的所有重建质量指标。

    Args:
        pred: 2D numpy array (H, W), 预测图像
        target: 2D numpy array (H, W), 真实值

    Returns:
        dict: PSNR, SSIM, LPIPS, ROI_PSNR, ROI_SSIM
    """
    if pred.max() > 1.5:
        pred = pred / 255.0
    if target.max() > 1.5:
        target = target / 255.0

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
    """可训练参数量 (M)。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_flops(model, input_shape=(1, 8, 1, 96, 96), device='cpu', verbose=False):
    """
    使用 fvcore 测量 FLOPs (G)。
    若未安装 fvcore，返回 None。
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch
        model.eval()
        x = torch.randn(*input_shape).to(device)
        flops = FlopCountAnalysis(model, x)
        return flops.total() / 1e9
    except ImportError:
        if verbose:
            print("  [info] fvcore not installed, FLOPs = N/A.  Install via: pip install fvcore")
        return None


def measure_fps(model, input_shape=(1, 8, 1, 96, 96), device='cpu', warmup=10, repeats=100):
    """
    测量推理帧率 FPS (frames per second)。
    输入 clip = [1, 8, 1, 96, 96]，模型输出中取中间帧，所以 FPS 即 clip/s。
    """
    import torch
    import time
    model.eval()
    x = torch.randn(*input_shape).to(device)
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # timed runs
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return repeats / elapsed


# ═══════════════════════════════════════════════════════════
# 6. 边缘保留 — Edge PSNR (触觉压力边界保真度)
# ═══════════════════════════════════════════════════════════

def compute_edge_psnr(pred, target, data_range=1.0):
    """
    在 Laplacian 梯度域计算 PSNR，衡量压力边界/纹理边缘的保真度。
    触觉重建的核心难点不是平坦区，而是接触边界的锐利程度。
    """
    from scipy.ndimage import laplace as _laplacian
    pred_edge = _laplacian(pred.astype(np.float64))
    target_edge = _laplacian(target.astype(np.float64))
    edge_range = max(pred_edge.max() - pred_edge.min(),
                     target_edge.max() - target_edge.min(), 1e-8)
    # 将边缘图映射到 [0, edge_range] 做 PSNR
    mse = np.mean((pred_edge - target_edge) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(edge_range ** 2 / mse)


# ═══════════════════════════════════════════════════════════
# 7. 时序一致性 — Temporal PSNR (证明 DSTTimeBlock 有效性)
# ═══════════════════════════════════════════════════════════

def compute_temporal_psnr(preds, targets, data_range=1.0):
    """
    对重建帧序列计算帧间差分的 PSNR。
    衡量重建序列的时序平滑性，证明 DSTTimeBlock 确实在利用时间信息。

    Args:
        preds: list of 2D numpy arrays, 重建帧序列
        targets: list of 2D numpy arrays, GT 帧序列

    Returns:
        float: Temporal PSNR, 取所有相邻帧对的平均值
    """
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


# ═══════════════════════════════════════════════════════════
# 8. 一站式效率获取接口
# ═══════════════════════════════════════════════════════════

def get_efficiency_metrics(model, input_shape=(1, 8, 1, 96, 96), device='cpu',
                           verbose=True):
    """
    一站式获取计算效率指标。

    Returns:
        dict: Params (M), FLOPs (G, optional), FPS
    """
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
