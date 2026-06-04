"""
Metrics module for LSDUNet evaluation.
Computes global, low-rank/sparse, ROI, background suppression, and rank-tracking metrics.
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
                # Allow unverified SSL for model download in restricted networks
                ssl._create_default_https_context = ssl._create_unverified_context
                _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
                if torch.cuda.is_available():
                    _lpips_fn = _lpips_fn.cuda()
        except Exception:
            _lpips_fn = False
    return _lpips_fn


# ═══════════════════════════════════════════════════════════
# 1. 全局指标 Global Metrics
# ═══════════════════════════════════════════════════════════

def compute_psnr(pred, target, data_range=1.0):
    """PSNR (Peak Signal-to-Noise Ratio), higher is better."""
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)
    return _psnr(target, pred, data_range=data_range)


def compute_ssim(pred, target, data_range=1.0):
    """SSIM (Structural Similarity), higher is better."""
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)
    return _ssim(target, pred, data_range=data_range)


def compute_lpips(pred, target):
    """
    LPIPS (Learned Perceptual Image Patch Similarity), lower is better.

    Requires: pip install lpips
    Returns float if available, else None.
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
# 2. SVD-based Low-rank / Sparse Decomposition
# ═══════════════════════════════════════════════════════════

def svd_decompose(img, energy_threshold=0.30):
    """
    Decompose a 2D image into low-rank L and sparse S using SVD thresholding.

    Args:
        img: 2D numpy array (H, W)
        energy_threshold: fraction of singular values to retain for L (default 30%)
                          or fraction of energy to retain

    Returns:
        L: low-rank component (H, W)
        S: sparse component = img - L (H, W)
        k: number of singular values retained
    """
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    # Keep top energy_threshold fraction of singular values
    k = max(1, int(len(s) * energy_threshold))
    s_k = s.copy()
    s_k[k:] = 0
    L = U @ np.diag(s_k) @ Vt
    S = img - L
    return L, S, k


def effective_rank(img, energy_threshold=0.95):
    """
    Effective rank: number of singular values needed to explain
    `energy_threshold` fraction of total variance.

    Args:
        img: 2D numpy array (H, W)
        energy_threshold: fraction of energy to retain (default 0.95)

    Returns:
        rank: effective rank (integer)
    """
    _, s, _ = np.linalg.svd(img, full_matrices=False)
    s2 = s ** 2
    cumsum = np.cumsum(s2)
    total = cumsum[-1]
    if total == 0:
        return 0
    rank = int(np.searchsorted(cumsum, energy_threshold * total) + 1)
    return min(rank, len(s))


# ═══════════════════════════════════════════════════════════
# 3. Low-rank / Sparse Component Metrics
# ═══════════════════════════════════════════════════════════

def compute_lowrank_sparse_metrics(pred, target, energy_threshold=0.30):
    """
    Compute PSNR/SSIM/RelErr for low-rank (L) and sparse (S) components.

    Args:
        pred, target: 2D numpy arrays (H, W)
        energy_threshold: fraction of SVs for low-rank component

    Returns:
        dict with keys: PSNR_L, SSIM_L, RelErr_L, PSNR_S, SSIM_S, RelErr_S, k_L, k_S
    """
    L_pred, S_pred, k_pred = svd_decompose(pred, energy_threshold)
    L_target, S_target, k_target = svd_decompose(target, energy_threshold)

    psnr_L = compute_psnr(L_pred, L_target)
    ssim_L = compute_ssim(L_pred, L_target)

    # RelErr = ||pred - target||_F / ||target||_F (Frobenius norm ratio)
    rel_err_L = np.linalg.norm(L_pred - L_target, 'fro') / (np.linalg.norm(L_target, 'fro') + 1e-8)

    psnr_S = compute_psnr(S_pred, S_target)
    ssim_S = compute_ssim(S_pred, S_target)
    rel_err_S = np.linalg.norm(S_pred - S_target, 'fro') / (np.linalg.norm(S_target, 'fro') + 1e-8)

    return {
        'PSNR_L': psnr_L, 'SSIM_L': ssim_L, 'RelErr_L': rel_err_L,
        'PSNR_S': psnr_S, 'SSIM_S': ssim_S, 'RelErr_S': rel_err_S,
        'k_L': k_pred, 'k_target': k_target,
    }


# ═══════════════════════════════════════════════════════════
# 4. ROI (Region of Interest) — 接触区域
# ═══════════════════════════════════════════════════════════

def compute_roi_mask(img, method='otsu', threshold=None):
    """
    Compute binary mask for the contact region (ROI) in a tactile image.

    Args:
        img: 2D numpy array (H, W), normalized to [0, 1]
        method: 'otsu' or 'threshold'
        threshold: if method='threshold', use this value (default: mean + 0.3*std)

    Returns:
        mask: binary 2D array, 1 = contact region, 0 = background
    """
    img_clipped = np.clip(img, 0, 1).astype(np.float64)

    if method == 'otsu':
        # Scale to [0, 255] for Otsu
        img_uint8 = (img_clipped * 255).astype(np.uint8)
        thresh = threshold_otsu(img_uint8)
        mask = (img_uint8 > thresh).astype(np.float32)
        # Clean up mask with morphological operations
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.float32)
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5))).astype(np.float32)
    elif method == 'threshold':
        if threshold is None:
            threshold = np.mean(img_clipped) + 0.3 * np.std(img_clipped)
        mask = (img_clipped > threshold).astype(np.float32)
    else:
        raise ValueError(f"Unknown ROI method: {method}")

    # If mask is all zeros, fallback to center region
    if mask.sum() == 0:
        h, w = img.shape
        y, x = np.ogrid[:h, :w]
        center = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2) < min(h, w) / 3
        mask = center.astype(np.float32)

    return mask


def compute_roi_psnr(pred, target, mask=None, data_range=1.0):
    """PSNR computed only within the ROI (contact region)."""
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
    """SSIM computed only within the ROI, approximated by masking."""
    if mask is None:
        mask = compute_roi_mask(target)
    if mask.sum() < 1:
        return 0.0
    # Apply mask: set background to 0 in both images
    pred_masked = pred * mask
    target_masked = target * mask
    return _ssim(target_masked, pred_masked, data_range=data_range)


# ═══════════════════════════════════════════════════════════
# 5. SCRG / BSF — 背景抑制效果
# ═══════════════════════════════════════════════════════════

def compute_scr(img, mask):
    """
    Signal-to-Clutter Ratio.
    SCR = mean(|signal_in_ROI|²) / mean(|background|²)
    """
    roi = mask > 0.5
    bg = ~roi
    signal_power = np.mean(img[roi] ** 2) if roi.sum() > 0 else 0
    bg_power = np.mean(img[bg] ** 2) if bg.sum() > 0 else 1e-8
    if bg_power == 0:
        return 100.0
    return signal_power / bg_power


def compute_scr_gain(pred, target, mask=None):
    """
    Signal-to-Clutter Ratio Gain.
    SCRG = SCR(pred) / SCR(target)

    SCRG > 1 means the reconstruction improves signal relative to background.
    """
    if mask is None:
        mask = compute_roi_mask(target)
    scr_pred = compute_scr(pred, mask)
    scr_target = compute_scr(target, mask)
    if scr_target == 0:
        return 0.0
    return scr_pred / scr_target


def compute_bsf(pred, target, mask=None):
    """
    Background Suppression Factor.
    BSF = std(background_target) / std(background_pred)

    BSF > 1 means background noise is reduced by reconstruction.
    """
    if mask is None:
        mask = compute_roi_mask(target)
    bg = mask < 0.5
    if bg.sum() < 2:
        return 1.0
    bg_std_target = np.std(target[bg])
    bg_std_pred = np.std(pred[bg])
    if bg_std_pred == 0:
        return 100.0
    return bg_std_target / bg_std_pred


# ═══════════════════════════════════════════════════════════
# 6. rank(L̂) vs. 迭代次数
# ═══════════════════════════════════════════════════════════

def compute_rank_vs_iterations(intermediate_preds, energy_threshold=0.30):
    """
    Compute effective rank of the **low-rank component L̂** at each iteration.

    For each intermediate prediction, first decompose via SVD into L + S,
    then compute effective_rank of L.

    Args:
        intermediate_preds: list of 2D numpy arrays, one per iteration
        energy_threshold: fraction of SVs for low-rank decomposition

    Returns:
        ranks: list of effective ranks of L at each iteration
    """
    ranks = []
    for pred in intermediate_preds:
        L, _, _ = svd_decompose(pred, energy_threshold=energy_threshold)
        rank = effective_rank(L, energy_threshold=0.95)
        ranks.append(rank)
    return ranks


# ═══════════════════════════════════════════════════════════
# 7. 接触力 RMSE / 滑动检测（需力传感器/滑动标注数据）
# ═══════════════════════════════════════════════════════════

def compute_contact_force_rmse(pred_force, target_force):
    """
    Contact force RMSE.

    Args:
        pred_force, target_force: force values (scalar or array)

    Returns:
        rmse: root mean squared error
    """
    return np.sqrt(np.mean((np.array(pred_force) - np.array(target_force)) ** 2))


def detect_slip_displacement(pred_frames, threshold=5.0):
    """
    Simple slip detection via frame-to-frame displacement.
    Uses optical-flow-like center-of-mass shift between consecutive frames.

    Args:
        pred_frames: list of 2D numpy arrays (T, H, W), consecutive predicted frames
        threshold: displacement threshold (pixels) for slip detection

    Returns:
        displacements: list of displacement magnitudes between consecutive frames
        slip_detected: list of bool, True if displacement > threshold
    """
    displacements = []
    slip_detected = []
    for i in range(1, len(pred_frames)):
        prev = pred_frames[i - 1]
        curr = pred_frames[i]

        # Center of mass shift as proxy for slip
        prev_mask = prev > (prev.mean() + 0.3 * prev.std())
        curr_mask = curr > (curr.mean() + 0.3 * curr.std())

        if prev_mask.sum() > 0 and curr_mask.sum() > 0:
            cy_prev, cx_prev = ndimage.center_of_mass(prev_mask)
            cy_curr, cx_curr = ndimage.center_of_mass(curr_mask)
            disp = np.sqrt((cy_curr - cy_prev) ** 2 + (cx_curr - cx_prev) ** 2)
        else:
            disp = 0.0

        displacements.append(disp)
        slip_detected.append(disp > threshold)

    return displacements, slip_detected


# ═══════════════════════════════════════════════════════════
# Comprehensive evaluation wrapper
# ═══════════════════════════════════════════════════════════

def evaluate_all(pred, target, intermediate_preds=None, force_pair=None):
    """
    Compute all metrics for a single image pair.

    Args:
        pred: 2D numpy array (H, W), predicted image (range [0, 1] or [0, 255])
        target: 2D numpy array (H, W), ground truth image
        intermediate_preds: optional list of 2D arrays, intermediate predictions per iteration
        force_pair: optional tuple (pred_force, target_force)

    Returns:
        metrics: dict with all computed metrics
    """
    # Normalize to [0, 1]
    if pred.max() > 1.5:
        pred = pred / 255.0
    if target.max() > 1.5:
        target = target / 255.0

    metrics = {}

    # --- 1. 全局指标 ---
    metrics['PSNR'] = compute_psnr(pred, target)
    metrics['SSIM'] = compute_ssim(pred, target)
    metrics['LPIPS'] = compute_lpips(pred, target)

    # --- 2. 低秩/稀疏分量 ---
    ls_metrics = compute_lowrank_sparse_metrics(pred, target)
    metrics.update(ls_metrics)

    # --- 3. ROI ---
    roi_mask = compute_roi_mask(target)
    metrics['ROI_PSNR'] = compute_roi_psnr(pred, target, roi_mask)
    metrics['ROI_SSIM'] = compute_roi_ssim(pred, target, roi_mask)

    # --- 4. SCRG / BSF ---
    metrics['SCRG'] = compute_scr_gain(pred, target, roi_mask)
    metrics['BSF'] = compute_bsf(pred, target, roi_mask)

    # --- 5. rank(L̂) vs. iterations ---
    if intermediate_preds is not None and len(intermediate_preds) > 0:
        metrics['ranks'] = compute_rank_vs_iterations(intermediate_preds)
    else:
        metrics['ranks'] = []

    # --- 6. 接触力 RMSE ---
    if force_pair is not None:
        metrics['Force_RMSE'] = compute_contact_force_rmse(force_pair[0], force_pair[1])
    else:
        metrics['Force_RMSE'] = None

    return metrics


# ─── Formatting helpers ───

def format_metrics_row(metrics, prefix=''):
    """Format a single metrics dict into a readable string."""
    parts = []
    if 'PSNR' in metrics:
        parts.append(f"PSNR={metrics['PSNR']:.2f}")
    if 'SSIM' in metrics:
        parts.append(f"SSIM={metrics['SSIM']:.4f}")
    if 'LPIPS' in metrics and metrics['LPIPS'] is not None:
        parts.append(f"LPIPS={metrics['LPIPS']:.4f}")
    if 'ROI_PSNR' in metrics:
        parts.append(f"ROI-PSNR={metrics['ROI_PSNR']:.2f}")
    if 'ROI_SSIM' in metrics:
        parts.append(f"ROI-SSIM={metrics['ROI_SSIM']:.4f}")
    if 'PSNR_L' in metrics:
        parts.append(f"PSNR_L={metrics['PSNR_L']:.2f}  SSIM_L={metrics['SSIM_L']:.4f}  RelErr(L)={metrics['RelErr_L']:.4f}")
    if 'PSNR_S' in metrics:
        parts.append(f"PSNR_S={metrics['PSNR_S']:.2f}  SSIM_S={metrics['SSIM_S']:.4f}  RelErr(S)={metrics['RelErr_S']:.4f}")
    if 'SCRG' in metrics:
        parts.append(f"SCRG={metrics['SCRG']:.3f}  BSF={metrics['BSF']:.3f}")
    if 'Force_RMSE' in metrics and metrics['Force_RMSE'] is not None:
        parts.append(f"Force_RMSE={metrics['Force_RMSE']:.4f}")
    if prefix:
        return f"  {prefix} " + "  ".join(parts)
    return "  ".join(parts)


def format_rank_summary(all_ranks_per_sample):
    """
    Compute average rank across samples for each iteration.

    Args:
        all_ranks_per_sample: list of lists, each inner list is ranks per iteration

    Returns:
        avg_ranks: average rank per iteration
    """
    if not all_ranks_per_sample:
        return []
    max_iters = max(len(r) for r in all_ranks_per_sample)
    avg_ranks = []
    for i in range(max_iters):
        iter_vals = [r[i] for r in all_ranks_per_sample if i < len(r)]
        if iter_vals:
            avg_ranks.append(np.mean(iter_vals))
        else:
            avg_ranks.append(0)
    return avg_ranks