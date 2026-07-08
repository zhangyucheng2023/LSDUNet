"""Shared evaluation utilities for all LSDUNet eval scripts.

Extracted from eval.py to eliminate code duplication across:
  eval.py, eval_noise.py, eval_mechanism.py, eval_generalization.py, compute_visualization.py
"""

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from model.model_3d import LSDUNet
from data_processor import collect_sequences, SequenceVolumeDataset

# ─── Mechanism analysis dict keys (shared across eval_mechanism & compute_visualization) ───
KEY_BASIS_WEIGHTS = 'basis_weights'
KEY_SPACE_ATTN = 'space_attn'
KEY_HISTORY_ATTN = 'history_attn'  # (a.k.a. lrta_attn in older code)
KEY_UNCERTAINTY = 'uncertainty'
KEY_ITER_DATA = 'iter_data'
KEY_INTERMEDIATES = 'intermediates'

# ─── Dataset paths (single source of truth) ───
DATASET_PATHS = {
    'toucHD':        './dataset/toucHD/test',
    'tacquad':       './dataset/TacQuad',
    'yuan18':        './dataset/PhyTouch',
    'visgel':        './dataset/visgel',
    'touch_and_go':  './dataset/touch_and_go',
    'tacquad_real':  './dataset/TacQuad_real',
}


def _get_device():
    """Safe device detection for eval scripts (GPU or CPU fallback)."""
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            _ = torch.zeros(1).to(device)
            return device
        except Exception:
            pass
    return torch.device("cpu")


def load_model(cs_ratio, checkpoint_path, iter_num=6, model_dim=64, patch=32,
               use_cache=True):
    """Load LSDUNet from checkpoint, with optional singleton cache for speed.

    Args:
        cs_ratio: CS sensing ratio
        checkpoint_path: path to .pth checkpoint
        iter_num, model_dim, patch: model architecture params
        use_cache: if True, cache the model globally to avoid reloading for
                   different datasets at the same ratio

    Returns:
        (model, device) tuple. Model is in eval() mode.
    """
    device = _get_device()

    if use_cache:
        global _cached_model, _cached_ratio
        if '_cached_model' not in globals():
            globals()['_cached_model'] = None
            globals()['_cached_ratio'] = 10000
        cached = globals()['_cached_model']
        cached_ratio = globals()['_cached_ratio']
        if cached is not None and cs_ratio == cached_ratio:
            return cached, device

    model = LSDUNet(ratio=cs_ratio, iter_num=iter_num,
                    model_dim=model_dim, patch=patch).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt = state if 'model_state_dict' not in state else state['model_state_dict']
    result = model.load_state_dict(ckpt, strict=False)

    if result.missing_keys:
        print(f"[WARN] Checkpoint missing {len(result.missing_keys)} keys "
              f"(new modules will be randomly initialized):")
        groups = {}
        for k in result.missing_keys:
            prefix = k.split('.')[0]
            groups.setdefault(prefix, []).append(k)
        for prefix, keys in groups.items():
            print(f"  - {prefix}: {len(keys)} params (e.g. {keys[0]})")
    if result.unexpected_keys:
        print(f"[WARN] Checkpoint has {len(result.unexpected_keys)} unexpected keys "
              f"(old modules removed, ignored)")
    if not result.missing_keys and not result.unexpected_keys:
        print("[OK] Checkpoint loaded with full parameter match")

    model.eval()

    if use_cache:
        globals()['_cached_model'] = model
        globals()['_cached_ratio'] = cs_ratio

    return model, device


def load_frame(path, image_size=224):
    """Load single frame as RGB normalized tensor [3, H, W]."""
    img = Image.open(path).convert('RGB')
    resize = max(256, image_size)
    img_t = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])(img)
    return img_t


def build_temporal_clip(frame_buffer, center_idx, clip_len=8):
    """Build [clip_len, 3, H, W] clip centered at center_idx, edge replication."""
    n = len(frame_buffer)
    half = clip_len // 2
    indices = []
    for offset in range(-half, half):
        idx = max(0, min(n - 1, center_idx + offset))
        indices.append(idx)
    clip = torch.stack([frame_buffer[i] for i in indices], dim=0)
    clip = clip.unsqueeze(0)  # [1, T, C, H, W]
    return clip, indices


def make_eval_dataloader(val_dir, num_frames=8, image_size=224, batch_size=1):
    """Create a DataLoader for evaluation on a given directory."""
    seqs = collect_sequences(val_dir, min_frames=num_frames)
    if not seqs:
        raise FileNotFoundError(f"No sequences with >= {num_frames} frames in {val_dir}")
    transform = transforms.Compose([
        transforms.Resize((max(256, image_size), max(256, image_size))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    ds = SequenceVolumeDataset(seqs, num_frames=num_frames, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader


def eval_3d_temporal(frame_paths, model, device, return_intermediates=False):
    """Sliding window evaluation with T_clip=8."""
    T_clip = 8
    n_frames = len(frame_paths)
    frame_buffer = [load_frame(p) for p in frame_paths]

    preds = []
    all_intermediates = []

    for i in range(n_frames):
        clip, _ = build_temporal_clip(frame_buffer, i, T_clip)
        with torch.no_grad():
            if return_intermediates:
                out, result = model(clip.to(device), return_intermediates=True)
                if isinstance(result, (list, tuple)):
                    inter_preds = [im[0, T_clip // 2].permute(1, 2, 0).cpu().numpy()
                                   for im in result]
                else:
                    inter_preds = []
                all_intermediates.append(inter_preds)
            else:
                out = model(clip.to(device))
        pred = out[0, T_clip // 2].permute(1, 2, 0).cpu().numpy()
        preds.append(pred)

    targets = [f.permute(1, 2, 0).cpu().numpy() for f in frame_buffer]

    if return_intermediates:
        return preds, targets, all_intermediates
    return preds, targets
