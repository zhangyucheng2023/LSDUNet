"""LSDUNet unified evaluation entry point.

All evaluation tasks are accessible via --mode:

  standard          Full evaluation on all datasets (default)
  cross-domain      High-resolution (448) + cross-domain generalization with ECE/Brier
  noise             Noise robustness + ConvTokenizer3D vs LinearTokenizer ablation
  interpret         Mechanism analysis & visualization (use --submode export|render|all)

Usage:
  python eval.py                                              # standard eval
  python eval.py --mode cross-domain --image_size 448         # cross-domain generalization
  python eval.py --mode noise --full                          # noise + ablation
  python eval.py --mode interpret --submode export --ratio 0.10
  python eval.py --mode interpret --submode render --ckpt trained_model/lsdunet_0.10.pth
  python eval.py --mode interpret --submode all --ckpt trained_model/lsdunet_0.10.pth
"""
import os
import sys
import csv
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from data_processor import (collect_ycb_by_object, collect_sequences,
                            collect_visgel_sequences, SequenceVolumeDataset)
from model.model_3d import LSDUNet
from metrics import (evaluate_all, get_efficiency_metrics, compute_temporal_psnr,
                     compute_roi_mask, compute_ece, compute_brier_score, _get_lpips)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# Section 1: Shared evaluation utilities
# ═══════════════════════════════════════════════════════════════════

# Mechanism analysis dict keys
KEY_BASIS_WEIGHTS = 'basis_weights'
KEY_SPACE_ATTN = 'space_attn'
KEY_HISTORY_ATTN = 'history_attn'
KEY_UNCERTAINTY = 'uncertainty'
KEY_ITER_DATA = 'iter_data'
KEY_INTERMEDIATES = 'intermediates'

# Dataset paths (single source of truth)
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

    Returns (model, device) tuple. Model is in eval() mode.
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
    """Build [1, clip_len, 3, H, W] clip centered at center_idx, edge replication."""
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


# ═══════════════════════════════════════════════════════════════════
# Section 2: Standard evaluation helpers
# ═══════════════════════════════════════════════════════════════════

PER_OBJECT_COLUMNS = [
    'Object', 'Mode', 'N', 'Ratio',
    'PSNR', 'SSIM', 'LPIPS', 'Edge_PSNR',
    'ROI_PSNR', 'ROI_SSIM', 'Temporal_PSNR',
    'Params(M)', 'FLOPs(G)', 'FPS',
]
PER_SAMPLE_COLUMNS = PER_OBJECT_COLUMNS + ['Image']

_FRAME_KEYS = ['PSNR', 'SSIM', 'LPIPS', 'Edge_PSNR', 'ROI_PSNR', 'ROI_SSIM']


def _round_v(val, ndigits=4):
    if val is None:
        return 'N/A'
    return round(val, ndigits)


def write_object_row(writer, obj_name, mode, n, ratio, metrics, eff=None):
    row = [
        obj_name, mode, n, ratio,
        _round_v(metrics['PSNR']), _round_v(metrics['SSIM']),
        _round_v(metrics['LPIPS']), _round_v(metrics['Edge_PSNR']),
        _round_v(metrics['ROI_PSNR']), _round_v(metrics['ROI_SSIM']),
        _round_v(metrics.get('Temporal_PSNR')),
        _round_v(eff['Params'], 3) if eff else 'N/A',
        _round_v(eff['FLOPs'], 3) if (eff and eff['FLOPs'] is not None) else 'N/A',
        _round_v(eff['FPS'], 1) if eff else 'N/A',
    ]
    writer.writerow(row)


def write_sample_row(writer, obj_name, mode, impath, ratio, m, eff=None,
                     temporal_psnr=None):
    row = [
        obj_name, mode, 1, ratio,
        _round_v(m['PSNR']), _round_v(m['SSIM']),
        _round_v(m['LPIPS']), _round_v(m['Edge_PSNR']),
        _round_v(m['ROI_PSNR']), _round_v(m['ROI_SSIM']),
        _round_v(temporal_psnr),
        _round_v(eff['Params'], 3) if eff else 'N/A',
        _round_v(eff['FLOPs'], 3) if (eff and eff['FLOPs'] is not None) else 'N/A',
        _round_v(eff['FPS'], 1) if eff else 'N/A',
        impath,
    ]
    writer.writerow(row)


def _accumulate_frame_metrics(accum, metrics):
    for k in _FRAME_KEYS:
        if metrics[k] is not None:
            accum.setdefault(k, 0.0)
            accum[k] += metrics[k]
        else:
            accum.setdefault(k + '_miss', 0)
            accum[k + '_miss'] += 1
    return accum


def _finalize_frame_metrics(accum, n):
    result = {}
    for k in _FRAME_KEYS:
        miss = accum.get(k + '_miss', 0)
        if n > miss:
            result[k] = accum.get(k, 0.0) / (n - miss)
        else:
            result[k] = None
    return result


def _print_subject(name, n, accum, temporal_psnr=None):
    avg = _finalize_frame_metrics(accum, n)
    print(f"\n{'─' * 100}")
    print(f"  {name}  (N={n})")
    print(f"  {'─' * 80}")
    print(f"  [全局]        PSNR={avg['PSNR']:.2f} dB  |  SSIM={avg['SSIM']:.4f}"
          + (f"  |  LPIPS={avg['LPIPS']:.4f}" if avg['LPIPS'] is not None
             else f"  |  LPIPS=N/A"))
    print(f"  [边缘保留]    Edge-PSNR={avg['Edge_PSNR']:.2f} dB")
    print(f"  [接触区域]    ROI-PSNR={avg['ROI_PSNR']:.2f} dB  |  ROI-SSIM={avg['ROI_SSIM']:.4f}")
    if temporal_psnr is not None:
        print(f"  [时序一致]    Temporal-PSNR={temporal_psnr:.2f} dB")
    return avg


def _run_temporal_eval_seqs(sequences, mode_tag, label, model, device,
                            sampling_rate, eff, obj_writer, sample_writer,
                            summary_writer):
    """对已加载的序列字典 ({dir: [frame_paths]}) 执行时序滑动窗口评估。"""
    if not sequences:
        return

    print("\n\n" + "=" * 100)
    print(f"=== {label} Temporal Evaluation [ratio={sampling_rate}] ===")
    print("=" * 100)

    t_accum = {}
    t_total_frames = 0

    for seq_dir, frame_paths in sorted(sequences.items()):
        seq_name = os.path.basename(seq_dir)
        n = len(frame_paths)

        preds, targets = eval_3d_temporal(frame_paths, model, device)
        t_psnr = compute_temporal_psnr(preds, targets)

        s_accum = {}
        for i in range(n):
            metrics = evaluate_all(preds[i], targets[i])
            _accumulate_frame_metrics(s_accum, metrics)
            _accumulate_frame_metrics(t_accum, metrics)
            write_sample_row(sample_writer, seq_name, mode_tag,
                             frame_paths[i], sampling_rate,
                             metrics, eff, temporal_psnr=t_psnr)

        avg = _print_subject(f"Seq: {seq_name}", n, s_accum,
                             temporal_psnr=t_psnr)
        avg['Temporal_PSNR'] = t_psnr
        write_object_row(obj_writer, seq_name, mode_tag, n,
                         sampling_rate, avg, eff)
        t_total_frames += n

    if t_total_frames > 0:
        overall_t = _finalize_frame_metrics(t_accum, t_total_frames)
        overall_t['Temporal_PSNR'] = None
        write_object_row(obj_writer, 'OVERALL', mode_tag,
                         t_total_frames, sampling_rate, overall_t, eff)
        summary_writer.writerow([
            'OVERALL', mode_tag, t_total_frames, sampling_rate,
            _round_v(overall_t['PSNR']), _round_v(overall_t['SSIM']),
            _round_v(overall_t['LPIPS']), _round_v(overall_t['Edge_PSNR']),
            _round_v(overall_t['ROI_PSNR']), _round_v(overall_t['ROI_SSIM']),
            'N/A',
            _round_v(eff['Params'], 3),
            _round_v(eff['FLOPs'], 3) if eff['FLOPs'] is not None else 'N/A',
            _round_v(eff['FPS'], 1),
        ])

        print(f"\n{'═' * 100}")
        print(f"  [ratio={sampling_rate}] {label} OVERALL  (frames={t_total_frames})")
        print(f"  {'═' * 100}")
        print(f"  [全局]        PSNR={overall_t['PSNR']:.2f} dB  |  SSIM={overall_t['SSIM']:.4f}"
              + (f"  |  LPIPS={overall_t['LPIPS']:.4f}" if overall_t['LPIPS'] is not None
                 else f"  |  LPIPS=N/A"))
        print(f"  [边缘保留]    Edge-PSNR={overall_t['Edge_PSNR']:.2f} dB")
        print(f"  [接触区域]    ROI-PSNR={overall_t['ROI_PSNR']:.2f} dB  |  ROI-SSIM={overall_t['ROI_SSIM']:.4f}")
        print(f"{'═' * 100}")


# ═══════════════════════════════════════════════════════════════════
# Section 3: Standard mode — full evaluation on all datasets
# ═══════════════════════════════════════════════════════════════════

def run_standard_eval(args):
    """Full evaluation across all datasets and CS ratios."""
    cs_ratios = [0.01, 0.04, 0.10, 0.25, 0.50]

    # ─── 数据集定义 ───
    # TacQuad 仿真物体 (Physiclear 物体多视角序列)
    tacquad_root = './dataset/tacquad/train'
    tacquad_objects = collect_ycb_by_object(tacquad_root)
    tactile_objs = {k: v for k, v in tacquad_objects.items() if not k.endswith('_gt')}
    gt_objs = {k.replace('_gt', ''): v for k, v in tacquad_objects.items() if k.endswith('_gt')}

    for obj_name in sorted(tactile_objs.keys())[:5]:
        imgs = tactile_objs[obj_name]
        print(f"[TacQuad|{obj_name}] {len(imgs)} imgs")
    print(f"[TacQuad] Total objects: {len(tactile_objs)}")

    # TacQuad 真实机器人采集数据
    tacquad_real_root = './dataset/tacquad/test/real/gelsight'
    tacquad_real_images = sorted(
        [os.path.join(tacquad_real_root, f) for f in os.listdir(tacquad_real_root)
         if f.lower().endswith(('.jpg', '.png')) and 'Zone.Identifier' not in f]
    ) if os.path.isdir(tacquad_real_root) else []
    if tacquad_real_images:
        print(f"[TacQuad-Real] {len(tacquad_real_images)} frames")

    # Yuan18 布料触觉时序数据 (test split)
    yuan18_test_root = './dataset/yuan18/test'
    yuan18_test_seqs = {}
    if os.path.isdir(yuan18_test_root):
        for seq_name in sorted(os.listdir(yuan18_test_root)):
            seq_dir = os.path.join(yuan18_test_root, seq_name)
            if os.path.isdir(seq_dir):
                frames = sorted(
                    [os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
                     if f.lower().endswith(('.jpg', '.png'))],
                    key=lambda p: int(''.join(filter(str.isdigit,
                                       os.path.splitext(os.path.basename(p))[0])) or 0)
                )
                if len(frames) >= 8:
                    yuan18_test_seqs[seq_dir] = frames
        if yuan18_test_seqs:
            total_frames = sum(len(v) for v in yuan18_test_seqs.values())
            print(f"[Yuan18] {len(yuan18_test_seqs)} test sequences, {total_frames} frames")

    # VisGel 触觉时序数据 (全部 recording)
    visgel_root = './dataset/visgel/images/touch'
    visgel_test_seqs = {}
    if os.path.isdir(visgel_root):
        visgel_test_seqs = collect_visgel_sequences(visgel_root, min_frames=8)
        if visgel_test_seqs:
            total_frames = sum(len(v) for v in visgel_test_seqs.values())
            print(f"[VisGel] {len(visgel_test_seqs)} test sequences, {total_frames} frames")

    # Touch and Go 触觉时序数据 (全部序列)
    tag_root = './dataset/touch_and_go'
    tag_test_seqs = collect_sequences(tag_root, min_frames=8) if os.path.isdir(tag_root) else {}
    if tag_test_seqs:
        total_frames = sum(len(v) for v in tag_test_seqs.values())
        print(f"[TouchAndGo] {len(tag_test_seqs)} test sequences, {total_frames} frames")

    # ─── 汇总: 所有采样率的 OVERALL 对比 ───
    summary_csv = open('./eval_results/summary_all_ratios.csv', 'w', newline='')
    summary_writer = csv.writer(summary_csv)
    summary_writer.writerow(PER_OBJECT_COLUMNS)

    for sampling_rate in cs_ratios:
        ckpt = f'./trained_model/lsdunet_{sampling_rate}.pth'

        if not os.path.exists(ckpt):
            print(f"\n[SKIP] Checkpoint not found: {ckpt}")
            continue

        model, device = load_model(sampling_rate, ckpt)
        print(f"\n{'═' * 100}")
        print(f"Model: LSDUNet | Ratio: {sampling_rate}")
        print(f"Checkpoint: {ckpt}")
        print(f"{'═' * 100}")

        # ─── 一次性计算效率指标 ───
        eff = get_efficiency_metrics(model, device=device)

        # ─── 输出目录 ───
        out_dir = f'./eval_results/ratio_{sampling_rate}'
        os.makedirs(out_dir, exist_ok=True)

        obj_csv = open(os.path.join(out_dir, 'per_object.csv'), 'w', newline='')
        obj_writer = csv.writer(obj_csv)
        obj_writer.writerow(PER_OBJECT_COLUMNS)

        sample_csv = open(os.path.join(out_dir, 'per_sample.csv'), 'w', newline='')
        sample_writer = csv.writer(sample_csv)
        sample_writer.writerow(PER_SAMPLE_COLUMNS)

        # ═══════════════════════════════════════════════════════════════
        # Mode 1: self-reference reconstruction
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 100)
        print(f"=== Per-object evaluation (self-reference) [ratio={sampling_rate}] ===")
        print("=" * 100)

        m1_accum = {}
        m1_total = 0

        for obj_name in sorted(tactile_objs.keys()):
            imgs = tactile_objs[obj_name]  # 评估全序列，不再限制前20帧
            n = len(imgs)

            accum = {}
            preds, targets = eval_3d_temporal(imgs, model, device)
            t_psnr = compute_temporal_psnr(preds, targets)

            for i, impath in enumerate(imgs):
                metrics = evaluate_all(preds[i], targets[i])
                _accumulate_frame_metrics(accum, metrics)
                _accumulate_frame_metrics(m1_accum, metrics)
                write_sample_row(sample_writer, obj_name, 'self-ref', impath,
                                 sampling_rate, metrics, eff, temporal_psnr=t_psnr)

            avg = _print_subject(f"Object: {obj_name}", n, accum, temporal_psnr=t_psnr)
            avg['Temporal_PSNR'] = t_psnr
            write_object_row(obj_writer, obj_name, 'self-ref', n,
                             sampling_rate, avg, eff)
            m1_total += n

        # Mode 1 全局汇总
        if m1_total > 0:
            overall_m1 = _finalize_frame_metrics(m1_accum, m1_total)
            overall_m1['Temporal_PSNR'] = None
            write_object_row(obj_writer, 'OVERALL', 'self-ref', m1_total,
                             sampling_rate, overall_m1, eff)
            summary_writer.writerow([
                'OVERALL', 'self-ref', m1_total,
                sampling_rate,
                _round_v(overall_m1['PSNR']), _round_v(overall_m1['SSIM']),
                _round_v(overall_m1['LPIPS']), _round_v(overall_m1['Edge_PSNR']),
                _round_v(overall_m1['ROI_PSNR']), _round_v(overall_m1['ROI_SSIM']),
                'N/A',
                _round_v(eff['Params'], 3),
                _round_v(eff['FLOPs'], 3) if eff['FLOPs'] is not None else 'N/A',
                _round_v(eff['FPS'], 1),
            ])

            print(f"\n{'═' * 100}")
            print(f"  [ratio={sampling_rate}] SELF-REF OVERALL  (frames={m1_total})")
            print(f"  {'═' * 100}")
            print(f"  [全局]        PSNR={overall_m1['PSNR']:.2f} dB  |  SSIM={overall_m1['SSIM']:.4f}"
                  + (f"  |  LPIPS={overall_m1['LPIPS']:.4f}" if overall_m1['LPIPS'] is not None
                     else f"  |  LPIPS=N/A"))
            print(f"  [边缘保留]    Edge-PSNR={overall_m1['Edge_PSNR']:.2f} dB")
            print(f"  [接触区域]    ROI-PSNR={overall_m1['ROI_PSNR']:.2f} dB  |  ROI-SSIM={overall_m1['ROI_SSIM']:.4f}")
            print(f"{'═' * 100}")

        # ═══════════════════════════════════════════════════════════════
        # Mode 2: tactile -> gt_height_map
        # ═══════════════════════════════════════════════════════════════
        print("\n\n" + "=" * 100)
        print(f"=== Per-object evaluation (tactile -> gt_height_map) [ratio={sampling_rate}] ===")
        print("=" * 100)

        g_accum = {}
        paired_total = 0

        for obj_name in sorted(tactile_objs.keys()):
            if obj_name not in gt_objs:
                continue
            imgs = tactile_objs[obj_name]
            gt_maps = gt_objs[obj_name]
            paired = min(len(imgs), len(gt_maps))
            if paired == 0:
                continue

            tactile_imgs = imgs[:paired]
            preds, _ = eval_3d_temporal(tactile_imgs, model, device)

            # Temporal PSNR 在有 heightmap GT 时不可靠（重建的是高度图，不是触觉图）。
            # 仅 Mode 1 和 Mode 3 的 self-reference 模式计算此指标。
            t_psnr = None

            o_accum = {}
            for i in range(paired):
                heightmap = np.load(gt_maps[i]).astype(np.float32)
                h_min, h_max = heightmap.min(), heightmap.max()
                if h_max > h_min:
                    heightmap = (heightmap - h_min) / (h_max - h_min)
                else:
                    heightmap = np.zeros_like(heightmap)
                heightmap = transforms.Resize((224, 224))(
                    torch.from_numpy(heightmap).unsqueeze(0)
                ).squeeze().numpy()

                metrics = evaluate_all(preds[i], heightmap)
                _accumulate_frame_metrics(o_accum, metrics)
                _accumulate_frame_metrics(g_accum, metrics)  # 全局：逐样本累加
                write_sample_row(sample_writer, obj_name, 'tactile2height',
                                 imgs[i], sampling_rate, metrics, eff,
                                 temporal_psnr=t_psnr)

            avg = _print_subject(f"Object: {obj_name}", paired, o_accum,
                                 temporal_psnr=t_psnr)
            avg['Temporal_PSNR'] = t_psnr
            write_object_row(obj_writer, obj_name, 'tactile2height', paired,
                             sampling_rate, avg, eff)
            paired_total += paired

        # ─── 全局汇总 ───
        if paired_total > 0:
            overall = _finalize_frame_metrics(g_accum, paired_total)
            # 全局 Temporal PSNR 取所有物体的平均值
            overall['Temporal_PSNR'] = None  # 跨物体不聚合 Temporal PSNR
            write_object_row(obj_writer, 'OVERALL', 'tactile2height',
                             paired_total, sampling_rate, overall, eff)
            summary_writer.writerow([
                'OVERALL', 'tactile2height', paired_total,
                sampling_rate,
                _round_v(overall['PSNR']), _round_v(overall['SSIM']),
                _round_v(overall['LPIPS']), _round_v(overall['Edge_PSNR']),
                _round_v(overall['ROI_PSNR']), _round_v(overall['ROI_SSIM']),
                'N/A',  # temporal PSNR not aggregated across objects
                _round_v(eff['Params'], 3),
                _round_v(eff['FLOPs'], 3) if eff['FLOPs'] is not None else 'N/A',
                _round_v(eff['FPS'], 1),
            ])

            print(f"\n{'═' * 100}")
            print(f"  [ratio={sampling_rate}] OVERALL SUMMARY  (pairs={paired_total})")
            print(f"  {'═' * 100}")
            print(f"  [全局]        PSNR={overall['PSNR']:.2f} dB  |  SSIM={overall['SSIM']:.4f}"
                  + (f"  |  LPIPS={overall['LPIPS']:.4f}" if overall['LPIPS'] is not None
                     else f"  |  LPIPS=N/A"))
            print(f"  [边缘保留]    Edge-PSNR={overall['Edge_PSNR']:.2f} dB")
            print(f"  [接触区域]    ROI-PSNR={overall['ROI_PSNR']:.2f} dB  |  ROI-SSIM={overall['ROI_SSIM']:.4f}")
            print(f"{'═' * 100}")

        # ═══════════════════════════════════════════════════════════════
        # Mode 3a: Touch and Go 时序评估 (142序列, 全部)
        # ═══════════════════════════════════════════════════════════════
        _run_temporal_eval_seqs(tag_test_seqs, 'temporal-tag',
                                'TouchAndGo', model, device, sampling_rate, eff,
                                obj_writer, sample_writer, summary_writer)

        # ═══════════════════════════════════════════════════════════════
        # Mode 3b: Yuan18 布料触觉时序评估
        # ═══════════════════════════════════════════════════════════════
        _run_temporal_eval_seqs(yuan18_test_seqs, 'temporal-yuan18',
                                'Yuan18-Cloth', model, device, sampling_rate, eff,
                                obj_writer, sample_writer, summary_writer)

        # ═══════════════════════════════════════════════════════════════
        # Mode 4: TacQuad 真实机器人评估 (self-reference)
        # ═══════════════════════════════════════════════════════════════
        if tacquad_real_images and len(tacquad_real_images) >= 4:
            print("\n\n" + "=" * 100)
            print(f"=== TacQuad Real-Robot Evaluation [ratio={sampling_rate}] ===")
            print("=" * 100)

            n = len(tacquad_real_images)
            preds, targets = eval_3d_temporal(
                tacquad_real_images, model, device)
            t_psnr = compute_temporal_psnr(preds, targets)
            r_accum = {}
            for i in range(n):
                metrics = evaluate_all(preds[i], targets[i])
                _accumulate_frame_metrics(r_accum, metrics)
                write_sample_row(sample_writer, 'tacquad-real', 'real-robot',
                                 tacquad_real_images[i], sampling_rate,
                                 metrics, eff, temporal_psnr=t_psnr)

            avg = _print_subject('TacQuad-Real', n, r_accum, temporal_psnr=t_psnr)
            avg['Temporal_PSNR'] = t_psnr
            write_object_row(obj_writer, 'tacquad-real', 'real-robot', n,
                             sampling_rate, avg, eff)
            summary_writer.writerow([
                'tacquad-real', 'real-robot', n, sampling_rate,
                _round_v(avg['PSNR']), _round_v(avg['SSIM']),
                _round_v(avg['LPIPS']), _round_v(avg['Edge_PSNR']),
                _round_v(avg['ROI_PSNR']), _round_v(avg['ROI_SSIM']),
                _round_v(t_psnr),
                _round_v(eff['Params'], 3),
                _round_v(eff['FLOPs'], 3) if eff['FLOPs'] is not None else 'N/A',
                _round_v(eff['FPS'], 1),
            ])

        # ═══════════════════════════════════════════════════════════════
        # Mode 5: VisGel 时序评估
        # ═══════════════════════════════════════════════════════════════
        _run_temporal_eval_seqs(visgel_test_seqs, 'temporal-visgel',
                                'VisGel', model, device, sampling_rate, eff,
                                obj_writer, sample_writer, summary_writer)

        obj_csv.close()
        sample_csv.close()
        print(f"\nResults saved to: {out_dir}/")

    summary_csv.close()
    print(f"\nCross-ratio summary saved to: ./eval_results/summary_all_ratios.csv")


# ═══════════════════════════════════════════════════════════════════
# Section 4: Cross-domain generalization mode
# ═══════════════════════════════════════════════════════════════════

def run_cross_domain_eval(args):
    """Evaluate on higher-resolution (448) + cross-domain datasets with ECE/Brier."""
    from trainer import valid_3d

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ratios = [float(x.strip()) for x in args.ratios.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    file_exists = os.path.exists(args.out_csv)
    out_file = open(args.out_csv, 'a' if file_exists else 'w', newline='')
    writer = csv.writer(out_file)
    if not file_exists:
        writer.writerow(['ckpt', 'ratio', 'dataset', 'image_size',
                         'PSNR', 'SSIM', 'LPIPS', 'Edge_PSNR', 'ROI_PSNR', 'ROI_SSIM',
                         'ECE', 'Brier'])

    try:
        for ratio in ratios:
            ckpt = f'./trained_model/lsdunet_{ratio:.2f}.pth'
            if not os.path.exists(ckpt):
                print(f"  [skip] {ckpt} not found")
                continue
            model, _ = load_model(ratio, ckpt, iter_num=args.iter_num,
                                  model_dim=args.model_dim, patch=args.patch)
            model = model.to(device)

            for ds_name in datasets:
                ds_path = DATASET_PATHS.get(ds_name, ds_name)
                if not os.path.isdir(ds_path):
                    print(f"  [skip] dataset path not found: {ds_path}")
                    continue
                loader = make_eval_dataloader(ds_path, args.num_frames, args.image_size)
                if loader is None:
                    continue
                print(f"  [{ds_name}] {len(loader.dataset)} volumes at {args.image_size}×{args.image_size}")
                result = valid_3d(loader, model, device, ddp=False, ema=None,
                                  collect_uncertainty=True)
                psnr, ssim, lpips, edge_psnr, roi_psnr, roi_ssim, ece, brier = result
                print(f"  ratio={ratio:.2f} dataset={ds_name} size={args.image_size} | "
                      f"PSNR={psnr:.2f} SSIM={ssim:.4f} ECE={ece:.4f} Brier={brier:.4f}")
                writer.writerow([os.path.basename(ckpt), f'{ratio:.2f}', ds_name, args.image_size,
                                 f'{psnr:.4f}', f'{ssim:.6f}',
                                 f'{lpips:.6f}' if lpips is not None else 'N/A',
                                 f'{edge_psnr:.4f}', f'{roi_psnr:.4f}', f'{roi_ssim:.6f}',
                                 f'{ece:.4f}', f'{brier:.4f}'])
                out_file.flush()
    finally:
        out_file.close()
    print(f"\nResults saved to {args.out_csv}")


# ═══════════════════════════════════════════════════════════════════
# Section 5: Noise robustness mode + ConvTokenizer3D vs LinearTokenizer ablation
# ═══════════════════════════════════════════════════════════════════

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
NOISE_CS_RATIOS = [0.01, 0.04, 0.10, 0.25, 0.50]

NOISE_RESULT_COLUMNS = [
    'Ratio', 'Sigma', 'Dataset', 'N_Frames',
    'PSNR_clean', 'PSNR_noisy', 'PSNR_drop',
    'SSIM_clean', 'SSIM_noisy', 'SSIM_drop',
    'Edge_PSNR_clean', 'Edge_PSNR_noisy', 'Edge_PSNR_drop',
    'Temporal_PSNR_clean', 'Temporal_PSNR_noisy', 'Temporal_PSNR_drop',
    'Tokenizer',
]


def add_gaussian_noise(images, sigma, seed=42):
    """Inject zero-mean Gaussian noise, clamp to [0, 1]."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    noise = torch.randn(images.shape, generator=rng) * sigma
    noisy = torch.clamp(images + noise, 0.0, 1.0)
    return noisy


class LinearTokenizer(nn.Module):
    """Ablation: pure 1x1x1 conv projection, no edge enhancement."""
    def __init__(self, in_ch=3, dim=16):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


def eval_noise_robustness(frame_paths, model, device, sigma):
    """Evaluate reconstruction quality under Gaussian noise σ."""
    T_clip = 8
    n_frames = len(frame_paths)
    frame_buffer = [load_frame(p) for p in frame_paths]

    clips = []
    indices = []
    for i in range(n_frames):
        clip, idx = build_temporal_clip(frame_buffer, i, T_clip)
        clips.append(clip)
        indices.append(idx)

    clean_preds = []
    noisy_preds = []
    clean_targets = [f.permute(1, 2, 0).cpu().numpy() for f in frame_buffer]

    for i in range(n_frames):
        clean_clip = clips[i]
        noisy_clip = add_gaussian_noise(clean_clip, sigma)

        with torch.no_grad():
            out_clean = model(clean_clip.to(device))
            out_noisy = model(noisy_clip.to(device))

        clean_pred = out_clean[0, T_clip // 2].permute(1, 2, 0).cpu().numpy()
        noisy_pred = out_noisy[0, T_clip // 2].permute(1, 2, 0).cpu().numpy()
        clean_preds.append(clean_pred)
        noisy_preds.append(noisy_pred)

    metrics_clean = [evaluate_all(clean_preds[i], clean_targets[i]) for i in range(n_frames)]
    metrics_noisy = [evaluate_all(noisy_preds[i], clean_targets[i]) for i in range(n_frames)]

    t_psnr_clean = compute_temporal_psnr(clean_preds, clean_targets)
    t_psnr_noisy = compute_temporal_psnr(noisy_preds, clean_targets)

    return metrics_clean, metrics_noisy, t_psnr_clean, t_psnr_noisy


def _avg_metric(metrics_list, key):
    vals = [m[key] for m in metrics_list if m.get(key) is not None]
    return np.mean(vals) if vals else np.nan


def run_noise_eval(dataset_name, sequences, model, device, ratio, noise_levels,
                   tokenizer_name, csv_writer):
    """Run noise robustness evaluation for all sequences in a dataset."""
    for sigma in noise_levels:
        all_clean = []
        all_noisy = []
        total_t_clean = []
        total_t_noisy = []
        total_frames = 0

        for seq_dir, frame_paths in tqdm(
            sorted(sequences.items()),
            desc=f"[{tokenizer_name}|r={ratio}|σ={sigma:.3f}] {dataset_name}"
        ):
            metrics_clean, metrics_noisy, t_clean, t_noisy = \
                eval_noise_robustness(frame_paths, model, device, sigma)

            all_clean.extend(metrics_clean)
            all_noisy.extend(metrics_noisy)
            if t_clean is not None:
                total_t_clean.append(t_clean)
            if t_noisy is not None:
                total_t_noisy.append(t_noisy)
            total_frames += len(frame_paths)

        # 汇聚指标
        row = [
            ratio,
            sigma,
            dataset_name,
            total_frames,
            _avg_metric(all_clean, 'PSNR'),
            _avg_metric(all_noisy, 'PSNR'),
            _avg_metric(all_clean, 'PSNR') - _avg_metric(all_noisy, 'PSNR'),
            _avg_metric(all_clean, 'SSIM'),
            _avg_metric(all_noisy, 'SSIM'),
            _avg_metric(all_clean, 'SSIM') - _avg_metric(all_noisy, 'SSIM'),
            _avg_metric(all_clean, 'Edge_PSNR'),
            _avg_metric(all_noisy, 'Edge_PSNR'),
            _avg_metric(all_clean, 'Edge_PSNR') - _avg_metric(all_noisy, 'Edge_PSNR'),
            np.mean(total_t_clean) if total_t_clean else np.nan,
            np.mean(total_t_noisy) if total_t_noisy else np.nan,
            (np.mean(total_t_clean) - np.mean(total_t_noisy)) if total_t_clean else np.nan,
            tokenizer_name,
        ]
        row = [round(v, 4) if isinstance(v, float) else v for v in row]
        csv_writer.writerow(row)


def run_noise_eval_main(args):
    """Noise robustness evaluation entry point."""
    noise_levels = [float(x.strip()) for x in args.noise.split(',')]
    cs_ratios = [float(x.strip()) for x in args.ratios.split(',')]

    if args.single:
        cs_ratios = [float(args.single)]

    # ─── 加载测试数据集 ───
    datasets = {}
    for name, root in [
        ('Yuan18', './dataset/yuan18/test'),
        ('VisGel', './dataset/visgel/images/touch'),
        ('TouchAndGo', './dataset/touch_and_go'),
    ]:
        if os.path.isdir(root):
            if 'visgel' in root.lower():
                seqs = collect_visgel_sequences(root, min_frames=8)
            else:
                seqs = collect_sequences(root, min_frames=8)
            if seqs:
                total = sum(len(v) for v in seqs.values())
                datasets[name] = seqs
                print(f"[{name}] {len(seqs)} seqs, {total} frames")

    if not datasets:
        print("No test datasets found!")
        return

    # ─── 确定评估模式 ───
    if args.full:
        modes = ['ConvTokenizer3D', 'LinearTokenizer']
    elif args.ablation:
        modes = ['LinearTokenizer']
    else:
        modes = ['ConvTokenizer3D']

    results_dir = './eval_results/noise'
    os.makedirs(results_dir, exist_ok=True)

    # 每个 ratio 独立输出
    for ratio in cs_ratios:
        ckpt = f'./trained_model/lsdunet_{ratio:.2f}.pth'
        if not os.path.exists(ckpt):
            print(f"[SKIP] No checkpoint: {ckpt}")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Noise Robustness: ratio={ratio:.2f}")
        print(f"{'=' * 80}")

        for mode in modes:
            # 加载模型
            print(f"\n  >>> Tokenizer: {mode}")

            if mode == 'LinearTokenizer':
                # 消融: 用 LinearTokenizer 替换 ConvTokenizer3D
                base_model, device = load_model(ratio, ckpt)
                base_model.tokenizer = LinearTokenizer(in_ch=3,
                                                       dim=base_model.model_dim)
                model = base_model.to(device)
                model.eval()
            else:
                model, device = load_model(ratio, ckpt)

            tokenizer_tag = mode

            csv_path = os.path.join(results_dir,
                                    f'noise_robustness_r{ratio:.2f}_{mode}.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(NOISE_RESULT_COLUMNS)

            for ds_name, ds_seqs in datasets.items():
                run_noise_eval(
                    ds_name, ds_seqs, model, device,
                    ratio, noise_levels, tokenizer_tag, csv_writer,
                )
                csv_file.flush()

            csv_file.close()
            print(f"  Results saved to: {csv_path}")

    # ─── 跨 ratio 汇总 ───
    summary_path = os.path.join(results_dir, 'noise_summary_all.csv')
    summary_file = open(summary_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(NOISE_RESULT_COLUMNS)

    all_csvs = sorted([f for f in os.listdir(results_dir) if f.endswith('.csv')])
    for fname in all_csvs:
        if fname == 'noise_summary_all.csv':
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    summary_writer.writerow(row)

    summary_file.close()
    print(f"\nSummary saved to: {summary_path}")
    print("Done.")


# ═══════════════════════════════════════════════════════════════════
# Section 6: Interpret mode — mechanism analysis & visualization
# ═══════════════════════════════════════════════════════════════════

def extract_sequence_mechanisms(frame_paths, model, device, out_dir):
    """Run sliding-window inference, save all mechanism data per frame."""
    T_clip = 8
    n_frames = len(frame_paths)
    frame_buffer = [load_frame(p) for p in frame_paths]

    basis_all = []
    targets_list = []
    preds_list = []
    dst_raw = []
    dst_mean = []
    lrta_all = []

    for i in tqdm(range(n_frames), desc='extracting'):
        clip, _ = build_temporal_clip(frame_buffer, i, T_clip)
        with torch.no_grad():
            out, mech = model(clip.to(device), return_mechanism=True)

        pred = out[0, T_clip // 2].permute(1, 2, 0).cpu().numpy()
        target = frame_buffer[i].permute(1, 2, 0).cpu().numpy()
        targets_list.append(target)
        preds_list.append(pred)

        bw = mech['basis_weights'].numpy()
        center_start = (T_clip // 2) * 3
        basis_all.append(bw[center_start:center_start + 3].mean(axis=0))

        for iter_key in sorted(mech['iter_data'].keys(),
                               key=lambda k: int(k.split('_')[1])):
            iter_data = mech['iter_data'][iter_key]
            s_attn = iter_data['space_attn'].numpy()[0]
            s_attn_center = s_attn[:, T_clip // 2]
            l_attn = iter_data['lrta_attn'].numpy()[0]

            iter_idx = int(iter_key.split('_')[1])
            while len(dst_raw) <= iter_idx:
                dst_raw.append([])
                dst_mean.append([])
                lrta_all.append([])
            dst_raw[iter_idx].append(s_attn_center)
            dst_mean[iter_idx].append(s_attn_center.mean(axis=(0, 1)))
            lrta_all[iter_idx].append(l_attn)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'basis_weights.npy'), np.stack(basis_all, axis=0))
    np.save(os.path.join(out_dir, 'targets.npy'), np.stack(targets_list, axis=0))
    np.save(os.path.join(out_dir, 'preds.npy'), np.stack(preds_list, axis=0))

    for idx in range(len(dst_mean)):
        np.save(os.path.join(out_dir, f'dst_attn_{idx}.npy'),
                np.stack(dst_mean[idx], axis=0))
        np.save(os.path.join(out_dir, f'lrta_attn_{idx}.npy'),
                np.stack(lrta_all[idx], axis=0))
        if idx == 0:
            np.save(os.path.join(out_dir, f'dst_attn_{idx}_full.npy'),
                    np.stack(dst_raw[idx], axis=0))

    print(f"  Saved {n_frames} frames to {out_dir}")


def run_interpret_export(args):
    """Export mechanism data for multiple datasets."""
    ckpt = args.ckpt or f'./trained_model/lsdunet_{args.ratio:.2f}.pth'
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    model, device = load_model(args.ratio, ckpt, iter_num=args.iter_num,
                               model_dim=args.model_dim, patch=args.patch)
    print(f"Model loaded: ratio={args.ratio:.2f}")

    dataset_names = [d.strip() for d in args.datasets.split(',')]
    out_root = f'./eval_results/mechanism_ratio_{args.ratio:.2f}'

    if 'toucHD' in dataset_names:
        root = './dataset/toucHD/train'
        if os.path.isdir(root):
            seqs = collect_sequences(root, min_frames=8)
            for seq_dir, frame_paths in sorted(seqs.items())[:args.max_seqs]:
                fp = frame_paths if args.max_frames <= 0 else frame_paths[:args.max_frames]
                extract_sequence_mechanisms(fp, model, device,
                                            os.path.join(out_root, 'toucHD',
                                                         os.path.basename(seq_dir)))

    if 'TacQuad' in dataset_names:
        root = './dataset/tacquad/train'
        if os.path.isdir(root):
            objs = collect_ycb_by_object(root)
            tactile = {k: v for k, v in objs.items() if not k.endswith('_gt')}
            for obj_name in sorted(tactile.keys())[:args.max_seqs]:
                fp = tactile[obj_name]
                if args.max_frames > 0:
                    fp = fp[:args.max_frames]
                extract_sequence_mechanisms(fp, model, device,
                                            os.path.join(out_root, 'TacQuad', obj_name))

    if 'yuan18' in dataset_names:
        root = './dataset/yuan18/test'
        if os.path.isdir(root):
            seqs = collect_sequences(root, min_frames=8)
            for seq_dir, frame_paths in sorted(seqs.items())[:args.max_seqs]:
                fp = frame_paths if args.max_frames <= 0 else frame_paths[:args.max_frames]
                extract_sequence_mechanisms(fp, model, device,
                                            os.path.join(out_root, 'yuan18',
                                                         os.path.basename(seq_dir)))

    if 'visgel' in dataset_names:
        root = './dataset/visgel/images/touch'
        if os.path.isdir(root):
            seqs = collect_visgel_sequences(root, min_frames=8)
            for rec_dir, frame_paths in sorted(seqs.items())[:args.max_seqs]:
                fp = frame_paths if args.max_frames <= 0 else frame_paths[:args.max_frames]
                extract_sequence_mechanisms(fp, model, device,
                                            os.path.join(out_root, 'visgel',
                                                         os.path.basename(rec_dir)))

    print(f"\nDone. Mechanism data saved to: {out_root}")


def get_sample(val_dir, num_frames=8, image_size=224, idx=0):
    """Load a single volume sample for visualization."""
    loader = make_eval_dataloader(val_dir, num_frames=num_frames,
                                  image_size=image_size, batch_size=1)
    vol, _ = loader.dataset[idx]
    return vol.unsqueeze(0)


@torch.no_grad()
def run_interpret_render(args):
    """Render paper figures from a single sample."""
    import matplotlib.pyplot as plt

    ckpt = args.ckpt or f'./trained_model/lsdunet_{args.ratio:.2f}.pth'
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    model, device = load_model(args.ratio, ckpt,
                               iter_num=args.iter_num, model_dim=args.model_dim,
                               patch=args.patch)
    vol = get_sample(args.val_dir, num_frames=args.num_frames,
                     image_size=args.image_size, idx=args.sample_idx).to(device)

    os.makedirs(args.out, exist_ok=True)

    # 1. Per-iteration reconstruction (deep unfolding progression)
    intermediates = []
    out = model(vol, return_intermediates=True)
    if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], list):
        final_out, intermediates = out[0], out[1]
    else:
        final_out = out[0] if isinstance(out, tuple) else out

    n_iters = len(intermediates)
    if n_iters > 0:
        fig, axes = plt.subplots(1, n_iters + 2, figsize=(3 * (n_iters + 2), 3))
        axes[0].imshow(vol[0, 0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Input (frame 0)')
        axes[0].axis('off')
        for i, inter in enumerate(intermediates):
            axes[i + 1].imshow(inter[0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            axes[i + 1].set_title(f'Iter {i + 1}')
            axes[i + 1].axis('off')
        axes[-1].imshow(final_out[0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[-1].set_title('Final')
        axes[-1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'deep_unfolding_progression.png'), dpi=150)
        plt.close()
        print(f"  [saved] deep_unfolding_progression.png ({n_iters} iters)")

    # 2. Mechanism visualization (offsets, attention, basis weights)
    out_mech = model(vol, return_mechanism=True, return_uncertainty=True)
    if isinstance(out_mech, tuple) and len(out_mech) == 2:
        final_out, mech_data = out_mech

        bw = mech_data.get('basis_weights', None)
        if bw is not None:
            bw = bw[0].numpy()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(bw)), bw)
            ax.set_xlabel('Basis index')
            ax.set_ylabel('Weight')
            ax.set_title('AdaptiveSModule basis weights (input-dependent)')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'basis_weights.png'), dpi=150)
            plt.close()
            print(f"  [saved] basis_weights.png")

        for i, key in enumerate([k for k in mech_data.keys() if k.startswith('iter_')]):
            mech = mech_data[key]
            s_attn = mech.get('space_attn', None)
            if s_attn is not None:
                mag = s_attn[0].abs().mean(dim=0).cpu().numpy()
                T_len = mag.shape[0]
                fig, axes = plt.subplots(1, T_len, figsize=(2 * T_len, 2))
                if T_len == 1:
                    axes = [axes]
                for t in range(T_len):
                    axes[t].imshow(mag[t], cmap='hot')
                    axes[t].set_title(f't={t}')
                    axes[t].axis('off')
                plt.suptitle(f'DeformableSpatialBlock attention (iter {i})')
                plt.tight_layout()
                plt.savefig(os.path.join(args.out, f'deform_attn_iter{i}.png'), dpi=150)
                plt.close()
                print(f"  [saved] deform_attn_iter{i}.png")

            h_attn = mech.get('history_attn', None)
            if h_attn is not None:
                attn = h_attn[0].cpu().numpy()
                fig, ax = plt.subplots(figsize=(6, 3))
                im = ax.imshow(attn, aspect='auto', cmap='viridis')
                ax.set_xlabel('Time step')
                ax.set_ylabel('Query index')
                ax.set_title(f'History compressor attention (iter {i})')
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(args.out, f'history_attn_iter{i}.png'), dpi=150)
                plt.close()
                print(f"  [saved] history_attn_iter{i}.png")

        # 3. Uncertainty heatmap
        unc = mech_data.get('uncertainty', None)
        if unc is not None:
            unc_np = unc[0].squeeze().cpu().numpy()
            T_len = unc_np.shape[0]
            fig, axes = plt.subplots(1, T_len, figsize=(2 * T_len, 2))
            if T_len == 1:
                axes = [axes]
            for t in range(T_len):
                axes[t].imshow(unc_np[t], cmap='magma')
                axes[t].set_title(f't={t}')
                axes[t].axis('off')
            plt.suptitle('Per-pixel uncertainty (predicted σ²)')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'uncertainty_heatmap.png'), dpi=150)
            plt.close()
            print(f"  [saved] uncertainty_heatmap.png")

    print(f"\nAll visualizations saved to {args.out}/")


def run_interpret_main(args):
    """Interpret mode dispatcher: export | render | all."""
    if args.submode == 'export':
        run_interpret_export(args)
    elif args.submode == 'render':
        run_interpret_render(args)
    elif args.submode == 'all':
        run_interpret_export(args)
        run_interpret_render(args)
    else:
        print("[ERROR] --submode required for interpret mode "
              "(choices: export, render, all)")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _add_model_args(parser):
    parser.add_argument('--iter_num', type=int, default=6)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--patch', type=int, default=32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSDUNet evaluation (unified)')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'cross-domain', 'noise', 'interpret'],
                        help='standard: full eval | cross-domain: high-res+ECE/Brier '
                             '| noise: noise robustness | interpret: mechanism analysis')
    parser.add_argument('--submode', type=str, default=None,
                        choices=['export', 'render', 'all'],
                        help='Submode for --mode interpret (export|render|all)')

    # Cross-domain mode params
    parser.add_argument('--ratios', type=str, default='0.01,0.04,0.10,0.25,0.50',
                        help='CS ratios (cross-domain / noise mode)')
    parser.add_argument('--datasets', type=str,
                        default='touch_and_go,visgel,tacquad,yuan18',
                        help='datasets (cross-domain mode) or dataset list (interpret export)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size (default 224; cross-domain mode uses 448)')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--out_csv', type=str, default='results/generalization.csv',
                        help='output CSV (cross-domain mode)')

    # Noise mode params
    parser.add_argument('--noise', type=str, default=','.join(map(str, NOISE_LEVELS)),
                        help='noise std list, comma-separated (noise mode)')
    parser.add_argument('--ablation', action='store_true',
                        help='noise mode: only LinearTokenizer ablation')
    parser.add_argument('--full', action='store_true',
                        help='noise mode: both ConvTokenizer3D and LinearTokenizer')
    parser.add_argument('--single', type=str, default=None,
                        help='noise mode: single ratio checkpoint')

    # Interpret mode params
    parser.add_argument('--ratio', type=float, default=0.10,
                        help='CS ratio (interpret mode)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path (interpret render/all mode)')
    parser.add_argument('--val_dir', type=str, default='dataset/touch_and_go',
                        help='val dir (interpret render mode)')
    parser.add_argument('--out', type=str, default='vis',
                        help='output dir (interpret render mode)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='sample index (interpret render mode)')
    parser.add_argument('--max_seqs', type=int, default=3,
                        help='max sequences per dataset (interpret export mode)')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='max frames per sequence, 0=all (interpret export mode)')

    _add_model_args(parser)
    args = parser.parse_args()

    # ─── Dispatch ───
    if args.mode == 'cross-domain':
        run_cross_domain_eval(args)
    elif args.mode == 'noise':
        # Override NOISE_CS_RATIOS default with --ratios if user provided
        run_noise_eval_main(args)
    elif args.mode == 'interpret':
        run_interpret_main(args)
    else:
        # standard mode (default)
        run_standard_eval(args)
