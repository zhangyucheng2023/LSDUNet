import os
import csv
import warnings
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model.model_3d import LSDUNet
from data_processor import collect_ycb_by_object, collect_sequences, collect_visgel_sequences

warnings.filterwarnings("ignore")

model_3d = None
old_rate_3d = 10000


def load_3d_model(cs_ratio, checkpoint_path, iter_num=8, model_dim=64, patch=32, num_heads=8):
    global model_3d, old_rate_3d
    # 安全设备检测
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            _ = torch.zeros(1).to(device)
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    if model_3d is None or cs_ratio != old_rate_3d:
        model_3d = LSDUNet(ratio=cs_ratio, iter_num=iter_num,
                           model_dim=model_dim, patch=patch,
                           num_heads=num_heads).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        result = model_3d.load_state_dict(checkpoint, strict=False)
        if result.missing_keys:
            print(f"[WARN] Checkpoint missing {len(result.missing_keys)} keys "
                  f"(new modules will be randomly initialized):")
            # 按模块分组显示
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
        model_3d.eval()
        old_rate_3d = cs_ratio
    return model_3d, device


def _load_frame(path):
    """加载单帧并转为灰度归一化 tensor [1, H, W]"""
    img = Image.open(path).convert('L')
    img_t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])(img)                                              # [1, H, W], already [0,1]
    return img_t


def _build_temporal_clip(frame_buffer, center_idx, clip_len=8):
    """
    从预加载的帧缓冲区中，以 center_idx 为中心构建 [clip_len, 1, H, W] 时序片段。
    边界帧使用边缘复制补齐。
    """
    n = len(frame_buffer)
    half = clip_len // 2
    indices = []
    for offset in range(-half, half):
        idx = max(0, min(n - 1, center_idx + offset))
        indices.append(idx)
    clip = torch.stack([frame_buffer[i] for i in indices], dim=0)
    clip = clip.unsqueeze(0)
    return clip, indices


def eval_3d_temporal(frame_paths, model, device, return_intermediates=False,
                      return_uncertainty=False):
    """
    时序数据集滑动窗口评估。
    对每一帧 i，以其为中心构建 8 帧滑动窗口 [i-4, ..., i+3] 输入模型，
    取中间第 5 帧（索引 4）作为第 i 帧的重建结果。边界帧边缘复制补齐。
    """
    T_clip = 8  # 与训练时 num_frames 一致
    n_frames = len(frame_paths)
    frame_buffer = [_load_frame(p) for p in frame_paths]

    preds = []
    all_intermediates = []
    log_vars = [] if return_uncertainty else None

    for i in range(n_frames):
        clip, _ = _build_temporal_clip(frame_buffer, i, T_clip)
        with torch.no_grad():
            if return_intermediates:
                out, intermediates = model(clip.to(device), return_intermediates=True)
                inter_preds = [im[0, T_clip // 2, 0, :, :].cpu().numpy() for im in intermediates]
                all_intermediates.append(inter_preds)
            elif return_uncertainty:
                out, log_var = model(clip.to(device), return_uncertainty=True)
                log_vars.append(log_var[0, T_clip // 2, 0, :, :].cpu().numpy())
            else:
                out = model(clip.to(device))
        pred = out[0, T_clip // 2, 0, :, :].cpu().numpy()
        preds.append(pred)

    targets = [f.squeeze(0).cpu().numpy() for f in frame_buffer]

    if return_intermediates:
        return preds, targets, all_intermediates
    if return_uncertainty:
        return preds, targets, log_vars
    return preds, targets


def save_uncertainty_heatmaps(preds, targets, log_vars, seq_name, out_dir, n_samples=5):
    """
    自动生成不确定性可视化热力图。
    为每个序列选取前 n_samples 帧，生成三列并排图：
    [输入帧 | 重建帧 | 不确定性热力图 (σ = exp(log_var/2))]"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = min(n_samples, len(preds))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        target = targets[i]
        pred = preds[i]
        log_var = log_vars[i]
        std = np.exp(log_var / 2)  # σ = exp(log_var/2)

        # 目标帧（Ground Truth）
        axes[i, 0].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Target (Frame {i})')
        axes[i, 0].axis('off')

        # 重建帧
        axes[i, 1].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Reconstruction')
        axes[i, 1].axis('off')

        # 不确定性热力图
        im = axes[i, 2].imshow(std, cmap='hot', vmin=0, vmax=std.max())
        axes[i, 2].set_title(f'Uncertainty σ')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'uncertainty_{seq_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [uncertainty] Saved heatmap: {save_path}')


# ─── 列名定义 ───
PER_OBJECT_COLUMNS = [
    'Object', 'Mode', 'N', 'Ratio',
    'PSNR', 'SSIM', 'LPIPS', 'Edge_PSNR',
    'ROI_PSNR', 'ROI_SSIM', 'Temporal_PSNR',
    'Params(M)', 'FLOPs(G)', 'FPS',
]

PER_SAMPLE_COLUMNS = PER_OBJECT_COLUMNS + ['Image']


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


# ─── 聚合辅助 ───

_FRAME_KEYS = ['PSNR', 'SSIM', 'LPIPS', 'Edge_PSNR', 'ROI_PSNR', 'ROI_SSIM']


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
    from metrics import evaluate_all, compute_temporal_psnr

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


if __name__ == "__main__":
    from metrics import (evaluate_all, get_efficiency_metrics,
                         compute_temporal_psnr)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_uncertainty', action='store_true',
                        help='Generate uncertainty heatmaps (β-σ) after evaluation')
    parser.add_argument('--uncertainty_ratio', default=0.10, type=float,
                        help='Which CS ratio to use for uncertainty viz (default: 0.10)')
    args = parser.parse_args()

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

        model, device = load_3d_model(sampling_rate, ckpt)
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
                heightmap = transforms.Resize((96, 96))(
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

    # ═══════════════════════════════════════════════════════════════
    # 不确定性可视化（独立于主评估，仅生成热力图）
    # ═══════════════════════════════════════════════════════════════
    if args.save_uncertainty:
        print("\n" + "=" * 100)
        print("=== Uncertainty Visualization (β-NLL Heteroscedastic) ===")
        print("=" * 100)

        ckpt = f'./trained_model/lsdunet_{args.uncertainty_ratio}.pth'
        if not os.path.exists(ckpt):
            print(f"[SKIP] Checkpoint not found: {ckpt}")
        else:
            model, device = load_3d_model(args.uncertainty_ratio, ckpt)
            unc_dir = f'./eval_results/uncertainty'
            os.makedirs(unc_dir, exist_ok=True)

            # 选取代表性序列：Touch and Go (跨域时序), TacQuad (自参考), Yuan18 (布料)
            sample_sources = {}

            # Touch and Go: 取前 3 个序列
            if tag_test_seqs:
                for i, (seq_dir, frames) in enumerate(sorted(tag_test_seqs.items())[:3]):
                    sample_sources[f'Tag_{os.path.basename(seq_dir)}'] = frames

            # TacQuad: 取第一个物体
            if tactile_objs:
                first_obj = sorted(tactile_objs.keys())[0]
                sample_sources[f'TacQuad_{first_obj}'] = tactile_objs[first_obj]

            # Yuan18: 取前 2 个序列
            if yuan18_test_seqs:
                for i, (seq_dir, frames) in enumerate(sorted(yuan18_test_seqs.items())[:2]):
                    sample_sources[f'Yuan18_{os.path.basename(seq_dir)}'] = frames

            for name, frames in sample_sources.items():
                if len(frames) < 8:
                    continue
                print(f"\nGenerating uncertainty heatmap for: {name} ({len(frames)} frames)...")
                preds, targets, log_vars = eval_3d_temporal(
                    frames, model, device, return_uncertainty=True)

                # 保存 log_var 原始数据
                np.save(os.path.join(unc_dir, f'{name}_preds.npy'), np.array(preds))
                np.save(os.path.join(unc_dir, f'{name}_targets.npy'), np.array(targets))
                np.save(os.path.join(unc_dir, f'{name}_logvars.npy'), np.array(log_vars))

                # 自动生成热力图
                save_uncertainty_heatmaps(preds, targets, log_vars, name, unc_dir)

            print(f"\nUncertainty results saved to: {unc_dir}/")
