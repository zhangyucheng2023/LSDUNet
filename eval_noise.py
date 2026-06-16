#!/usr/bin/env python3
"""
动态视频抗噪评估 — 证明 ConvTokenizer3D 的物理抗噪先验
==========================================================
向测试集连续触觉序列注入高频白噪声，在不同噪声强度下评估重建稳定性。
对比 ConvTokenizer3D vs LinearTokenizer（纯线性投影）的噪声鲁棒性。

用法:
    python eval_noise.py                          # 使用默认噪声等级
    python eval_noise.py --noise 0.0,0.05,0.10,0.20  # 自定义噪声等级
    python eval_noise.py --ablation               # 仅评估消融模型
    python eval_noise.py --full                   # 同时对比两种 tokenizer
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.model_3d import LSDUNet
from metrics import evaluate_all, compute_temporal_psnr, get_efficiency_metrics
from eval import load_3d_model, eval_3d_temporal, _load_frame, _build_temporal_clip
from data_processor import collect_sequences, collect_visgel_sequences


# ═══════════════════════════════════════════════════════════
# 噪声注入工具
# ═══════════════════════════════════════════════════════════

def add_gaussian_noise(images, sigma, seed=42):
    """
    向连续帧序列注入零均值高斯白噪声（模拟高频传感器底噪）。

    Args:
        images: [T, 1, H, W] tensor, normalized to [0, 1]
        sigma: noise standard deviation (relative to [0,1] range)
               σ=0.05 → ~3% pixel values corrupted beyond ±0.1
               σ=0.10 → ~32% pixel values corrupted beyond ±0.1
        seed: random seed for reproducibility

    Returns:
        noisy: [T, 1, H, W] tensor, clamped to [0, 1]
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    noise = torch.randn(images.shape, generator=rng) * sigma
    noisy = torch.clamp(images + noise, 0.0, 1.0)
    return noisy


# ═══════════════════════════════════════════════════════════
# 消融 Tokenizer: 纯线性投影 (无边缘分支、无噪声抑制)
# ═══════════════════════════════════════════════════════════

class LinearTokenizer(nn.Module):
    """
    消融对照: 用单层 1x1x1 卷积替代 ConvTokenizer3D。
    没有 3D 卷积边缘增强，没有多尺度特征提取，
    等价于标准 ViT 的线性 Patch 投影。
    """
    def __init__(self, in_ch=1, dim=16):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


# ═══════════════════════════════════════════════════════════
# 噪声鲁棒性评估核心逻辑
# ═══════════════════════════════════════════════════════════

def eval_noise_robustness(frame_paths, model, device, sigma):
    """
    对一条时序序列进行噪声注入 + 重建评估。

    Args:
        frame_paths: list of image paths
        model: LSDUNet model
        device: torch device
        sigma: noise level

    Returns:
        metrics: list of dicts, one per frame (clean metrics)
        metrics_noisy: list of dicts, one per frame (noisy metrics)
        temporal_psnr_clean, temporal_psnr_noisy
    """
    T_clip = 8  # 与训练时 num_frames 一致
    n_frames = len(frame_paths)
    frame_buffer = [_load_frame(p) for p in frame_paths]

    # 预加载所有滑窗片段
    clips = []
    indices = []
    for i in range(n_frames):
        clip, idx = _build_temporal_clip(frame_buffer, i, T_clip)
        clips.append(clip)
        indices.append(idx)

    # 批量推理
    clean_preds = []
    noisy_preds = []
    clean_targets = [f.squeeze(0).cpu().numpy() for f in frame_buffer]

    for i in range(n_frames):
        clean_clip = clips[i]
        noisy_clip = add_gaussian_noise(clean_clip, sigma)

        with torch.no_grad():
            out_clean = model(clean_clip.to(device))
            out_noisy = model(noisy_clip.to(device))

        clean_pred = out_clean[0, T_clip // 2, 0, :, :].cpu().numpy()
        noisy_pred = out_noisy[0, T_clip // 2, 0, :, :].cpu().numpy()
        clean_preds.append(clean_pred)
        noisy_preds.append(noisy_pred)

    # 逐帧指标
    metrics_clean = [evaluate_all(clean_preds[i], clean_targets[i]) for i in range(n_frames)]
    metrics_noisy = [evaluate_all(noisy_preds[i], clean_targets[i]) for i in range(n_frames)]

    t_psnr_clean = compute_temporal_psnr(clean_preds, clean_targets)
    t_psnr_noisy = compute_temporal_psnr(noisy_preds, clean_targets)

    return metrics_clean, metrics_noisy, t_psnr_clean, t_psnr_noisy


# ═══════════════════════════════════════════════════════════
# 评估入口
# ═══════════════════════════════════════════════════════════

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
CS_RATIOS = [0.01, 0.04, 0.10, 0.25, 0.50]

RESULT_COLUMNS = [
    'Ratio', 'Sigma', 'Dataset', 'N_Frames',
    'PSNR_clean', 'PSNR_noisy', 'PSNR_drop',
    'SSIM_clean', 'SSIM_noisy', 'SSIM_drop',
    'Edge_PSNR_clean', 'Edge_PSNR_noisy', 'Edge_PSNR_drop',
    'Temporal_PSNR_clean', 'Temporal_PSNR_noisy', 'Temporal_PSNR_drop',
    'Tokenizer',
]


def _avg_metric(metrics_list, key):
    vals = [m[key] for m in metrics_list if m.get(key) is not None]
    return np.mean(vals) if vals else np.nan


def run_noise_eval(dataset_name, sequences, model, device, ratio, noise_levels,
                   tokenizer_name, csv_writer):
    """对单个数据集的所有序列执行噪声鲁棒性评估"""
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


def main():
    parser = argparse.ArgumentParser(description='LSDUNet 动态视频抗噪评估')
    parser.add_argument('--noise', type=str, default=','.join(map(str, NOISE_LEVELS)),
                        help='噪声标准差列表, 逗号分隔')
    parser.add_argument('--ratios', type=str, default=','.join(map(str, CS_RATIOS)),
                        help='CS 压缩比列表')
    parser.add_argument('--ablation', action='store_true',
                        help='仅评估 LinearTokenizer 消融模型 (跳过 ConvTokenizer3D)')
    parser.add_argument('--full', action='store_true',
                        help='同时评估 ConvTokenizer3D 和 LinearTokenizer')
    parser.add_argument('--single', type=str, default=None,
                        help='仅评估单个压缩比的 checkpoint')
    args = parser.parse_args()

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
                base_model, device = load_3d_model(ratio, ckpt)
                base_model.tokenizer = LinearTokenizer(in_ch=1,
                                                       dim=base_model.model_dim)
                model = base_model.to(device)
                model.eval()
            else:
                model, device = load_3d_model(ratio, ckpt)

            tokenizer_tag = mode

            csv_path = os.path.join(results_dir,
                                    f'noise_robustness_r{ratio:.2f}_{mode}.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(RESULT_COLUMNS)

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
    summary_writer.writerow(RESULT_COLUMNS)

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


if __name__ == '__main__':
    main()
