"""
LSDUNet 评估适配脚本 — 接入 LSDU-COMP 统一对比框架。
用 lsdu_adapter 的数据加载和指标计算，结果写入 all_baselines.csv。

LSDUNet 与 baseline 的区别：
  - baseline: 外部 mask 压缩 → model(y, mask, Phi_s) → 重建
  - LSDUNet:  CS 采样在模型内部 → model(volume) → 重建 (无需外部 mask)

用法:
  python eval_touchd.py                          # 评估 3 个 ratio
  python eval_touchd.py --ratios 0.10            # 评估单个 ratio
  python eval_touchd.py --ckpt_dir trained_model # 指定 checkpoint 目录
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

# 接入 LSDU-COMP 统一框架
sys.path.insert(0, '/home/tuf/LSDU-COMP')
from lsdu_adapter import (
    collect_touchd_sequences, ToucHDVolumeDataset,
    compute_all_metrics, compute_temporal_psnr,
    count_params, measure_flops, measure_fps, measure_latency,
    save_results_csv, RESULTS_DIR,
)

# LSDUNet 模型
sys.path.insert(0, '/home/tuf/LSDUNet')
from model.model_3d import LSDUNet


def parse_args():
    p = argparse.ArgumentParser(description='LSDUNet eval via LSDU-COMP adapter')
    p.add_argument('--ratios', default='0.01,0.10,0.50', type=str)
    p.add_argument('--ckpt_dir', default='/home/tuf/LSDUNet/trained_model', type=str)
    p.add_argument('--val_dir', default='/home/tuf/LSDUNet/dataset/touch_and_go', type=str)
    p.add_argument('--num_frames', default=8, type=int)
    p.add_argument('--image_size', default=224, type=int)
    p.add_argument('--batch_size', default=1, type=int)
    p.add_argument('--model_dim', default=64, type=int)
    p.add_argument('--iter_num', default=6, type=int)
    p.add_argument('--patch', default=32, type=int)
    p.add_argument('--ls_rank', default=4, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    return p.parse_args()


def load_lsdunet(ratio, ckpt_path, args, device):
    """加载 LSDUNet checkpoint。"""
    model = LSDUNet(
        cs_ratio=ratio,
        model_dim=args.model_dim,
        iter_num=args.iter_num,
        patch=args.patch,
        num_frames=args.num_frames,
        ls_rank=args.ls_rank,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, val_loader, device):
    """用 adapter 的 compute_all_metrics 评估，与 baseline 完全一致。"""
    metrics_sum = {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0,
                   'Edge_PSNR': 0, 'ROI_PSNR': 0, 'ROI_SSIM': 0}
    lpips_count = 0
    n_frames = 0
    t_psnr_sum = 0
    t_psnr_count = 0

    for inputs, _ in tqdm(val_loader, desc='eval'):
        inputs = inputs.to(device)  # [B, T, 3, H, W]
        B, T, C, H, W = inputs.shape
        outputs, _ = model(inputs, return_uncertainty=True)  # [B, T, 3, H, W]

        pred_np = outputs.float().cpu().numpy()
        tgt_np = inputs.float().cpu().numpy()

        for b in range(B):
            preds_b = [pred_np[b, t].transpose(1, 2, 0) for t in range(T)]
            tgts_b = [tgt_np[b, t].transpose(1, 2, 0) for t in range(T)]
            t_psnr = compute_temporal_psnr(preds_b, tgts_b)
            if t_psnr is not None:
                t_psnr_sum += t_psnr
                t_psnr_count += 1
            for t in range(T):
                m = compute_all_metrics(preds_b[t], tgts_b[t])
                for k in metrics_sum:
                    if k == 'LPIPS' and m[k] is None:
                        continue
                    metrics_sum[k] += m[k]
                if m['LPIPS'] is not None:
                    lpips_count += 1
                n_frames += 1

    result = {k: v / n_frames for k, v in metrics_sum.items()}
    result['LPIPS'] = metrics_sum['LPIPS'] / lpips_count if lpips_count > 0 else None
    result['Temporal_PSNR'] = t_psnr_sum / t_psnr_count if t_psnr_count > 0 else None
    return result


def measure_efficiency(model, args, device):
    """测效率：LSDUNet 单输入前向 model(volume) → (outputs, uncertainty)。"""
    model.eval()
    x = torch.randn(1, args.num_frames, 3, args.image_size, args.image_size, device=device)
    # warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # FPS / Latency
    repeats = 50
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps = repeats / elapsed if elapsed > 0 else 0
    latency_ms = elapsed / repeats * 1000
    # FLOPs (单输入前向，fvcore 支持)
    flops = measure_flops(model, (1, args.num_frames, 3, args.image_size, args.image_size), device)
    return flops, fps, latency_ms


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ratios = [float(x.strip()) for x in args.ratios.split(',')]

    # 加载评估数据（与 baseline 完全一致：touch_and_go）
    val_seqs = collect_touchd_sequences(args.val_dir, min_frames=args.num_frames)
    val_ds = ToucHDVolumeDataset(val_seqs, num_frames=args.num_frames,
                                 image_size=args.image_size, grayscale=False, augment=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f'Val: {len(set(s[0] for s in val_ds.samples))} seqs → {len(val_ds)} volumes')

    for ratio in ratios:
        ckpt = os.path.join(args.ckpt_dir, f'lsdunet_{ratio:.2f}.pth')
        if not os.path.exists(ckpt):
            print(f'[skip] {ckpt} not found')
            continue

        print(f'\n{"="*60}')
        print(f'LSDUNet | ratio={ratio:.2f}')
        print(f'{"="*60}')

        model = load_lsdunet(ratio, ckpt, args, device)
        print(f'Params: {count_params(model):.3f}M')

        start_t = time.time()
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start_t

        flops, fps, lat = measure_efficiency(model, args, device)
        eff = {'Params': count_params(model), 'FLOPs': flops, 'FPS': fps, 'Latency_ms': lat}

        print(f'PSNR={metrics["PSNR"]:.2f} SSIM={metrics["SSIM"]:.4f} '
              f'LPIPS={metrics["LPIPS"]:.4f if metrics["LPIPS"] else "N/A"} '
              f'Edge={metrics["Edge_PSNR"]:.2f} ROI={metrics["ROI_PSNR"]:.2f} '
              f'FPS={fps:.1f}')

        save_results_csv('LSDUNet', ratio, metrics, eff, elapsed,
                         temporal_psnr=metrics.get('Temporal_PSNR'))

    print(f'\nDone. Results: {os.path.join(RESULTS_DIR, "all_baselines.csv")}')


if __name__ == '__main__':
    main()
