"""Cross-domain generalization evaluation script.

Evaluates a trained LSDUNet checkpoint on:
  1. Higher-resolution inputs (448×448) — tests stride-2 adaptive tokenizer
  2. Cross-domain datasets (visgel, tacquad, yuan18) — zero-shot transfer
  3. Reports all metrics: PSNR, SSIM, LPIPS, Edge-PSNR, ROI-PSNR, ROI-SSIM,
     ECE, Brier Score

Usage:
  python eval_generalization.py --ckpt trained_model/lsdunet_0.10.pth \
      --ratios 0.01,0.04,0.10,0.25,0.50 \
      --datasets visgel,tacquad,yuan18 --image_size 448
"""
import os
import argparse
import csv
import torch
import numpy as np
from trainer import valid_3d
from metrics import count_parameters, measure_flops, measure_fps
from eval_common import (load_model, make_eval_dataloader, DATASET_PATHS)


def evaluate_one(ckpt_path, ratio, val_dir, image_size, num_frames, device,
                 iter_num=6, model_dim=64, patch=32):
    model, _ = load_model(ratio, ckpt_path, iter_num=iter_num,
                          model_dim=model_dim, patch=patch, use_cache=False)
    model = model.to(device)

    loader = make_eval_dataloader(val_dir, num_frames, image_size)
    if loader is None:
        return None

    print(f"  [{val_dir}] {len(loader.dataset)} volumes at {image_size}×{image_size}")
    result = valid_3d(loader, model, device, ddp=False, ema=None, collect_uncertainty=True)
    return result


def main(args):
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
            ckpt = args.ckpt.replace(f'lsdunet_{args.ckpt_ratio:.2f}.pth',
                                      f'lsdunet_{ratio:.2f}.pth') if args.ckpt_ratio else args.ckpt
            if not os.path.exists(ckpt):
                print(f"  [skip] {ckpt} not found")
                continue
            for ds_name in datasets:
                ds_path = DATASET_PATHS.get(ds_name, ds_name)
                if not os.path.isdir(ds_path):
                    print(f"  [skip] dataset path not found: {ds_path}")
                    continue
                result = evaluate_one(ckpt, ratio, ds_path, args.image_size, args.num_frames,
                                      device, iter_num=args.iter_num, model_dim=args.model_dim,
                                      patch=args.patch)
                if result is None:
                    continue
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint pattern, e.g. trained_model/lsdunet_0.10.pth')
    parser.add_argument('--ckpt_ratio', type=float, default=None,
                        help='ratio in ckpt filename (auto-replaced for each ratio)')
    parser.add_argument('--ratios', type=str, default='0.01,0.04,0.10,0.25,0.50')
    parser.add_argument('--datasets', type=str, default='touch_and_go,visgel,tacquad,yuan18')
    parser.add_argument('--image_size', type=int, default=448,
                        help='higher resolution to test adaptivity (default 448)')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--out_csv', type=str, default='results/generalization.csv')
    # Model architecture params (must match training config)
    parser.add_argument('--iter_num', type=int, default=6, help='deep unfolding iterations')
    parser.add_argument('--model_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--patch', type=int, default=32, help='CS sampling patch size')
    args = parser.parse_args()
    main(args)
