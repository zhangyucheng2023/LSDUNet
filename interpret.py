"""LSDUNet interpretability analysis — mechanism data export & visualization.

Combines:
  1. Mechanism data extraction (basis weights, attention maps, .npy export)
  2. Paper figure rendering (deep unfolding, deformable attn, uncertainty heatmaps)

Usage:
  # Export mechanism data as .npy for external analysis
  python interpret.py export --ratio 0.10 --datasets toucHD,TacQuad

  # Render paper figures (matplotlib PNGs)
  python interpret.py render --ckpt trained_model/lsdunet_0.10.pth --val_dir dataset/touch_and_go

  # Both (export then render)
  python interpret.py all --ratio 0.10
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

from data_processor import collect_sequences, collect_ycb_by_object, collect_visgel_sequences
from eval_common import (load_model, load_frame, build_temporal_clip as build_clip,
                         make_eval_dataloader, DATASET_PATHS,
                         KEY_SPACE_ATTN, KEY_HISTORY_ATTN,
                         KEY_BASIS_WEIGHTS, KEY_UNCERTAINTY, KEY_ITER_DATA)


# ═══════════════════════════════════════════════════════════════════
# Mode 1: Export mechanism data as .npy
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
        clip = build_clip(frame_buffer, i, T_clip)
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


def run_export(args):
    """Export mechanism data for multiple datasets."""
    ckpt = f'./trained_model/lsdunet_{args.ratio:.2f}.pth'
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


# ═══════════════════════════════════════════════════════════════════
# Mode 2: Render paper figures (matplotlib)
# ═══════════════════════════════════════════════════════════════════

def get_sample(val_dir, num_frames=8, image_size=224, idx=0):
    """Load a single volume sample for visualization."""
    loader = make_eval_dataloader(val_dir, num_frames=num_frames,
                                  image_size=image_size, batch_size=1)
    vol, _ = loader.dataset[idx]
    return vol.unsqueeze(0)


@torch.no_grad()
def run_render(args):
    """Render paper figures from a single sample."""
    import matplotlib.pyplot as plt

    model, device = load_model(args.ratio, args.ckpt,
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


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def add_model_args(parser):
    parser.add_argument('--iter_num', type=int, default=6)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--patch', type=int, default=32)


def main():
    parser = argparse.ArgumentParser(description='LSDUNet interpretability analysis')
    sub = parser.add_subparsers(dest='mode', required=True)

    # export subcommand
    p_export = sub.add_parser('export', help='Export mechanism data as .npy')
    p_export.add_argument('--ratio', type=float, default=0.10)
    p_export.add_argument('--datasets', type=str, default='toucHD,TacQuad,yuan18,visgel')
    p_export.add_argument('--max_seqs', type=int, default=3)
    p_export.add_argument('--max_frames', type=int, default=0)
    add_model_args(p_export)

    # render subcommand
    p_render = sub.add_parser('render', help='Render paper figures (matplotlib PNGs)')
    p_render.add_argument('--ckpt', type=str, required=True)
    p_render.add_argument('--val_dir', type=str, default='dataset/touch_and_go')
    p_render.add_argument('--out', type=str, default='vis')
    p_render.add_argument('--ratio', type=float, default=0.10)
    p_render.add_argument('--num_frames', type=int, default=8)
    p_render.add_argument('--image_size', type=int, default=224)
    p_render.add_argument('--sample_idx', type=int, default=0)
    add_model_args(p_render)

    # all subcommand (export + render)
    p_all = sub.add_parser('all', help='Export mechanism data then render figures')
    p_all.add_argument('--ckpt', type=str, required=True)
    p_all.add_argument('--val_dir', type=str, default='dataset/touch_and_go')
    p_all.add_argument('--out', type=str, default='vis')
    p_all.add_argument('--ratio', type=float, default=0.10)
    p_all.add_argument('--datasets', type=str, default='toucHD')
    p_all.add_argument('--num_frames', type=int, default=8)
    p_all.add_argument('--image_size', type=int, default=224)
    p_all.add_argument('--sample_idx', type=int, default=0)
    add_model_args(p_all)

    args = parser.parse_args()

    if args.mode == 'export':
        run_export(args)
    elif args.mode == 'render':
        run_render(args)
    elif args.mode == 'all':
        run_export(args)
        run_render(args)


if __name__ == '__main__':
    main()
