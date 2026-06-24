"""Extract mechanism analysis data for paper figures.

Output per sequence (saved as .npy):
  basis_weights.npy        [n_frames, K]  – meta-net softmax over K basis matrices
  targets.npy              [n_frames, H, W] – input frames for overlay
  preds.npy                [n_frames, H, W] – reconstructions
  Per iteration i (0..7):
    dst_attn_i.npy         [n_frames, H, W] – spatial attention (mean over heads & points)
    dst_attn_i_full.npy    [n_frames, num_heads, num_points, H, W] – raw attention (i=0 only)
    lrta_attn_i.npy        [n_frames, num_heads, num_queries, T] – temporal cross-attention

Usage:
  python eval_mechanism.py                                    # all datasets, ratio=0.10
  python eval_mechanism.py --ratio 0.25                       # single ratio
  python eval_mechanism.py --datasets toucHD,TacQuad          # specific datasets
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

from model.model_3d import LSDUNet
from data_processor import collect_sequences, collect_ycb_by_object, collect_visgel_sequences


def load_model(cs_ratio, checkpoint_path, device='cuda:0'):
    model = LSDUNet(ratio=cs_ratio, iter_num=8, model_dim=64, patch=32, num_heads=8).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f"  [WARN] {len(result.missing_keys)} missing keys (new modules random init)")
    model.eval()
    return model


def load_frame(path):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert('L')
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])(img)
    return t


def build_clip(frame_buffer, center_idx, clip_len=8):
    n = len(frame_buffer)
    half = clip_len // 2
    indices = []
    for offset in range(-half, half):
        idx = max(0, min(n - 1, center_idx + offset))
        indices.append(idx)
    clip = torch.stack([frame_buffer[i] for i in indices], dim=0)
    clip = clip.unsqueeze(0)  # [1, T, 1, H, W]
    return clip


def extract_sequence_mechanisms(frame_paths, model, device, out_dir):
    """Run sliding-window inference, save all mechanism data per frame."""
    T_clip = 8
    n_frames = len(frame_paths)
    frame_buffer = [load_frame(p) for p in frame_paths]

    # ─── accumulate per-frame data ───
    basis_all = []       # [n, K]
    targets_list = []    # [n, H, W]
    preds_list = []      # [n, H, W]
    dst_raw = []         # per-iter: [n, H, P, H, W]
    dst_mean = []        # per-iter: [n, H, W]
    lrta_all = []        # per-iter: [n, H, Q, T]

    for i in tqdm(range(n_frames), desc='extracting'):
        clip = build_clip(frame_buffer, i, T_clip)
        with torch.no_grad():
            out, mech = model(clip.to(device), return_mechanism=True)

        # Centre-frame reconstruction & target
        pred = out[0, T_clip // 2, 0, :, :].cpu().numpy()
        target = frame_buffer[i].squeeze(0).cpu().numpy()
        targets_list.append(target)
        preds_list.append(pred)

        # Basis weights: [B*T, K] → take centre-frame weights
        bw = mech['basis_weights'].numpy()  # [8, K]
        basis_all.append(bw[T_clip // 2])

        # Per-iteration attention data
        for iter_key in sorted(mech['iter_data'].keys(),
                               key=lambda k: int(k.split('_')[1])):
            iter_data = mech['iter_data'][iter_key]

            # Spatial attn: [B, H, T, P, H, W]
            s_attn = iter_data['space_attn'].numpy()[0]  # [H, T, P, H, W]
            s_attn_center = s_attn[:, T_clip // 2]       # [H, P, H, W]

            # LRTA attn: [B, H, Q, T]
            l_attn = iter_data['lrta_attn'].numpy()[0]   # [H, Q, T]

            iter_idx = int(iter_key.split('_')[1])
            while len(dst_raw) <= iter_idx:
                dst_raw.append([])
                dst_mean.append([])
                lrta_all.append([])
            dst_raw[iter_idx].append(s_attn_center)       # [H, P, H, W]
            dst_mean[iter_idx].append(s_attn_center.mean(axis=(0, 1)))  # [H, W]
            lrta_all[iter_idx].append(l_attn)             # [H, Q, T]

    # ─── stack & save ───
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'basis_weights.npy'),
            np.stack(basis_all, axis=0))
    np.save(os.path.join(out_dir, 'targets.npy'),
            np.stack(targets_list, axis=0))
    np.save(os.path.join(out_dir, 'preds.npy'),
            np.stack(preds_list, axis=0))

    for idx in range(len(dst_mean)):
        np.save(os.path.join(out_dir, f'dst_attn_{idx}.npy'),
                np.stack(dst_mean[idx], axis=0))
        np.save(os.path.join(out_dir, f'lrta_attn_{idx}.npy'),
                np.stack(lrta_all[idx], axis=0))
        # raw spatial attn — only for iteration 0 (too large otherwise)
        if idx == 0:
            np.save(os.path.join(out_dir, f'dst_attn_{idx}_full.npy'),
                    np.stack(dst_raw[idx], axis=0))

    print(f"  Saved {n_frames} frames to {out_dir}")
    return basis_all, dst_mean, lrta_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.10,
                        help='CS sensing rate for mechanism analysis')
    parser.add_argument('--datasets', type=str, default='toucHD,TacQuad,yuan18,visgel',
                        help='Comma-separated datasets to analyze')
    parser.add_argument('--max_seqs', type=int, default=3,
                        help='Max sequences per dataset')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per sequence (0=all)')
    args = parser.parse_args()

    ckpt = f'./trained_model/lsdunet_{args.ratio:.2f}.pth'
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ratio, ckpt, device)
    print(f"Model loaded: ratio={args.ratio:.2f}")

    dataset_names = [d.strip() for d in args.datasets.split(',')]
    out_root = f'./eval_results/mechanism_ratio_{args.ratio:.2f}'

    # ─── ToucHD ───
    if 'toucHD' in dataset_names:
        root = './dataset/toucHD/train'
        if os.path.isdir(root):
            seqs = collect_sequences(root, min_frames=8)
            seq_items = sorted(seqs.items())[:args.max_seqs]
            print(f"\n=== ToucHD: {len(seq_items)} seqs ===")
            for seq_dir, frame_paths in seq_items:
                seq_name = os.path.basename(seq_dir)
                fp = frame_paths if args.max_frames <= 0 else frame_paths[:args.max_frames]
                out_dir = os.path.join(out_root, 'toucHD', seq_name)
                extract_sequence_mechanisms(fp, model, device, out_dir)

    # ─── TacQuad ───
    if 'TacQuad' in dataset_names:
        root = './dataset/tacquad/train'
        if os.path.isdir(root):
            objs = collect_ycb_by_object(root)
            tactile = {k: v for k, v in objs.items() if not k.endswith('_gt')}
            for obj_name in sorted(tactile.keys())[:args.max_seqs]:
                fp = tactile[obj_name]
                if args.max_frames > 0:
                    fp = fp[:args.max_frames]
                out_dir = os.path.join(out_root, 'TacQuad', obj_name)
                extract_sequence_mechanisms(fp, model, device, out_dir)

    # ─── Yuan18 ───
    if 'yuan18' in dataset_names:
        root = './dataset/yuan18/test'
        if os.path.isdir(root):
            seqs = collect_sequences(root, min_frames=8)
            seq_items = sorted(seqs.items())[:args.max_seqs]
            print(f"\n=== Yuan18: {len(seq_items)} seqs ===")
            for seq_dir, frame_paths in seq_items:
                seq_name = os.path.basename(seq_dir)
                fp = frame_paths if args.max_frames <= 0 else frame_paths[:args.max_frames]
                out_dir = os.path.join(out_root, 'yuan18', seq_name)
                extract_sequence_mechanisms(fp, model, device, out_dir)

    # ─── VisGel ───
    if 'visgel' in dataset_names:
        root = './dataset/visgel/images/touch'
        if os.path.isdir(root):
            seqs = collect_visgel_sequences(root, min_frames=8)
            seq_items = sorted(seqs.items())[:args.max_seqs]
            print(f"\n=== VisGel: {len(seq_items)} seqs ===")
            for rec_dir, frame_paths in seq_items:
                rec_name = os.path.basename(rec_dir)
                fp = frame_paths if args.max_frames <= 0 else frame_paths[:args.max_frames]
                out_dir = os.path.join(out_root, 'visgel', rec_name)
                extract_sequence_mechanisms(fp, model, device, out_dir)

    print(f"\nDone. Mechanism data saved to: {out_root}")


if __name__ == '__main__':
    main()
