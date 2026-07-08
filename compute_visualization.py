"""Visualization tool for LSDUNet interpretability figures (paper appendix).

Generates:
  1. Per-iteration reconstruction (deep unfolding progression)
  2. Deformable spatial attention maps (offset magnitudes)
  3. TactileHistoryCompressor query weights & temporal attention
  4. AdaptiveSModule basis weights (sampling matrix interpretability)
  5. Uncertainty heatmaps (per-pixel variance)

Usage:
  python compute_visualization.py --ckpt trained_model/lsdunet_0.10.pth \
      --val_dir dataset/touch_and_go --out vis/
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model.model_3d import LSDUNet
from data_processor import collect_sequences, SequenceVolumeDataset
import torchvision.transforms as T


def load_model(ckpt_path, ratio, device, iter_num=6, model_dim=64, patch=32):
    model = LSDUNet(ratio=ratio, iter_num=iter_num, model_dim=model_dim,
                    patch=patch).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state if 'model_state_dict' not in state else state['model_state_dict'])
    model.eval()
    return model


def get_sample(val_dir, num_frames=8, image_size=224, idx=0):
    seqs = collect_sequences(val_dir, min_frames=num_frames)
    if not seqs:
        raise FileNotFoundError(f"No sequences with >= {num_frames} frames in {val_dir}")
    transform = T.Compose([T.Resize((256, 256)), T.CenterCrop(image_size), T.ToTensor()])
    ds = SequenceVolumeDataset(seqs, num_frames=num_frames, transform=transform)
    vol, _ = ds[idx]
    return vol.unsqueeze(0)  # [1, T, C, H, W]


@torch.no_grad()
def visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, args.ratio, device,
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
        # basis_weights (sampling matrix selection)
        bw = mech_data.get('basis_weights', None)
        if bw is not None:
            bw = bw[0].numpy()  # [num_basis]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(bw)), bw)
            ax.set_xlabel('Basis index')
            ax.set_ylabel('Weight')
            ax.set_title('AdaptiveSModule basis weights (input-dependent)')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'basis_weights.png'), dpi=150)
            plt.close()
            print(f"  [saved] basis_weights.png")

        # Per-iteration spatial attention (offset magnitude)
        for i, key in enumerate([k for k in mech_data.keys() if k.startswith('iter_')]):
            mech = mech_data[key]
            s_attn = mech.get('space_attn', None)
            if s_attn is not None:
                # offset magnitude averaged over channels
                mag = s_attn[0].abs().mean(dim=0).cpu().numpy()  # [T, H, W]
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
                attn = h_attn[0].cpu().numpy()  # [num_queries, T]
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
            unc_np = unc[0].squeeze().cpu().numpy()  # [T, H, W]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='path to LSDUNet checkpoint')
    parser.add_argument('--val_dir', type=str, default='dataset/touch_and_go')
    parser.add_argument('--out', type=str, default='vis', help='output directory')
    parser.add_argument('--ratio', type=float, default=0.10)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--sample_idx', type=int, default=0)
    # Model architecture params (must match training config)
    parser.add_argument('--iter_num', type=int, default=6, help='deep unfolding iterations')
    parser.add_argument('--model_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--patch', type=int, default=32, help='CS sampling patch size')
    args = parser.parse_args()
    visualize(args)
