import os
import csv
import warnings
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model.model_3d import LSDUNet
from data_processor import collect_ycb_by_object
from utils import rgb_to_ycbcr

warnings.filterwarnings("ignore")

model_3d = None
old_rate_3d = 10000


def load_3d_model(cs_ratio, checkpoint_path, iter_num=8, model_dim=16, patch=32):
    global model_3d, old_rate_3d
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_3d is None or cs_ratio != old_rate_3d:
        model_3d = LSDUNet(ratio=cs_ratio, iter_num=iter_num,
                            model_dim=model_dim, patch=patch).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_3d.load_state_dict(checkpoint, strict=True)
        model_3d.eval()
        old_rate_3d = cs_ratio
    return model_3d, device


def eval_3d_single(tactile_path, model, device, return_intermediates=False):
    img = Image.open(tactile_path).convert('RGB')
    img_t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])(img)
    x_ch = rgb_to_ycbcr(img_t)[0:1, :, :] / 255.0
    x_3d = x_ch.unsqueeze(0).unsqueeze(1).expand(1, 4, 1, 128, 128)
    with torch.no_grad():
        if return_intermediates:
            out, intermediates = model(x_3d.to(device), return_intermediates=True)
        else:
            out, _ = model(x_3d.to(device))
    pred = out[0, 2, 0, :, :].cpu().numpy()
    target = x_ch.cpu().numpy()
    if return_intermediates:
        inter_preds = [im[0, 2, 0, :, :].cpu().numpy() for im in intermediates]
        return pred, target, x_ch.cpu().numpy(), inter_preds
    return pred, target, x_ch.cpu().numpy()


# ─── 列名定义 ───
PER_OBJECT_COLUMNS = [
    'Object', 'Mode', 'N',
    'PSNR', 'SSIM', 'LPIPS',
    'PSNR_L', 'SSIM_L', 'RelErr_L',
    'PSNR_S', 'SSIM_S', 'RelErr_S',
    'ROI_PSNR', 'ROI_SSIM',
    'SCRG', 'BSF',
    'Force_RMSE',
]

PER_SAMPLE_COLUMNS = PER_OBJECT_COLUMNS + ['Image']


def write_object_row(writer, obj_name, mode, n, metrics):
    row = [
        obj_name, mode, n,
        round(metrics['PSNR'], 4), round(metrics['SSIM'], 4),
        round(metrics['LPIPS'], 4) if metrics['LPIPS'] is not None else 'N/A',
        round(metrics['PSNR_L'], 4), round(metrics['SSIM_L'], 4), round(metrics['RelErr_L'], 4),
        round(metrics['PSNR_S'], 4), round(metrics['SSIM_S'], 4), round(metrics['RelErr_S'], 4),
        round(metrics['ROI_PSNR'], 4), round(metrics['ROI_SSIM'], 4),
        round(metrics['SCRG'], 4), round(metrics['BSF'], 4),
        round(metrics['Force_RMSE'], 4) if metrics['Force_RMSE'] is not None else 'N/A',
    ]
    writer.writerow(row)


def write_sample_row(writer, obj_name, mode, impath, m):
    row = [
        obj_name, mode, 1,
        round(m['PSNR'], 4), round(m['SSIM'], 4),
        round(m['LPIPS'], 4) if m['LPIPS'] is not None else 'N/A',
        round(m['PSNR_L'], 4), round(m['SSIM_L'], 4), round(m['RelErr_L'], 4),
        round(m['PSNR_S'], 4), round(m['SSIM_S'], 4), round(m['RelErr_S'], 4),
        round(m['ROI_PSNR'], 4), round(m['ROI_SSIM'], 4),
        round(m['SCRG'], 4), round(m['BSF'], 4),
        round(m['Force_RMSE'], 4) if m['Force_RMSE'] is not None else 'N/A',
        impath,
    ]
    writer.writerow(row)


if __name__ == "__main__":
    from metrics import evaluate_all, format_rank_summary

    cs_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    ycb_root = './dataset/test/sim'
    ycb_objects = collect_ycb_by_object(ycb_root)
    tactile_objs = {k: v for k, v in ycb_objects.items() if not k.endswith('_gt')}
    gt_objs = {k.replace('_gt', ''): v for k, v in ycb_objects.items() if k.endswith('_gt')}

    for obj_name in sorted(tactile_objs.keys()):
        has_gt = obj_name in gt_objs and len(gt_objs[obj_name]) > 0
        imgs = tactile_objs[obj_name]
        status = f"{len(imgs)} imgs, GT available" if has_gt else f"{len(imgs)} imgs"
        print(f"[YCB-Sight|{obj_name}] {status}")

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
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'═' * 100}")
        print(f"Model: LSDUNet | Ratio: {sampling_rate} | Params: {params:,}")
        print(f"Checkpoint: {ckpt}")
        print(f"{'═' * 100}")

    # ─── 输出目录 ───
        out_dir = f'./eval_results/ratio_{sampling_rate}'
        os.makedirs(out_dir, exist_ok=True)

        # ─── CSV: per-object summary ───
        obj_csv = open(os.path.join(out_dir, 'per_object.csv'), 'w', newline='')
        obj_writer = csv.writer(obj_csv)
        obj_writer.writerow(PER_OBJECT_COLUMNS)

        # ─── CSV: per-sample detail ───
        sample_csv = open(os.path.join(out_dir, 'per_sample.csv'), 'w', newline='')
        sample_writer = csv.writer(sample_csv)
        sample_writer.writerow(PER_SAMPLE_COLUMNS)

        # ─── CSV: rank vs iterations ───
        rank_csv = open(os.path.join(out_dir, 'rank_vs_iter.csv'), 'w', newline='')
        rank_writer = csv.writer(rank_csv)
        rank_writer.writerow(['Object', 'Iteration', 'Avg_Rank'])

        # ═══════════════════════════════════════════════════════════════
        # Evaluation Mode 1: self-reference reconstruction (tactile -> reconstruction)
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 100)
        print(f"=== Per-object evaluation (tactile -> reconstruction self-reference) [ratio={sampling_rate}] ===")
        print("=" * 100)

        for obj_name in sorted(tactile_objs.keys()):
            imgs = tactile_objs[obj_name][:20]
            n = len(imgs)

            sum_psnr, sum_ssim, sum_lpips = 0.0, 0.0, 0.0
            sum_psnr_L, sum_ssim_L, sum_rel_err_L = 0.0, 0.0, 0.0
            sum_psnr_S, sum_ssim_S, sum_rel_err_S = 0.0, 0.0, 0.0
            sum_roi_psnr, sum_roi_ssim = 0.0, 0.0
            sum_scr_gain, sum_bsf = 0.0, 0.0
            all_ranks = []
            lpips_count = 0

            for impath in imgs:
                pred, target, _, inter_preds = eval_3d_single(
                    impath, model, device, return_intermediates=True)
                metrics = evaluate_all(pred, target, intermediate_preds=inter_preds)

                # ── 写入逐样本记录 ──
                write_sample_row(sample_writer, obj_name, 'self-ref', impath, metrics)

                sum_psnr += metrics['PSNR']
                sum_ssim += metrics['SSIM']
                if metrics['LPIPS'] is not None:
                    sum_lpips += metrics['LPIPS']
                    lpips_count += 1
                sum_psnr_L += metrics['PSNR_L']
                sum_ssim_L += metrics['SSIM_L']
                sum_rel_err_L += metrics['RelErr_L']
                sum_psnr_S += metrics['PSNR_S']
                sum_ssim_S += metrics['SSIM_S']
                sum_rel_err_S += metrics['RelErr_S']
                sum_roi_psnr += metrics['ROI_PSNR']
                sum_roi_ssim += metrics['ROI_SSIM']
                sum_scr_gain += metrics['SCRG']
                sum_bsf += metrics['BSF']
                if metrics['ranks']:
                    all_ranks.append(metrics['ranks'])

            # ── 写入物体级汇总 ──
            avg_metrics = {
                'PSNR': sum_psnr / n, 'SSIM': sum_ssim / n,
                'LPIPS': sum_lpips / lpips_count if lpips_count > 0 else None,
                'PSNR_L': sum_psnr_L / n, 'SSIM_L': sum_ssim_L / n, 'RelErr_L': sum_rel_err_L / n,
                'PSNR_S': sum_psnr_S / n, 'SSIM_S': sum_ssim_S / n, 'RelErr_S': sum_rel_err_S / n,
                'ROI_PSNR': sum_roi_psnr / n, 'ROI_SSIM': sum_roi_ssim / n,
                'SCRG': sum_scr_gain / n, 'BSF': sum_bsf / n,
                'Force_RMSE': None,
            }
            write_object_row(obj_writer, obj_name, 'self-ref', n, avg_metrics)

            # ── 写入 rank vs iter ──
            if all_ranks:
                avg_ranks = format_rank_summary(all_ranks)
                for i, r in enumerate(avg_ranks):
                    rank_writer.writerow([obj_name, i + 1, round(float(r), 2)])

            print(f"\n{'─' * 100}")
            print(f"  Object: {obj_name}  (imgs={n})")
            print(f"  {'─' * 80}")
            print(f"  [全局]        PSNR={sum_psnr/n:.2f} dB  |  SSIM={sum_ssim/n:.4f}"
                  + (f"  |  LPIPS={sum_lpips/lpips_count:.4f}" if lpips_count > 0 else f"  |  LPIPS=N/A"))
            print(f"  [低秩分量 L]  PSNR_L={sum_psnr_L/n:.2f} dB  |  SSIM_L={sum_ssim_L/n:.4f}  |  RelErr(L)={sum_rel_err_L/n:.4f}")
            print(f"  [稀疏分量 S]  PSNR_S={sum_psnr_S/n:.2f} dB  |  SSIM_S={sum_ssim_S/n:.4f}  |  RelErr(S)={sum_rel_err_S/n:.4f}")
            print(f"  [接触区域]    ROI-PSNR={sum_roi_psnr/n:.2f} dB  |  ROI-SSIM={sum_roi_ssim/n:.4f}")
            print(f"  [背景抑制]    SCRG={sum_scr_gain/n:.3f}  |  BSF={sum_bsf/n:.3f}")
            print(f"  [接触力RMSE]  N/A (无接触力标注)")
            if all_ranks:
                avg_ranks = format_rank_summary(all_ranks)
                rank_str = "  ".join([f"iter{i+1}={int(r)}" for i, r in enumerate(avg_ranks)])
                print(f"  [rank(L̂) vs iter]  avg rank:  {rank_str}")

        # ═══════════════════════════════════════════════════════════════
        # Evaluation Mode 2: tactile -> gt_height_map
        # ═══════════════════════════════════════════════════════════════
        print("\n\n" + "=" * 100)
        print(f"=== Per-object evaluation (tactile -> gt_height_map) [ratio={sampling_rate}] ===")
        print("=" * 100)

        g_psnr, g_ssim, g_lpips = 0.0, 0.0, 0.0
        g_psnr_L, g_ssim_L, g_rel_err_L = 0.0, 0.0, 0.0
        g_psnr_S, g_ssim_S, g_rel_err_S = 0.0, 0.0, 0.0
        g_roi_psnr, g_roi_ssim = 0.0, 0.0
        g_scr_gain, g_bsf = 0.0, 0.0
        g_all_ranks = []
        g_lpips_count = 0
        paired_total = 0

        for obj_name in sorted(tactile_objs.keys()):
            if obj_name not in gt_objs:
                continue
            imgs = tactile_objs[obj_name]
            gt_maps = gt_objs[obj_name]
            paired = min(len(imgs), len(gt_maps))
            if paired == 0:
                continue

            o_psnr, o_ssim, o_lpips = 0.0, 0.0, 0.0
            o_psnr_L, o_ssim_L, o_rel_err_L = 0.0, 0.0, 0.0
            o_psnr_S, o_ssim_S, o_rel_err_S = 0.0, 0.0, 0.0
            o_roi_psnr, o_roi_ssim = 0.0, 0.0
            o_scr_gain, o_bsf = 0.0, 0.0
            o_ranks = []
            o_lpips_count = 0

            for i in range(paired):
                pred, _, _, inter_preds = eval_3d_single(
                    imgs[i], model, device, return_intermediates=True)
                heightmap = np.load(gt_maps[i]).astype(np.float32)
                h_min, h_max = heightmap.min(), heightmap.max()
                if h_max > h_min:
                    heightmap = (heightmap - h_min) / (h_max - h_min)
                else:
                    heightmap = np.zeros_like(heightmap)
                heightmap = transforms.Resize((128, 128))(
                    torch.from_numpy(heightmap).unsqueeze(0)
                ).squeeze().numpy()

                metrics = evaluate_all(pred, heightmap, intermediate_preds=inter_preds)

                # ── 写入逐样本记录 ──
                write_sample_row(sample_writer, obj_name, 'tactile2height', imgs[i], metrics)

                o_psnr += metrics['PSNR']
                o_ssim += metrics['SSIM']
                if metrics['LPIPS'] is not None:
                    o_lpips += metrics['LPIPS']
                    o_lpips_count += 1
                o_psnr_L += metrics['PSNR_L']
                o_ssim_L += metrics['SSIM_L']
                o_rel_err_L += metrics['RelErr_L']
                o_psnr_S += metrics['PSNR_S']
                o_ssim_S += metrics['SSIM_S']
                o_rel_err_S += metrics['RelErr_S']
                o_roi_psnr += metrics['ROI_PSNR']
                o_roi_ssim += metrics['ROI_SSIM']
                o_scr_gain += metrics['SCRG']
                o_bsf += metrics['BSF']
                if metrics['ranks']:
                    o_ranks.append(metrics['ranks'])

            # ── 写入物体级汇总 ──
            avg_metrics = {
                'PSNR': o_psnr / paired, 'SSIM': o_ssim / paired,
                'LPIPS': o_lpips / o_lpips_count if o_lpips_count > 0 else None,
                'PSNR_L': o_psnr_L / paired, 'SSIM_L': o_ssim_L / paired, 'RelErr_L': o_rel_err_L / paired,
                'PSNR_S': o_psnr_S / paired, 'SSIM_S': o_ssim_S / paired, 'RelErr_S': o_rel_err_S / paired,
                'ROI_PSNR': o_roi_psnr / paired, 'ROI_SSIM': o_roi_ssim / paired,
                'SCRG': o_scr_gain / paired, 'BSF': o_bsf / paired,
                'Force_RMSE': None,
            }
            write_object_row(obj_writer, obj_name, 'tactile2height', paired, avg_metrics)

            print(f"\n{'─' * 100}")
            print(f"  Object: {obj_name}  (pairs={paired})")
            print(f"  {'─' * 80}")
            print(f"  [全局]        PSNR={o_psnr/paired:.2f} dB  |  SSIM={o_ssim/paired:.4f}"
                  + (f"  |  LPIPS={o_lpips/o_lpips_count:.4f}" if o_lpips_count > 0 else f"  |  LPIPS=N/A"))
            print(f"  [低秩分量 L]  PSNR_L={o_psnr_L/paired:.2f} dB  |  SSIM_L={o_ssim_L/paired:.4f}  |  RelErr(L)={o_rel_err_L/paired:.4f}")
            print(f"  [稀疏分量 S]  PSNR_S={o_psnr_S/paired:.2f} dB  |  SSIM_S={o_ssim_S/paired:.4f}  |  RelErr(S)={o_rel_err_S/paired:.4f}")
            print(f"  [接触区域]    ROI-PSNR={o_roi_psnr/paired:.2f} dB  |  ROI-SSIM={o_roi_ssim/paired:.4f}")
            print(f"  [背景抑制]    SCRG={o_scr_gain/paired:.3f}  |  BSF={o_bsf/paired:.3f}")
            print(f"  [接触力RMSE]  N/A (无接触力标注)")
            if o_ranks:
                avg_ranks = format_rank_summary(o_ranks)
                rank_str = "  ".join([f"iter{i+1}={int(r)}" for i, r in enumerate(avg_ranks)])
                print(f"  [rank(L̂) vs iter]  avg rank:  {rank_str}")

            g_psnr += o_psnr
            g_ssim += o_ssim
            g_lpips += o_lpips
            g_lpips_count += o_lpips_count
            g_psnr_L += o_psnr_L
            g_ssim_L += o_ssim_L
            g_rel_err_L += o_rel_err_L
            g_psnr_S += o_psnr_S
            g_ssim_S += o_ssim_S
            g_rel_err_S += o_rel_err_S
            g_roi_psnr += o_roi_psnr
            g_roi_ssim += o_roi_ssim
            g_scr_gain += o_scr_gain
            g_bsf += o_bsf
            g_all_ranks.extend(o_ranks)
            paired_total += paired

        # ─── 当前采样率全局汇总 ───
        if paired_total > 0:
            overall_metrics = {
                'PSNR': g_psnr / paired_total, 'SSIM': g_ssim / paired_total,
                'LPIPS': g_lpips / g_lpips_count if g_lpips_count > 0 else None,
                'PSNR_L': g_psnr_L / paired_total, 'SSIM_L': g_ssim_L / paired_total, 'RelErr_L': g_rel_err_L / paired_total,
                'PSNR_S': g_psnr_S / paired_total, 'SSIM_S': g_ssim_S / paired_total, 'RelErr_S': g_rel_err_S / paired_total,
                'ROI_PSNR': g_roi_psnr / paired_total, 'ROI_SSIM': g_roi_ssim / paired_total,
                'SCRG': g_scr_gain / paired_total, 'BSF': g_bsf / paired_total,
                'Force_RMSE': None,
            }
            write_object_row(obj_writer, 'OVERALL', 'tactile2height', paired_total, overall_metrics)
            # ── 写入跨采样率汇总 ──
            summary_writer.writerow([
                f'ratio={sampling_rate}', 'tactile2height', paired_total,
                round(overall_metrics['PSNR'], 4), round(overall_metrics['SSIM'], 4),
                round(overall_metrics['LPIPS'], 4) if overall_metrics['LPIPS'] is not None else 'N/A',
                round(overall_metrics['PSNR_L'], 4), round(overall_metrics['SSIM_L'], 4), round(overall_metrics['RelErr_L'], 4),
                round(overall_metrics['PSNR_S'], 4), round(overall_metrics['SSIM_S'], 4), round(overall_metrics['RelErr_S'], 4),
                round(overall_metrics['ROI_PSNR'], 4), round(overall_metrics['ROI_SSIM'], 4),
                round(overall_metrics['SCRG'], 4), round(overall_metrics['BSF'], 4), 'N/A',
            ])

            # ── 全局 rank vs iter ──
            if g_all_ranks:
                avg_ranks = format_rank_summary(g_all_ranks)
                for i, r in enumerate(avg_ranks):
                    rank_writer.writerow(['OVERALL', i + 1, round(float(r), 2)])

            print(f"\n{'═' * 100}")
            print(f"  [ratio={sampling_rate}] OVERALL SUMMARY  (pairs={paired_total})")
            print(f"  {'═' * 100}")
            print(f"  [全局]        PSNR={g_psnr/paired_total:.2f} dB  |  SSIM={g_ssim/paired_total:.4f}"
                  + (f"  |  LPIPS={g_lpips/g_lpips_count:.4f}" if g_lpips_count > 0 else f"  |  LPIPS=N/A"))
            print(f"  [低秩分量 L]  PSNR_L={g_psnr_L/paired_total:.2f} dB  |  SSIM_L={g_ssim_L/paired_total:.4f}  |  RelErr(L)={g_rel_err_L/paired_total:.4f}")
            print(f"  [稀疏分量 S]  PSNR_S={g_psnr_S/paired_total:.2f} dB  |  SSIM_S={g_ssim_S/paired_total:.4f}  |  RelErr(S)={g_rel_err_S/paired_total:.4f}")
            print(f"  [接触区域]    ROI-PSNR={g_roi_psnr/paired_total:.2f} dB  |  ROI-SSIM={g_roi_ssim/paired_total:.4f}")
            print(f"  [背景抑制]    SCRG={g_scr_gain/paired_total:.3f}  |  BSF={g_bsf/paired_total:.3f}")
            print(f"  [接触力RMSE]  N/A (无接触力标注)")
            if g_all_ranks:
                avg_ranks = format_rank_summary(g_all_ranks)
                rank_str = "  ".join([f"iter{i+1}={int(r)}" for i, r in enumerate(avg_ranks)])
                print(f"  [rank(L̂) vs iter]  avg rank:  {rank_str}")
            print(f"{'═' * 100}")

        # ─── 关闭当前采样率的 CSV ───
        obj_csv.close()
        sample_csv.close()
        rank_csv.close()
        print(f"\nResults saved to: {out_dir}/")

    # ─── 关闭汇总 CSV ───
    summary_csv.close()
    print(f"\nCross-ratio summary saved to: ./eval_results/summary_all_ratios.csv")
