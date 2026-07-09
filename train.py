import os
import warnings
import csv

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import *

setup_seed(3407)
# TF32: faster matmul on Blackwell/Ampere GPUs with negligible precision loss
torch.set_float32_matmul_precision('high')
import argparse
from model.model_3d import *
import torch.optim as optim
from data_processor import data_loader_3d
from trainer import train_3d, valid_3d, EMA
import time

warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layerscale_weights(model):
    """Collect average LayerScale weights from all DSTLayer modules."""
    w_time_vals, w_space_vals, w_lrta_vals, w_ffn_vals = [], [], [], []
    for _, module in model.named_modules():
        if isinstance(module, DSTLayer):
            w_time_vals.append(module.w_time.data.mean().item())
            w_space_vals.append(module.w_space.data.mean().item())
            w_lrta_vals.append(module.w_history.data.mean().item())  # history replaces lrta
            w_ffn_vals.append(module.w_ffn.data.mean().item())
    if not w_time_vals:
        return 0.0, 0.0, 0.0, 0.0
    return (sum(w_time_vals) / len(w_time_vals),
            sum(w_space_vals) / len(w_space_vals),
            sum(w_lrta_vals) / len(w_lrta_vals),
            sum(w_ffn_vals) / len(w_ffn_vals))


def main(cs_ratio):
    ck_file_name = f'lsdunet_{cs_ratio:.2f}.pth'
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = (rank == 0)

    os.makedirs(args.save_dir, exist_ok=True)
    model_dir = "./%s/%s-3d_ratio_%.2f" % (args.save_dir, args.model, cs_ratio)
    if is_main:
        os.makedirs(model_dir, exist_ok=True)

    model = LSDUNet(ratio=cs_ratio, iter_num=args.iter_num,
                     model_dim=args.model_dim, patch=args.patch,
                     in_ch=3, ls_rank=args.ls_rank,
                     num_frames=args.num_frames).to(device)

    # Resume from checkpoint
    start_epoch = 1
    best_val_psnr = 0.0
    resume_ckpt = None
    if args.resume:
        ckpt_path = os.path.join(model_dir, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            start_epoch = resume_ckpt['epoch'] + 1
            best_val_psnr = resume_ckpt['best_val_psnr']
            if is_main:
                print(f"[Resume] Loaded checkpoint from epoch {resume_ckpt['epoch']}, "
                      f"best PSNR: {best_val_psnr:.4f}, resuming at epoch {start_epoch}")
        elif is_main:
            print(f"[Resume] No checkpoint found at {ckpt_path}, starting from scratch")

    # Wrap with DDP
    # find_unused_parameters=True: gdb[0].iter_gate is never used (first iteration
    # has no x_prev), so its params don't receive gradient. True allows DDP to
    # handle this without hanging.
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        base_model = model.module
    else:
        base_model = model

    # EMA (disabled in debug mode, warmup decay for faster early tracking)
    ema = EMA(base_model, decay=args.ema_decay) if not args.debug else None

    # 学习率分组: 不同模块用不同学习率
    cs_params = list(base_model.adaptive_s.parameters())
    sfeat_params = list(base_model.S_feat.parameters())
    ls_params = []
    for gdb in base_model.gdb:
        ls_params.extend(list(gdb.ls_decomp.parameters()))
    special_ids = set(id(p) for p in cs_params + sfeat_params + ls_params)
    other_params = [p for p in model.parameters() if id(p) not in special_ids]

    optimizer = optim.AdamW([
        {'params': cs_params, 'lr': args.lr * 0.1},      # CS 采样矩阵: 小 lr 避免剧烈变化
        {'params': sfeat_params, 'lr': args.lr * 0.5},   # 特征 CS: 中 lr
        {'params': ls_params, 'lr': args.lr * 0.5},      # L+S 分解: 中 lr
        {'params': other_params, 'lr': args.lr},          # 其他: 标准 lr
    ], lr=args.lr, weight_decay=args.wd)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warm_epochs,
                                       eta_min=args.flr)

    def warmup_lambda(epoch):
        if epoch < args.warm_epochs:
            return float(epoch + 1) / max(args.warm_epochs, 1)
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    # Resume: load optimizer and scheduler states
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        warmup_scheduler.load_state_dict(resume_ckpt['warmup_scheduler_state_dict'])
        main_scheduler.load_state_dict(resume_ckpt['main_scheduler_state_dict'])

    train_loader, val_loader, train_sampler, val_sampler = data_loader_3d(
        args, train_dir=args.train_data, val_dir=args.val_dir,
        train_split=args.train_split or None,
        val_split=args.val_split or None,
        ddp=dist.is_initialized())

    if is_main:
        print("\n" + "=" * 70)
        print("Model: LSDUNet (3D Spatiotemporal) | DDP: %d GPU(s)" % world_size)
        print("Sensing Rate: %.2f\nEpoch: %d\nInitial LR: %f\nParameter: %.0f" % (
            cs_ratio, args.epochs, args.lr, count_parameters(model)))
        print("Volume frames: %d" % args.num_frames)
        n_train_seqs = len(set(s[0] for s in train_loader.dataset.samples))
        n_val_seqs = len(set(s[0] for s in val_loader.dataset.samples))
        print("Train: %d seqs → %d volumes" % (n_train_seqs, len(train_loader.dataset)))
        print("Val:   %d seqs → %d volumes" % (n_val_seqs, len(val_loader.dataset)))
        print("Effective batch: %d (per GPU: %d × grad_accum: %d × %d GPUs)" %
              (args.batch_size * args.grad_accum * world_size,
               args.batch_size, args.grad_accum, world_size))

    if is_main:
        print('Start training---------------------------------------------------------')

    # ---- 日志初始化 ----
    log_path = os.path.join(model_dir, 'train_log.csv')
    if is_main:
        writer = SummaryWriter(log_dir=model_dir)
        log_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
        log_file = open(log_path, 'a' if (args.resume and log_exists) else 'w', newline='')
        csv_writer = csv.writer(log_file)
        if not (args.resume and log_exists):
            csv_writer.writerow(['epoch', 'train_loss', 'loss_type', 'val_psnr', 'val_ssim', 'val_lpips',
                                 'val_edge_psnr', 'val_roi_psnr', 'val_roi_ssim',
                                 'lr', 'best_psnr', 'time_s',
                                 'w_time', 'w_space', 'w_lrta', 'w_ffn'])
        log_file.flush()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            start_ = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            if is_main:
                print('current lr {:.5e}'.format(current_lr))

            # Set epoch for DistributedSampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            loss = train_3d(train_loader, base_model, optimizer, device,
                            grad_clip=args.grad_clip, grad_accum=args.grad_accum,
                            ddp_model=model if dist.is_initialized() else None,
                            ema=ema,
                            w_edge=args.w_edge, w_freq=args.w_freq,
                            w_ssim=args.w_ssim,
                            w_ortho=args.w_ortho, w_nll=args.w_nll,
                            w_lowrank=args.w_lowrank, w_sparse=args.w_sparse,
                            epoch=epoch, warm_epochs=args.warm_epochs)

            if epoch < args.warm_epochs:
                warmup_scheduler.step()
            else:
                main_scheduler.step()

            if is_main:
                print_data = "[%02d/%02d] Train Loss: %.5f" % (epoch, args.epochs, loss)
                print(print_data)
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('LR', current_lr, epoch)

            # Validation (skip on non-validation epochs, except last epoch)
            # Collect uncertainty calibration only on the final epoch to save time.
            do_validate = (epoch % args.val_interval == 0) or (epoch == args.epochs)
            collect_unc = (epoch == args.epochs) and not args.debug
            if do_validate:
                val_result = valid_3d(val_loader, base_model, test_device,
                                      ddp=dist.is_initialized(), ema=ema,
                                      collect_uncertainty=collect_unc)
                val_psnr, val_ssim = val_result[0], val_result[1]
                val_lpips = val_result[2]
                val_edge_psnr = val_result[3]
                val_roi_psnr = val_result[4]
                val_roi_ssim = val_result[5]
                val_ece = val_result[6] if len(val_result) > 6 else None
                val_brier = val_result[7] if len(val_result) > 7 else None

                if is_main:
                    print("Val--PSNR: %.2f--SSIM: %.4f" % (val_psnr, val_ssim), end='')
                    if val_lpips is not None:
                        print("--LPIPS: %.4f" % val_lpips, end='')
                    print("--Edge: %.2f--ROI_PSNR: %.2f--ROI_SSIM: %.4f" %
                          (val_edge_psnr, val_roi_psnr, val_roi_ssim), end='')
                    if val_ece is not None:
                        print("--ECE: %.4f--Brier: %.4f" % (val_ece, val_brier), end='')
                    print()
                    writer.add_scalar('Metrics/val_psnr', val_psnr, epoch)
                    writer.add_scalar('Metrics/val_ssim', val_ssim, epoch)
                    if val_lpips is not None:
                        writer.add_scalar('Metrics/val_lpips', val_lpips, epoch)
                    writer.add_scalar('Metrics/val_edge_psnr', val_edge_psnr, epoch)
                    writer.add_scalar('Metrics/val_roi_psnr', val_roi_psnr, epoch)
                    writer.add_scalar('Metrics/val_roi_ssim', val_roi_ssim, epoch)
                    if val_ece is not None:
                        writer.add_scalar('Metrics/val_ece', val_ece, epoch)
                        writer.add_scalar('Metrics/val_brier', val_brier, epoch)

                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    if is_main:
                        model_to_save = model.module if dist.is_initialized() else model
                        torch.save(model_to_save.state_dict(), "%s/best.pth" % model_dir)
                        torch.save(model_to_save.state_dict(), "./trained_model/%s" % ck_file_name)
                        print("  >> Best model saved (PSNR: %.2f)" % val_psnr)
            else:
                val_psnr = val_ssim = val_lpips = None
                val_edge_psnr = val_roi_psnr = val_roi_ssim = None
                if is_main:
                    print("Val--skipped (val_interval=%d)" % args.val_interval)

            elapsed = time.time() - start_
            if is_main:
                print('Running time: {:.2f} seconds'.format(elapsed))
                print()

            # Save training checkpoint for resume
            if is_main:
                model_to_save = model.module if dist.is_initialized() else model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                    'main_scheduler_state_dict': main_scheduler.state_dict(),
                    'best_val_psnr': best_val_psnr,
                }
                torch.save(checkpoint, os.path.join(model_dir, 'checkpoint.pth'))

            # LayerScale weights
            w_time, w_space, w_lrta, w_ffn = get_layerscale_weights(model)
            if is_main:
                writer.add_scalar('Weights/w_time', w_time, epoch)
                writer.add_scalar('Weights/w_space', w_space, epoch)
                writer.add_scalar('Weights/w_lrta', w_lrta, epoch)
                writer.add_scalar('Weights/w_ffn', w_ffn, epoch)

                csv_writer.writerow([epoch, f'{loss:.6f}', 'MSE',
                                     f'{val_psnr:.4f}' if val_psnr is not None else 'N/A',
                                     f'{val_ssim:.6f}' if val_ssim is not None else 'N/A',
                                     f'{val_lpips:.6f}' if val_lpips is not None else 'N/A',
                                     f'{val_edge_psnr:.4f}' if val_edge_psnr is not None else 'N/A',
                                     f'{val_roi_psnr:.4f}' if val_roi_psnr is not None else 'N/A',
                                     f'{val_roi_ssim:.6f}' if val_roi_ssim is not None else 'N/A',
                                     f'{current_lr:.2e}',
                                     f'{best_val_psnr:.4f}', f'{elapsed:.1f}',
                                     f'{w_time:.6e}', f'{w_space:.6e}',
                                    f'{w_lrta:.6e}', f'{w_ffn:.6e}'])
                log_file.flush()
    finally:
        # Ensure log file and writer are closed even on exception/crash
        if is_main:
            log_file.close()
            writer.close()

    # ─── 训练结束后计算效率指标 (仅 rank 0) ───
    from metrics import get_efficiency_metrics
    model_for_eval = model.module if dist.is_initialized() else model
    if is_main:
        eff = get_efficiency_metrics(
            model_for_eval,
            input_shape=(1, args.num_frames, 3, args.image_size, args.image_size),
            device=test_device, verbose=False)
    else:
        eff = {'Params': 0.0, 'FLOPs': None, 'FPS': 0.0}
    eff['Params'] = sum(p.numel() for p in model_for_eval.parameters() if p.requires_grad) / 1e6

    if is_main:
        print('Training finished. Best val PSNR: %.2f | Params: %.3fM | FLOPs: %sG | FPS: %.1f' %
              (best_val_psnr, eff['Params'],
               f'{eff["FLOPs"]:.3f}' if eff['FLOPs'] is not None else 'N/A',
               eff['FPS']))
    if dist.is_initialized():
        dist.barrier()
    return best_val_psnr, eff


if __name__ == '__main__':
    # ─── DDP 初始化 ───
    if 'LOCAL_RANK' in os.environ:
        local_rank, world_size = ddp_setup()
    else:
        local_rank, world_size = 0, 1

    # 初始化 device（必须在 DDP 之后）
    device = init_device()
    test_device = device

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSDUNet', help='model name')
    parser.add_argument('--warm_epochs', default=5, type=int, help='linear warmup epochs')
    parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size (batch=8 fits 16GB at 9.15 GiB)')
    parser.add_argument('--image-size', default=16 * 14, type=int, metavar='N', help='(default: 224)')
    parser.add_argument('--num_frames', default=8, type=int, help='frames per 3D volume')
    parser.add_argument('--max_frames_per_seq', default=500, type=int, help='max frames per sequence (0=unlimited)')
    parser.add_argument('--grad_accum', default=2, type=int, help='gradient accumulation steps')
    parser.add_argument('--patch', default=32, type=int, help='CS sampling patch size')
    parser.add_argument('--lr', '--learning_rate', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--flr', '--final_learning_rate', default=1e-5, type=float, help='final learning rate')
    parser.add_argument('--wd', '--weight_decay', default=0.05, type=float, help='AdamW weight decay')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping norm')
    parser.add_argument('--save_dir', help='trained models', default='trained_model', type=str)
    parser.add_argument('--iter_num', type=int, default=6, help='deep unfolding iterations (lite: 4, original: 8)')
    parser.add_argument('--model_dim', type=int, default=64, help='feature dimension (lite: 32, original: 64)')
    parser.add_argument('--train_data', type=str, default='dataset/toucHD/train', help='train dataset path (ToucHD gelsight, 142 seqs)')
    parser.add_argument('--val_dir', type=str, default='dataset/touch_and_go', help='val dataset path (Touch and Go, 142 seqs)')
    parser.add_argument('--train_split', type=str, default='', help='frame-level split file for train filtering')
    parser.add_argument('--val_split', type=str, default='', help='frame-level split file for val filtering')
    parser.add_argument('--ratios', type=str, default='0.01,0.04,0.10,0.25,0.50',
                        help='CS sensing rates, comma-separated (default: 0.01,0.04,0.10,0.25,0.50)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from last checkpoint in model_dir')
    parser.add_argument('--val_interval', default=5, type=int,
                        help='validate every N epochs (5=every 5 epochs, always validates last epoch)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode: disable EMA and other production-only features')
    # Loss weights (for grid-search / ablation study)
    parser.add_argument('--w_edge', type=float, default=0.1, help='weight for Sobel edge loss')
    parser.add_argument('--w_freq', type=float, default=0.01, help='weight for DWT wavelet loss')
    parser.add_argument('--w_ssim', type=float, default=0.1, help='weight for differentiable SSIM loss')
    parser.add_argument('--w_ortho', type=float, default=0.01, help='weight for sampling-matrix orthogonality loss')
    parser.add_argument('--w_nll', type=float, default=0.01, help='weight for uncertainty NLL loss')
    parser.add_argument('--w_lowrank', type=float, default=0.01, help='weight for L+S low-rank regularization')
    parser.add_argument('--w_sparse', type=float, default=0.01, help='weight for L+S sparsity regularization')
    parser.add_argument('--ls_rank', type=int, default=4, help='rank for L+S decomposition bottleneck')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay (final, after warmup)')
    args = parser.parse_args()

    cs_ratios = [float(x.strip()) for x in args.ratios.split(',')]
    total_start = time.time()

    # 汇总日志 (仅 rank 0)
    summary_path = os.path.join(args.save_dir, 'summary_all_ratios.csv')
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = (rank == 0)

    if is_main:
        file_exists = os.path.exists(summary_path)
        summary_file = open(summary_path, 'a', newline='')
        summary_writer = csv.writer(summary_file)
        if not file_exists:
            summary_writer.writerow(['sensing_rate', 'best_val_psnr', 'total_params',
                                     'FLOPs(G)', 'FPS', 'total_time_s'])
            summary_file.flush()

    try:
        for cs_ratio in cs_ratios:
            ratio_start = time.time()
            if is_main:
                print("\n" + "=" * 70)
                print(">>> 开始训练: sensing_rate = %.2f" % cs_ratio)
                print("=" * 70)
            best_psnr, eff = main(cs_ratio)

            # 显式释放上一个 ratio 的显存，避免碎片化
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - ratio_start
            if is_main:
                print(">>> sensing_rate = %.2f 完成, 耗时: %.1f 分钟, Best PSNR: %.2f" %
                      (cs_ratio, elapsed / 60, best_psnr))
                summary_writer.writerow([f'{cs_ratio:.2f}', f'{best_psnr:.4f}',
                                         f'{eff["Params"]:.3f}',
                                         f'{eff["FLOPs"]:.3f}' if eff['FLOPs'] is not None else 'N/A',
                                         f'{eff["FPS"]:.1f}',
                                         f'{elapsed:.1f}'])
                summary_file.flush()
    finally:
        if is_main:
            summary_file.close()

    if is_main:
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print("全部压缩比训练完成! 总耗时: %.1f 分钟" % (total_elapsed / 60))
        print("汇总日志: %s" % summary_path)
        print("=" * 70)

    if dist.is_initialized():
        dist.destroy_process_group()
