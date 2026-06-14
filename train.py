import warnings
import csv

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils import *

setup_seed(3407)
import argparse
from model.model_3d import *
import torch.optim as optim
from data_processor import data_loader_3d
from trainer import train_3d, valid_3d
import time

warnings.filterwarnings("ignore")


def loss_fun(X, Y):
    return F.mse_loss(X, Y)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(cs_ratio):
    ck_file_name = f'lsdunet_{cs_ratio}.pth'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_dir = "./%s/%s-3d_ratio_%.2f" % (args.save_dir, args.model, cs_ratio)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model = LSDUNet(ratio=cs_ratio, iter_num=args.iter_num,
                     model_dim=args.model_dim, patch=args.patch).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs + 1, eta_min=args.flr)

    train_loader, val_loader = data_loader_3d(
        args, train_dir=args.train_data, val_dir=args.val_dir,
        train_split=args.train_split or None,
        val_split=args.val_split or None)

    criterion = loss_fun

    print("\n" + "=" * 70)
    print("Model: LSDUNet (3D Spatiotemporal)")
    print("Sensing Rate: %.2f\nEpoch: %d\nInitial LR: %f\nParameter: %.0f" % (
        cs_ratio, args.epochs, args.lr, count_parameters(model)))
    print("Volume frames: %d" % args.num_frames)
    n_train_seqs = len(set(s[0] for s in train_loader.dataset.samples))
    n_val_seqs = len(set(s[0] for s in val_loader.dataset.samples))
    print("Train: %d seqs → %d volumes" % (n_train_seqs, len(train_loader.dataset)))
    print("Val:   %d seqs → %d volumes" % (n_val_seqs, len(val_loader.dataset)))

    best_val_psnr = 0
    print('Start training---------------------------------------------------------')

    # ---- 日志初始化 ----
    writer = SummaryWriter(log_dir=model_dir)
    log_path = os.path.join(model_dir, 'train_log.csv')
    log_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips',
                         'val_edge_psnr', 'val_roi_psnr', 'val_roi_ssim',
                         'lr', 'best_psnr', 'time_s'])

    for epoch in range(1, args.epochs + 1):
        start_ = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print('current lr {:.5e}'.format(current_lr))
        loss = train_3d(train_loader, model, criterion, optimizer, device)

        scheduler.step()
        print_data = "[%02d/%02d] Train Loss: %.5f" % (epoch, args.epochs, loss)
        print(print_data)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if epoch % 1 == 0:
            val_result = valid_3d(val_loader, model, test_device)
            val_psnr, val_ssim = val_result[0], val_result[1]
            val_lpips = val_result[2]
            val_edge_psnr = val_result[3]
            val_roi_psnr = val_result[4]
            val_roi_ssim = val_result[5]
            print("Val--PSNR: %.2f--SSIM: %.4f" % (val_psnr, val_ssim), end='')
            if val_lpips is not None:
                print("--LPIPS: %.4f" % val_lpips, end='')
            print("--Edge: %.2f--ROI_PSNR: %.2f--ROI_SSIM: %.4f" %
                  (val_edge_psnr, val_roi_psnr, val_roi_ssim))
            writer.add_scalar('Metrics/val_psnr', val_psnr, epoch)
            writer.add_scalar('Metrics/val_ssim', val_ssim, epoch)
            if val_lpips is not None:
                writer.add_scalar('Metrics/val_lpips', val_lpips, epoch)
            writer.add_scalar('Metrics/val_edge_psnr', val_edge_psnr, epoch)
            writer.add_scalar('Metrics/val_roi_psnr', val_roi_psnr, epoch)
            writer.add_scalar('Metrics/val_roi_ssim', val_roi_ssim, epoch)

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), "%s/best.pth" % model_dir)
                torch.save(model.state_dict(), "./trained_model/%s" % ck_file_name)
                print("  >> Best model saved (PSNR: %.2f)" % val_psnr)

        elapsed = time.time() - start_
        print('Running time: {:.2f} seconds'.format(elapsed))
        print()

        csv_writer.writerow([epoch, f'{loss:.6f}', f'{val_psnr:.4f}',
                             f'{val_ssim:.6f}',
                             f'{val_lpips:.6f}' if val_lpips is not None else 'N/A',
                             f'{val_edge_psnr:.4f}', f'{val_roi_psnr:.4f}',
                             f'{val_roi_ssim:.6f}',
                             f'{current_lr:.2e}',
                             f'{best_val_psnr:.4f}', f'{elapsed:.1f}'])

    log_file.close()
    writer.close()

    # ─── 训练结束后计算效率指标 ───
    from metrics import get_efficiency_metrics
    eff = get_efficiency_metrics(model, device=test_device, verbose=False)

    print('Training finished. Best val PSNR: %.2f | Params: %.3fM | FLOPs: %sG | FPS: %.1f' %
          (best_val_psnr, eff['Params'],
           f'{eff["FLOPs"]:.3f}' if eff['FLOPs'] is not None else 'N/A',
           eff['FPS']))
    return best_val_psnr, eff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSDUNet', help='model name')
    parser.add_argument('--warm_epochs', default=1, type=int, help='number of epochs to warm up')
    parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')
    parser.add_argument('--image-size', default=16 * 6, type=int, metavar='N', help='(default: 96)')
    parser.add_argument('--num_frames', default=4, type=int, help='frames per 3D volume')
    parser.add_argument('--max_frames_per_seq', default=200, type=int, help='max frames per sequence (0=unlimited)')
    parser.add_argument('--patch', default=32, type=int, help='CS sampling patch size')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--flr', '--final_learning_rate', default=1e-6, type=float, help='final learning rate')
    parser.add_argument('--save_dir', help='trained models', default='trained_model', type=str)
    parser.add_argument('--iter_num', type=int, default=8, help='3D iteration count')
    parser.add_argument('--model_dim', type=int, default=16, help='feature dimension')
    parser.add_argument('--train_data', type=str, default='dataset/toucHD/train', help='train dataset path (ToucHD gelsight, 142 seqs)')
    parser.add_argument('--val_dir', type=str, default='dataset/touch_and_go', help='val dataset path (Touch and Go, 142 seqs)')
    parser.add_argument('--train_split', type=str, default='', help='frame-level split file for train filtering')
    parser.add_argument('--val_split', type=str, default='', help='frame-level split file for val filtering')
    parser.add_argument('--ratios', type=str, default='0.01,0.04,0.10,0.25,0.50',
                        help='CS sensing rates, comma-separated (default: 0.01,0.04,0.10,0.25,0.50)')
    args = parser.parse_args()

    cs_ratios = [float(x.strip()) for x in args.ratios.split(',')]
    total_start = time.time()

    # 汇总日志
    summary_path = os.path.join(args.save_dir, 'summary_all_ratios.csv')
    summary_file = open(summary_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['sensing_rate', 'best_val_psnr', 'total_params',
                             'FLOPs(G)', 'FPS', 'total_time_s'])

    for cs_ratio in cs_ratios:
        ratio_start = time.time()
        print("\n" + "=" * 70)
        print(">>> 开始训练: sensing_rate = %.2f" % cs_ratio)
        print("=" * 70)
        best_psnr, eff = main(cs_ratio)
        elapsed = time.time() - ratio_start
        print(">>> sensing_rate = %.2f 完成, 耗时: %.1f 分钟, Best PSNR: %.2f" %
              (cs_ratio, elapsed / 60, best_psnr))
        summary_writer.writerow([f'{cs_ratio:.2f}', f'{best_psnr:.4f}',
                                 f'{eff["Params"]:.3f}',
                                 f'{eff["FLOPs"]:.3f}' if eff['FLOPs'] is not None else 'N/A',
                                 f'{eff["FPS"]:.1f}',
                                 f'{elapsed:.1f}'])
        summary_file.flush()

    summary_file.close()
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("全部压缩比训练完成! 总耗时: %.1f 分钟" % (total_elapsed / 60))
    print("汇总日志: %s" % summary_path)
    print("=" * 70)
