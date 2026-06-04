from utils import *

from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Optional LPIPS
try:
    import ssl as _ssl
    import warnings as _warnings
    _ssl._create_default_https_context = _ssl._create_unverified_context
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        import lpips as _lpips_lib
        _lpips_model = _lpips_lib.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
    HAS_LPIPS = True
except Exception:
    _lpips_model = None
    HAS_LPIPS = False


def compute_lpips_batch(pred, target):
    """Compute LPIPS for a batch of 2D images. pred, target: [N, H, W]"""
    if not HAS_LPIPS or _lpips_model is None:
        return None
    pred_t = pred.unsqueeze(1).repeat(1, 3, 1, 1)  # [N, 3, H, W]
    target_t = target.unsqueeze(1).repeat(1, 3, 1, 1)
    if pred_t.max() > 1.5:
        pred_t = pred_t / 255.0
        target_t = target_t / 255.0
    device = next(_lpips_model.parameters()).device
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    with torch.no_grad():
        return _lpips_model(pred_t, target_t).mean().item()


def train_3d(train_loader, model, criterion, optimizer, device):
    model.train()
    sum_loss = 0
    scaler = GradScaler(device.type)
    pbar = tqdm(train_loader, desc='train', dynamic_ncols=True)

    for inputs, _ in pbar:
        inputs = inputs.to(device)
        B, T, C, H, W = inputs.shape
        x_flat = inputs.reshape(B * T, C, H, W)
        y_ch = rgb_to_ycbcr(x_flat)[:, 0, :, :].view(B, T, 1, H, W) / 255.
        optimizer.zero_grad()

        with autocast(device.type):
            outputs, _ = model(y_ch)
            loss = criterion(outputs, y_ch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return sum_loss / len(train_loader)


def valid_3d(val_loader, model, device):
    sum_psnr = 0
    sum_ssim = 0
    sum_lpips = 0
    lpips_count = 0
    model.eval()
    pbar = tqdm(val_loader, desc='valid', dynamic_ncols=True)
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            B, T, C, H, W = inputs.shape
            x_flat = inputs.reshape(B * T, C, H, W)
            y_ch = rgb_to_ycbcr(x_flat)[:, 0, :, :].view(B, T, 1, H, W) / 255.
            outputs = model(y_ch)
            pred = outputs[0]
            target = y_ch
            mse = F.mse_loss(pred, target)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            mid = T // 2
            for b in range(B):
                sum_ssim += ssim(pred[b, mid, 0].cpu().numpy(),
                                 target[b, mid, 0].cpu().numpy(), data_range=1)
                if HAS_LPIPS:
                    lpips_val = compute_lpips_batch(
                        pred[b, mid, 0].cpu().unsqueeze(0),
                        target[b, mid, 0].cpu().unsqueeze(0))
                    if lpips_val is not None:
                        sum_lpips += lpips_val
                        lpips_count += 1
            postfix = {'psnr': f'{psnr:.2f}', 'ssim': f'{sum_ssim / (iters * B + B):.4f}'}
            if lpips_count > 0:
                postfix['lpips'] = f'{sum_lpips / lpips_count:.4f}'
            pbar.set_postfix(postfix)
    ret = (sum_psnr / max(len(val_loader), 1), sum_ssim / max(len(val_loader.dataset), 1))
    if lpips_count > 0:
        ret = ret + (sum_lpips / lpips_count,)
    return ret if len(ret) > 2 else ret


def valid_3d_single(valid_loader, model, device):
    sum_psnr = 0
    sum_ssim = 0
    sum_lpips = 0
    lpips_count = 0
    model.eval()
    pbar = tqdm(valid_loader, desc='valid', dynamic_ncols=True)
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            x_ch = rgb_to_ycbcr(inputs)[:, 0, :, :].unsqueeze(1) / 255.
            B, C, H, W = x_ch.shape
            T = getattr(model, 'num_frames', 4)
            x_3d = x_ch.unsqueeze(1).expand(B, T, C, H, W).reshape(B, T, C, H, W)
            outputs = model(x_3d)
            pred = outputs[0][:, T // 2, :, :, :]
            target = x_ch
            mse = F.mse_loss(pred, target)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(pred.squeeze().cpu().numpy(), target.squeeze().cpu().numpy(), data_range=1)
            if HAS_LPIPS:
                lpips_val = compute_lpips_batch(
                    pred.squeeze().cpu().unsqueeze(0),
                    target.squeeze().cpu().unsqueeze(0))
                if lpips_val is not None:
                    sum_lpips += lpips_val
                    lpips_count += 1
            postfix = {'psnr': f'{psnr:.2f}', 'ssim': f'{sum_ssim / (iters + 1):.4f}'}
            if lpips_count > 0:
                postfix['lpips'] = f'{sum_lpips / lpips_count:.4f}'
            pbar.set_postfix(postfix)
    ret = (sum_psnr / len(valid_loader), sum_ssim / len(valid_loader))
    if lpips_count > 0:
        ret = ret + (sum_lpips / lpips_count,)
    return ret if len(ret) > 2 else ret
