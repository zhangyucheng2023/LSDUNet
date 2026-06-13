import os
import torch
import numpy as np
import random
from math import log10, exp
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_count

# 安全设备检测：尝试 CUDA 计算，失败则回退 CPU
def _get_device(device_id="cuda:0"):
    if torch.cuda.is_available():
        try:
            d = torch.device(device_id)
            t = torch.zeros(2).to(d)
            t = t + 1  # 执行实际运算，验证 kernel 可用
            return d
        except Exception:
            pass
    return torch.device("cpu")

device = _get_device("cuda:0")
test_device = _get_device("cuda:0")
print(f"[Device] Using: {device}")

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2):
    return ssim_count(img1.squeeze().cpu().numpy(), img2.squeeze().cpu().numpy(), data_range=1)
