"""
LSDUNet — 混合架构：浅层 3D 卷积 Token 化 + 深层 DST 可变形时空注意力
ConvTokenizer3D: 轻量 3D 卷积提取高频边缘特征 + 抑制传感器噪声
DST Attention: 时间块捕捉滑动轨迹 + 空间块重构压力分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *


# ───────────────── Layer Norm ─────────────────
class LayerNorm3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = x.view(b * t, c, -1).transpose(1, 2)
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        x = x.transpose(1, 2).view(b * t, c, h, w)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x


# ───────────────── FFN3D ─────────────────
class FFN3D(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()
        self.norm = LayerNorm3D(dim)
        self.fc1 = nn.Conv3d(dim, dim * hidden, 1)
        self.pos_spatial = nn.Conv3d(dim * hidden, dim * hidden, (1, 3, 3),
                                     padding=(0, 1, 1), groups=dim * hidden)
        self.pos_temporal = nn.Conv3d(dim * hidden, dim * hidden, (3, 1, 1),
                                      padding=(1, 0, 0), groups=dim * hidden)
        self.fc2 = nn.Conv3d(dim * hidden, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos_spatial(x)) + self.act(self.pos_temporal(x))
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════
# 新增：浅层 3D 卷积 Token 化模块
# ═══════════════════════════════════════════════════
class ConvTokenizer3D(nn.Module):
    """
    浅层 3D 卷积 Token 化（替代原 proj_in 的简单1层投影）
    用轻量级 3D 卷积提取触觉图像的高频边缘特征（物体轮廓、压力梯度），
    起到"将卷积投影作为 Token 接入"的作用，同时抑制高频传感器噪声。
    """
    def __init__(self, in_ch=1, dim=16):
        super().__init__()
        # --- 主分支：渐进式 3D 卷积特征提取 ---
        self.stem1 = nn.Sequential(
            nn.Conv3d(in_ch, dim // 4, kernel_size=(3, 5, 5),
                      padding=(1, 2, 2)),
            nn.BatchNorm3d(dim // 4),
            nn.GELU(),
        )
        self.stem2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(dim // 2),
            nn.GELU(),
        )
        self.stem3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.GELU(),
        )
        # --- 边缘增强分支：Laplacian-like 高频提取 ---
        # Sobel-style 空间边缘 + 帧间差分捕捉压力梯度
        self.edge_spatial = nn.Conv3d(in_ch, dim // 4, kernel_size=(1, 3, 3),
                                      padding=(0, 1, 1), bias=False)
        self.edge_temporal = nn.Conv3d(in_ch, dim // 4, kernel_size=(3, 1, 1),
                                       padding=(1, 0, 0), bias=False)
        self.edge_fuse = nn.Conv3d(dim // 2, dim // 4, kernel_size=1)
        # --- 融合 ---
        self.fusion = nn.Conv3d(dim + dim // 4, dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        # 主分支
        f1 = self.stem1(x)      # [B, d/4, T, H, W]
        f2 = self.stem2(f1)     # [B, d/2, T, H, W]
        f3 = self.stem3(f2)     # [B, d, T, H, W]
        # 边缘分支
        e_sp = self.edge_spatial(x)  # 空间边缘
        e_tp = self.edge_temporal(x)  # 帧间差分（滑动痕迹）
        e = self.edge_fuse(torch.cat([e_sp, e_tp], dim=1))
        e = F.gelu(e)
        # 融合主特征 + 边缘增强
        out = self.fusion(torch.cat([f3, e], dim=1))
        return out


# ═══════════════════════════════════════════════════
# 新增：DST 时空可变形注意力模块
# ═══════════════════════════════════════════════════
class DSTTimeBlock(nn.Module):
    """
    时间块注意力 — 捕捉"滑动轨迹"
    使用多尺度时间卷积提取帧间运动特征，建模接触模式的时序演化。
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = LayerNorm3D(dim)

        # 多尺度时间卷积核：小核捕捉快速运动，大核捕捉慢速滑动
        self.tconv_small = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1),
                                     padding=(1, 0, 0), groups=dim)
        self.tconv_large = nn.Conv3d(dim, dim, kernel_size=(5, 1, 1),
                                     padding=(2, 0, 0), groups=dim)
        self.tconv_fuse = nn.Conv3d(dim * 2, dim, kernel_size=1)

        # 时间注意力门控
        self.t_gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim),
            nn.Sigmoid(),
        )
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_norm = self.norm(x)
        v = self.v_proj(x_norm)

        # 多尺度时序特征提取
        t_small = self.tconv_small(v)  # 快速运动
        t_large = self.tconv_large(v)  # 慢速滑动
        t_feat = self.tconv_fuse(torch.cat([t_small, t_large], dim=1))

        # 门控机制：自适应选择关注帧间变化
        gate = self.t_gate(t_feat)
        out = gate * t_feat + (1 - gate) * v
        out = self.out_proj(out)
        return out


class DSTSpaceBlock(nn.Module):
    """
    空间块可变形注意力 — 重构"压力分布形状"
    学习空间采样偏移，自适应地关注接触区域的几何形状。
    使用 grid_sample 实现 2D 空间可变形。
    """
    def __init__(self, dim, num_heads=4, num_points=9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_points = num_points
        self.norm = LayerNorm3D(dim)

        # 生成 2D 空间偏移 [B, H*P*2, T, H, W]
        self.offset_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim),
            nn.GELU(),
            nn.Conv3d(dim, num_heads * num_points * 2, kernel_size=1),
        )
        self.attn_weight = nn.Sequential(
            nn.Conv3d(dim, num_heads * num_points, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_norm = self.norm(x)
        v = self.v_proj(x_norm)
        v = v.reshape(B, self.num_heads, self.head_dim, T, H, W)

        # 空间偏移 [B, H*P*2, T, H, W]
        offsets = self.offset_conv(x_norm)
        # [B, H, P, 2, T, H, W]
        offsets = offsets.reshape(B, self.num_heads, self.num_points, 2, T, H, W)
        offsets = torch.tanh(offsets) * 0.5

        # 注意力 [B, H, P, T, H, W]
        attn = self.attn_weight(x_norm)
        attn = attn.reshape(B, self.num_heads, self.num_points, T, H, W)

        # 参考网格 [H, W]
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid_ref = torch.stack([gx, gy], dim=-1)  # [H, W, 2]
        grid_ref = grid_ref.view(1, 1, 1, 1, H, W, 2)  # [1, 1, 1, 1, H, W, 2]

        # 采样网格 = 参考 + 偏移
        # offsets: [B, H, P, 2, T, H, W] → [B, H, T, P, H, W, 2]
        sample_grid = grid_ref + offsets.permute(0, 1, 4, 2, 5, 6, 3)  # [B, H, T, P, H, W, 2]
        # 合并 B*H*T 为 batch 维度做 grid_sample
        sample_grid = sample_grid.reshape(B * self.num_heads * T, self.num_points, H, W, 2)

        v_flat = v.permute(0, 1, 3, 2, 4, 5)  # [B, H, T, C', H, W]
        v_flat = v_flat.reshape(B * self.num_heads * T, self.head_dim, H, W)

        # 向量化 grid_sample：合并所有采样点到 batch 维，单次调用
        BHT, P, H_grid, W_grid, _ = sample_grid.shape
        sample_grid_flat = sample_grid.reshape(BHT * P, H_grid, W_grid, 2)
        v_flat_expanded = v_flat.unsqueeze(1).expand(-1, P, -1, -1, -1).reshape(BHT * P, self.head_dim, H_grid, W_grid)
        sampled_flat = F.grid_sample(v_flat_expanded, sample_grid_flat, mode='bilinear',
                                     padding_mode='border', align_corners=True)
        sampled = sampled_flat.reshape(BHT, P, self.head_dim, H_grid, W_grid)  # [B*H*T, P, C', H, W]

        # 注意力加权聚合
        attn_flat = attn.permute(0, 1, 3, 2, 4, 5)  # [B, H, T, P, H, W]
        attn_flat = attn_flat.reshape(B * self.num_heads * T, self.num_points, H, W)

        out = (sampled * attn_flat.unsqueeze(2)).sum(dim=1)  # [B*H*T, C', H, W]
        out = out.reshape(B, self.num_heads, T, self.head_dim, H, W)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, C, T, H, W)
        out = self.out_proj(out)
        return out


class DSTLayer(nn.Module):
    """
    DST 混合注意力层：时间块 → 空间块 → FFN
    替换原架构的 CATrans3D + WLTrans3D
    """
    def __init__(self, dim, num_heads=4, num_offset_points=9):
        super().__init__()
        self.dst_time = DSTTimeBlock(dim, num_heads)
        self.dst_space = DSTSpaceBlock(dim, num_heads, num_offset_points)
        self.ffn = FFN3D(dim, hidden=4)
        # 可学习残差权重（稳定训练）
        self.w_time = nn.Parameter(0.05 * torch.ones(1, dim, 1, 1, 1))
        self.w_space = nn.Parameter(0.05 * torch.ones(1, dim, 1, 1, 1))
        self.w_ffn = nn.Parameter(0.05 * torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        # 时间可变形注意力 — 捕捉滑动轨迹
        x = self.w_time * self.dst_time(x) + x
        # 空间可变形注意力 — 重构压力分布
        x = self.w_space * self.dst_space(x) + x
        # FFN
        x = self.w_ffn * self.ffn(x) + x
        return x


# ═══════════════════════════════════════════════════
# 保留：梯度去噪块 (GDB3D) — 深度展开优化核心
# ═══════════════════════════════════════════════════

class GRAD3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv2p = nn.Conv3d(dim, 1, kernel_size=1)
        self.conv2f = nn.Conv3d(1, dim, kernel_size=1)
        self.res = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

    def forward(self, x, y, S, R):
        B, C, T, H, W = x.shape
        x_proj = self.conv2p(x)
        x_proj_flat = x_proj.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
        y_flat = y.view(B * T, *y.shape[-3:])
        err = y_flat - S(x_proj_flat)
        de_flat = R(err)
        De = de_flat.view(B, T, *de_flat.shape[-3:]).permute(0, 2, 1, 3, 4)
        De = self.conv2f(De)
        De = De + self.res(De)
        x = x + De
        return x


class DENO3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.GELU(),
        )
        self.res = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )
        self.conv_out = nn.Conv3d(dim * 2, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def _up2(self, x):
        _, _, _, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        _, T, C, _, _ = x.size()
        x = x.view(-1, C, h, w)
        x = F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')
        x = x.view(-1, T, C, h * 2, w * 2).permute(0, 2, 1, 3, 4)
        return x

    def forward(self, x, x_as=None):
        x_down0 = x
        x_down1 = self.down1(x_down0) + (x_as[2] if x_as else 0)
        x_down2 = self.down2(x_down1) + (x_as[1] if x_as else 0)
        x_down3 = self.down3(x_down2) + (x_as[0] if x_as else 0)

        x_up0 = x_down3
        x_up1 = self.mix(self._up2(x_up0) + x_down2)
        x_up2 = self.mix(self._up2(x_up1) + x_down1)
        x_up3 = self.mix(self._up2(x_up2) + x_down0)

        x_out = x_up3 + self.res(x_up3)
        x = self.conv_out(torch.cat([x, x_out], dim=1))
        return x, [x_up0, x_up1, x_up2]


class GDB3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.grad = GRAD3D(dim)
        self.deno = DENO3D(dim)

    def forward(self, x, y, S, R, x_as_old):
        x = self.grad(x, y, S, R)
        x, x_as_new = self.deno(x, x_as_old)
        return x, x_as_new


class SModule(nn.Module):
    def __init__(self, patch, s_weight):
        super().__init__()
        self.patch = patch
        self.s_weight = s_weight

    def forward(self, x):
        return F.conv2d(x, self.s_weight, stride=self.patch)


class RModule(nn.Module):
    def __init__(self, patch, s_weight):
        super().__init__()
        self.patch = patch
        self.s_weight = s_weight

    def forward(self, y):
        return F.conv_transpose2d(y, self.s_weight, stride=self.patch)


# ═══════════════════════════════════════════════════
# LSDUNet 主模型
# ═══════════════════════════════════════════════════
class LSDUNet(nn.Module):
    def __init__(self, ratio, iter_num=8, model_dim=16, patch=32):
        super().__init__()

        self.model_dim = model_dim
        self.iter_num = iter_num
        self.patch = patch
        self.full_dim = patch ** 2
        self.cs_dim = int(ratio * self.full_dim)

        self.s_weight = nn.Parameter(
            kaiming_normal_(torch.Tensor(self.cs_dim, 1, self.patch, self.patch))
        )
        self.S = SModule(self.patch, self.s_weight)
        self.R = RModule(self.patch, self.s_weight)

        # ── 替换：ConvTokenizer3D 替代原 proj_in ──
        self.tokenizer = ConvTokenizer3D(in_ch=1, dim=self.model_dim)

        self.proj_out = nn.Conv3d(self.model_dim, 1, kernel_size=1)

        # GDB 梯度去噪 + DST 混合注意力
        self.gdb = nn.ModuleList([GDB3D(model_dim) for _ in range(self.iter_num)])
        self.dst = nn.ModuleList([DSTLayer(model_dim) for _ in range(self.iter_num)])

    def forward(self, x, return_intermediates=False):
        # x: [B, T, 1, H, W]
        B, T, _, H, W = x.shape
        x_flat = x.view(B * T, 1, H, W)

        y = self.S(x_flat)
        x_r = self.R(y)

        # 重塑为 3D 格式 [B, 1, T, H, W]
        x_r = x_r.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4)

        # ── ConvTokenizer: 浅层3D卷积提取边缘+纹理Token ──
        x = self.tokenizer(x_r)

        # ── 8轮迭代展开: GDB(梯度优化) + DST(时空注意力) ──
        x_as = None
        intermediates = []  # 中间结果列表（用于 rank 追踪）
        for i in range(self.iter_num):
            x, x_as = self.gdb[i](x, y, self.S, self.R, x_as)
            x = self.dst[i](x)
            if return_intermediates:
                # 投影到图像空间获取中间重建结果
                x_proj = self.proj_out(x)
                x_img = x_proj.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
                intermediates.append(x_img)

        x = self.proj_out(x)
        x = x.permute(0, 2, 1, 3, 4)

        if return_intermediates:
            return x, intermediates
        return x
