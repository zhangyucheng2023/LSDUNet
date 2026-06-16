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

    域不变性设计 (方案 C)：使用 GroupNorm 替代 BatchNorm3d。
    GroupNorm 对每个样本独立归一化，不依赖跨 batch 统计量，
    因此推理时不受训练域 (ToucHD) 的 running stats 影响，
    天然支持跨传感器 (TacQuad/Yuan18) 的域泛化。
    """
    def __init__(self, in_ch=1, dim=16):
        super().__init__()
        # --- 主分支：渐进式 3D 卷积特征提取 ---
        self.stem1 = nn.Sequential(
            nn.Conv3d(in_ch, dim // 4, kernel_size=(3, 5, 5),
                      padding=(1, 2, 2)),
            nn.GroupNorm(num_groups=2, num_channels=dim // 4),
            nn.GELU(),
        )
        self.stem2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=4, num_channels=dim // 2),
            nn.GELU(),
        )
        self.stem3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.GroupNorm(num_groups=4, num_channels=dim),
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
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = LayerNorm3D(dim)

        # 三尺度时间卷积核：小核抓快速运动，中核抓慢速滑动，大核抓整体趋势
        self.tconv_small = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1),
                                     padding=(1, 0, 0), groups=dim)
        self.tconv_medium = nn.Conv3d(dim, dim, kernel_size=(5, 1, 1),
                                      padding=(2, 0, 0), groups=dim)
        self.tconv_large = nn.Conv3d(dim, dim, kernel_size=(7, 1, 1),
                                     padding=(3, 0, 0), groups=dim)
        self.tconv_fuse = nn.Conv3d(dim * 3, dim, kernel_size=1)

        # 时间注意力门控
        self.t_gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim),
            nn.Sigmoid(),
        )
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        _ = x.shape
        x_norm = self.norm(x)
        v = self.v_proj(x_norm)

        # 三尺度时序特征提取
        t_small = self.tconv_small(v)   # k=3: 快速运动
        t_medium = self.tconv_medium(v)  # k=5: 慢速滑动
        t_large = self.tconv_large(v)   # k=7: 整体趋势
        t_feat = self.tconv_fuse(torch.cat([t_small, t_medium, t_large], dim=1))

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
    def __init__(self, dim, num_heads=8, num_points=9):
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
    DST 混合注意力层：时间块 → 空间块 → 长程时序 → FFN
    替换原架构的 CATrans3D + WLTrans3D
    """
    def __init__(self, dim, num_heads=8, num_offset_points=9):
        super().__init__()
        self.dst_time = DSTTimeBlock(dim, num_heads)
        self.dst_space = DSTSpaceBlock(dim, num_heads, num_offset_points)
        self.lrta = LongRangeTemporalAttention(dim, num_queries=3, num_heads=num_heads)
        self.ffn = FFN3D(dim, hidden=4)
        # LayerScale: 可学习逐通道残差权重，初始化为极小值（Touvron et al., ICCV 2021）
        self.w_time = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_space = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_lrta = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_ffn = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        # 时间可变形注意力 — 捕捉滑动轨迹
        x = self.w_time * self.dst_time(x) + x
        # 空间可变形注意力 — 重构压力分布
        x = self.w_space * self.dst_space(x) + x
        # 长程时序注意力 — 全局时序依赖
        x = self.w_lrta * self.lrta(x) + x
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
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

    def forward(self, x, y, S, R, s_dyn):
        B, _, T, H, W = x.shape
        x_proj = self.conv2p(x)
        x_proj_flat = x_proj.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
        y_flat = y.view(B * T, *y.shape[-3:])
        # 使用与初始采样相同的 s_dyn 计算反投影残差
        y_proj, _ = S(x_proj_flat, s_fixed=s_dyn)
        err = y_flat - y_proj
        de_flat = R(err, s_dyn)
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
        self.mix1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.GELU(),
        )
        self.mix2 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.GELU(),
        )
        self.mix3 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.GELU(),
        )
        self.res = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(num_groups=4, num_channels=dim),
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
        x_up1 = self.mix1(self._up2(x_up0) + x_down2)
        x_up2 = self.mix2(self._up2(x_up1) + x_down1)
        x_up3 = self.mix3(self._up2(x_up2) + x_down0)

        x_out = x_up3 + self.res(x_up3)
        x = self.conv_out(torch.cat([x, x_out], dim=1))
        return x, [x_up0, x_up1, x_up2]


class GDB3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.grad = GRAD3D(dim)
        self.deno = DENO3D(dim)

    def forward(self, x, y, S, R, s_dyn, x_as_old):
        x = self.grad(x, y, S, R, s_dyn)
        x, x_as_new = self.deno(x, x_as_old)
        return x, x_as_new


# ═══════════════════════════════════════════════════
# D. 自适应采样矩阵 — 内容感知的重要性分配
# ═══════════════════════════════════════════════════
class AdaptiveSModule(nn.Module):
    """
    内容自适应采样 + 多基矩阵动态组合 (Mixture-of-Basis)

    方案 B 设计：
    - 学习 K 个基采样矩阵（不同传感器模式），而非单一固定矩阵
    - 轻量 meta-network 根据输入统计量动态组合基矩阵
    - 正交正则化鼓励基矩阵学习不同的采样模式
    - 域不变性：InstanceNorm2d 消除跨传感器强度差异
    - 渐进式重要性预测器保留空间细节

    参数开销：meta-net ~1K，几乎不影响推理速度。
    """
    def __init__(self, patch, cs_dim, num_basis=4):
        super().__init__()
        self.patch = patch
        self.cs_dim = cs_dim
        self.num_basis = num_basis

        # K 个基采样矩阵 (Mixture-of-Experts)
        self.s_basis = nn.Parameter(
            kaiming_normal_(torch.Tensor(num_basis, cs_dim, 1, patch, patch))
        )

        # Meta-network: 根据输入统计量预测基矩阵权重
        # 输入 [B, 1, H, W] → 输出 [B, K]
        self.meta_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, num_basis),
            nn.Softmax(dim=-1),
        )

        # 重要性预测器（含域归一化）
        self.importance_net = nn.Sequential(
            nn.InstanceNorm2d(1, affine=True),
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, cs_dim, 1),
            nn.Sigmoid(),
        )

    def _compose_sampling(self, x):
        """根据输入动态组合基矩阵 → [B, cs_dim, 1, patch, patch]"""
        B = x.size(0)
        basis_weights = self.meta_net(x)  # [B, K]
        # 加权求和: [B, K] @ [K, cs_dim*patch*patch] → [B, cs_dim, 1, patch, patch]
        s_dyn = torch.einsum('bk,kchw->bchw', basis_weights,
                             self.s_basis.squeeze(2)).unsqueeze(2)
        return s_dyn

    def _group_conv2d(self, x, s_dyn):
        """Group conv2d: 每个样本使用自己的采样矩阵"""
        B, _, H, W = x.shape
        s_grp = s_dyn.reshape(B * self.cs_dim, 1, self.patch, self.patch)
        x_grp = x.view(1, B, H, W)
        y = F.conv2d(x_grp, s_grp, stride=self.patch, groups=B)
        return y.view(B, self.cs_dim, y.shape[-2], y.shape[-1])

    def forward(self, x, s_fixed=None):
        """
        x: [B, 1, H, W]
        s_fixed: 可选固定采样矩阵 [B, cs_dim, 1, patch, patch]，
                 用于梯度反投影步骤中保持与初始采样一致的矩阵。
        返回 (y, s_dynamic)
        """
        if s_fixed is not None:
            s_dyn = s_fixed
        else:
            s_dyn = self._compose_sampling(x)

        # 采样
        y = self._group_conv2d(x, s_dyn)

        # 重要性加权
        imp = self.importance_net(x)
        imp = F.adaptive_avg_pool2d(imp, (y.shape[-2], y.shape[-1]))
        y = y * imp
        return y, s_dyn

    def ortho_loss(self):
        """正交正则化：鼓励 K 个基矩阵学习不同的采样模式"""
        basis_flat = self.s_basis.view(self.num_basis, -1)
        basis_norm = F.normalize(basis_flat, dim=1)
        ortho = basis_norm @ basis_norm.T
        target = torch.eye(self.num_basis, device=ortho.device)
        return (ortho - target).pow(2).mean()


class RModule(nn.Module):
    """重建模块：使用 per-sample 采样矩阵做反投影 (方案 B)"""
    def __init__(self, patch):
        super().__init__()
        self.patch = patch

    def forward(self, y, s_dyn):
        # y: [B, cs_dim, H/P, W/P]
        # s_dyn: [B, cs_dim, 1, P, P]
        B, cs_dim, Hq, Wq = y.shape
        y_grp = y.view(1, B * cs_dim, Hq, Wq)
        s_grp = s_dyn.reshape(B * cs_dim, 1, self.patch, self.patch)
        x_r = F.conv_transpose2d(y_grp, s_grp, stride=self.patch, groups=B)
        return x_r.view(B, 1, x_r.shape[-2], x_r.shape[-1])


# ═══════════════════════════════════════════════════
# F. 长程时序依赖 — 可学习查询跨注意力
# ═══════════════════════════════════════════════════
class LongRangeTemporalAttention(nn.Module):
    """
    长程时序依赖建模：用一组可学习时序查询 (Q) 跨注意力到所有 T 帧，
    捕捉超出 8 帧局部窗口的全局时序演化模式。
    复杂度 O(Q×T)，Q=3 个查询覆盖不同时间尺度（快划/慢划/全局）。

    设计动机：触觉滑动信号（快划/慢划）的时序模式在空间上高度一致——
    整个传感器经历相同的接触-滑动-脱离过程，因此空间池化后做
    跨注意力是合理的归纳偏置。精细的空间结构由 DSTSpaceBlock 独立建模。
    若需空间感知的长程时序，可扩展为分组空间池化（如 2×2 网格区域）。
    """
    def __init__(self, dim, num_queries=3, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = LayerNorm3D(dim)

        # 可学习时序查询 — 快划/慢划/全局三个时间尺度
        self.temporal_queries = nn.Parameter(
            torch.randn(num_queries, dim) * 0.02
        )

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Conv3d(dim, dim * 2, 1)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_norm = self.norm(x)

        # 可学习查询跨注意力
        queries = self.temporal_queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        q = self.q_proj(queries).view(B, self.num_queries, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, H, Q, D]

        # 空间池化：整个传感器面的时序演化高度一致
        kv = self.kv_proj(x_norm).mean(dim=[-2, -1])  # [B, 2C, T]
        k, v = kv.chunk(2, dim=1)
        k = k.permute(0, 2, 1).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Q, T]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, Q, D]
        out = out.permute(0, 2, 1, 3).reshape(B, self.num_queries, C)  # [B, Q, C]

        # 聚合查询结果 → 广播回时空维度
        out = out.mean(dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, C, 1, 1, 1]
        out = out.expand(-1, -1, T, H, W)
        out = self.proj(out)

        return out


# ═══════════════════════════════════════════════════
# LSDUNet 主模型
# ═══════════════════════════════════════════════════
class LSDUNet(nn.Module):
    def __init__(self, ratio, iter_num=8, model_dim=64, patch=32, num_heads=8):
        super().__init__()

        self.model_dim = model_dim
        self.iter_num = iter_num
        self.patch = patch
        self.num_heads = num_heads
        self.full_dim = patch ** 2
        self.cs_dim = int(ratio * self.full_dim)

        # ── D. 自适应采样矩阵：多基动态组合 (方案 B) ──
        self.adaptive_s = AdaptiveSModule(self.patch, self.cs_dim)
        self.R = RModule(self.patch)

        # ── 替换：ConvTokenizer3D 替代原 proj_in ──
        self.tokenizer = ConvTokenizer3D(in_ch=1, dim=self.model_dim)

        # ── H. 不确定性量化：均值头 + 方差头 ──
        self.proj_out = nn.Conv3d(self.model_dim, 1, kernel_size=1)       # 均值 μ
        self.proj_var = nn.Conv3d(self.model_dim, 1, kernel_size=1)       # 对数方差 log σ²

        # GDB 梯度去噪 + DST 混合注意力（含长程时序）
        self.gdb = nn.ModuleList([GDB3D(model_dim) for _ in range(self.iter_num)])
        self.dst = nn.ModuleList([DSTLayer(model_dim, num_heads=num_heads) for _ in range(self.iter_num)])

    def forward(self, x, return_intermediates=False, return_uncertainty=False):
        # x: [B, T, 1, H, W]
        B, T, _, H, W = x.shape
        x_flat = x.view(B * T, 1, H, W)

        y, s_dyn = self.adaptive_s(x_flat)   # 多基动态采样，返回 (测量值, 采样矩阵)
        x_r = self.R(y, s_dyn)            # 反投影，使用 per-sample 采样矩阵

        # 重塑为 3D 格式 [B, 1, T, H, W]
        x_r = x_r.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4)

        # ── ConvTokenizer: 浅层3D卷积提取边缘+纹理Token ──
        x = self.tokenizer(x_r)

        # ── 8轮迭代展开: GDB(梯度优化) + DST(时空注意力+长程时序) ──
        x_as = None
        intermediates = []  # 中间结果列表（用于 rank 追踪）
        for i in range(self.iter_num):
            x, x_as = self.gdb[i](x, y, self.adaptive_s, self.R, s_dyn, x_as)
            x = self.dst[i](x)
            if return_intermediates:
                # 投影到图像空间获取中间重建结果
                x_proj = self.proj_out(x)
                x_img = x_proj.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
                intermediates.append(x_img)

        # 均值预测
        x_mean = self.proj_out(x)
        x_mean = x_mean.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]

        if return_uncertainty:
            # 对数方差预测 — clamp 到安全范围防止数值溢出
            log_var = self.proj_var(x)
            log_var = torch.clamp(log_var, min=-10, max=10)
            log_var = log_var.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
            if return_intermediates:
                return x_mean, log_var, intermediates
            return x_mean, log_var

        if return_intermediates:
            return x_mean, intermediates
        return x_mean
