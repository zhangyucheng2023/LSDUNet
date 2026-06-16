"""LSDUNet: 3D Conv Tokenizer + DST deformable attention + deep unfolding CS reconstruction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *


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


class ConvTokenizer3D(nn.Module):
    """3D conv tokenizer with edge branch, GroupNorm for domain invariance."""
    def __init__(self, in_ch=1, dim=16):
        super().__init__()
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
        self.edge_spatial = nn.Conv3d(in_ch, dim // 4, kernel_size=(1, 3, 3),
                                      padding=(0, 1, 1), bias=False)
        self.edge_temporal = nn.Conv3d(in_ch, dim // 4, kernel_size=(3, 1, 1),
                                       padding=(1, 0, 0), bias=False)
        self.edge_fuse = nn.Conv3d(dim // 2, dim // 4, kernel_size=1)
        self.fusion = nn.Conv3d(dim + dim // 4, dim, kernel_size=1)

    def forward(self, x):
        # [B, 1, T, H, W]
        f1 = self.stem1(x)
        f2 = self.stem2(f1)
        f3 = self.stem3(f2)
        e_sp = self.edge_spatial(x)
        e_tp = self.edge_temporal(x)
        e = self.edge_fuse(torch.cat([e_sp, e_tp], dim=1))
        e = F.gelu(e)
        out = self.fusion(torch.cat([f3, e], dim=1))
        return out


class DSTTimeBlock(nn.Module):
    """Multi-scale temporal conv with gating."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = LayerNorm3D(dim)

        self.tconv_small = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1),
                                     padding=(1, 0, 0), groups=dim)
        self.tconv_medium = nn.Conv3d(dim, dim, kernel_size=(5, 1, 1),
                                      padding=(2, 0, 0), groups=dim)
        self.tconv_large = nn.Conv3d(dim, dim, kernel_size=(7, 1, 1),
                                     padding=(3, 0, 0), groups=dim)
        self.tconv_fuse = nn.Conv3d(dim * 3, dim, kernel_size=1)

        self.t_gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim),
            nn.Sigmoid(),
        )
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        x_norm = self.norm(x)
        v = self.v_proj(x_norm)
        t_small = self.tconv_small(v)
        t_medium = self.tconv_medium(v)
        t_large = self.tconv_large(v)
        t_feat = self.tconv_fuse(torch.cat([t_small, t_medium, t_large], dim=1))
        gate = self.t_gate(t_feat)
        out = gate * t_feat + (1 - gate) * v
        out = self.out_proj(out)
        return out


class DSTSpaceBlock(nn.Module):
    """Deformable spatial attention via grid_sample."""
    def __init__(self, dim, num_heads=8, num_points=9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_points = num_points
        self.norm = LayerNorm3D(dim)

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

        offsets = self.offset_conv(x_norm)
        offsets = offsets.reshape(B, self.num_heads, self.num_points, 2, T, H, W)
        offsets = torch.tanh(offsets) * 0.5

        attn = self.attn_weight(x_norm)
        attn = attn.reshape(B, self.num_heads, self.num_points, T, H, W)

        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid_ref = torch.stack([gx, gy], dim=-1)
        grid_ref = grid_ref.view(1, 1, 1, 1, H, W, 2)

        sample_grid = grid_ref + offsets.permute(0, 1, 4, 2, 5, 6, 3)
        sample_grid = sample_grid.reshape(B * self.num_heads * T, self.num_points, H, W, 2)

        v_flat = v.permute(0, 1, 3, 2, 4, 5)
        v_flat = v_flat.reshape(B * self.num_heads * T, self.head_dim, H, W)

        BHT, P, H_grid, W_grid, _ = sample_grid.shape
        sample_grid_flat = sample_grid.reshape(BHT * P, H_grid, W_grid, 2)
        v_flat_expanded = v_flat.unsqueeze(1).expand(-1, P, -1, -1, -1).reshape(BHT * P, self.head_dim, H_grid, W_grid)
        sampled_flat = F.grid_sample(v_flat_expanded, sample_grid_flat, mode='bilinear',
                                     padding_mode='border', align_corners=True)
        sampled = sampled_flat.reshape(BHT, P, self.head_dim, H_grid, W_grid)

        attn_flat = attn.permute(0, 1, 3, 2, 4, 5)
        attn_flat = attn_flat.reshape(B * self.num_heads * T, self.num_points, H, W)

        out = (sampled * attn_flat.unsqueeze(2)).sum(dim=1)
        out = out.reshape(B, self.num_heads, T, self.head_dim, H, W)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, C, T, H, W)
        out = self.out_proj(out)
        return out


class DSTLayer(nn.Module):
    """DST hybrid: TimeBlock → SpaceBlock → LRTA → FFN."""
    def __init__(self, dim, num_heads=8, num_offset_points=9):
        super().__init__()
        self.dst_time = DSTTimeBlock(dim, num_heads)
        self.dst_space = DSTSpaceBlock(dim, num_heads, num_offset_points)
        self.lrta = LongRangeTemporalAttention(dim, num_queries=3, num_heads=num_heads)
        self.ffn = FFN3D(dim, hidden=4)
        self.w_time = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_space = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_lrta = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_ffn = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        x = self.w_time * self.dst_time(x) + x
        x = self.w_space * self.dst_space(x) + x
        x = self.w_lrta * self.lrta(x) + x
        x = self.w_ffn * self.ffn(x) + x
        return x


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


class AdaptiveSModule(nn.Module):
    """Multi-basis dynamic sampling matrix with meta-network."""
    def __init__(self, patch, cs_dim, num_basis=4):
        super().__init__()
        self.patch = patch
        self.cs_dim = cs_dim
        self.num_basis = num_basis

        # K 个基采样矩阵 (Mixture-of-Experts)
        self.s_basis = nn.Parameter(
            kaiming_normal_(torch.Tensor(num_basis, cs_dim, 1, patch, patch))
        )

        self.meta_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, num_basis),
            nn.Softmax(dim=-1),
        )

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
        B = x.size(0)
        basis_weights = self.meta_net(x)
        s_dyn = torch.einsum('bk,kchw->bchw', basis_weights,
                             self.s_basis.squeeze(2)).unsqueeze(2)
        return s_dyn

    def _group_conv2d(self, x, s_dyn):
        B, _, H, W = x.shape
        s_grp = s_dyn.reshape(B * self.cs_dim, 1, self.patch, self.patch)
        x_grp = x.view(1, B, H, W)
        y = F.conv2d(x_grp, s_grp, stride=self.patch, groups=B)
        return y.view(B, self.cs_dim, y.shape[-2], y.shape[-1])

    def forward(self, x, s_fixed=None):
        if s_fixed is not None:
            s_dyn = s_fixed
        else:
            s_dyn = self._compose_sampling(x)

        y = self._group_conv2d(x, s_dyn)

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
    """Back-projection with per-sample sampling matrix."""
    def __init__(self, patch):
        super().__init__()
        self.patch = patch

    def forward(self, y, s_dyn):
        B, cs_dim, Hq, Wq = y.shape
        y_grp = y.view(1, B * cs_dim, Hq, Wq)
        s_grp = s_dyn.reshape(B * cs_dim, 1, self.patch, self.patch)
        x_r = F.conv_transpose2d(y_grp, s_grp, stride=self.patch, groups=B)
        return x_r.view(B, 1, x_r.shape[-2], x_r.shape[-1])


class LongRangeTemporalAttention(nn.Module):
    """Learnable temporal queries cross-attending to all frames."""
    def __init__(self, dim, num_queries=3, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = LayerNorm3D(dim)

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
        queries = self.temporal_queries.unsqueeze(0).expand(B, -1, -1)
        q = self.q_proj(queries).view(B, self.num_queries, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        kv = self.kv_proj(x_norm).mean(dim=[-2, -1])
        k, v = kv.chunk(2, dim=1)
        k = k.permute(0, 2, 1).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, self.num_queries, C)

        out = out.mean(dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        out = out.expand(-1, -1, T, H, W)
        out = self.proj(out)

        return out


class LSDUNet(nn.Module):
    """Deep unfolding CS reconstruction with multi-basis sampling, 3D conv tokenizer, DST attention and uncertainty prediction."""
    def __init__(self, ratio, iter_num=8, model_dim=64, patch=32, num_heads=8):
        super().__init__()

        self.model_dim = model_dim
        self.iter_num = iter_num
        self.patch = patch
        self.num_heads = num_heads
        self.full_dim = patch ** 2
        self.cs_dim = int(ratio * self.full_dim)

        self.adaptive_s = AdaptiveSModule(self.patch, self.cs_dim)
        self.R = RModule(self.patch)

        self.tokenizer = ConvTokenizer3D(in_ch=1, dim=self.model_dim)

        self.proj_out = nn.Conv3d(self.model_dim, 1, kernel_size=1)
        self.proj_var = nn.Conv3d(self.model_dim, 1, kernel_size=1)

        self.gdb = nn.ModuleList([GDB3D(model_dim) for _ in range(self.iter_num)])
        self.dst = nn.ModuleList([DSTLayer(model_dim, num_heads=num_heads) for _ in range(self.iter_num)])

    def forward(self, x, return_intermediates=False, return_uncertainty=False):
        B, T, _, H, W = x.shape
        x_flat = x.view(B * T, 1, H, W)

        y, s_dyn = self.adaptive_s(x_flat)
        x_r = self.R(y, s_dyn)

        x_r = x_r.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4)

        x = self.tokenizer(x_r)

        x_as = None
        intermediates = []
        for i in range(self.iter_num):
            x, x_as = self.gdb[i](x, y, self.adaptive_s, self.R, s_dyn, x_as)
            x = self.dst[i](x)
            if return_intermediates:
                x_proj = self.proj_out(x)
                x_img = x_proj.permute(0, 2, 1, 3, 4)
                intermediates.append(x_img)

        x_mean = self.proj_out(x)
        x_mean = x_mean.permute(0, 2, 1, 3, 4)

        if return_uncertainty:
            log_var = self.proj_var(x)
            log_var = torch.clamp(log_var, min=-10, max=10)
            log_var = log_var.permute(0, 2, 1, 3, 4)
            if return_intermediates:
                return x_mean, log_var, intermediates
            return x_mean, log_var

        if return_intermediates:
            return x_mean, intermediates
        return x_mean
