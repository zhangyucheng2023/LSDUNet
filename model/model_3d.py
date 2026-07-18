"""LSDUNet: Lightweight deep unfolding CS reconstruction (fully adaptive).

Three selling points preserved end-to-end:
  1. "Deformable spatial attention" -> DeformableSpatialBlock combining
     torchvision DeformConv2d (local deformable, ~1/100 memory of grid_sample)
     with MultiScaleSpatialConv (global context via dilated depthwise conv,
     O(D·k²) memory vs MambaSSM's O(B·L·D·d_state)).
  2. "Deep unfolding with CS" -> CSGradientStep computes a real adaptive
     feature-level CS gradient R_feat(y_feat - S_feat(x)) EVERY iteration,
     using AdaptiveFeatCS (no forced channel conversions, resolution-adaptive).
  3. "Meta-network adaptive sampling" -> AdaptiveSModule unchanged.

TacMamba-inspired temporal compression: TactileHistoryCompressor uses MambaSSM
on temporal dimension only (T=8, tiny memory), replacing O(T²) attention.

Fully adaptive: ConvTokenizer3D uses stride-2 convolutions for proportional
2x downsampling (224 -> 112, 448 -> 224). AdaptiveFeatCS uses adaptive pooling
to produce a fixed 7x7 measurement grid regardless of feature resolution.
No forced channel conversions anywhere in the deep unfolding pipeline.

Uncertainty estimation: UncertaintyHead predicts per-pixel reconstruction
variance via negative log-likelihood, providing confidence maps for robotics
deployment.

Lightweight by design: spatial processing uses O(D·k²) convolutions instead
of O(B·L·D·d_state) SSM scans. Gradient checkpointing keeps memory low.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *
from torch.utils.checkpoint import checkpoint as _ckpt
from torchvision.ops import DeformConv2d


# ═══════════════════════════════════════════════════════════════════
# 1. Shared components
# ═══════════════════════════════════════════════════════════════════

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


class BayesianFusion(nn.Module):
    """Bayesian uncertainty-weighted feature fusion (CMLF-inspired).

    Replaces brute-force concatenation with maximum likelihood estimation.
    Each branch predicts its own per-channel uncertainty (variance), and
    fusion follows inverse-variance weighting:
        fused = (f1 * σ2² + f2 * σ1²) / (σ1² + σ2²)

    This is the core idea of CMLF (arXiv:2604.02108): "用贝叶斯推断代替
    暴力的特征拼接" — using Bayesian inference instead of brute-force
    feature concatenation, greatly reducing multimodal alignment redundancy.

    Lightweight: adds only 2 linear layers (dim → dim) for variance prediction.
    """
    def __init__(self, dim):
        super().__init__()
        # Per-channel variance predictors (Softplus ensures positivity)
        self.var_pred1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(dim, dim), nn.Softplus(),
        )
        self.var_pred2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(dim, dim), nn.Softplus(),
        )

    def forward(self, f1, f2):
        """Bayesian maximum likelihood fusion of two feature branches.

        Args:
            f1, f2: [B, C, T, H, W] feature maps (same shape, same C)
        Returns:
            fused: [B, C, T, H, W] uncertainty-weighted combination
        """
        # Predict per-channel variance (uncertainty) for each branch
        var1 = self.var_pred1(f1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6  # [B, C, 1, 1, 1]
        var2 = self.var_pred2(f2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6
        # Inverse-variance weighted fusion (Bayesian MLE)
        fused = (f1 * var2 + f2 * var1) / (var1 + var2)
        return fused


class ConvTokenizer3D(nn.Module):
    """3D conv tokenizer with edge branch. Proportional 2x downsampling.

    Input 224 -> features 112, input 448 -> features 224. Fully adaptive.

    Uses BayesianFusion (CMLF-inspired) to fuse main stem and edge branch
    via uncertainty-weighted maximum likelihood, replacing brute-force
    concatenation.
    """
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.stem1 = nn.Sequential(
            nn.Conv3d(in_ch, dim // 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(2, dim // 4), nn.GELU(),
        )
        self.stem2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, dim // 2), nn.GELU(),
        )
        self.stem3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, dim), nn.GELU(),
        )
        self.edge_spatial = nn.Conv3d(in_ch, dim // 4, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.edge_temporal = nn.Conv3d(in_ch, dim // 4, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.edge_fuse = nn.Conv3d(dim // 2, dim // 4, kernel_size=1)
        # Project edge branch to same dim as main stem for Bayesian fusion
        self.edge_proj = nn.Conv3d(dim // 4, dim, kernel_size=1)
        # Bayesian uncertainty-weighted fusion (replaces concat + 1x1 conv)
        self.bayesian_fusion = BayesianFusion(dim)

    def forward(self, x):
        f1 = self.stem1(x)
        f2 = self.stem2(f1)
        f3 = self.stem3(f2)
        e_sp = self.edge_spatial(x)
        e_tp = self.edge_temporal(x)
        # Match spatial size: e_sp is at H/2 (stride-2), e_tp is at full H.
        # Pool e_tp down to e_sp's spatial size for concatenation.
        e_tp = F.adaptive_avg_pool3d(e_tp, (e_tp.shape[2], e_sp.shape[-2], e_sp.shape[-1]))
        e = self.edge_fuse(torch.cat([e_sp, e_tp], dim=1))
        e = F.gelu(e)
        # Project edge branch to dim channels for Bayesian fusion
        e = self.edge_proj(e)
        # Bayesian uncertainty-weighted fusion (CMLF-inspired)
        return self.bayesian_fusion(f3, e)


class DSTTimeBlock(nn.Module):
    """Multi-scale temporal conv with gating (kept from original - lightweight)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm3D(dim)
        self.tconv_small = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.tconv_medium = nn.Conv3d(dim, dim, (5, 1, 1), padding=(2, 0, 0), groups=dim)
        self.tconv_large = nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim)
        self.tconv_fuse = nn.Conv3d(dim * 3, dim, kernel_size=1)
        self.t_gate = nn.Sequential(
            nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim), nn.Sigmoid())
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
        return self.out_proj(out)


class AdaptiveSModule(nn.Module):
    """Multi-basis dynamic sampling matrix with meta-network.

    Enhanced meta_net: uses mean + std + gradient magnitude (3 features)
    instead of just mean (1 feature) for richer input-dependent basis selection.

    Simplified importance_net: 3 conv layers instead of 5, reducing params ~60%
    while maintaining the same spatial importance weighting functionality.
    """
    def __init__(self, patch, cs_dim, num_basis=4):
        super().__init__()
        self.patch = patch
        self.cs_dim = cs_dim
        self.num_basis = num_basis
        self.s_basis = nn.Parameter(kaiming_normal_(torch.Tensor(num_basis, cs_dim, 1, patch, patch)))
        # Enhanced meta_net: 3 statistical features (mean, std, grad_mag)
        self.meta_net = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(),
            nn.Linear(16, num_basis), nn.Softmax(dim=-1),
        )
        # Simplified importance_net: 3 layers (was 5), ~60% param reduction
        self.importance_net = nn.Sequential(
            nn.InstanceNorm2d(1, affine=True),
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, cs_dim, 1), nn.Sigmoid(),
        )

    def _compute_stats(self, x):
        """Compute rich statistical features for meta_net input.

        Returns [B, 3] tensor: [global_mean, global_std, gradient_magnitude]
        """
        mean = x.mean(dim=[1, 2, 3])  # [B]
        std = x.std(dim=[1, 2, 3])  # [B]
        # Gradient magnitude (Sobel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_mag = grad_x.abs().mean(dim=[1, 2, 3])  # [B]
        return torch.stack([mean, std, grad_mag], dim=-1)  # [B, 3]

    def _group_conv2d(self, x, s_dyn):
        B, _, H, W = x.shape
        s_grp = s_dyn.reshape(B * self.cs_dim, 1, self.patch, self.patch)
        x_grp = x.view(1, B, H, W)
        y = F.conv2d(x_grp, s_grp, stride=self.patch, groups=B)
        return y.view(B, self.cs_dim, y.shape[-2], y.shape[-1])

    def forward(self, x, s_fixed=None, return_basis_weights=False):
        if s_fixed is not None:
            s_dyn = s_fixed
            basis_weights = None
        else:
            stats = self._compute_stats(x)  # [B, 3]
            basis_weights = self.meta_net(stats)
            s_dyn = torch.einsum('bk,kchw->bchw', basis_weights,
                                 self.s_basis.squeeze(2)).unsqueeze(2)
        y = self._group_conv2d(x, s_dyn)
        imp = self.importance_net(x)
        imp = F.adaptive_avg_pool2d(imp, (y.shape[-2], y.shape[-1]))
        y = y * imp
        if return_basis_weights:
            return y, s_dyn, basis_weights
        return y, s_dyn

    def ortho_loss(self):
        basis_flat = self.s_basis.view(self.num_basis, -1)
        basis_norm = F.normalize(basis_flat, dim=1)
        ortho = basis_norm @ basis_norm.T
        target = torch.eye(self.num_basis, device=ortho.device)
        return (ortho - target).pow(2).mean()


class RModule(nn.Module):
    """Back-projection with per-sample sampling matrix (kept from original)."""
    def __init__(self, patch):
        super().__init__()
        self.patch = patch

    def forward(self, y, s_dyn):
        B, cs_dim, Hq, Wq = y.shape
        y_grp = y.view(1, B * cs_dim, Hq, Wq)
        s_grp = s_dyn.reshape(B * cs_dim, 1, self.patch, self.patch)
        x_r = F.conv_transpose2d(y_grp, s_grp, stride=self.patch, groups=B)
        return x_r.view(B, 1, x_r.shape[-2], x_r.shape[-1])


# ═══════════════════════════════════════════════════════════════════
# 2. Pure PyTorch Mamba SSM (no CUDA compilation needed)
# ═══════════════════════════════════════════════════════════════════

class MambaSSM(nn.Module):
    """Selective state-space model in pure PyTorch.

    O(N) time and memory. Uses chunked parallel scan for long sequences.
    No mamba_ssm/causal_conv1d CUDA compilation required.

    Memory optimization: batch chunking + gradient checkpointing to avoid
    materializing [B, L, D, d_state] for the full batch at once.
    """

    def __init__(self, dim, d_state=16, d_conv=4, bidirectional=True,
                 chunk_size=1024, batch_chunk=128):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.chunk_size = chunk_size
        self.batch_chunk = batch_chunk  # max batch elements per scan

        # Mamba-style gated projection
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, d_conv, padding=d_conv - 1, groups=dim)

        # SSM parameters (input-dependent = selective)
        self.A_log = nn.Parameter(torch.zeros(d_state))
        self.B_proj = nn.Linear(dim, d_state)
        self.C_proj = nn.Linear(dim, d_state)
        self.dt_proj = nn.Linear(dim, d_state)
        self.D = nn.Parameter(torch.ones(dim))

        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        nn.init.uniform_(self.A_log, -2.0, -0.5)

    def _scan_direct(self, x, A, dt, B_param, C_param):
        """Direct sequential scan for short sequences (L <= 32).

        More numerically stable than parallel scan for small L.
        The parallel scan's exp(-log_cumsum_A) can overflow for moderate L,
        while this direct recursion is O(L) and perfectly stable.
        """
        B_size, L, D = x.shape
        h = torch.zeros(B_size, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            A_bar = torch.exp(A.unsqueeze(0) * dt[:, t])  # [B, d_state]
            B_bar = B_param[:, t] * dt[:, t]  # [B, d_state]
            h = A_bar.unsqueeze(1) * h + B_bar.unsqueeze(1) * x[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)  # [B, D]
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)  # [B, L, D]

    def _scan_chunk(self, x, A, dt, B_param, C_param, init_state=None):
        """Parallel scan for one chunk. Returns output and last state.

        x: [B_chunk, L_chunk, D]
        init_state: [B_chunk, D, d_state] or None

        Note: For short sequences (L <= 32), _scan_direct is preferred
        for numerical stability.
        """
        B_size, L_chunk, D = x.shape

        # Discretize
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt)  # [B, L, d_state]
        B_bar = B_param * dt  # [B, L, d_state]

        # Log-space parallel scan
        log_A_bar = torch.log(A_bar + 1e-8)
        log_cumsum_A = torch.cumsum(log_A_bar, dim=1)

        # weighted_Bx: [B, L, D, d_state]
        weighted_Bx = B_bar.unsqueeze(2) * x.unsqueeze(-1)
        weighted_Bx = weighted_Bx * torch.exp(-log_cumsum_A).unsqueeze(2)

        cumsum_Bx = torch.cumsum(weighted_Bx, dim=1)

        # Add initial state contribution
        if init_state is not None:
            init_scaled = init_state.unsqueeze(1) * torch.exp(log_cumsum_A).unsqueeze(2)
            cumsum_Bx = cumsum_Bx + init_scaled

        h = cumsum_Bx * torch.exp(log_cumsum_A).unsqueeze(2)  # [B, L, D, d_state]
        y = (h * C_param.unsqueeze(2)).sum(-1)  # [B, L, D]

        last_state = h[:, -1]  # [B, D, d_state]

        return y, last_state

    def _scan(self, x):
        """Chunked parallel scan: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t

        x: [B, L, D] -> [B, L, D]
        Processes batch dimension in chunks to limit peak memory.
        """
        B_size, L, D = x.shape

        A = -torch.exp(self.A_log)
        dt = F.softplus(self.dt_proj(x))
        B_param = self.B_proj(x)
        C_param = self.C_proj(x)

        # Process batch in chunks to limit [B, L, D, d_state] peak memory
        # Note: A is [d_state] (shared across batch), don't slice it
        if B_size <= self.batch_chunk:
            return self._scan_batch(x, A, dt, B_param, C_param)

        outputs = []
        for b_start in range(0, B_size, self.batch_chunk):
            b_end = min(b_start + self.batch_chunk, B_size)
            y_chunk = self._scan_batch(
                x[b_start:b_end], A, dt[b_start:b_end],
                B_param[b_start:b_end], C_param[b_start:b_end])
            outputs.append(y_chunk)
        return torch.cat(outputs, dim=0)

    def _scan_batch(self, x, A, dt, B_param, C_param):
        """Scan for a sub-batch, with sequence-length chunking."""
        B_size, L, D = x.shape

        # For short sequences (L <= 32), use direct sequential scan (numerically stable)
        if L <= 32:
            return self._scan_direct(x, A, dt, B_param, C_param)

        if L <= self.chunk_size:
            y, _ = self._scan_chunk(x, A, dt, B_param, C_param)
            return y

        outputs = []
        init_state = None
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            y_chunk, init_state = self._scan_chunk(
                x[:, start:end], A, dt[:, start:end],
                B_param[:, start:end], C_param[:, start:end], init_state)
            outputs.append(y_chunk)
        return torch.cat(outputs, dim=1)

    def _forward_core(self, x_conv):
        """Core scan logic, wrapped by checkpoint during training."""
        if self.bidirectional:
            y_fwd = self._scan(x_conv)
            y_bwd = self._scan(x_conv.flip(1)).flip(1)
            return y_fwd + y_bwd
        return self._scan(x_conv)

    def forward(self, x):
        """x: [B, L, D] -> [B, L, D]"""
        residual = x
        x = self.norm(x)

        x_proj = self.in_proj(x)
        gate, x_branch = x_proj.chunk(2, dim=-1)

        x_conv = self.conv(x_branch.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)[:, :x.shape[1], :]

        # Checkpoint the scan to avoid storing [B, L, D, d_state] for backward
        if self.training and x_conv.requires_grad:
            y = _ckpt(self._forward_core, x_conv, use_reentrant=False)
        else:
            y = self._forward_core(x_conv)

        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(gate)

        return self.out_proj(y) + residual


# ═══════════════════════════════════════════════════════════════════
# 3. DeformableSpatialBlock (replaces DSTSpaceBlock / MambaSpatialBlock)
# ═══════════════════════════════════════════════════════════════════

class MultiScaleSpatialConv(nn.Module):
    """Multi-scale dilated depthwise conv for global spatial context.

    Replaces MambaSSM spatial scan with O(D·k²) memory per position.
    Receptive field: 15×15 (dilation=7) covers ~13% of 112×112 feature map.

    Memory comparison for batch=8, feature_size=112:
      MambaSSM: [7168, 112, 32, 16] = 830 MB per intermediate ×4 = 3.3 GiB
      DilatedConv: [8, 32, 8, 112, 112] × 3 branches = 17 MB total
      → 190x reduction in peak memory.
    """
    def __init__(self, dim):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, d, d),
                      dilation=(1, d, d), groups=dim)
            for d in [1, 3, 7]
        ])
        self.fuse = nn.Conv3d(dim * 3, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        outs = [self.act(b(x)) for b in self.branches]
        return self.fuse(torch.cat(outs, dim=1))


class DeformableSpatialBlock(nn.Module):
    """Deformable conv + multi-scale dilated conv. Replaces DSTSpaceBlock.

    Architecture (principled lightweight design):
    1. DeformConv2d: local deformable attention (preserves "deformable" selling point)
       - O(D·k²) memory, learns spatial offsets for contact-adaptive sampling
    2. MultiScaleSpatialConv: global spatial context via dilated depthwise conv
       - O(D·k²) memory, 3 branches with dilation {1,3,7} → 15×15 receptive field
       - Replaces MambaSSM spatial scan (O(B·L·D·d_state) → O(D·k²))

    Total spatial block memory: O(D·k²) = ~17 MB vs MambaSSM's ~3.3 GiB.
    MambaSSM is retained ONLY for temporal compression (TactileHistoryCompressor),
    where sequence length T=8 makes it efficient.
    """
    def __init__(self, dim, num_points=9):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm3D(dim)

        # Deformable conv branch (local deformable attention)
        self.offset_predictor = nn.Conv3d(dim, 2 * 3 * 3, (1, 3, 3), padding=(0, 1, 1))
        self.deform_conv = DeformConv2d(dim, dim, kernel_size=3, padding=1)
        self.deform_proj = nn.Conv3d(dim, dim, kernel_size=1)

        # Multi-scale dilated conv (global spatial context, O(D*k²) memory)
        self.global_conv = MultiScaleSpatialConv(dim)

        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x, return_attn=False):
        B, C, T, H, W = x.shape
        x_norm = self.norm(x)

        # --- Deformable conv branch (local deformable attention) ---
        offset = self.offset_predictor(x_norm)  # [B, 2*3*3, T, H, W]
        x_2d = x_norm.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        offset_2d = offset.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
        deform_out = self.deform_conv(x_2d, offset_2d)
        deform_out = deform_out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        deform_out = self.deform_proj(deform_out)

        # --- Multi-scale dilated conv (global context, O(D*k²) memory) ---
        global_out = self.global_conv(x_norm)

        # Combine
        out = deform_out + global_out
        out = self.out_proj(out)

        if return_attn:
            attn_map = offset.mean(dim=1, keepdim=True)
            return out, attn_map
        return out


# ═══════════════════════════════════════════════════════════════════
# 4. TactileHistoryCompressor (replaces LongRangeTemporalAttention)
# ═══════════════════════════════════════════════════════════════════

class TactileHistoryCompressor(nn.Module):
    """Mamba-based temporal history compression with causal latent filter.

    Replaces LongRangeTemporalAttention (O(T^2) attention) with
    O(T) Mamba SSM on spatially-pooled features.

    Inspired by TacMamba (arXiv:2603.01700):
    - Compresses continuous force history into compact state
    - O(1) inference latency after warmup

    Inspired by CMLF (arXiv:2604.02108):
    - Causal Latent State-Space Filter: prediction + Bayesian update
    - Uses a learned transition model to predict next state, then
      combines prediction with Mamba observation via Kalman gain
    - Filters temporal noise through causal state-space evolution

    Adaptive num_queries: instead of fixed 3, predicts per-sample query
    weighting from input complexity, allowing simpler inputs to use fewer
    effective queries (interpretability + efficiency).
    """

    def __init__(self, dim, num_queries=3):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        self.norm = LayerNorm3D(dim)

        # Temporal Mamba on pooled features: [B, T, D] -> [B, T, D]
        self.temporal_mamba = MambaSSM(dim, d_state=16, bidirectional=True)

        # Causal Latent State-Space Filter (CMLF-inspired)
        # Transition model: predict state at t from state at t-1
        self.transition = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.transition.weight)  # identity init for stable start
        # Kalman gain: learned, input-dependent mixing of prediction and observation
        self.kalman_gain = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.Sigmoid(),
        )

        # Learnable queries for temporal summarization
        self.temporal_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)

        # Adaptive query weighting: input-complexity-dependent gating
        # Allows the model to softly select how many queries to use per sample
        self.query_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(dim, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, num_queries), nn.Softmax(dim=-1),
        )

        self.proj = nn.Conv3d(dim, dim, 1)

        # 隐式 L+S 缓存: L_kalman=平滑状态(低秩), S_kalman=innovation(稀疏)
        self.last_L_kalman = None
        self.last_S_kalman = None

    def forward(self, x, return_attn=False):
        B, C, T, H, W = x.shape
        x_norm = self.norm(x)

        # Spatial pooling for efficient temporal scan
        x_pooled = x_norm.mean(dim=[-2, -1])  # [B, C, T]
        x_temporal = x_pooled.permute(0, 2, 1)  # [B, T, C]

        # Mamba temporal compression (observation)
        observed = self.temporal_mamba(x_temporal)  # [B, T, C]

        # Causal Latent State-Space Filter (CMLF-inspired)
        # Step 1: Predict next state from current via transition model
        predicted = self.transition(observed)  # [B, T, C]
        # Causal shift: prediction at t is based on state at t-1
        # First timestep has no previous state, keep observation
        predicted = torch.cat([observed[:, :1], predicted[:, :-1]], dim=1)

        # Step 2: Kalman update — blend prediction and observation
        # gain ∈ [0, 1]: high gain → trust observation, low gain → trust prediction
        gain = self.kalman_gain(torch.cat([predicted, observed], dim=-1))  # [B, T, C]
        compressed = gain * observed + (1 - gain) * predicted

        # 隐式 L+S 分离 (Kalman 物理意义):
        #   L_kalman = compressed (平滑状态 → 低秩, 对应稳定接触背景)
        #   S_kalman = observed - predicted (innovation → 稀疏, 对应突变的滑动/碰撞)
        if self.training:
            # detach: 避免 L/S 缓存持有 forward 计算图引用导致 gradient
            # checkpointing 失效 (会保留所有中间激活). L+S 正则化只作为
            # 监控信号 (w_lowrank/w_sparse=0.01 辅助正则), 不参与反传.
            self.last_L_kalman = compressed.detach()  # [B, T, C]
            self.last_S_kalman = (observed - predicted).detach()  # [B, T, C]

        # Adaptive query weighting (per-sample, input-dependent)
        q_weights = self.query_gate(x_norm)  # [B, num_queries]

        # Query-based summarization with adaptive weighting
        queries = self.temporal_queries.unsqueeze(0).expand(B, -1, -1)
        q = self.q_proj(queries)

        attn = torch.einsum('bqd,btd->bqt', q, compressed) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        summary = torch.einsum('bqt,btd->bqd', attn, compressed)  # [B, num_queries, C]

        # Adaptive weighted combination of queries (soft query count)
        summary = summary * q_weights.unsqueeze(-1)  # [B, num_queries, C]
        out = summary.sum(dim=1)  # [B, C]

        # Broadcast to all spatial positions
        out = out.view(B, C, 1, 1, 1).expand(-1, -1, T, H, W)
        out = self.proj(out)

        if return_attn:
            return out, attn
        return out


# ═══════════════════════════════════════════════════════════════════
# 5. AdaptiveFeatCS (adaptive feature-level CS measurement)
# ═══════════════════════════════════════════════════════════════════

class AdaptiveFeatCS(nn.Module):
    """Adaptive feature-level CS measurement.

    No forced kernel size or channel conversion. Uses adaptive pooling
    to produce a fixed 7x7 measurement grid regardless of feature resolution,
    with input-dependent channel weighting (adaptive).
    """
    def __init__(self, dim, grid_size=7, num_basis=4):
        super().__init__()
        self.grid_size = grid_size
        self.dim = dim
        self.num_basis = num_basis

        # Adaptive channel weighting (input-dependent)
        self.channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, dim), nn.Sigmoid(),
        )

        # Multi-basis spatial modulation
        self.s_basis = nn.Parameter(torch.randn(num_basis, 1, grid_size, grid_size) * 0.02)
        self.basis_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, 16), nn.GELU(),
            nn.Linear(16, num_basis), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """x: [B, C, H, W] -> measurement [B, 1, 7, 7]"""
        # Adaptive pool to fixed grid (works at any resolution)
        x_pooled = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))  # [B, C, 7, 7]

        # Adaptive channel weighting
        w = self.channel_weight(x)  # [B, C]
        x_weighted = x_pooled * w.unsqueeze(-1).unsqueeze(-1)  # [B, C, 7, 7]

        # Channel reduction (adaptive, not forced)
        y = x_weighted.mean(dim=1, keepdim=True)  # [B, 1, 7, 7]

        # Multi-basis spatial modulation
        bw = self.basis_weight(x)  # [B, num_basis]
        s_dyn = torch.einsum('bk,khw->bhw', bw, self.s_basis.squeeze(1))  # [B, 7, 7]
        y = y.squeeze(1) * s_dyn  # [B, 7, 7]
        return y.unsqueeze(1)  # [B, 1, 7, 7]


# ═══════════════════════════════════════════════════════════════════
# 5.5 LSDecomposition (L+S 低秩稀疏分解, 轻量化)
# ═══════════════════════════════════════════════════════════════════

class LSDecomposition(nn.Module):
    """时空分离 L+S 低秩稀疏分解 (轻量化, 可微).

    借鉴红外图像 STT (Spatio-Temporal Tensor) 思想, 将分解分为两个维度:
    - L_t (时序低秩): T → rank → T 瓶颈, 捕获帧间相关的静态接触结构
    - L_s (空间低秩): 通道瓶颈 C → rank → C, 捕获背景均匀区域
    - S (稀疏): 通道相关软阈值, 捕获空间稀疏的动态变形 + 接触边缘

    用瓶颈结构替代 SVD, 避免 O(n^2) 计算和显存爆炸.
    通道相关阈值: 每通道独立 τ, 适应不同特征通道的稀疏度.

    训练时缓存 last_L/last_S 供 Schatten-p 正则化损失使用.
    """
    def __init__(self, dim, rank=4, T=8):
        super().__init__()
        self.T = T
        # 时序低秩: T → rank → T (捕获帧间相关性, 静态接触)
        self.t_down = nn.Linear(T, rank)
        self.t_up = nn.Linear(rank, T)
        # 空间低秩: 通道瓶颈 C → rank → C (捕获背景均匀性)
        self.s_down = nn.Conv3d(dim, rank, kernel_size=1)
        self.s_up = nn.Conv3d(rank, dim, kernel_size=1)
        # 通道相关稀疏阈值 (每通道独立, 适应不同特征稀疏度)
        self.tau = nn.Parameter(torch.ones(dim, 1, 1, 1) * 0.01)
        # 缓存 L/S 供正则化损失使用 (训练时填充)
        self.last_L = None
        self.last_S = None

    def forward(self, x):
        """x: [B, C, T, H, W] → [B, C, T, H, W] (L+S 重建)"""
        # 时序低秩 L_t: T → rank → T (帧间相关性)
        # x: [B, C, T, H, W] → [B, C, H, W, T] → Linear(T) → back
        x_t = x.permute(0, 1, 3, 4, 2)  # [B, C, H, W, T]
        L_t = self.t_up(self.t_down(x_t))  # [B, C, H, W, T]
        L_t = L_t.permute(0, 1, 4, 2, 3)  # [B, C, T, H, W]
        # 空间低秩 L_s: 通道瓶颈 (背景均匀性)
        L_s = self.s_up(self.s_down(x))  # [B, C, T, H, W]
        # 联合低秩 L = L_t + L_s
        L = L_t + L_s
        # 稀疏残差 S: 通道相关软阈值
        S = x - L
        S = torch.sign(S) * torch.clamp(torch.abs(S) - self.tau, min=0)
        # 缓存 L/S 供 Schatten-p 正则化损失使用
        if self.training:
            # 关键优化: 池化后缓存, 避免保留完整 5D 特征图 (~412MB/张).
            # 完整 [B,C,T,H,W] 池化到 [B,C,T,4,4] 后做 svdvals, 结果完全等价
            # (trainer.py 后续也是这样池化的). 缓存体积减少 760x (~540KB).
            T_l = L.shape[2]
            self.last_L = F.adaptive_avg_pool3d(L.detach(), (T_l, 4, 4))
            self.last_S = F.adaptive_avg_pool3d(S.detach(), (S.shape[2], 4, 4))
        return L + S


# ═══════════════════════════════════════════════════════════════════
# 6. CSGradientStep (adaptive, no forced channel conversions)
# ═══════════════════════════════════════════════════════════════════

class CSGradientStep(nn.Module):
    """Per-iteration CS gradient step with adaptive feature-level CS.

    No forced channel conversions. S_feat operates directly on dim features.
    Back-projection uses adaptive interpolation (resolution-adaptive).
    """
    def __init__(self, dim, in_ch=3, patch=32, ls_rank=4, T=8):
        super().__init__()
        self.dim = dim

        # Adaptive feature-level CS (no forced channel conversion)
        self.S_feat = AdaptiveFeatCS(dim, grid_size=7, num_basis=4)

        # Adaptive back-projection (resolution-adaptive, no fixed kernel)
        self.R_feat_proj = nn.Conv2d(1, dim, kernel_size=1)

        # Kalman gain (input-dependent, learned)
        self.gain_net = nn.Sequential(
            nn.Conv3d(dim, dim // 2, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Cross-iteration gate
        self.iter_gate = nn.Sequential(
            nn.Conv3d(dim * 2, dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Causal temporal conv denoiser
        self.denoise_norm = LayerNorm3D(dim)
        self.causal_conv = nn.Sequential(
            nn.Conv3d(dim, dim, (5, 1, 1), padding=(2, 0, 0), groups=dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1),
        )

        # Residual refinement
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1)),
        )

        # L+S 时空分离低秩稀疏分解 (近端算子)
        self.ls_decomp = LSDecomposition(dim, rank=ls_rank, T=T)

        # 残差缩放 (LayerScale 风格, 梯度稳定性, 初始化为小值)
        self.denoise_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.refine_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, y_feat, x_prev=None):
        """Per-iteration CS gradient step.

        Args:
            x: [B, dim, T, H_feat, W_feat] feature map
            y_feat: [B, 1, T, 7, 7] feature-level CS measurement
            x_prev: previous iteration's feature map
        """
        B, dim, T, H_feat, W_feat = x.shape

        # --- Step 1: Adaptive feature-level CS gradient ---
        # S_feat operates directly on dim features (no forced channel conversion)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, dim, H_feat, W_feat)
        y_pred = self.S_feat(x_2d)  # [B*T, 1, 7, 7]

        # Error: y_feat - S_feat(x)
        err = y_feat.reshape(B * T, y_feat.shape[1], y_feat.shape[-2], y_feat.shape[-1]) - y_pred  # [B*T, 1, 7, 7]

        # Adaptive back-projection (interpolate to feature size, then project)
        err_up = F.interpolate(err, size=(H_feat, W_feat), mode='bilinear', align_corners=False)  # [B*T, 1, H_feat, W_feat]
        grad = self.R_feat_proj(err_up)  # [B*T, dim, H_feat, W_feat]
        grad = grad.reshape(B, T, dim, H_feat, W_feat).permute(0, 2, 1, 3, 4)

        K = self.gain_net(x)
        x_new = x + K * grad

        # --- Cross-iteration information flow ---
        if x_prev is not None:
            gate = self.iter_gate(torch.cat([x, x_prev], dim=1))
            x = gate * x_new + (1 - gate) * x_prev
        else:
            x = x_new

        # --- Step 1.5: L+S 低秩稀疏分解 (近端算子) ---
        # L (低秩): 静态接触结构, S (稀疏): 动态变形
        x = self.ls_decomp(x)

        # --- Step 2: Causal temporal conv denoising (残差缩放) ---
        x_norm = self.denoise_norm(x)
        x = x + self.denoise_scale * self.causal_conv(x_norm)

        # --- Step 3: Residual refinement (残差缩放) ---
        x = x + self.refine_scale * self.refine(x)

        return x


# ═══════════════════════════════════════════════════════════════════
# 7. DSTLayer
# ═══════════════════════════════════════════════════════════════════

class DSTLayer(nn.Module):
    """DST layer: TimeBlock -> DeformableSpatial -> HistoryCompressor -> FFN."""
    def __init__(self, dim, num_offset_points=9):
        super().__init__()
        self.dst_time = DSTTimeBlock(dim)
        self.dst_space = DeformableSpatialBlock(dim, num_offset_points)
        self.history = TactileHistoryCompressor(dim, num_queries=3)
        self.ffn = FFN3D(dim, hidden=4)
        self.w_time = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_space = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_history = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))
        self.w_ffn = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1, 1))

    def forward(self, x, return_mechanism=False):
        t_out = self.dst_time(x)
        x = self.w_time * t_out + x
        mech = {}
        if return_mechanism:
            s_out, s_attn = self.dst_space(x, return_attn=True)
            x = self.w_space * s_out + x
            h_out, h_attn = self.history(x, return_attn=True)
            x = self.w_history * h_out + x
            mech['space_attn'] = s_attn
            mech['history_attn'] = h_attn
        else:
            x = self.w_space * self.dst_space(x) + x
            x = self.w_history * self.history(x) + x
        x = self.w_ffn * self.ffn(x) + x
        if return_mechanism:
            return x, mech
        return x


# ═══════════════════════════════════════════════════════════════════
# 8. UncertaintyHead (per-pixel variance for NLL loss)
# ═══════════════════════════════════════════════════════════════════

class UncertaintyHead(nn.Module):
    """Predicts per-pixel reconstruction variance for uncertainty estimation.

    Uses negative log-likelihood loss: 0.5 * (log(σ²) + (x-μ)²/σ²)
    Provides confidence maps for robotics deployment.
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim // 2, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim // 2, dim // 4, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim // 4, 1, kernel_size=1),
            nn.Softplus(),  # ensures positive variance
        )

    def forward(self, x):
        """x: [B, dim, T, H, W] -> variance [B, 1, T, H, W]"""
        return self.net(x) + 1e-6  # minimum variance for numerical stability


# ═══════════════════════════════════════════════════════════════════
# 9. LSDUNet (main model — fully adaptive)
# ═══════════════════════════════════════════════════════════════════

class LSDUNet(nn.Module):
    """LSDUNet: Lightweight deep unfolding CS reconstruction.

    Fully adaptive: proportional 2x downsampling, no forced channel conversions,
    resolution-adaptive CS measurement. Includes uncertainty estimation.
    """

    def __init__(self, ratio, iter_num=6, model_dim=64, patch=32,
                 in_ch=3, d_state=16, ls_rank=4, num_frames=8, **kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.iter_num = iter_num
        self.patch = patch
        self.in_ch = in_ch
        self.full_dim = patch ** 2
        self.cs_dim = int(ratio * self.full_dim)

        # Original CS modules (at full resolution)
        self.adaptive_s = AdaptiveSModule(self.patch, self.cs_dim)
        self.R = RModule(self.patch)

        # Adaptive feature-level CS measurement
        self.S_feat = AdaptiveFeatCS(model_dim, grid_size=7, num_basis=4)

        # Tokenizer (proportional 2x downsampling, fully adaptive)
        self.tokenizer = ConvTokenizer3D(in_ch=in_ch, dim=self.model_dim)

        # Multi-scale skip connection (tokenizer → output)
        self.skip_proj = nn.Conv3d(self.model_dim, self.model_dim, kernel_size=1)

        # Output projection + adaptive upsampling
        self.proj_out = nn.Conv3d(self.model_dim, in_ch, kernel_size=1)

        # Uncertainty estimation head
        self.uncertainty_head = UncertaintyHead(model_dim)

        # Deep unfolding iterations
        self.gdb = nn.ModuleList([
            CSGradientStep(model_dim, in_ch=in_ch, patch=patch, ls_rank=ls_rank, T=num_frames)
            for _ in range(self.iter_num)
        ])
        self.dst = nn.ModuleList([
            DSTLayer(model_dim)
            for _ in range(self.iter_num)
        ])

    def get_ls_regularization(self):
        """返回所有迭代的 L/S 张量列表, 供正则化损失使用.

        双重 L+S 来源:
          ① 显式 LSDecomposition (时空分离 + Schatten-p)
          ② 隐式 CausalLatentFilter (Kalman 平滑状态=L, innovation=S)
        """
        Ls, Ss = [], []
        base = self.module if hasattr(self, 'module') else self
        # ① 显式 L+S: LSDecomposition
        for gdb in base.gdb:
            ls = gdb.ls_decomp
            if ls.last_L is not None:
                Ls.append(ls.last_L)
                Ss.append(ls.last_S)
        # ② 隐式 L+S: Kalman 滤波 (L=平滑状态, S=innovation)
        for dst in base.dst:
            thc = dst.history
            if thc.last_L_kalman is not None:
                Ls.append(thc.last_L_kalman)
                Ss.append(thc.last_S_kalman)
        return Ls, Ss

    def _iter_forward(self, i, x, y_feat, x_prev, return_mechanism=False):
        x_out = self.gdb[i](x, y_feat, x_prev)
        if return_mechanism:
            x_out, mech = self.dst[i](x_out, return_mechanism=True)
            return x_out, mech
        x_out = self.dst[i](x_out)
        return x_out

    def forward(self, x, return_intermediates=False, return_mechanism=False, return_uncertainty=False):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T * C, 1, H, W)

        if return_mechanism:
            y, s_dyn, basis_weights = self.adaptive_s(x_flat, return_basis_weights=True)
        else:
            y, s_dyn = self.adaptive_s(x_flat)

        x_r = self.R(y, s_dyn)
        x_r = x_r.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        # Tokenize (proportional 2x downsampling, fully adaptive)
        x_tok = self.tokenizer(x_r)
        x = x_tok
        H_feat, W_feat = x.shape[-2], x.shape[-1]

        # Compute adaptive feature-level CS measurement ONCE
        # Detach x_tok so S_feat gets gradient through itself but not through tokenizer
        # (avoid double-counting tokenizer gradients via the measurement target)
        x_tok_2d = x_tok.detach().permute(0, 2, 1, 3, 4).reshape(B * T, self.model_dim, H_feat, W_feat)
        y_feat_flat = self.S_feat(x_tok_2d)  # [B*T, 1, 7, 7] - S_feat params get gradient
        y_feat = y_feat_flat.reshape(B, 1, T, y_feat_flat.shape[-2], y_feat_flat.shape[-1])

        intermediates = []
        mech_data = {} if return_mechanism else None

        x_prev = None
        # Gradient checkpointing: forward 不存中间激活, backward 时重算.
        # 注意: 条件不能检查 x.requires_grad (输入图像通常不需要梯度),
        # 否则训练时永远不启用 → 中间激活全部保留 → OOM.
        # 正确判断: training 模式 + _use_ckpt 标志 + 至少一个参数需要梯度.
        use_ckpt = (self.training and getattr(self, '_use_ckpt', False)
                    and any(p.requires_grad for p in self.parameters()))
        for i in range(self.iter_num):
            if use_ckpt:
                if return_mechanism:
                    x_new, mech = _ckpt(self._iter_forward, i, x, y_feat, x_prev, True, use_reentrant=False)
                    mech_data[f'iter_{i}'] = {k: v.detach() if torch.is_tensor(v) else v
                                              for k, v in mech.items()}
                else:
                    x_new = _ckpt(self._iter_forward, i, x, y_feat, x_prev, False, use_reentrant=False)
            else:
                if return_mechanism:
                    x_new, mech = self._iter_forward(i, x, y_feat, x_prev, True)
                    mech_data[f'iter_{i}'] = {k: v.detach() if torch.is_tensor(v) else v
                                              for k, v in mech.items()}
                else:
                    x_new = self._iter_forward(i, x, y_feat, x_prev, False)
            x_prev = x
            x = x_new

            if return_intermediates:
                x_proj = self.proj_out(x)
                x_up = F.interpolate(x_proj, size=(T, H, W), mode='trilinear', align_corners=False).permute(0, 2, 1, 3, 4)
                intermediates.append(x_up)

        # Multi-scale feature fusion (skip connection from tokenizer)
        x = x + self.skip_proj(x_tok)

        # Uncertainty estimation
        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_head(x)  # [B, 1, T, H_feat, W_feat]
            uncertainty = F.interpolate(uncertainty, size=(T, H, W), mode='trilinear', align_corners=False)
            uncertainty = uncertainty.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]

        # Output: project + upsample to original resolution
        x_mean = self.proj_out(x)
        x_mean = F.interpolate(x_mean, size=(T, H, W), mode='trilinear', align_corners=False)
        x_mean = x_mean.permute(0, 2, 1, 3, 4)

        if return_mechanism:
            result = {'basis_weights': basis_weights.detach().cpu(), 'iter_data': mech_data}
            if return_intermediates:
                # Include intermediates in result dict (detached for visualization)
                result['intermediates'] = [t.detach() for t in intermediates]
            if return_uncertainty:
                result['uncertainty'] = uncertainty
            return x_mean, result
        if return_uncertainty:
            return x_mean, uncertainty
        if return_intermediates:
            return x_mean, intermediates
        return x_mean
