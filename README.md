# LSDUNet — Learned Spatial-temporal Deep Unfolding Network

基于压缩感知 + 深度展开的轻量化触觉视频重建网络。融合 TacMamba (时序压缩) 与 CMLF (贝叶斯融合 + 因果状态空间滤波)，2.637M 参数，支持 4×4090 DDP 训练。

## 项目结构

```
LSDUNet/
├── model/
│   ├── __init__.py
│   └── model_3d.py              # 核心模型 (LSDUNet + 15个子模块)
├── data_processor.py             # 数据集加载与预处理
├── trainer.py                    # 训练/验证循环 (EMA + 多术语损失)
├── train.py                      # 训练入口 (DDP支持)
├── eval.py                       # 统一评估入口 (--mode standard|cross-domain|noise|interpret)
├── metrics.py                    # 评估指标 (含ECE/Brier校准)
├── utils.py                      # 工具函数
├── run_4x4090.sh                 # 4×4090 DDP启动脚本
├── requirements.txt
└── LICENSE
```

## 核心贡献

### 1. 自适应 CS 采样 — AdaptiveSModule (增强版)

触觉图像分块压缩采样，K=4 个基矩阵动态组合。

| 组件 | 机制 | 改进 |
|------|------|------|
| `meta_net` | 3维统计特征 (均值 + 标准差 + 梯度幅值) → 基权重 | 原1维标量→3维，内容自适应更丰富 |
| `importance_net` | 3层CNN空间重要性加权 → Sigmoid | 原5层→3层，参数减少62% |
| `ortho_loss` | 正交正则化鼓励基矩阵多样性 | — |

### 2. 贝叶斯特征融合 — BayesianFusion ★CMLF

替换暴力 `cat+conv` 拼接，用逆方差加权最大似然估计融合主分支与边缘分支：

```
σ₁² = Softplus(Linear(pool(f_main)))
σ₂² = Softplus(Linear(pool(f_edge)))
fused = (f_main·σ₂² + f_edge·σ₁²) / (σ₁² + σ₂²)
```

仅增加 2 个 Linear 层 (~8.3K 参数)，将多模态对齐冗余降到最低。

### 3. DST 时空层 — DSTTimeBlock + DeformableSpatialBlock

| 组件 | 机制 | 空间复杂度 |
|------|------|-----------|
| `DSTTimeBlock` | 多尺度时间卷积 [3,5,7] + Sigmoid 门控 | O(D·T) |
| `DeformableSpatialBlock` | 可变形卷积 (DeformConv2d) + 多尺度膨胀卷积 [d=1,3,7] | O(D·k²) |
| `FFN3D` | 3D卷积 + 时空深度可分离卷积增强 | O(D²) |

DeformConv2d 替代 grid_sample，显存仅需 ~1/100；膨胀深度卷积替代 MambaSSM 空间扫描，显存降低约 190×。

### 4. 时序历史压缩 — TactileHistoryCompressor ★TacMamba + CMLF

用 O(T) MambaSSM 替代 O(T²) 长程注意力，融合两条论文思想：

```
observed   = MambaSSM(x_temporal)          ← TacMamba: O(T) 替代 O(T²) 注意力
predicted  = shift(transition(observed))   ← CMLF:  因果状态预测
gain       = Sigmoid(MLP(pred, observed)) ← CMLF:  卡尔曼增益 (输入依赖)
compressed = gain·observed + (1-gain)·predicted  ← 贝叶斯卡尔曼更新
summary    = attention(adaptive_queries, compressed)
```

- MambaSSM: 仅用于 T=8 时序维度，纯 PyTorch 实现 (无需 CUDA 编译)
- `_scan_direct`: T≤32 时使用直接递归扫描，避免 parallel scan 的数值溢出
- 自适应查询权重：输入复杂度相关的 soft query count

### 5. 深度展开 — CSGradientStep (6次迭代)

每次迭代执行：

```
① 特征级CS梯度:   grad = R_feat(y_feat - S_feat(x))
② 卡尔曼更新:      x_new = x + K(x)·grad
③ 跨迭代门控:      x = gate·(x_new, x_prev)
④ 因果时序去噪:    x += causal_conv(x)
```

特征级 CS 测量 (`AdaptiveFeatCS`) 只算一次但每次迭代都用，7×7 网格分辨率自适应。

### 6. 不确定性估计 — UncertaintyHead

Softplus 预测每像素方差，NLL 损失训练，ECE/Brier Score 评估校准质量，为机器人部署提供置信度图。

## 损失函数

| 损失项 | 权重 | 公式 |
|--------|------|------|
| MSE 重建 | `w=1.0` | `MSE(output, target)` |
| Sobel 边缘 | `w=0.1` | `MSE(Sobel(output), Sobel(target))` |
| FFT 频率 | `w=0.01` | `\|FFT(output) - FFT(target)\|` |
| 不确定性 NLL | `w=0.01` | `0.5·(log(σ²) + (x-μ)²/σ²)` |
| 正交正则化 | `w=0.01` | `ortho_loss()` (基矩阵之间) |

## 数据流

```
输入: [B, 8, 3, 224, 224]
 │
 ├─① 像素级CS测量 (AdaptiveSModule)
 │   224×224 → 分组卷积(S·x) → [B, cs_dim, 7, 7]
 │   meta_net(mean+std+grad) → 4组基矩阵动态组合
 │   importance_net → 空间重要性加权
 │
 ├─② 初始反投影 (RModule)
 │   S^T·y → [B, 1, 224, 224] → reshape [B, 3, 8, 224, 224]
 │
 ├─③ Tokenization + BayesianFusion ★CMLF
 │   主分支: Conv3D stride-2 → [B, 64, 8, 112, 112]
 │   边缘分支: spatial + temporal edges → edge_proj
 │   贝叶斯融合: inverse-variance weighted MLE
 │
 ├─④ 特征级CS测量 (AdaptiveFeatCS) — 只算一次
 │   x_tok → adaptive_pool(7×7) → [B, 1, 8, 7, 7]
 │
 ├─⑤ 深度展开循环 (×6)
 │   for i in 0..5:
 │   ├─ CSGradientStep: 梯度修正 + 卡尔曼更新 + 因果去噪
 │   │   y_pred = S_feat(x) → err → grad → x_new
 │   │   gate(x_new, x_prev) → causal_conv → x
 │   └─ DSTLayer
 │       ├─ DSTTimeBlock:    多尺度时间卷积 [3,5,7]
 │       ├─ DeformableSpatial: 可变形卷积 + 膨胀卷积 [d=1,3,7]
 │       └─ TactileHistoryCompressor ★TacMamba+CMLF
 │           MambaSSM → 因果预测 → 卡尔曼更新 → 查询摘要 → 广播
 │
 ├─⑥ 跳跃连接: x += skip_proj(x_tok)
 │
 ├─⑦ 不确定性: UncertaintyHead → [B, 8, 1, 224, 224]
 │
 └─⑧ 输出: proj_out + trilinear upsample → [B, 8, 3, 224, 224]
```

## 总结

| 亮点 | 来源 | 参数量 |
|------|------|--------|
| BayesianFusion 贝叶斯融合 | CMLF启发 (arXiv:2604.02108) | ~8.3K |
| CausalLatentFilter 因果滤波 | CMLF启发 (arXiv:2604.02108) | ~8.5K |
| MambaSSM 时序压缩 | TacMamba启发 (arXiv:2603.01700) | ~21K |
| DeformableSpatialBlock 可变形注意力 | 原创 | ~170K/层 |
| AdaptiveFeatCS 自适应特征CS | 原创 | ~1.1K |
| UncertaintyHead 不确定性估计 | 原创 | ~23K |

轻量化设计：纯前馈，无VAE/Diffusion；空间卷积 O(D·k²) 非 O(B·L·D·d_state)；时序 Mamba O(T) 非 O(T²)。

## 训练

```bash
# 单卡
python train.py

# 4× GPU DDP (推荐)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py --ratios 0.10 --val_interval 5
```

依次训练 5 个压缩比：`[0.01, 0.04, 0.10, 0.25, 0.50]`，每轮 150 epoch。

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 150 | 训练轮数 |
| `--batch_size` | 8 | 单卡批次大小 (224×224×8帧, ~9-16GB显存) |
| `--grad_accum` | 2 | 梯度累积 (effective batch = 8×2×4GPUs = 64) |
| `--image_size` | 224 | 训练分辨率 |
| `--num_frames` | 8 | 3D 体积帧数 |
| `--patch` | 32 | CS 采样分块 |
| `--iter_num` | 6 | 深度展开迭代次数 |
| `--model_dim` | 64 | 特征维度 |
| `--lr` | 2e-4 | 初始学习率 (AdamW) |
| `--flr` | 1e-5 | 终点学习率 (余弦退火) |
| `--warm_epochs` | 5 | 线性 warmup 轮数 |
| `--wd` | 0.05 | AdamW 权重衰减 |
| `--grad_clip` | 1.0 | 梯度裁剪 |
| `--val_interval` | 5 | 验证间隔 (epoch) |
| `--resume` | False | 从 checkpoint.pth 恢复训练 |

### 损失权重 (可调参数)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--w_edge` | 0.1 | Sobel 边缘损失 |
| `--w_freq` | 0.01 | FFT 频率损失 |
| `--w_ortho` | 0.01 | 正交正则化 |
| `--w_nll` | 0.01 | 不确定性 NLL 损失 |

## GPU 需求

bf16 在 Ampere/Ada/Blackwell (sm≥80) 上自动启用，fp16+GradScaler 在旧卡上回退。

| GPU | 显存 | 配置 | 备注 |
|-----|------|------|------|
| **A100 / H100** | 80 GB | `batch=16, grad_accum=1` | **最佳选择** |
| RTX 4090 / 3090 | 24 GB | `batch=8, grad_accum=2 (4卡)` | 4×4090 DDP 推荐 |
| RTX 3090 (单卡) | 24 GB | `batch=4, grad_accum=4` | 有效batch=16 |
| RTX 4060 Ti | 16 GB | `batch=2, grad_accum=8` | 有效batch=16 |
| 更小显存 | <16 GB | `batch=1, grad_accum=16` | 仅测试用 |

参数量: **2.637M** (ratio=0.10)，梯度checkpointing全程开启。

## 评估

```bash
# 标准评估 (全数据集, 全压缩比)
python eval.py

# 跨域泛化 (高分辨率 + ECE/Brier)
python eval.py --mode cross-domain --image_size 448

# 噪声鲁棒性评估
python eval.py --mode noise
python eval.py --mode noise --full                # 完整消融对比 (ConvTok vs LinearTok)

# 机制分析与可视化 (--submode export|render|all)
python eval.py --mode interpret --submode export --ratio 0.10
python eval.py --mode interpret --submode render --ckpt trained_model/lsdunet_0.10.pth
python eval.py --mode interpret --submode all --ckpt trained_model/lsdunet_0.10.pth
```

## 安装

```bash
pip install -r requirements.txt
```

## 依赖

- Python >= 3.12
- PyTorch >= 2.0
- torchvision, scikit-image, scipy, lpips, tqdm, matplotlib
