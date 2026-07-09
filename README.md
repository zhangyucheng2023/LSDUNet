# LSDUNet — Learned Spatial-temporal Deep Unfolding Network

基于压缩感知 + 深度展开 + L+S 低秩稀疏分解的轻量化触觉视频重建网络。融合 TacMamba (时序压缩) 与 CMLF (贝叶斯融合 + 因果状态空间滤波)，2.640M 参数，支持 4×4090 DDP 训练。

## 项目结构

```
LSDUNet/
├── model/model_3d.py             # 核心模型 (LSDUNet + 子模块)
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

## 核心模块

### 1. 自适应 CS 采样 — AdaptiveSModule

触觉图像分块压缩采样，K=4 个基矩阵动态组合。

- `meta_net`: 3维统计特征 (均值 + 标准差 + 梯度幅值) → 基权重
- `importance_net`: 3层CNN空间重要性加权 → Sigmoid
- `ortho_loss`: 正交正则化鼓励基矩阵多样性

### 2. 贝叶斯特征融合 — BayesianFusion (CMLF)

替换 `cat+conv` 拼接，用逆方差加权最大似然估计融合主分支与边缘分支：

```
σ₁² = Softplus(Linear(pool(f_main)))
σ₂² = Softplus(Linear(pool(f_edge)))
fused = (f_main·σ₂² + f_edge·σ₁²) / (σ₁² + σ₂²)
```

仅增加 2 个 Linear 层 (~8.3K 参数)。

### 3. DST 时空层

| 组件 | 机制 | 复杂度 |
|------|------|--------|
| `DSTTimeBlock` | 多尺度时间卷积 [3,5,7] + Sigmoid 门控 | O(D·T) |
| `DeformableSpatialBlock` | 可变形卷积 + 多尺度膨胀卷积 [d=1,3,7] | O(D·k²) |
| `FFN3D` | 3D卷积 + 时空深度可分离卷积 | O(D²) |

### 4. 时序历史压缩 — TactileHistoryCompressor (TacMamba + CMLF)

用 O(T) MambaSSM 替代 O(T²) 长程注意力：

```
observed   = MambaSSM(x_temporal)          ← TacMamba: O(T) 时序压缩
predicted  = shift(transition(observed))   ← CMLF:  因果状态预测
gain       = Sigmoid(MLP(pred, observed)) ← CMLF:  卡尔曼增益
compressed = gain·observed + (1-gain)·predicted
```

- 纯 PyTorch 实现 (无需 CUDA 编译)
- `_scan_direct`: T≤32 时直接递归扫描，避免 parallel scan 数值溢出

### 5. 深度展开 — CSGradientStep (6次迭代)

每次迭代：特征级CS梯度修正 → 卡尔曼更新 → 跨迭代门控 → **L+S 低秩稀疏分解** → 因果时序去噪。

#### L+S 低秩稀疏分解 — LSDecomposition

每次迭代中将特征分解为低秩 + 稀疏两部分（近端算子）：

```
L = up(down(x))                          ← 通道瓶颈近似低秩 (静态接触结构)
S = sign(x-L) · max(|x-L| - τ, 0)       ← 软阈值稀疏化 (动态变形)
x = L + S
```

- 低秩 L 捕获时序一致的静态接触区域
- 稀疏 S 捕获空间稀疏的动态变形/滑动
- 用 1×1×1 卷积瓶颈替代 SVD，避免 O(n²) 开销
- 6 次迭代共增加 3.5K 参数 (0.13%)

### 6. 不确定性估计 — UncertaintyHead

Softplus 预测每像素方差，NLL 损失训练，ECE/Brier Score 评估校准质量。

## 损失函数

| 损失项 | 参数 | 默认 | 公式 |
|--------|------|------|------|
| MSE 重建 | — | 1.0 | `MSE(output, target)` |
| Sobel 边缘 | `--w_edge` | 0.1 | `MSE(Sobel(output), Sobel(target))` |
| DWT 小波 | `--w_freq` | 0.01 | 多级 Haar 小波 L1 损失 |
| SSIM 结构 | `--w_ssim` | 0.1 | `1 - SSIM(output, target)` (可微) |
| 不确定性 NLL | `--w_nll` | 0.01 | `0.5·(log(σ²) + (x-μ)²/σ²)` |
| 正交正则化 | `--w_ortho` | 0.01 | `ortho_loss()` (基矩阵之间) |
| L+S 低秩 | `--w_lowrank` | 0.01 | Schatten-0.5 范数 `Σσ^0.5` (池化SVD, 精确促进低秩) |
| L+S 稀疏 | `--w_sparse` | 0.01 | `‖S‖_1` (鼓励稀疏残差) |

辅助损失在 warmup 阶段线性增长（避免训练初期不稳定）。DWT 小波损失替代 FFT：多尺度局部时频、边缘保持、O(N) 复杂度、支持 bf16。

L+S 时空分离分解：时序低秩（T→rank→T 瓶颈，捕获帧间静态接触）+ 空间低秩（通道瓶颈，捕获背景均匀区域）+ 通道相关软阈值稀疏（动态变形+接触边缘）。Schatten-0.5 范数比核范数更精确促进低秩，借鉴红外图像 NRIU 非凸秩近似。

## 数据流

```
输入 [B, 8, 3, 224, 224]
  │
  ├─ AdaptiveSModule: 分组卷积 S·x → [B, cs_dim, 7, 7]
  │                   meta_net(均值+标准差+梯度) → 4组基矩阵动态组合
  │
  ├─ RModule: S^T·y → [B, 3, 8, 224, 224]
  │
  ├─ ConvTokenizer3D + BayesianFusion → [B, 64, 8, 112, 112]
  │
  ├─ AdaptiveFeatCS: x_tok → adaptive_pool(7×7) → [B, 1, 8, 7, 7]  (只算一次)
  │
  ├─ 深度展开循环 ×6:
  │   ├─ CSGradientStep: 梯度修正 + 卡尔曼更新 + 门控
  │   │   └─ LSDecomposition: L(低秩) + S(稀疏) 分解
  │   └─ DSTLayer: DSTTimeBlock + DeformableSpatial + TactileHistoryCompressor
  │
  ├─ 跳跃连接: x += skip_proj(x_tok)
  ├─ UncertaintyHead → [B, 8, 1, 224, 224]
  └─ proj_out + trilinear upsample → [B, 8, 3, 224, 224]
```

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
| `--batch_size` | 8 | 单卡批次大小 |
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
| `--ls_rank` | 4 | L+S 分解瓶颈维度 |
| `--ema_decay` | 0.999 | EMA decay (warmup 后) |
| `--w_lowrank` | 0.01 | L+S 低秩正则化权重 |
| `--w_sparse` | 0.01 | L+S 稀疏正则化权重 |
| `--resume` | False | 从 checkpoint.pth 恢复训练 |

### GPU 需求

bf16 在 Ampere/Ada/Blackwell (sm≥80) 上自动启用，fp16+GradScaler 在旧卡上回退。

| GPU | 显存 | 推荐配置 |
|-----|------|----------|
| A100 / H100 | 80 GB | `batch=16, grad_accum=1` |
| RTX 4090 / 3090 | 24 GB | `batch=8, grad_accum=2 (4卡)` |
| 更小显存 | <16 GB | `batch=1, grad_accum=16` (仅测试) |

参数量: **2.640M** (ratio=0.10)

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
