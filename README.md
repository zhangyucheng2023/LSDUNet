# LSDUNet — Learned Spatial-temporal Deep Unfolding Network for Tactile Image Compressed Sensing

基于**压缩感知 + 低秩稀疏分解深度展开**的触觉图像重建网络，使用极少量测量值重建高质量触觉压力分布图。采用混合架构：**浅层 3D 卷积 Token 化 + 深层 DST (Divided Space-Time) 可变形时空注意力 + 长程时序注意力 + β-NLL 异方差不确定性量化**。

## 项目结构

```
LSDUNet/
├── model/
│   ├── __init__.py      # 模块导出
│   └── model_3d.py      # 核心模型 (LSDUNet, AdaptiveSModule, ConvTokenizer3D, DSTLayer, LongRangeTemporalAttention)
├── data_processor.py     # 数据集加载与预处理
├── trainer.py            # 训练/验证循环
├── train.py              # 训练入口脚本
├── eval.py               # 评估脚本 (含不确定性可视化)
├── eval_noise.py         # 动态视频抗噪评估
├── metrics.py            # 评估指标 (ROI-PSNR, Edge-PSNR, Temporal-PSNR, 效率指标)
├── utils.py              # 工具函数 (设备、种子、色彩空间转换)
├── requirements.txt      # Python 依赖
└── LICENSE               # MIT License
```

## 核心思路

### 1. 自适应压缩感知 (CS) 采样 — 多基矩阵动态组合

触觉图像被分块压缩采样，通过**内容感知的多基矩阵动态组合**降低测量维度：
       
```
y = S(x) · imp(x)   [自适应采样: AdaptiveSModule, 重要性加权]
ẋ = R(y, s_dyn)      [初始反投影: RModule, per-sample ConvTranspose2d]
```

- **多基矩阵动态组合 (Mixture-of-Basis)**: 学习 K=4 个基采样矩阵，轻量 meta-network (~1K 参数) 根据输入统计量动态组合。不同传感器模式自动选择不同的基矩阵组合，实现跨域泛化
- **正交正则化**: 鼓励 K 个基矩阵学习互不重叠的采样模式，最大化组合多样性
- **重要性预测器**: 轻量 CNN 预测逐 patch 信号能量，高能量区域（接触面）获得更多有效测量，等效于 variable-rate CS
- **域不变性**: InstanceNorm2d 消除不同 GelSight 传感器之间的增益/曝光差异
- 输入尺寸经 `patch=32` 分块，压缩比由 `sensing_rate` 控制

### 2. 浅层 3D 卷积 Token 化 (ConvTokenizer3D)

替代传统 ViT 的线性 Patch 划分，用**三层渐进式 3D 卷积**替代简单投影，起到"卷积投影作为 Token 接入"的作用：

| 阶段 | 卷积核 | 归一化 | 作用 |
|------|--------|--------|------|
| `stem1` | `(3, 5, 5)` | GroupNorm(2,4) | 大空间感受野提取轮廓 |
| `stem2` | `(3, 3, 3)` | GroupNorm(4,8) | 时空联合特征细化 |
| `stem3` | `(1, 3, 3)` | GroupNorm(4,16) | 逐帧空间特征压缩 |

- **域不变性**: 使用 GroupNorm 替代 BatchNorm3d，对每个样本独立归一化，不依赖跨 batch 统计量，推理时不受训练域 (ToucHD) 的 running stats 影响，天然支持跨传感器泛化
- 同时配备**边缘增强分支**：Sobel-style 空间边缘卷积 + 帧间差分卷积，专门捕捉压力梯度和物体轮廓，抑制高频传感器噪声

### 3. DST 可变形时空注意力 + 长程时序 (DSTLayer)

每轮迭代按 **时间 → 空间 → 长程时序 → FFN** 顺序执行，配合 LayerScale 逐通道残差权重 (Touvron et al., ICCV 2021) 稳定训练：

| 阶段 | 组件 | 机制 | 作用 |
|------|------|------|------|
| 时间块 | `DSTTimeBlock` | 多尺度时间卷积 (3×1×1 / 5×1×1) + Sigmoid 门控 | 捕捉**滑动轨迹**，建模接触模式的时序演化 |
| 空间块 | `DSTSpaceBlock` | 可变形注意力：学习 2D 采样偏移 + `grid_sample` + Softmax 聚合 | 重构**压力分布形状**，自适应不规则接触区域 |
| 长程时序 | `LongRangeTemporalAttention` | 3 个可学习时序查询 (快划/慢划/全局) 跨注意力到全部 T 帧 | 捕捉超出局部窗口的**全局时序依赖** |
| 前馈 | `FFN3D` | 3D 卷积 + 空间/时间深度可分离卷积增强 | 通道混合与非线性变换 |

**LayerScale 残差权重**: 所有残差分支权重初始化为 `1e-5`，初始时网络退化为恒等映射，训练中逐步学习各模块贡献，训练结束后可分析各模块的最终权重。

### 4. 梯度去噪块 (GDB3D) — 低秩稀疏分解深度展开

模仿传统压缩感知的低秩稀疏分解优化问题，深度展开为 8 轮神经网络：

```
目标: min_x  ½ ||y - Sx||₂² + λ_low·σ_low(x) + λ_sparse·||x||₁
```

其中 `σ_low(x)` 为低秩正则（触觉背景的全局结构），`||x||₁` 为稀疏正则（接触区域的局部压力分布）。

| 子模块 | 功能 | 对应优化项 |
|--------|------|------------|
| `GRAD3D` | 计算测量残差 `y − S(x)`，反投影得到梯度修正量 | 数据一致性项 `½ ||y - Sx||₂²` |
| `DENO3D` | U-Net 风格 3 级下采样 → 瓶颈 5×5 卷积 → 3 级上采样去噪 | 先验项近端算子 `prox(λ·P(x))` |

- **8 轮迭代**逐步细化：从粗略初始反投影 `S^T y` 到高质量重建
- 每轮均使用当前动态采样矩阵 `s_dyn` 保持数据一致性
- DST 时空注意力隐式学习低秩时序结构 + 稀疏空间结构
- 所有 3D 卷积使用 `GroupNorm`，不依赖 batch 统计量，天然跨域泛化

### 5. 深度展开迭代

将 8 轮 CS 迭代优化展开为端到端网络，每轮交替执行梯度修正与注意力增强：

```
for i = 1..8:
    x = GDB3D[i](x, y, S, R)   # 梯度修正 + 去噪
    x = DSTLayer[i](x)          # 时间 → 空间 → 长程时序 → FFN
```

### 6. 不确定性量化 (β-NLL Heteroscedastic Uncertainty)

遵循 Kendall & Gal (NeurIPS 2017) 的异方差不确定性框架，并采用 Seitzer et al. (ICLR 2022) 的 β-NLL 改进：

- **双头输出**: `proj_out` 预测均值 μ，`proj_var` 预测对数方差 log σ²，输出端 clamp 到 [-10, 10] 防止数值溢出
- **β-NLL 损失**: `L = 0.5 × (log σ² + (μ − y)² / σ²) × stop_grad(σ^(2β))`，β=0.5 为推荐值
- **方差坍缩抑制**: β-NLL 通过 stop_grad 权重项防止方差趋近于 0，比标准 NLL warm-up 更原则性
- **NLL warm-up**: 前 10 epochs 用纯 MSE 稳定 mean head，之后开启 β-NLL 联合训练
- **正交正则化**: 训练损失附加 `0.01 × ortho_loss`，鼓励基矩阵学习不同采样模式
- 评估时仅使用 mean head，零额外推理开销；`--save_uncertainty` 可生成 σ 热力图

## 数据流

```
触觉图像序列 [B, T, 1, H, W]
        │
        ▼
  ┌──────────────┐
  │AdaptiveSModule│  ← 多基动态 CS 采样 + 重要性加权 → (y, s_dyn)
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │   RModule    │  ← 初始反投影重建 (per-sample s_dyn)
  └──────┬───────┘
         │
  ┌──────▼────────────┐
  │ ConvTokenizer3D   │  ← 浅层 3D 卷积 + GroupNorm + 边缘增强
  └──────┬────────────┘
         │
  ┌──────▼──────────────────────────────────────────┐
  │ for i = 1..8:                                    │
  │    GDB3D[i] + DSTLayer[i] (时间/空间/长程/FFN)    │
  └──────┬──────────────────────────────────────────┘
         │
  ┌──────▼──────┐  ┌──────▼──────┐
  │  proj_out   │  │  proj_var   │  ← 均值 μ / 对数方差 log σ²
  └─────────────┘  └─────────────┘
```

## 关键设计决策

| 设计 | 动机 |
|------|------|
| 3D 体积（8 帧堆叠） | 利用帧间时间冗余提升重建质量 |
| 多基矩阵动态组合 (K=4) | 不同传感器模式自动选择不同基矩阵，实现跨域泛化 |
| 正交正则化 (ortho_loss) | 鼓励基矩阵学习互不重叠的采样模式 |
| GroupNorm 替代 BatchNorm3d | 消除跨传感器域的 running stats 依赖，天然支持跨域泛化 |
| CS 纯线性采样（无激活） | 保证数据一致性误差的严密物理可解释性 |
| 低秩稀疏分解深度展开 | 触觉背景低秩 + 接触区域稀疏，GDB3D + DST 隐式建模 |
| 深度展开 8 轮 | 平衡重建精度与计算开销 |
| 可变形空间注意力 | 触觉接触区域形状不规则，固定 grid 无法自适应 |
| 时序门控 | 过滤无意义的帧间传感器噪声 |
| 长程时序注意力 (3 queries) | 快划/慢划/全局三个时间尺度捕捉全局时序演化 |
| LayerScale 残差权重 (1e-5) | 初始退化为恒等映射，保证深度展开数值稳定性 |
| β-NLL 异方差不确定性 (β=0.5) | 输出逐像素重建置信度，抑制方差坍缩 |
| 边缘增强分支 | 保留物体物理轮廓，抑制触觉传感器高频噪声 |

## 训练

```bash
python train.py                # 默认 β-NLL (β=0.5)
python train.py --beta_nll 0   # 标准 NLL (Kendall & Gal)
python train.py --beta_nll 0.5 # β-NLL (Seitzer et al., ICLR 2022 推荐)
```

默认依次训练 5 个压缩比：`[0.01, 0.04, 0.10, 0.25, 0.50]`，每个均训练 150 轮。

### 数据集

本项目使用 5 个公开触觉数据集进行训练、验证和跨域泛化评估。所有数据集均为直接下载自原始来源的公开数据，无需额外预处理脚本。

| 用途 | 数据集 | 规模 | 采集设备 | 来源 |
|------|--------|------|----------|------|
| 训练集 | ToucHD | 142 序列，每序列 ~779 帧 | 5 种 GelSight 传感器 | [HuggingFace](https://huggingface.co/datasets/xxuan01/BAAI/ToucHD-Force) |
| 验证集 | Touch and Go | 142 序列 | GelSight | 公开数据集 |
| 测试集 | TacQuad | 3 种场景 (sim/real/models) | GelSight | 公开数据集 |
| 测试集 | Yuan18 | 布匹抓取测试，含 GelSight 视频 | GelSight | 公开数据集 |
| 测试集 | VisGel | 2 序列 | GelSight | 公开数据集 |

**数据目录结构**：
```
dataset/
├── toucHD/              # 训练集
│   └── train/           # 142 个序列目录 (obj000_speed1 ~ obj141_speed2)
│       └── objXXX_speedY/  # 每序列约 779 帧 PNG 灰度图
├── touch_and_go/        # 验证集 (142 个序列目录)
├── tacquad/             # 测试集 (test/sim, real, models)
├── yuan18/              # 测试集 (test/ 含 GelSight 视频帧 + metadata)
└── visgel/              # 测试集 (images/ 含 2 个序列)
```

**ToucHD 数据集详情**：
- 论文: [ToucHD: Large-Scale Tactile Hierarchical Dynamic Dataset](https://arxiv.org/abs/2602.09617)
- 包含 722,436 触觉-力配对样本
- 5 种 GelSight 类触觉传感器，71 种不同压头
- 4 个方向 (前后左右) 的滑动运动，3D 接触力序列
- 像素级接触压力分布，8-bit 灰度 PNG

**Yuan18 数据集详情**：
- 布匹抓取与滑动检测数据集
- 含 GelSight 视频 (`GelSight_video.mp4`) + 背景帧 (`background.png`)
- 帧率 30 FPS，可用 `ffmpeg` 提取帧
- 含布匹物理属性元数据 (柔软度、厚度、拉伸性等)

**数据预处理**：
- 输入: 8-bit 灰度 PNG，`Grayscale() → Resize(128,128) → CenterCrop(96) → ToTensor()`
- 输出: `[B, T, 1, 96, 96]` float32 张量，值域 [0, 1]
- 时序构建: 滑动窗口采样，每 8 帧组成一个 3D 体积输入
- 无需额外预处理脚本，数据集直接下载后即可使用

**数据下载**：
```bash
# ToucHD (HuggingFace)
huggingface-cli download --repo-type dataset xxuan01/BAAI/ToucHD-Force --local-dir dataset/toucHD

# 其他数据集请联系作者或从原始来源获取
```

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 150 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--num_frames` | 8 | 3D 体积帧数 |
| `--patch` | 32 | CS 采样分块大小 |
| `--iter_num` | 8 | 深度展开迭代次数 |
| `--model_dim` | 64 | 特征维度 |
| `--num_heads` | 8 | DST 注意力头数 |
| `--lr` | 2e-4 | 初始学习率 |
| `--flr` | 1e-6 | 最终学习率（CosineAnnealing） |
| `--wd` | 0.05 | AdamW 权重衰减 |
| `--grad_clip` | 1.0 | 梯度裁剪范数 |
| `--nll_warmup` | 10 | NLL 预热轮数（前 N epochs 用 MSE） |
| `--beta_nll` | 0.5 | β-NLL 参数 (0=NLL, 0.5=β-NLL 推荐) |
| `--train_data` | `dataset/toucHD/train` | 训练集路径 |
| `--val_dir` | `dataset/touch_and_go` | 验证集路径 |

### 损失函数与优化

- **损失函数**: β-NLL (Seitzer et al., ICLR 2022) with heteroscedastic uncertainty，warm-up 期间退化为 MSE；附加 `0.01 × ortho_loss` 正交正则化
- **优化器**: AdamW + CosineAnnealingLR 调度 + Linear Warmup (5 epochs)
- **混合精度**: AMP (`torch.cuda.amp`) 在 GPU 环境自动启用，CPU 回退标准训练
- **评估指标**: PSNR + SSIM + LPIPS + Edge-PSNR + ROI-PSNR + ROI-SSIM

### 训练日志

训练过程自动记录以下内容：

| 记录方式 | 内容 |
|----------|------|
| `train_log.csv` | epoch, train_loss, loss_type, val_psnr, val_ssim, val_lpips, val_edge_psnr, val_roi_psnr, val_roi_ssim, lr, best_psnr, time_s |
| TensorBoard | Loss/train, LR, Metrics/val_psnr, val_ssim, val_lpips, val_edge_psnr, val_roi_psnr, val_roi_ssim |
| `best.pth` | 验证集 PSNR 最优的模型权重 |
| `trained_model/lsdunet_{ratio}.pth` | 最终模型权重（与 best.pth 相同） |

## 评估

### 标准评估

```bash
python eval.py                      # 标准评估
python eval.py --save_uncertainty   # 评估 + 自动生成不确定性热力图
python eval.py --save_uncertainty --uncertainty_ratio 0.10  # 指定 CS 比
```

对训练好的模型在测试集上评估，输出 PSNR、SSIM、LPIPS、Edge-PSNR、ROI-PSNR、Temporal-PSNR 及推理效率（FPS、FLOPs）。

支持多数据集评估：tacquad、yuan18、visgel、touch_and_go（跨域泛化测试）。

`--save_uncertainty` 选项会在评估完成后，自动对代表性序列（ToucHD、TacQuad、Yuan18 各取前几个序列）生成三列并排热力图 `[Target | Reconstruction | σ Heatmap]`，并保存 `.npy` 原始数据（preds, targets, log_vars）。

> **注意**: 评估脚本使用 `strict=False` 加载模型权重，兼容新旧 checkpoint。若加载旧 checkpoint，新增模块（自适应采样、长程时序、方差头）将被随机初始化，建议使用新训练的模型。

### 动态视频抗噪评估

```bash
# 仅 ConvTokenizer3D（默认）
python eval_noise.py

# 完整消融对比（ConvTokenizer3D vs LinearTokenizer）
python eval_noise.py --full

# 仅消融模型
python eval_noise.py --ablation

# 单压缩比
python eval_noise.py --single 0.10
```

向连续触觉测试序列注入多等级高斯白噪声（σ ∈ [0, 0.20]），评估重建稳定性：

| 噪声等级 | σ=0 | σ=0.02 | σ=0.05 | σ=0.10 | σ=0.15 | σ=0.20 |

输出指标（逐帧 clean vs noisy）：
- **PSNR_drop** = PSNR_clean − PSNR_noisy（越小 = 抗噪越强）
- **Edge_PSNR_drop**: 边缘保留对噪声的鲁棒性
- **Temporal_PSNR_drop**: 时序一致性对噪声的鲁棒性

消融对照 `LinearTokenizer`（纯 1×1×1 卷积投影，无边缘分支），证明 ConvTokenizer3D 的 Sobel-style 空间边缘 + 帧间差分算子提供了**物理抗噪先验**。

## 硬件兼容性

- **自动设备检测**: 运行时通过实际 CUDA 运算测试验证 GPU 可用性，失败自动回退 CPU
- **RTX 5080 (Blackwell, sm_120)**: 需 PyTorch nightly + CUDA 13.0+
- **CPU 训练**: 无 GPU 时自动禁用 AMP，使用标准反向传播

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd LSDUNet

# 安装依赖
pip install -r requirements.txt
```

## 依赖

- Python >= 3.12
- PyTorch >= 2.0（推荐 2.5+，RTX 5080 需 CUDA 13.0+）
- torchvision >= 0.15
- scikit-image >= 0.21
- scipy >= 1.10
- lpips >= 0.1.4
- tqdm, tensorboard, matplotlib
- 完整依赖见 `requirements.txt`

## 参考文献

| 方法 | 论文 |
|------|------|
| β-NLL 不确定性量化 | Seitzer et al. "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." ICLR, 2022. |
| 异方差不确定性 | Kendall & Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS, 2017. |
| LayerScale | Touvron et al. "Going Deeper with Image Transformers." ICCV, 2021. |
| 深度展开 | Gregor & LeCun. "Learning Fast Approximations of Sparse Coding." ICML, 2010. |
| ToucHD 数据集 | "ToucHD: Large-Scale Tactile Hierarchical Dynamic Dataset." arXiv:2602.09617. |