# LSDUNet — Learned Spatial-temporal Deep Unfolding Network for Tactile Image Compressed Sensing

基于**压缩感知 + 深度展开优化**的触觉图像重建网络，使用极少量测量值重建高质量触觉压力分布图。采用混合架构：**浅层 3D 卷积 Token 化 + 深层 DST (Divided Space-Time) 可变形时空注意力 + 长程时序注意力 + 不确定性量化**。

## 项目结构

```
LSDUNet/
├── model/
│   └── model_3d.py      # 核心模型定义 (LSDUNet, AdaptiveSModule, DSTLayer, LongRangeTemporalAttention)
├── data_processor.py     # 数据集加载与预处理
├── trainer.py            # 训练/验证循环
├── train.py              # 训练入口脚本
├── eval.py               # 评估脚本
├── eval_noise.py         # 动态视频抗噪评估
├── metrics.py            # 评估指标 (ROI-PSNR, Edge-PSNR, 效率指标)
└── utils.py              # 工具函数 (设备、种子、色彩空间转换)
```

## 核心思路

### 1. 自适应压缩感知 (CS) 采样

触觉图像被分块压缩采样，通过**内容感知的自适应采样矩阵**降低测量维度：

```
y = S(x) · imp(x)   [自适应采样: AdaptiveSModule, 重要性加权]
ẋ = R(y)             [初始反投影: RModule, ConvTranspose2d]
```

- **自适应采样矩阵 (AdaptiveSModule)**: 轻量 CNN 预测逐 patch 信号能量，高能量区域（接触面）获得更多有效测量，等效于 variable-rate CS
- 输入尺寸经 `patch=32` 分块，压缩比由 `sensing_rate` 控制
- 采样矩阵作为网络参数端到端学习，无需手工设计测量矩阵

### 2. 浅层 3D 卷积 Token 化 (ConvTokenizer3D)

替代传统 ViT 的线性 Patch 划分，用**三层渐进式 3D 卷积**替代简单投影，起到"卷积投影作为 Token 接入"的作用：

| 阶段 | 卷积核 | 作用 |
|------|--------|------|
| `stem1` | `(3, 5, 5)` | 大空间感受野提取轮廓 |
| `stem2` | `(3, 3, 3)` | 时空联合特征细化 |
| `stem3` | `(1, 3, 3)` | 逐帧空间特征压缩 |

同时配备**边缘增强分支**：Sobel-style 空间边缘卷积 + 帧间差分卷积，专门捕捉压力梯度和物体轮廓，抑制高频传感器噪声。

### 3. DST 可变形时空注意力 + 长程时序 (DSTLayer)

每轮迭代按 **时间 → 空间 → 长程时序 → FFN** 顺序执行，配合 LayerScale 逐通道残差权重 (Touvron et al., ICCV 2021) 稳定训练：

| 阶段 | 组件 | 机制 | 作用 |
|------|------|------|------|
| 时间块 | `DSTTimeBlock` | 多尺度时间卷积 (3×1×1 / 5×1×1) + Sigmoid 门控 | 捕捉**滑动轨迹**，建模接触模式的时序演化 |
| 空间块 | `DSTSpaceBlock` | 可变形注意力：学习 2D 采样偏移 + `grid_sample` + Softmax 聚合 | 重构**压力分布形状**，自适应不规则接触区域 |
| 长程时序 | `LongRangeTemporalAttention` | 3 个可学习时序查询 (快划/慢划/全局) 跨注意力到全部 T 帧 | 捕捉超出局部窗口的**全局时序依赖** |
| 前馈 | `FFN3D` | 3D 卷积 + 空间/时间深度可分离卷积增强 | 通道混合与非线性变换 |

**LayerScale 残差权重**: 所有残差分支权重初始化为 `1e-5`，初始时网络退化为恒等映射，训练中逐步学习各模块贡献，训练结束后可分析各模块的最终权重。

### 4. 梯度去噪块 (GDB3D)

深度展开优化框架，将迭代优化过程展开为神经网络层：

| 子模块 | 功能 |
|--------|------|
| `GRAD3D` | 计算测量残差 `y − S(x)`，反投影得到梯度修正量 |
| `DENO3D` | U-Net 风格 3 级下采样 → 瓶颈混合 (5×5 Conv) → 3 级上采样去噪 |

### 5. 深度展开迭代

将 8 轮 CS 迭代优化展开为端到端网络，每轮交替执行梯度修正与注意力增强：

```
for i = 1..8:
    x = GDB3D[i](x, y, S, R)   # 梯度修正 + 去噪
    x = DSTLayer[i](x)          # 时间 → 空间 → 长程时序 → FFN
```

### 6. 不确定性量化 (Heteroscedastic NLL)

遵循 Kendall & Gal (2017) 的异方差不确定性框架：

- **双头输出**: `proj_out` 预测均值 μ，`proj_var` 预测对数方差 log σ²
- **NLL 损失**: `L = 0.5 × (log σ² + (μ − y)² / σ²)`
- **NLL warm-up**: 前 10 epochs 用纯 MSE 稳定 mean head，之后开启 NLL 联合训练
- **log_var clamp**: 限制在 [-10, 10] 防止数值溢出
- 评估时仅使用 mean head，零额外推理开销

## 数据流

```
触觉图像序列 [B, T, 1, H, W]
        │
        ▼
  ┌──────────────┐
  │AdaptiveSModule│  ← 自适应 CS 采样 + 重要性加权
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │   RModule    │  ← 初始反投影重建
  └──────┬───────┘
         │
  ┌──────▼────────────┐
  │ ConvTokenizer3D   │  ← 浅层 3D 卷积提取边缘/纹理 Token
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
| 3D 体积（多帧堆叠） | 利用帧间时间冗余提升重建质量 |
| 自适应 CS 采样 | 高能量区域（接触面）获得更多有效测量，测量预算动态分配 |
| CS 纯线性采样（无激活） | 保证数据一致性误差的严密物理可解释性 |
| 深度展开 8 轮 | 平衡重建精度与计算开销 |
| 可变形空间注意力 | 触觉接触区域形状不规则，固定 grid 无法自适应 |
| 时序门控 | 过滤无意义的帧间传感器噪声 |
| 长程时序注意力 (3 queries) | 快划/慢划/全局三个时间尺度捕捉全局时序演化 |
| LayerScale 残差权重 (1e-5) | 初始退化为恒等映射，保证深度展开数值稳定性 |
| 异方差 NLL 不确定性 | 输出逐像素重建置信度，低置信度区域指示测量不足 |
| 边缘增强分支 | 保留物体物理轮廓，抑制触觉传感器高频噪声 |

## 训练

```bash
python train.py
```

默认依次训练 5 个压缩比：`[0.01, 0.04, 0.10, 0.25, 0.50]`，每个均训练 150 轮。

### 数据集

| 用途 | 数据集 | 说明 |
|------|--------|------|
| 训练集 | `dataset/toucHD/train` | ToucHD GelSight，142 条时序序列 |
| 验证集 | `dataset/touch_and_go` | Touch and Go，140 条时序序列 |
| 测试集 | tacquad / yuan18 / visgel | 跨域泛化评估 |

输入图片为 8-bit 灰度 PNG，经 `Grayscale() → Resize(128,128) → CenterCrop(96) → ToTensor()` 转为 `[B, T, 1, H, W]` float32 张量（值域 [0, 1]）。

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
| `--train_data` | `dataset/toucHD/train` | 训练集路径 |
| `--val_dir` | `dataset/touch_and_go` | 验证集路径 |

### 损失函数与优化

- **损失函数**: NLL (Negative Log-Likelihood) with heteroscedastic uncertainty，warm-up 期间退化为 MSE
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
python eval.py
```

对训练好的模型在测试集上评估，输出 PSNR、SSIM、LPIPS、Edge-PSNR、ROI-PSNR 及推理效率（FPS、FLOPs）。

支持多数据集评估：tacquad、yuan18、visgel（跨域泛化测试）以及 touch_and_go 等。

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

## 依赖

- PyTorch >= 2.5（RTX 5080 需 nightly + CUDA 13.0+）
- torchvision
- scikit-image
- tqdm
- tensorboard