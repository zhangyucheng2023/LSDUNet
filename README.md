# LSDUNet — Learned Spatial-temporal Deep Unfolding Network for Tactile Image Compressed Sensing

基于**压缩感知 + 深度展开优化**的触觉图像重建网络，使用极少量测量值重建高质量触觉压力分布图。采用混合架构：**浅层 3D 卷积 Token 化 + 深层 DST (Divided Space-Time) 可变形时空注意力**。

## 项目结构

```
LSDUNet/
├── model/
│   └── model_3d.py      # 核心模型定义 (LSDUNet)
├── data_processor.py     # 数据集加载与预处理
├── trainer.py            # 训练/验证循环
├── train.py              # 训练入口脚本
├── eval.py               # 评估脚本
└── utils.py              # 工具函数 (设备、种子、色彩空间转换)
```

## 核心思路

### 1. 压缩感知 (CS) 采样

触觉图像被分块压缩采样，通过可学习的卷积采样矩阵降低测量维度：

```
y = Φ(x)   [采样: SModule, Conv2d stride=patch]
ẋ = Φᵀ(y)  [初始反投影: RModule, ConvTranspose2d]
```

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

### 3. DST 可变形时空注意力 (DSTLayer)

每轮迭代按 **时间 → 空间 → FFN** 顺序执行，配合可学习残差权重稳定训练：

| 阶段 | 组件 | 机制 | 作用 |
|------|------|------|------|
| 时间块 | `DSTTimeBlock` | 多尺度时间卷积 (3×1×1 / 5×1×1) + Sigmoid 门控 | 捕捉**滑动轨迹**，建模接触模式的时序演化 |
| 空间块 | `DSTSpaceBlock` | 可变形注意力：学习 2D 采样偏移 + `grid_sample` + Softmax 聚合 | 重构**压力分布形状**，自适应不规则接触区域 |
| 前馈 | `FFN3D` | 3D 卷积 + 空间/时间深度可分离卷积增强 | 通道混合与非线性变换 |

与 TimeSformer 原始 DST 的对比：

| 特性 | TimeSformer | LSDUNet 改造版 |
|------|-------------|---------------|
| Token 化 | 2D Conv patchify | 3D Conv 多层渐进 + 边缘增强 |
| 时间注意力 | QKV Self-Attention | 多尺度时间卷积 + 门控 |
| 空间注意力 | QKV Self-Attention | 可变形注意力 (grid_sample) |
| 使用位置 | ViT Block 内部 | 与 GDB 交替的独立模块 |

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
    x = DSTLayer[i](x)          # 时序注意力 → 空间注意力 → FFN
```

## 数据流

```
触觉图像序列 [B, T, 1, H, W]
        │
        ▼
  ┌──────────┐
  │ SModule  │  ← 可学习 CS 采样
  └────┬─────┘
       │
  ┌────▼─────┐
  │ RModule  │  ← 初始反投影重建
  └────┬─────┘
       │
  ┌────▼──────────┐
  │ConvTokenizer3D│  ← 浅层 3D 卷积提取边缘/纹理 Token
  └────┬──────────┘
       │
  ┌────▼──────────────────────┐
  │ for i = 1..8:             │
  │    GDB3D[i]  + DSTLayer[i] │
  └────┬──────────────────────┘
       │
  ┌────▼──────┐
  │ proj_out  │  ← 1×1 Conv → [B, T, 1, H, W]
  └───────────┘
```

## 关键设计决策

| 设计 | 动机 |
|------|------|
| 3D 体积（多帧堆叠） | 利用帧间时间冗余提升重建质量 |
| CS 分块采样 (patch=32) | 降低参数量，压缩比可控 |
| CS 纯线性采样（无激活） | 保证数据一致性误差的严密物理可解释性 |
| 深度展开 8 轮 | 平衡重建精度与计算开销 |
| 可变形空间注意力 | 触觉接触区域形状不规则，固定 grid 无法自适应 |
| 时序门控 | 过滤无意义的帧间传感器噪声 |
| 可学习残差权重 (0.05) | 早期退化为恒等映射，保证深度展开数值稳定性 |
| 边缘增强分支 | 保留物体物理轮廓，抑制触觉传感器高频噪声 |

## 训练

```bash
python train.py
```

默认依次训练 5 个压缩比：`[0.01, 0.04, 0.10, 0.25, 0.50]`，每个均训练 80 轮。

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
| `--epochs` | 80 | 训练轮数 |
| `--batch_size` | 8 | 批次大小（RTX 5080 上限） |
| `--num_frames` | 4 | 3D 体积帧数 |
| `--patch` | 32 | CS 采样分块大小 |
| `--iter_num` | 8 | 深度展开迭代次数 |
| `--model_dim` | 16 | 特征维度 |
| `--lr` | 1e-4 | 初始学习率 |
| `--flr` | 1e-6 | 最终学习率（CosineAnnealing） |
| `--train_data` | `dataset/toucHD/train` | 训练集路径 |
| `--val_dir` | `dataset/touch_and_go` | 验证集路径 |

### 损失函数与优化

- **损失函数**: MSE (均方误差)
- **优化器**: Adam + CosineAnnealingLR 调度
- **混合精度**: AMP (`torch.cuda.amp`) 在 GPU 环境自动启用，CPU 回退标准训练
- **评估指标**: PSNR + SSIM + LPIPS + Edge-PSNR + ROI-PSNR

## 评估

```bash
python eval.py
```

对训练好的模型在测试集上评估，输出 PSNR、SSIM、LPIPS、Edge-PSNR、ROI-PSNR 及推理效率（FPS、FLOPs）。

支持多数据集评估：tacquad、yuan18、visgel（跨域泛化测试）以及 touch_and_go 等。

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
