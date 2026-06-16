# LSDUNet — Learned Spatial-temporal Deep Unfolding Network

基于压缩感知 + 低秩稀疏分解深度展开的触觉图像重建网络。混合架构：浅层 3D 卷积 Token 化 + DST 可变形时空注意力 + 长程时序注意力 + β-NLL 异方差不确定性量化。

## 项目结构

```
LSDUNet/
├── model/
│   ├── __init__.py
│   └── model_3d.py          # LSDUNet, AdaptiveSModule, ConvTokenizer3D, DSTLayer
├── data_processor.py         # 数据集加载与预处理
├── trainer.py                # 训练/验证循环
├── train.py                  # 训练入口
├── eval.py                   # 评估脚本 (含不确定性可视化)
├── eval_noise.py             # 动态视频抗噪评估
├── metrics.py                # 评估指标
├── utils.py                  # 工具函数
├── requirements.txt
└── LICENSE
```

## 核心思路

### 1. 自适应 CS 采样 — 多基矩阵动态组合

触觉图像分块压缩采样，K=4 个基矩阵由 meta-network 动态组合，正交正则化鼓励多样性，重要性预测器实现 variable-rate CS。

### 2. 浅层 3D 卷积 Token 化 (ConvTokenizer3D)

三层渐进式 3D 卷积 + 边缘增强分支 (Sobel + 帧间差分)，GroupNorm 替代 BatchNorm3d 实现跨传感器域泛化。

### 3. DST 时空注意力 + 长程时序 (DSTLayer)

时间 → 空间 → 长程时序 → FFN，LayerScale 残差权重 (1e-5) 稳定训练。

| 组件 | 机制 |
|------|------|
| DSTTimeBlock | 多尺度时间卷积 + Sigmoid 门控 |
| DSTSpaceBlock | 可变形注意力 + grid_sample |
| LongRangeTemporalAttention | 3 个可学习查询跨注意力到全部 T 帧 |
| FFN3D | 3D 卷积 + 时空深度可分离卷积增强 |

### 4. 梯度去噪块 (GDB3D) — 深度展开

8 轮迭代交替执行梯度修正 (GRAD3D) 与多尺度去噪 (DENO3D)，DST 隐式学习低秩时序结构 + 稀疏空间结构。

### 5. 不确定性量化 (β-NLL)

双头输出 (均值 μ + 对数方差 log σ²)，β-NLL 损失抑制方差坍缩。NLL warm-up (前 10 epoch MSE) 稳定训练。

## 训练

```bash
python train.py                   # 默认配置
python train.py --beta_nll 0.5    # β-NLL (推荐)
```

依次训练 5 个压缩比：`[0.01, 0.04, 0.10, 0.25, 0.50]`，每轮 150 epoch。

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 150 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--num_frames` | 8 | 3D 体积帧数 |
| `--patch` | 32 | CS 采样分块 |
| `--iter_num` | 8 | 深度展开迭代次数 |
| `--model_dim` | 64 | 特征维度 |
| `--num_heads` | 8 | DST 注意力头数 |
| `--lr` | 2e-4 | 初始学习率 |
| `--wd` | 0.05 | AdamW 权重衰减 |
| `--grad_clip` | 1.0 | 梯度裁剪 |
| `--nll_warmup` | 10 | NLL 预热轮数 |
| `--beta_nll` | 0.5 | β-NLL 参数 |

## 评估

```bash
python eval.py                      # 标准评估
python eval.py --save_uncertainty   # 含不确定性热力图
python eval_noise.py                # 动态视频抗噪
python eval_noise.py --full         # 完整消融对比
```

## 安装

```bash
pip install -r requirements.txt
```

## 依赖

- Python >= 3.12
- PyTorch >= 2.0
- torchvision, scikit-image, scipy, lpips, tqdm, tensorboard, matplotlib