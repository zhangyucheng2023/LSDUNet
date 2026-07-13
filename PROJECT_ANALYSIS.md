# LSDUNet 项目技术文档

## 一、项目概述

LSDUNet（Learned Spatial-temporal Deep Unfolding Network）是一个基于压缩感知（CS）+ 深度展开的轻量级触觉视频重建网络。针对触觉传感器的低采样率压缩测量，通过 6 次可学习迭代展开恢复高保真时空触觉序列。

- 参数量：2.641M
- 输入：[B, 8, 3, 224, 224]（8帧 RGB 触觉体积）
- 输出：同尺寸重建序列 + 逐像素不确定性图
- 训练硬件：2×RTX 4090 DDP，bf16 精度

---

## 二、文件结构

```
LSDUNet/
├── model/
│   └── model_3d.py       # 核心模型（17 个类）
├── data_processor.py      # 数据集加载与预处理
├── trainer.py             # 训练/验证循环（EMA + 8项损失）
├── train.py               # 训练入口（DDP + lr分组 + warmup）
├── eval.py                # 评估入口（standard / cross-domain / noise / interpret）
├── eval_touchd.py         # LSDU-COMP 统一对比框架适配
├── metrics.py             # 评估指标（PSNR/SSIM/LPIPS/ECE/Brier）
├── utils.py               # 工具函数
├── run_2x4090.sh          # 2×4090 DDP 启动脚本
├── run_4x4090.sh          # 4×4090 DDP 启动脚本
└── requirements.txt
```

---

## 三、模型结构（model_3d.py）

模型共 17 个类，按数据流顺序分为 6 个阶段：

### 阶段 1：像素级 CS 采样

#### `AdaptiveSModule`
- 功能：对输入触觉图像进行分块压缩采样，充当 CS 测量矩阵 Φ
- 机制：
  - K=4 个可学习基矩阵，通过 `meta_net` 动态组合
  - `meta_net` 输入 3 维统计特征（均值 + 标准差 + 梯度幅值），输出 4 组基权重
  - `importance_net`（3 层 CNN + Sigmoid）做空间重要性加权
  - `ortho_loss()` 正则化鼓励基矩阵正交
- 输入：[B·T·C, 1, H, W] → 输出：[B, cs_dim, 7, 7]

#### `RModule`
- 功能：CS 反投影，将压缩测量恢复到图像域
- 机制：S^T·y → [B, 1, 224, 224] → reshape [B, 3, 8, 224, 224]

### 阶段 2：特征编码 + 融合

#### `ConvTokenizer3D`
- 功能：将反投影图像编码为特征体积，主分支 + 边缘分支
- 主分支：Conv3D stride-2 → [B, 64, 8, 112, 112]
- 边缘分支：空间边缘 + 时序边缘 → edge_proj

#### `BayesianFusion`
- 功能：用贝叶斯逆方差加权最大似然估计融合主分支与边缘分支，替代暴力 cat+conv
- 公式：`fused = (f1·σ₂² + f2·σ₁²) / (σ₁² + σ₂²)`
- 方差预测：Softplus + 1e-6 保证正定和数值稳定
- 来源：CMLF 论文启发，仅 +8.3K 参数

### 阶段 3：特征级 CS 测量

#### `AdaptiveFeatCS`
- 功能：对特征体积做第二次 CS 测量（只算一次，每次迭代复用）
- 机制：adaptive_pool(7×7) → 通道加权 → 多基空间调制 → [B, 1, 8, 7, 7]

### 阶段 4：深度展开迭代（×6）

#### `CSGradientStep`（每次迭代的核心）
执行 PGD 深度展开的一步：
1. **梯度修正**：`grad = R_feat(y_feat - S_feat(x))`，特征级 CS 误差反传
2. **Kalman 更新**：`x_new = x + K(x)·grad`，K 为输入依赖的可学习增益
3. **跨迭代门控**：`x = gate·x_new + (1-gate)·x_prev`，门控融合相邻迭代
4. **L+S 分解**：`LSDecomposition` 时空分离低秩稀疏
5. **因果去噪**：causal_conv + refine，LayerScale 残差缩放保证梯度稳定

#### `LSDecomposition`（显式 L+S，在 CSGradientStep 内）
- **时序低秩 L_t**：T→rank→T 线性瓶颈，捕获帧间相关的静态接触
- **空间低秩 L_s**：C→rank→C 通道瓶颈，捕获背景均匀区域
- **稀疏 S**：通道相关软阈值 `sign(x)·max(|x|-τ, 0)`，每通道独立 τ
- 训练时缓存 L/S 供 Schatten-0.5 正则化使用

#### `DSTLayer`（每次迭代的先验去噪，在 CSGradientStep 之后）
包含三个子模块：

##### `DSTTimeBlock`
- 功能：多尺度时间卷积
- 机制：[3,5,7] 三尺度并行 1D 卷积 + Sigmoid 门控融合
- 复杂度：O(D·T)

##### `DeformableSpatialBlock`
- 功能：可变形空间特征提取，替代空间注意力
- 机制：
  - `DeformConv2d`：学习偏移量，采样点自适应包裹不规则接触面
  - `MultiScaleSpatialConv`：多尺度膨胀卷积 d=1,3,7，替代 MambaSSM 空间扫描
- 复杂度：O(D·k²)，显存比空间注意力降低数千倍

##### `TactileHistoryCompressor`
- 功能：时序历史压缩，替代 O(T²) 时序注意力
- 机制（融合 TacMamba + CMLF 两篇论文思想）：
  1. GAP 空间池化：[B,C,T,H,W] → [B,T,C]，避免数万 batch 维度
  2. MambaSSM 时序扫描：O(T) 状态空间模型，得到 observed
  3. CausalLatentFilter（CMLF 启发）：
     - 预测：`predicted = transition(observed)` + 因果移位
     - Kalman 更新：`gain = Sigmoid(MLP(predicted, observed))`
     - `compressed = gain·observed + (1-gain)·predicted`
  4. 自适应查询摘要：输入复杂度相关的 soft query count
  5. 广播回空间维度
- 隐式 L+S：L_kalman = compressed（平滑状态→低秩），S_kalman = observed - predicted（innovation→稀疏）

#### `MambaSSM`
- 功能：选择性状态空间模型，O(T) 时序扫描
- 实现：纯 PyTorch（无需 CUDA 编译）
- `_scan_direct`：T≤32 时用直接递归扫描，避免 parallel scan 的 exp 溢出

### 阶段 5：输出

#### `UncertaintyHead`
- 功能：预测逐像素重建方差，为机器人部署提供置信度图
- 机制：Softplus 预测方差 + 1e-6 下限，NLL 损失训练
- 评估：ECE / Brier Score 校准质量

#### 输出层
- skip connection + proj_out + trilinear 上采样 → [B, 8, 3, 224, 224]

---

## 四、训练流程（train.py + trainer.py）

### DDP 训练
- 2×4090，bf16 精度
- 有效 batch = 12（单卡）× 4（grad_accum）× 2（GPU）= 96
- find_unused_parameters=True（iter_gate 第一次迭代无梯度）

### 学习率分组
| 参数组 | 相对 lr | 原因 |
|--------|---------|------|
| CS 采样矩阵 | 0.1× | 避免测量矩阵剧烈变化 |
| 特征 CS / L+S | 0.5× | 新模块适中 |
| 其他 | 1.0× | 正常收敛 |

### EMA
- decay 从 0.9（warmup）线性增长到 0.999
- flatten buffer 单次 all_reduce（1 次 NCCL 调用替代 N 次）

### 损失函数（8 项）

| 损失项 | 权重 | 说明 |
|--------|------|------|
| MSE 重建 | 1.0 | 主损失 |
| Sobel 边缘 | 0.1 | 边缘保持 |
| DWT 小波 | 0.01 | 多尺度细节（Haar 小波，替代 FFT） |
| SSIM | 0.1 | 结构相似性（avg_pool 可微实现） |
| 不确定性 NLL | 0.01 | 0.5·(log σ² + (x-μ)²/σ²) |
| 正交正则化 | 0.01 | 基矩阵多样性 |
| L+S 低秩 | 0.01 | Schatten-0.5: Σσ^0.5（显式+隐式 L） |
| L+S 稀疏 | 0.01 | ‖S‖₁（显式+隐式 S） |

- 辅助损失在 warmup 阶段（前 5 epoch）线性增加：`aux_scale = min(1.0, (epoch+1)/warm_epochs)`
- L+S 正则化通过 `get_ls_regularization()` 收集双重来源的 L/S 张量

### 训练超参数

| 参数 | 值 |
|------|-----|
| epochs | 150 |
| lr | 2e-4（余弦退火到 1e-5） |
| warmup | 5 epoch（线性） |
| wd | 0.05（AdamW） |
| grad_clip | 1.0 |
| val_interval | 5 |
| iter_num | 6 |
| model_dim | 64 |
| num_frames | 8 |
| image_size | 224 |
| ls_rank | 4 |

### 压缩比
训练 3 个关键 ratio：0.01（低）、0.10（中）、0.50（高）

---

## 五、评估流程

### eval.py（统一入口）
```
python eval.py                                    # 标准评估
python eval.py --mode cross-domain --image_size 448  # 跨域泛化 + ECE/Brier
python eval.py --mode noise [--full]              # 噪声鲁棒性 + 消融
python eval.py --mode interpret --submode export|render|all  # 机制分析
```

### eval_touchd.py（LSDU-COMP 统一对比）
- 接入 LSDU-COMP 框架，与 10 个 baseline 在完全相同条件下对比
- 统一数据集（ToucHDVolumeDataset）、统一指标、统一 CSV 输出

### 指标
- 重建质量：PSNR / SSIM / LPIPS / Edge-PSNR / ROI-PSNR / ROI-SSIM / Temporal-PSNR
- 不确定性校准：ECE / Brier Score
- 效率：Params / FLOPs / FPS / Latency

---

## 六、L+S 双重分解机制

### 显式 L+S（LSDecomposition 模块）
在每次 CSGradientStep 迭代中：
- 时序低秩 L_t：T→rank→T 瓶颈
- 空间低秩 L_s：C→rank→C 瓶颈
- 稀疏 S：通道相关软阈值

### 隐式 L+S（CausalLatentFilter，Kalman 滤波）
在 TactileHistoryCompressor 中：
- L_kalman = compressed（平滑状态，对应低秩背景）
- S_kalman = observed - predicted（innovation，对应稀疏突变）

### 正则化
两种 L/S 都被 `get_ls_regularization()` 收集，统一受 Schatten-0.5 + L1 约束。Schatten-p (p=0.5) 比核范数更精确促进低秩，池化后小矩阵 SVD 计算开销可忽略。

---

## 七、数据流

```
输入: [B, 8, 3, 224, 224]
 │
 ├─① 像素级 CS 采样 (AdaptiveSModule)
 │   224×224 → 分组卷积(S·x) → [B, cs_dim, 7, 7]
 │   meta_net(mean+std+grad) → 4 组基矩阵动态组合
 │   importance_net → 空间重要性加权
 │
 ├─② 初始反投影 (RModule)
 │   S^T·y → [B, 1, 224, 224] → reshape [B, 3, 8, 224, 224]
 │
 ├─③ Tokenization + BayesianFusion
 │   主分支: Conv3D stride-2 → [B, 64, 8, 112, 112]
 │   边缘分支: spatial + temporal edges
 │   贝叶斯融合: 逆方差加权 MLE
 │
 ├─④ 特征级 CS 测量 (AdaptiveFeatCS) — 只算一次
 │   x_tok → adaptive_pool(7×7) → [B, 1, 8, 7, 7]
 │
 ├─⑤ 深度展开循环 (×6)
 │   for i in 0..5:
 │   ├─ CSGradientStep:
 │   │   ① 梯度修正: grad = R_feat(y_feat - S_feat(x))
 │   │   ② Kalman 更新: x_new = x + K(x)·grad
 │   │   ③ 门控: x = gate·x_new + (1-gate)·x_prev
 │   │   ④ L+S 分解: LSDecomposition (时序+空间低秩, 通道软阈值)
 │   │   ⑤ 因果去噪: causal_conv + LayerScale
 │   └─ DSTLayer:
 │       ├─ DSTTimeBlock: 多尺度时间卷积 [3,5,7]
 │       ├─ DeformableSpatialBlock: DeformConv + 膨胀卷积 [d=1,3,7]
 │       └─ TactileHistoryCompressor
 │           GAP → MambaSSM → 因果预测 → Kalman 更新 → 查询摘要 → 广播
 │
 ├─⑥ 跳跃连接: x += skip_proj(x_tok)
 │
 ├─⑦ 不确定性: UncertaintyHead → [B, 8, 1, 224, 224]
 │
 └─⑧ 输出: proj_out + trilinear upsample → [B, 8, 3, 224, 224]
```

---

## 八、模块来源

| 模块 | 来源 | 说明 |
|------|------|------|
| MambaSSM | Mamba (arXiv:2312.00752) | 纯 PyTorch 实现 + _scan_direct 短序列修复 |
| TactileHistoryCompressor | TacMamba (arXiv:2603.01700) | GAP 池化 + 自适应查询 |
| CausalLatentFilter | CMLF (arXiv:2604.02108) | Sigmoid 门控 + 显式 L/S 输出 |
| BayesianFusion | CMLF (arXiv:2604.02108) | 2 层 Linear 逆方差加权 |
| DeformConv2d | DCN (arXiv:1703.06203) | torchvision 调用 |
| AdaptiveSModule | 原创 | K=4 基矩阵 + meta_net(3维) + importance_net(3层) |
| LSDecomposition | 原创（红外 STT 启发） | 时空分离 + Schatten-0.5 |
| CSGradientStep | 原创（PGD Unfolding） | Kalman gain + 门控 + L+S + 因果去噪 |
| AdaptiveFeatCS | 原创 | adaptive_pool(7×7) + 多基调制 |
| UncertaintyHead | 原创 | Softplus 方差 + NLL + ECE/Brier |
| DWT 小波损失 | 原创 | Haar 小波替代 FFT |
| SSIM 损失 | 原创 | avg_pool 可微实现 |
| DSTTimeBlock | 原创 | [3,5,7] 多尺度 + Sigmoid 门控 |
| DeformableSpatialBlock | 原创 | DeformConv + MultiScaleSpatialConv 组合 |
