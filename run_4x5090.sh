#!/bin/bash
# Run LSDUNet training on 4×RTX 5090 32GB (DDP, Blackwell sm_120).
# Tuned for 5090: batch=12 (peak ~26.5GB, safe headroom), grad_accum=2 → effective=96.
# BF16 autocast + TF32 matmul enabled in code (Blackwell native support).
#
# Usage:
#   cd "$(dirname "$0")"
#   bash run_4x5090.sh                       # full run (3 ratios from scratch)
#   bash run_4x5090.sh --resume             # resume 0.01 from epoch 40, train 0.10/0.50 from scratch
#   bash run_4x5090.sh --ratios 0.10,0.50   # only train missing ratios
#   bash run_4x5090.sh --debug              # debug (disable EMA, fast smoke test)
#   BATCH=10 bash run_4x5090.sh              # override batch_size if OOM (peak ~22GB)

set -e

cd "$(dirname "$0")"

# 4×5090 DDP launcher
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
# Blackwell 5090: expandable_segments reduces fragmentation under big batch+ckpt
export TORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 实测显存 (修复 use_ckpt + 双重 L+S 缓存后, single GPU):
#   B=10 → peak 22.22 GB | B=12 → peak 26.45 GB | B=14 → ~30.8 GB (危险) | B=16 → OOM
# 默认 batch=12 (与 4090 历史配置一致, effective=96 最优), OOM 可降级 BATCH=10
BATCH=${BATCH:-12}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EFFECTIVE=$((BATCH * GRAD_ACCUM * 4))

# 5090 supports bfloat16 (Blackwell, compute capability 12.0)
# 3 key ratios: 0.01 (low), 0.10 (mid), 0.50 (high) — covers full CS range

echo "============================================================"
echo " LSDUNet: Training on 4×RTX 5090 32GB (DDP, Blackwell)"
echo " Effective batch: ${BATCH}×${GRAD_ACCUM}×4 = ${EFFECTIVE} | 150 epochs | image=224 | dim=64 | iter=6"
echo " Ratios: 0.01, 0.10, 0.50 (low/mid/high)"
echo " (OOM 时: BATCH=10 bash run_4x5090.sh)"
echo "============================================================"

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    train.py \
    --ratios 0.01,0.10,0.50 \
    --epochs 150 \
    --batch_size ${BATCH} \
    --grad_accum ${GRAD_ACCUM} \
    --warm_epochs 5 \
    --lr 2e-4 \
    --flr 1e-5 \
    --wd 0.05 \
    --grad_clip 1.0 \
    --model_dim 64 \
    --iter_num 6 \
    --num_frames 8 \
    --image-size 224 \
    --val_interval 5 \
    --ls_rank 4 \
    --w_lowrank 0.01 \
    --w_sparse 0.01 \
    --w_ssim 0.1 \
    --ema_decay 0.999 \
    "$@"

echo "============================================================"
echo " LSDUNet training finished."
echo "============================================================"
