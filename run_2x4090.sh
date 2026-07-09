#!/bin/bash
# Run LSDUNet training on 2×RTX 4090 (DDP).
# Aligned: 150 epochs, image_size=224, batch=12, grad_accum=4, model_dim=64, iter_num=6.
#
# Usage:
#   cd /home/tuf/LSDUNet
#   bash run_2x4090.sh                # full run (3 key ratios)
#   bash run_2x4090.sh --ratios 0.10  # single ratio
#   bash run_2x4090.sh --resume       # resume from checkpoint

set -e

cd /home/tuf/LSDUNet

# 2×4090 DDP launcher
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# 4090 supports bfloat16 (Ada Lovelace, compute capability 8.9)
# Effective batch = 12 (per GPU) × 4 (grad_accum) × 2 (GPUs) = 96
# 3 key ratios: 0.01 (low), 0.10 (mid), 0.50 (high) — covers full CS range

echo "============================================================"
echo " LSDUNet: Training on 2×RTX 4090 (DDP)"
echo " Effective batch: 12×4×2 = 96 | 150 epochs | image=224 | dim=64 | iter=6"
echo " Ratios: 0.01, 0.10, 0.50 (low/mid/high)"
echo "============================================================"

torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    train.py \
    --ratios 0.01,0.10,0.50 \
    --epochs 150 \
    --batch_size 12 \
    --grad_accum 4 \
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
