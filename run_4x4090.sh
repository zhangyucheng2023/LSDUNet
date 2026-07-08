#!/bin/bash
# Run LSDUNet training on 4×RTX 4090 (DDP).
# Aligned: 150 epochs, image_size=224, batch=8, grad_accum=1, model_dim=64, iter_num=6.
#
# Usage:
#   cd /home/tuf/LSDUNet
#   bash run_4x4090.sh              # full run
#   bash run_4x4090.sh --debug      # debug (disable EMA, fast smoke test)

set -e

cd /home/tuf/LSDUNet

# 4×4090 DDP launcher
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# 4090 supports bfloat16 (Ada Lovelace, compute capability 8.9)
# Effective batch = 8 (per GPU) × 4 (GPUs) × 1 (grad_accum) = 32

echo "============================================================"
echo " LSDUNet: Training on 4×RTX 4090 (DDP)"
echo " Effective batch: 8×4×1 = 32 | 150 epochs | image=224 | dim=64 | iter=6"
echo "============================================================"

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    train.py \
    --ratios 0.01,0.04,0.10,0.25,0.50 \
    --epochs 150 \
    --batch_size 8 \
    --grad_accum 1 \
    --warm_epochs 10 \
    --lr 2e-4 \
    --flr 1e-5 \
    --wd 0.05 \
    --grad_clip 1.0 \
    --model_dim 64 \
    --iter_num 6 \
    --num_frames 8 \
    --image-size 224 \
    --val_interval 5 \
    "$@"

echo "============================================================"
echo " LSDUNet training finished."
echo "============================================================"
