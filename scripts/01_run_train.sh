#!/bin/bash
# Train SAE on REVE activations using BCIC2a EEG data
#
# Usage: bash scripts/01_run_train.sh
#
# Adjust --db_path to your BCIC2a LMDB dataset location.
# The SAE is trained on activations from the second-to-last transformer block.

set -e

DB_PATH="${BCIC2A_DB_PATH:-/path/to/bcic2a.lmdb}"
MODEL_NAME="${REVE_MODEL:-brain-bzh/reve-base}"
DEVICE="${DEVICE:-cuda}"

python tasks/train_sae_eeg.py \
    --db_path "$DB_PATH" \
    --model_name "$MODEL_NAME" \
    --block_layer -2 \
    --module_name resid \
    --d_in 512 \
    --expansion_factor 16 \
    --batch_size 32 \
    --total_training_tokens 500000 \
    --l1_coefficient 1e-4 \
    --lr 3e-4 \
    --lr_scheduler_name constantwithwarmup \
    --lr_warm_up_steps 500 \
    --use_ghost_grads \
    --feature_sampling_window 500 \
    --dead_feature_window 250 \
    --seed 42 \
    --n_checkpoints 2 \
    --checkpoint_path checkpoints/bcic2a_sae \
    --device "$DEVICE" \
    --scale_div 100.0 \
    "$@"
