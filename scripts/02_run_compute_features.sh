#!/bin/bash
# Compute SAE feature statistics on BCIC2a data
#
# Usage: bash scripts/02_run_compute_features.sh
#
# Requires a trained SAE checkpoint.

set -e

DB_PATH="${BCIC2A_DB_PATH:-/path/to/bcic2a.lmdb}"
SAE_PATH="${SAE_PATH:-checkpoints/bcic2a_sae/final_eeg_sae_brain-bzh_reve-base_L-2_resid_8192.pt}"
MODEL_NAME="${REVE_MODEL:-brain-bzh/reve-base}"
DEVICE="${DEVICE:-cuda}"

python tasks/compute_sae_feature_data.py \
    --db_path "$DB_PATH" \
    --sae_path "$SAE_PATH" \
    --model_name "$MODEL_NAME" \
    --block_layer -2 \
    --module_name resid \
    --batch_size 64 \
    --num_top_trials 20 \
    --output_dir results/feature_data \
    --device "$DEVICE" \
    "$@"
