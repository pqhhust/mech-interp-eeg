#!/bin/bash
# Evaluate SAE features on BCIC2a with linear probing
#
# Usage: bash scripts/03_run_bcic2a_probe.sh
#
# Runs linear probe evaluation for a specific test subject.
# Also supports top-K feature masking and raw baseline.

set -e

DB_PATH="${BCIC2A_DB_PATH:-/path/to/bcic2a.lmdb}"
SAE_PATH="${SAE_PATH:-checkpoints/bcic2a_sae/final_eeg_sae_brain-bzh_reve-base_L-2_resid_8192.pt}"
MODEL_NAME="${REVE_MODEL:-brain-bzh/reve-base}"
DEVICE="${DEVICE:-cuda}"
TEST_SUBJECT="${TEST_SUBJECT:-9}"

python downstream/linear_probe.py \
    --db_path "$DB_PATH" \
    --sae_path "$SAE_PATH" \
    --model_name "$MODEL_NAME" \
    --block_layer -2 \
    --module_name resid \
    --test_subject "$TEST_SUBJECT" \
    --batch_size 64 \
    --pooling mean \
    --run_raw_baseline \
    --run_topk \
    --topk_list "10,50,100,500,1000" \
    --output_dir results/bcic2a \
    --device "$DEVICE" \
    "$@"
