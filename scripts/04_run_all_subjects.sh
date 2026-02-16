#!/bin/bash
# Run all BCIC2a subjects in leave-one-out fashion
#
# Usage: bash scripts/04_run_all_subjects.sh
#
# Evaluates each subject as test subject (1-9).

set -e

DB_PATH="${BCIC2A_DB_PATH:-/path/to/bcic2a.lmdb}"
SAE_PATH="${SAE_PATH:-checkpoints/bcic2a_sae/final_eeg_sae_brain-bzh_reve-base_L-2_resid_8192.pt}"
MODEL_NAME="${REVE_MODEL:-brain-bzh/reve-base}"
DEVICE="${DEVICE:-cuda}"

for SUBJECT in 1 2 3 4 5 6 7 8 9; do
    echo "==============================="
    echo "Testing subject $SUBJECT"
    echo "==============================="
    TEST_SUBJECT=$SUBJECT \
    BCIC2A_DB_PATH="$DB_PATH" \
    SAE_PATH="$SAE_PATH" \
    REVE_MODEL="$MODEL_NAME" \
    DEVICE="$DEVICE" \
    bash scripts/03_run_bcic2a_probe.sh
    echo ""
done

echo "All subjects done. Results in results/bcic2a/"
