#!/bin/bash
# ── Launch EEG-SAE Interactive Demo ──
#
# Usage:
#   bash scripts/05_run_demo.sh
#
# Prerequisites:
#   1. Train SAE:    bash scripts/01_run_train.sh
#   2. Compute features: bash scripts/02_run_compute_features.sh
#
# The demo will open a Gradio web interface at http://localhost:7860

set -e

SAE_PATH="${SAE_PATH:-checkpoints/sae_latest.pt}"
FEATURE_DATA_PATH="${FEATURE_DATA_PATH:-results/feature_data/sae_feature_data.npz}"
DB_PATH="${DB_PATH:-/path/to/bcic2a.lmdb}"
DEVICE="${DEVICE:-cpu}"
PORT="${PORT:-7860}"
TEST_SUBJECT="${TEST_SUBJECT:-9}"

echo "=== EEG-SAE Interactive Demo ==="
echo "SAE:          $SAE_PATH"
echo "Feature data: $FEATURE_DATA_PATH"
echo "Dataset:      $DB_PATH"
echo "Device:       $DEVICE"
echo "Port:         $PORT"
echo ""

python -m src.demo.app \
    --sae_path "$SAE_PATH" \
    --feature_data_path "$FEATURE_DATA_PATH" \
    --db_path "$DB_PATH" \
    --device "$DEVICE" \
    --port "$PORT" \
    --test_subject "$TEST_SUBJECT"
