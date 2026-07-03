#!/bin/bash
# ============================================================
# run_training.sh
# Training script for GNSS-Transformer
#
# This script:
# 1. Prepares data (if needed)
# 2. Runs Python training pipeline
# 3. Evaluates the trained model
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  GNSS-Transformer Training Pipeline"
echo "============================================"

# Activate virtual environment if exists
if [ -d "$PROJECT_DIR/venv" ]; then
    echo "[Step 0] Activating virtual environment..."
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Install dependencies
echo "[Step 1] Installing Python dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Prepare data (optional, skip if data already processed)
if [ "$1" != "--skip-prepare" ]; then
    echo ""
    echo "[Step 2] Preparing data..."
    bash "$SCRIPT_DIR/prepare_data.sh"
fi

# Run training
echo ""
echo "[Step 3] Starting training..."
cd "$PROJECT_DIR"

python3 -m python.training.train \
    --config "$PROJECT_DIR/config/train_config.json" \
    --data-dir "$PROJECT_DIR/data/processed" \
    --save-dir "$PROJECT_DIR/trained_model" \
    --log-dir "$PROJECT_DIR/logs" \
    --experiment "gnss_transformer_v1"

# Run evaluation
echo ""
echo "[Step 4] Evaluating model..."
python3 -m python.training.evaluate \
    --model-path "$PROJECT_DIR/trained_model/gnss_transformer_best.pth" \
    --config "$PROJECT_DIR/config/train_config.json" \
    --data-dir "$PROJECT_DIR/data/processed" \
    --output-dir "$PROJECT_DIR/results"

echo ""
echo "============================================"
echo "  Training pipeline complete!"
echo "============================================"
echo "Results saved to: $PROJECT_DIR/results/"
echo "Model saved to:   $PROJECT_DIR/trained_model/"
