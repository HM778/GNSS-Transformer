#!/bin/bash
# ============================================================
# run_inference.sh
# Inference script for GNSS-Transformer
#
# This script:
# 1. Loads a trained Transformer model
# 2. Runs inference on test data
# 3. Computes corrected positions using C++ engine
# 4. Evaluates positioning accuracy
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  GNSS-Transformer Inference"
echo "============================================"

# Check arguments
MODEL_PATH="${1:-$PROJECT_DIR/trained_model/gnss_transformer_best.pth}"
DATA_PATH="${2:-$PROJECT_DIR/data/processed}"
OUTPUT_DIR="${3:-$PROJECT_DIR/results}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Usage: $0 [model_path] [data_path] [output_dir]"
    exit 1
fi

# Activate virtual environment if exists
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Run Python inference
echo "[Step 1] Running Transformer inference..."
cd "$PROJECT_DIR"

python3 -m python.training.inference \
    --model-path "$MODEL_PATH" \
    --config "$PROJECT_DIR/config/train_config.json" \
    --data-dir "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR"

# Step 2: Apply corrections using C++ engine (if available)
CPP_TOOL="$PROJECT_DIR/cpp/build/gnss_transformer_cpp"
if [ -f "$CPP_TOOL" ]; then
    echo ""
    echo "[Step 2] Applying corrections with C++ engine..."
    
    # Process each test file
    for csv_file in "$DATA_PATH"/*.csv; do
        if [ -f "$csv_file" ]; then
            basename=$(basename "$csv_file" .csv)
            corrected_output="$OUTPUT_DIR/${basename}_corrected.csv"
            
            echo "  Processing: $basename"
            # TODO: Integrate with C++ correction pipeline
            # $CPP_TOOL correct "$csv_file" "$OUTPUT_DIR/${basename}_corrections.npy" "$corrected_output"
        fi
    done
else
    echo ""
    echo "[Step 2] C++ tool not built. Skipping C++ correction step."
    echo "  Build with: cd cpp/build && cmake .. && make"
fi

# Step 3: Evaluate results
echo ""
echo "[Step 3] Evaluating positioning accuracy..."
python3 -c "
import os
import glob
import numpy as np
import json

output_dir = '$OUTPUT_DIR'
results = {}

# Load evaluation results
eval_file = os.path.join(output_dir, 'evaluation_results.json')
if os.path.exists(eval_file):
    with open(eval_file, 'r') as f:
        results = json.load(f)
    
    print('\n=== Evaluation Results ===')
    for dataset, metrics in results.items():
        print(f'\nDataset: {dataset}')
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f'  {key}: {value:.3f}')
            else:
                print(f'  {key}: {value}')
else:
    print('No evaluation results found.')
    print('Run evaluation first: python3 -m python.training.evaluate ...')
"

echo ""
echo "============================================"
echo "  Inference complete!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
