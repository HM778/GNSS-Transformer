#!/bin/bash
# ============================================================
# prepare_data.sh
# Data preparation script for GNSS-Transformer
#
# This script:
# 1. Parses RINEX/UBX data using C++ tool
# 2. Converts to CSV format for Python training
# 3. Organizes data into train/val/test splits
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CPP_BUILD_DIR="$PROJECT_DIR/cpp/build"
DATA_DIR="$PROJECT_DIR/data"
RAW_DATA_DIR="$DATA_DIR/raw"
PROCESSED_DATA_DIR="$DATA_DIR/processed"

echo "============================================"
echo "  GNSS-Transformer Data Preparation"
echo "============================================"

# Create directories
mkdir -p "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR"

# Build C++ tools if needed
if [ ! -f "$CPP_BUILD_DIR/gnss_transformer_cpp" ]; then
    echo "[Step 1] Building C++ tools..."
    mkdir -p "$CPP_BUILD_DIR"
    cd "$CPP_BUILD_DIR"
    cmake ..
    make -j$(nproc)
    cd "$PROJECT_DIR"
fi

CPP_TOOL="$CPP_BUILD_DIR/gnss_transformer_cpp"

# Process RINEX files
process_rinex() {
    local dataset=$1
    local obs_file="$RAW_DATA_DIR/$dataset/$dataset.obs"
    local nav_file="$RAW_DATA_DIR/$dataset/$dataset.nav"
    local gt_file="$RAW_DATA_DIR/$dataset/$dataset.csv"
    local output_file="$PROCESSED_DATA_DIR/$dataset.csv"

    if [ -f "$obs_file" ]; then
        echo "[Step 2] Processing $dataset..."
        if [ -f "$nav_file" ] && [ -f "$gt_file" ]; then
            $CPP_TOOL process "$obs_file" "$nav_file" "$gt_file" "$output_file"
        else
            $CPP_TOOL parse "$obs_file" "" "$output_file"
        fi
        echo "  -> Output: $output_file"
    else
        echo "[Step 2] Skipping $dataset (no RINEX files found)"
    fi
}

# Process UBX files
process_ubx() {
    local dataset=$1
    local ubx_file="$RAW_DATA_DIR/$dataset/$dataset.ubx"
    local output_file="$PROCESSED_DATA_DIR/$dataset.csv"

    if [ -f "$ubx_file" ]; then
        echo "[Step 2] Processing UBX data for $dataset..."
        $CPP_TOOL parse-ubx "$ubx_file" "$output_file"
        echo "  -> Output: $output_file"
    fi
}

# Process all datasets
for dataset_dir in "$RAW_DATA_DIR"/*/; do
    dataset=$(basename "$dataset_dir")
    echo ""
    echo "--- Dataset: $dataset ---"
    process_rinex "$dataset"
    process_ubx "$dataset"
done

# Generate Python-compatible data splits
echo ""
echo "[Step 3] Generating data splits..."
python3 -c "
import os
import glob
import json

data_dir = '$PROCESSED_DATA_DIR'
splits = {
    'train': [],
    'val': [],
    'test': []
}

csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
print(f'Found {len(csv_files)} processed CSV files:')
for f in csv_files:
    print(f'  - {os.path.basename(f)}')

# Simple split: last dataset for test, second-to-last for val, rest for train
if len(csv_files) >= 3:
    splits['test'] = [csv_files[-1]]
    splits['val'] = [csv_files[-2]]
    splits['train'] = csv_files[:-2]
elif len(csv_files) == 2:
    splits['val'] = [csv_files[-1]]
    splits['train'] = csv_files[:-1]
elif len(csv_files) == 1:
    splits['train'] = csv_files

# Save splits config
splits_path = os.path.join(data_dir, 'splits.json')
with open(splits_path, 'w') as f:
    json.dump(splits, f, indent=2)
print(f'\nData splits saved to: {splits_path}')
print(f'  Train: {len(splits[\"train\"])} files')
print(f'  Val:   {len(splits[\"val\"])} files')
print(f'  Test:  {len(splits[\"test\"])} files')
"

echo ""
echo "============================================"
echo "  Data preparation complete!"
echo "============================================"
