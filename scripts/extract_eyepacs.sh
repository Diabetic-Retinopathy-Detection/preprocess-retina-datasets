#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="data/EyePACS-Kaggle/diabetic-retinopathy-detection"
OUTPUT_DIR="$DATASET_DIR/extracted"

7z x "$DATASET_DIR/train.zip.001" -o"$OUTPUT_DIR" -y
7z x "$DATASET_DIR/test.zip.001" -o"$OUTPUT_DIR" -y
