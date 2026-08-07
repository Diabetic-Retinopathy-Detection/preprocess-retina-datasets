#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="data/DDR-dataset"
OUTPUT_DIR="$DATASET_DIR"

7z x "$DATASET_DIR/ZIP/DDR-dataset.zip.001" -o"$OUTPUT_DIR" -y
