#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE_DIR="$ROOT_DIR/backbones/causal_forcing"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR:$BACKBONE_DIR${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$BACKBONE_DIR/checkpoints/chunkwise/causal_forcing.pt}"
PROMPT_PATH="${PROMPT_PATH:-$BACKBONE_DIR/prompts/example_prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_FOLDER="$OUTPUT_ROOT/causal_forcing_raw_seed${SEED}_${RUN_TAG}"
mkdir -p "$OUTPUT_FOLDER"

cd "$BACKBONE_DIR"
"$PYTHON" inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder "$OUTPUT_FOLDER" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$PROMPT_PATH" \
  --seed "$SEED" \
  --slepian_guidance_scale 0

echo "Saved Causal-Forcing raw videos to: $OUTPUT_FOLDER"
