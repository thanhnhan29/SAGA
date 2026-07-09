#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE_DIR="$ROOT_DIR/backbones/causvid"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR:$BACKBONE_DIR${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-$BACKBONE_DIR/wan_models}"
PROMPT_PATH="${PROMPT_PATH:-$BACKBONE_DIR/sample_dataset/MovieGenVideoBench.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

OUTPUT_FOLDER="$OUTPUT_ROOT/causvid_raw_seed${SEED}_${RUN_TAG}"
mkdir -p "$OUTPUT_FOLDER"

cd "$BACKBONE_DIR"
"$PYTHON" minimal_inference/autoregressive_inference.py \
  --config_path configs/wan_causal_dmd_warp_4step_cfg2.yaml \
  --checkpoint_folder "$CHECKPOINT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --prompt_file_path "$PROMPT_PATH" \
  --seed "$SEED" \
  --num_samples "$NUM_SAMPLES" \
  --slepian_guidance_scale 0

echo "Saved CausVid raw videos to: $OUTPUT_FOLDER"
