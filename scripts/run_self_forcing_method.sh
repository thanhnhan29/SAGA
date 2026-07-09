#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE_DIR="$ROOT_DIR/backbones/self_forcing"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR:$BACKBONE_DIR${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$BACKBONE_DIR/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-$BACKBONE_DIR/prompts/MovieGenVideoBench_extended.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

IDX="${IDX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

SLEPIAN_GUIDANCE_SCALE="${SLEPIAN_GUIDANCE_SCALE:-3}"
SLEPIAN_NW="${SLEPIAN_NW:-1.5}"
SLEPIAN_K_CUTOFF="${SLEPIAN_K_CUTOFF:-3}"
SLEPIAN_WINDOW="${SLEPIAN_WINDOW:-9}"
SLEPIAN_RECURRENCE_STEPS="${SLEPIAN_RECURRENCE_STEPS:-2}"
SLEPIAN_LOSS_TYPE="${SLEPIAN_LOSS_TYPE:-1}"

OUTPUT_FOLDER="$OUTPUT_ROOT/self_forcing_method_l${SLEPIAN_LOSS_TYPE}_g${SLEPIAN_GUIDANCE_SCALE}_idx${IDX}_n${MAX_SAMPLES}_seed${SEED}_${RUN_TAG}"
mkdir -p "$OUTPUT_FOLDER"

cd "$BACKBONE_DIR"
"$PYTHON" inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --output_folder "$OUTPUT_FOLDER" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$PROMPT_PATH" \
  --idx "$IDX" \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --save_with_index \
  --noise_type dual_ar \
  --slepian_guidance_scale "$SLEPIAN_GUIDANCE_SCALE" \
  --slepian_nw "$SLEPIAN_NW" \
  --slepian_k_cutoff "$SLEPIAN_K_CUTOFF" \
  --slepian_window "$SLEPIAN_WINDOW" \
  --slepian_recurrence_steps "$SLEPIAN_RECURRENCE_STEPS" \
  --slepian_loss_type "$SLEPIAN_LOSS_TYPE" \
  --use_ema

echo "Saved Self-Forcing SAGA videos to: $OUTPUT_FOLDER"
