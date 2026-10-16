#!/usr/bin/env bash
# Fase 7: fine-tuning conjunto (solo decoder del detector + módulo de asociación). Pensado para una GPU grande (A100).
# Uso: tracking/scripts/train_joint.sh [local|hpc] [argumentos extra de tracking.joint.train]
# Variables: DET_CKPT (por defecto $OUTPUT_ROOT/detector_r50vd_person/best.pth), ASSOC_CKPT (por defecto $OUTPUT_ROOT/assoc_graph/best.pth),
#            RUN_NAME (joint_graph), NUM_WORKERS (8)
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; shift || true
DET_CKPT="${DET_CKPT:-$OUTPUT_ROOT/detector_r50vd_person/best.pth}"; ASSOC_CKPT="${ASSOC_CKPT:-$OUTPUT_ROOT/assoc_graph/best.pth}"
RUN="$OUTPUT_ROOT/${RUN_NAME:-joint_graph}"; mkdir -p "$RUN"; cd "$PROJECT_ROOT"
python -m tracking.joint.train --det-ckpt "$DET_CKPT" --assoc-ckpt "$ASSOC_CKPT" --out "$RUN" --seed "$SEED" \
  --num-workers "${NUM_WORKERS:-8}" "$@" 2>&1 | tee -a "$RUN/train.log"
