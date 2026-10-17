#!/usr/bin/env bash
# Una semilla completa de una variante del módulo de asociación: entrena -> tracker online sobre val -> evalúa (TrackEval, global y cruces).
# Uso: tracking/scripts/run_one_seed.sh [local|hpc] <variante> <nombre_base> <semilla> [argumentos extra de tracking.assoc.train]
#   p. ej.: tracking/scripts/run_one_seed.sh hpc hist assoc_hist_A 1 --burn-in 10
# Resultado en $OUTPUT_ROOT/<nombre_base>_s<semilla>/ (best.pth, results/, eval.json). Variables: FEAT_DIR (obligatoria),
# BIRTH_THRESH (0.6), MARGIN (10), CROSS_IOU (0.3), MAX_LOST (30), MAX_LOST_OCC / MAX_LOST_OUT (por defecto = MAX_LOST; solo hist).
set -euo pipefail
ENVN="${1:-local}"; U="${2:?variante}"; BASE="${3:?nombre base}"; SEED_N="${4:?semilla}"; shift 4
source "$(dirname "$0")/../../setup_env/load.sh" "$ENVN"
: "${FEAT_DIR:?define FEAT_DIR (carpeta con train/ y val/ de features cacheadas)}"
cd "$PROJECT_ROOT"; VAL="$DANCETRACK_DIR/val"; RUN="${BASE}_s${SEED_N}"
RUN_NAME="$RUN" bash tracking/scripts/train_assoc.sh "$ENVN" "$U" "$@" --seed "$SEED_N"    # el --seed final gana al de SEED
python -m tracking.assoc.track --ckpt "$OUTPUT_ROOT/$RUN/best.pth" --cache "$FEAT_DIR/val" --out "$OUTPUT_ROOT/$RUN/results" \
  --birth-thresh "${BIRTH_THRESH:-0.6}" --max-lost "${MAX_LOST:-30}" \
  --max-lost-occluded "${MAX_LOST_OCC:-${MAX_LOST:-30}}" --max-lost-out "${MAX_LOST_OUT:-${MAX_LOST:-30}}"
python -m tracking.baseline.evaluate --results "$OUTPUT_ROOT/$RUN/results" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
  --seqs $(cd "$FEAT_DIR/val" && ls *.pt | sed 's/.pt$//') --out "$OUTPUT_ROOT/$RUN/eval.json" > /dev/null
echo "listo: $OUTPUT_ROOT/$RUN/eval.json"
