#!/usr/bin/env bash
# Fase 6: tracker online (Hungarian sobre log-probabilidades) + evaluación con el mismo protocolo de la Fase 4.
# Uso: tracking/scripts/eval_assoc.sh [local|hpc] <run1> [run2 ...]   (runs = carpetas en $OUTPUT_ROOT con best.pth)
# Requiere FEAT_DIR. Opcionales: BIRTH_THRESH (0.5), MAX_LOST (30), MARGIN (10), CROSS_IOU (0.3),
#   BASELINE_JSONS ("bytetrack=<json> ocsort=<json>") para añadir el baseline a la tabla.
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; shift
: "${FEAT_DIR:?define FEAT_DIR}"
cd "$PROJECT_ROOT"; VAL="$DANCETRACK_DIR/val"; ARGS=(${BASELINE_JSONS:-})
for R in "$@"; do
  D="$OUTPUT_ROOT/$R"
  python -m tracking.assoc.track --ckpt "$D/best.pth" --cache "$FEAT_DIR/val" --out "$D/results" \
    --birth-thresh "${BIRTH_THRESH:-0.5}" --max-lost "${MAX_LOST:-30}"
  python -m tracking.baseline.evaluate --results "$D/results" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
    --seqs $(cd "$FEAT_DIR/val" && ls *.pt | sed 's/.pt$//') --out "$D/eval.json" > /dev/null
  ARGS+=("$R=$D/eval.json")
done
python -m tracking.baseline.make_table "${ARGS[@]}" | tee "$OUTPUT_ROOT/assoc_table.md"
