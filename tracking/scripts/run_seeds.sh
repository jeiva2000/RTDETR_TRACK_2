#!/usr/bin/env bash
# Entrena y evalúa una variante con varias semillas y deja una tabla con media ± desviación.
# Uso: tracking/scripts/run_seeds.sh [local|hpc] <variante> <nombre_base> [argumentos extra de tracking.assoc.train]
#   p. ej.: tracking/scripts/run_seeds.sh local hist assoc_hist_A --burn-in 10
#           tracking/scripts/run_seeds.sh local hist assoc_hist_B --burn-in 10 --hist-aff max
# Variables: SEEDS ("1 2 3"), FEAT_DIR (obligatoria), BIRTH_THRESH (0.6), MARGIN (10), CROSS_IOU (0.3)
# Cada semilla guarda en $OUTPUT_ROOT/<nombre_base>_s<semilla>/ (best.pth, results/, eval.json). La tabla va a $OUTPUT_ROOT/<nombre_base>_seeds.md
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; U="${2:?variante}"; BASE="${3:?nombre base}"; shift 3
: "${FEAT_DIR:?define FEAT_DIR}"
cd "$PROJECT_ROOT"; VAL="$DANCETRACK_DIR/val"; JSONS=()
for S in ${SEEDS:-1 2 3}; do
  RUN="${BASE}_s${S}"
  RUN_NAME="$RUN" bash tracking/scripts/train_assoc.sh "${1:-local}" "$U" --seed "$S" "$@"   # el --seed final de "$@" no se usa: gana el de cada vuelta
  python -m tracking.assoc.track --ckpt "$OUTPUT_ROOT/$RUN/best.pth" --cache "$FEAT_DIR/val" --out "$OUTPUT_ROOT/$RUN/results" --birth-thresh "${BIRTH_THRESH:-0.6}"
  python -m tracking.baseline.evaluate --results "$OUTPUT_ROOT/$RUN/results" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
    --seqs $(cd "$FEAT_DIR/val" && ls *.pt | sed 's/.pt$//') --out "$OUTPUT_ROOT/$RUN/eval.json" > /dev/null
  JSONS+=("$OUTPUT_ROOT/$RUN/eval.json")
done
python -m tracking.baseline.make_table "$BASE=$(IFS=,; echo "${JSONS[*]}")" | tee "$OUTPUT_ROOT/${BASE}_seeds.md"
