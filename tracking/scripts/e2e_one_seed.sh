#!/usr/bin/env bash
# Módulo e2e, una semilla completa: entrena -> inferencia frame a frame sobre val -> TrackEval (global y cruces), como run_one_seed.sh.
# Uso: tracking/scripts/e2e_one_seed.sh [local|hpc] <nombre_base> <semilla> [-u clave=valor ...]
#   p. ej.: tracking/scripts/e2e_one_seed.sh hpc e2e_h1 1
#           tracking/scripts/e2e_one_seed.sh hpc e2e_h1_a02 1 -u appearance.alpha_app=0.2
# Variables: FEAT_DIR (obligatoria, con train/ y val/ de cache_features), BIRTH_SCORE (0.6: score mínimo para nacer, como el umbral de las
# variantes assoc), BIRTH_PROB (0.5: probabilidad mínima de la cabeza de nacimiento), MARGIN (10), CROSS_IOU (0.3).
# Resultado en $OUTPUT_ROOT/<base>_s<semilla>/ (best.pth, results/, eval.json); aggregate_seeds.sh <base> da la media ± desviación.
set -euo pipefail
ENVN="${1:-local}"; BASE="${2:?nombre base}"; SEED_N="${3:?semilla}"; shift 3
if [ "${1:-}" = "-u" ]; then shift; OV=("$@"); else OV=(); fi
source "$(dirname "$0")/../../setup_env/load.sh" "$ENVN"
: "${FEAT_DIR:?define FEAT_DIR (carpeta con train/ y val/ de features cacheadas)}"
cd "$PROJECT_ROOT"; VAL="$DANCETRACK_DIR/val"; RUN="${BASE}_s${SEED_N}"; OUT="$OUTPUT_ROOT/$RUN"
python -m tracking.e2e.train --train-cache "$FEAT_DIR/train" --val-cache "$FEAT_DIR/val" --out "$OUT" --seed "$SEED_N" \
  ${OV[@]+-u "${OV[@]}"} 2>&1 | tee -a "$OUTPUT_ROOT/$RUN.log"
python -m tracking.e2e.infer --ckpt "$OUT/best.pth" --cache "$FEAT_DIR/val" --out "$OUT/results" \
  -u infer.min_score="${BIRTH_SCORE:-0.6}" infer.birth_thr="${BIRTH_PROB:-0.5}"
python -m tracking.baseline.evaluate --results "$OUT/results" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
  --seqs $(cd "$FEAT_DIR/val" && ls *.pt | sed 's/.pt$//') --out "$OUT/eval.json" > /dev/null
echo "listo: $OUT/eval.json"
