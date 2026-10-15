#!/usr/bin/env bash
# Fase 6: entrena una variante del módulo de asociación sobre las features cacheadas (Fase 5).
# Uso: tracking/scripts/train_assoc.sh [local|hpc] <none|ema|gru|graph> [argumentos extra de tracking.assoc.train]
# Requiere FEAT_DIR (p. ej. $OUTPUT_ROOT/features/<detector>) con train/ y val/ generados por tracking.video.cache_features.
# Opcional: RUN_NAME (assoc_<variante>).
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; U="${2:?falta la variante: none|ema|gru|graph}"; shift 2
: "${FEAT_DIR:?define FEAT_DIR (carpeta con train/ y val/ de features cacheadas)}"
cd "$PROJECT_ROOT"
python -m tracking.assoc.train --update "$U" --train-cache "$FEAT_DIR/train" --val-cache "$FEAT_DIR/val" \
  --out "$OUTPUT_ROOT/${RUN_NAME:-assoc_$U}" --seed "$SEED" "$@" 2>&1 | tee -a "$OUTPUT_ROOT/${RUN_NAME:-assoc_$U}.log"
