#!/usr/bin/env bash
# Tabla con media ± desviación a partir de las semillas ya evaluadas de una o varias variantes.
# Uso: tracking/scripts/aggregate_seeds.sh [local|hpc] <base1> [<base2> ...]     (busca $OUTPUT_ROOT/<base>_s*/eval.json)
# Para añadir el baseline: BASELINE_JSONS="bytetrack=<json> ocsort=<json>"; salida en $OUTPUT_ROOT/seeds_table.md
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; shift
cd "$PROJECT_ROOT"; ARGS=(${BASELINE_JSONS:-})
for B in "$@"; do
  J=$(ls "$OUTPUT_ROOT/${B}"_s*/eval.json 2>/dev/null | paste -sd, -)
  [ -n "$J" ] || { echo "sin eval.json para $B"; continue; }
  ARGS+=("$B=$J")
done
python -m tracking.baseline.make_table "${ARGS[@]}" | tee "$OUTPUT_ROOT/seeds_table.md"
