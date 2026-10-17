#!/usr/bin/env bash
# Varias semillas de una variante, en serie, y tabla con media ± desviación.
# Uso: tracking/scripts/run_seeds.sh [local|hpc] <variante> <nombre_base> [argumentos extra de tracking.assoc.train]
#   p. ej.: tracking/scripts/run_seeds.sh local hist assoc_hist_A --burn-in 10
# Variables: SEEDS ("1 2 3"), y las de run_one_seed.sh. En el servidor es mejor el array de SLURM (slurm/seeds_assoc.sbatch).
set -euo pipefail
ENVN="${1:-local}"; U="${2:?variante}"; BASE="${3:?nombre base}"; shift 3
for S in ${SEEDS:-1 2 3}; do bash "$(dirname "$0")/run_one_seed.sh" "$ENVN" "$U" "$BASE" "$S" "$@"; done
bash "$(dirname "$0")/aggregate_seeds.sh" "$ENVN" "$BASE"
