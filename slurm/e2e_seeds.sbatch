#!/bin/bash
#SBATCH --job-name=e2e_seeds
#SBATCH --array=1-3                # una tarea por semilla (la semilla es SLURM_ARRAY_TASK_ID)
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                  # el entrenamiento carga las cachés enteras en RAM
#SBATCH --time=06:00:00            # sin medir
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Hitos H1-H6 (detector congelado, caché): una GPU por semilla. Añadir --partition / --account según el clúster. Desde la raíz del repo:
#   export FEAT_DIR=/ruta/features/detector_v1
#   BASE=e2e_h1 sbatch slurm/e2e_seeds.sbatch                                  # configuración por defecto (tracking/configs/e2e.yml)
#   BASE=e2e_h1_a02 OV="appearance.alpha_app=0.2" sbatch slurm/e2e_seeds.sbatch  # sobrescribiendo claves (separadas por espacios)
# Al terminar: bash tracking/scripts/aggregate_seeds.sh hpc e2e_h1 assoc_ema assoc_graph      (las dos últimas, de la Fase 6, mismas semillas)
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
: "${BASE:?define BASE}"; : "${FEAT_DIR:?define FEAT_DIR}"
# shellcheck disable=SC2086
if [ -n "${OV:-}" ]; then bash tracking/scripts/e2e_one_seed.sh hpc "$BASE" "$SLURM_ARRAY_TASK_ID" -u $OV
else bash tracking/scripts/e2e_one_seed.sh hpc "$BASE" "$SLURM_ARRAY_TASK_ID"; fi
