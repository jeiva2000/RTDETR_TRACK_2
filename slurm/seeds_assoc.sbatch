#!/bin/bash
#SBATCH --job-name=assoc_seeds
#SBATCH --array=1-3                # una tarea por semilla (la semilla es SLURM_ARRAY_TASK_ID)
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                  # el entrenamiento carga las cachés enteras en RAM
#SBATCH --time=06:00:00            # sin medir: en la workstation graph tardaba ~2 min/época (T=10); hist con burn-in puede doblar
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Añadir --partition / --account según el clúster. Desde la raíz del repo (con mkdir -p logs hecho):
#   export FEAT_DIR=/ruta/features/detector_v1        # sbatch hereda el entorno (--export=ALL por defecto)
#   VARIANT=hist  BASE=assoc_hist_A EXTRA="--burn-in 10"                  sbatch slurm/seeds_assoc.sbatch
#   VARIANT=hist  BASE=assoc_hist_B EXTRA="--burn-in 10 --hist-aff max"   sbatch slurm/seeds_assoc.sbatch
#   VARIANT=graph BASE=assoc_graph                                        sbatch slurm/seeds_assoc.sbatch   # referencia, mismas semillas
# Cada tarea entrena, corre el tracker online sobre val y evalúa (TrackEval). Al terminar las 3:
#   bash tracking/scripts/aggregate_seeds.sh hpc assoc_hist_A assoc_hist_B assoc_graph
# Opcionales: BIRTH_THRESH (0.6), MAX_LOST, MAX_LOST_OCC, MAX_LOST_OUT.
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
: "${VARIANT:?define VARIANT (none|ema|gru|graph|graph2|hist)}"; : "${BASE:?define BASE}"; : "${FEAT_DIR:?define FEAT_DIR}"
# shellcheck disable=SC2086
bash tracking/scripts/run_one_seed.sh hpc "$VARIANT" "$BASE" "$SLURM_ARRAY_TASK_ID" ${EXTRA:-}
