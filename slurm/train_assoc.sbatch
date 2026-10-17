#!/bin/bash
#SBATCH --job-name=assoc_train
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00            # sin medir; ajustar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Un único entrenamiento (sin evaluar), con la semilla de setup_env/paths.hpc.env (SEED). Desde la raíz del repo:
#   FEAT_DIR=/ruta/features VARIANT=hist RUN_NAME=assoc_hist_A EXTRA="--burn-in 10" sbatch slurm/train_assoc.sbatch
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
: "${VARIANT:?define VARIANT}"; : "${FEAT_DIR:?define FEAT_DIR}"
# shellcheck disable=SC2086
bash tracking/scripts/train_assoc.sh hpc "$VARIANT" ${EXTRA:-}
