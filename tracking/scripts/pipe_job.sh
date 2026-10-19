#!/bin/bash
# Envoltorio de un trabajo de SLURM enviado por e2e_pipeline.sh en modo "un trabajo por etapa" (alternativa a slurm/e2e.slurm, que ejecuta todo en una sola reserva).
# Uso (lo hace el orquestador): sbatch ... tracking/scripts/pipe_job.sh <etapa> [argumentos]
set -euo pipefail
cd "${PROJECT_ROOT:?define PROJECT_ROOT}"
module purge
module load ${CLUSTER_MODULES:-GCC/12.2.0 CUDA/11.8.0}
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" 2>/dev/null || eval "$(conda shell.bash hook)"
source setup_env/load.sh "${ENV_NAME:-hpc}"
conda activate "$CONDA_ENV"
bash tracking/scripts/pipe_stage.sh "${ENV_NAME:-hpc}" "$@"
