#!/bin/bash
# Envoltorio de un trabajo de SLURM: carga módulos, activa el entorno y ejecuta una etapa. Lo envía e2e_pipeline.sh con sbatch.
# Uso (lo hace el orquestador): sbatch ... tracking/scripts/pipe_job.sh <etapa> [argumentos]
set -euo pipefail
cd "${PROJECT_ROOT:?define PROJECT_ROOT}"
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh "${ENV_NAME:-hpc}"
conda activate "$CONDA_ENV"
bash tracking/scripts/pipe_stage.sh "${ENV_NAME:-hpc}" "$@"
