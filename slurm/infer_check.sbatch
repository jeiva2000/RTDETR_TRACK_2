#!/bin/bash
#SBATCH --job-name=rtdetr_infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Uso (desde la raíz del repo): sbatch --export=ALL,IMAGE=/ruta/img.jpg slurm/infer_check.sbatch
# Ajustar partición/cuenta según el clúster.
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
source setup_env/load.sh hpc
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT/rtdetrv2_pytorch"
python ../tracking/scripts/infer_image.py --image "$IMAGE"
