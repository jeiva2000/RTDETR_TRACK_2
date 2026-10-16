#!/bin/bash
#SBATCH --job-name=rtdetr_joint
#SBATCH --gres=gpu:a100:1          # ajustar a la sintaxis del clúster (p. ej. --gres=gpu:1 --constraint=a100)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00            # sin estimar; ajustar tras medir s/it
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Añadir --partition / --account según el clúster. Uso (desde la raíz del repo): sbatch slurm/train_joint.sbatch
# Variables opcionales (sbatch --export=ALL,RUN_NAME=...,DET_CKPT=...,ASSOC_CKPT=...): ver tracking/scripts/train_joint.sh
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
bash tracking/scripts/train_joint.sh hpc "$@"
