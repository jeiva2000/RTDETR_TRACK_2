#!/bin/bash
#SBATCH --job-name=rtdetr_joint
#SBATCH --gres=gpu:a100:1          # ajustar a la sintaxis del clúster (p. ej. --gres=gpu:1 --constraint=a100)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00            # sin medir; ver docs/SERVIDOR.md
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Fase 7: fine-tuning conjunto (solo decoder + módulo de asociación). UNA GPU. Añadir --partition / --account según el clúster.
# Desde la raíz del repo (con mkdir -p logs hecho), con la variante nueva (hist) como inicialización del módulo:
#   DET_CKPT=/ruta/detector/best.pth ASSOC_CKPT=$OUTPUT_ROOT/assoc_hist_A_s1/best.pth RUN_NAME=joint_hist \
#     EXTRA="--samples-per-epoch 400 --epochs 15" sbatch slurm/train_joint.sbatch
# (sbatch hereda el entorno; también valen argumentos tras el nombre del script.) Semilla: la de paths.hpc.env salvo --seed en EXTRA.
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
# shellcheck disable=SC2086
bash tracking/scripts/train_joint.sh hpc ${EXTRA:-} "$@"
