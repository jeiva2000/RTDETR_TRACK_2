#!/bin/bash
#SBATCH --job-name=rtdetr_cache
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00            # sin estimar; ajustar tras medir
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Añadir --partition / --account según el clúster. Desde la raíz del repo (con mkdir -p logs hecho):
#   DET_CKPT=/ruta/detector/best.pth FEAT_DIR=/ruta/features/detector_v1 sbatch slurm/cache_features.sbatch
# Cachea las top-K queries del detector congelado para train y val (Fase 5). Hay que repetirlo si cambia el detector.
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
: "${DET_CKPT:?define DET_CKPT}"; : "${FEAT_DIR:?define FEAT_DIR}"
python -m tracking.video.cache_features --ckpt "$DET_CKPT" --split val   --min-iou 0.5 --out "$FEAT_DIR/val"
python -m tracking.video.cache_features --ckpt "$DET_CKPT" --split train --min-iou 0.5 --out "$FEAT_DIR/train"
