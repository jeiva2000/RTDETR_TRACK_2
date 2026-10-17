#!/bin/bash
#SBATCH --job-name=joint_lambda
#SBATCH --array=0-3                # λ_assoc = 0 (control: solo detección), 1, 3, 10
#SBATCH --gres=gpu:a100:1          # ajustar a la sintaxis del clúster
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00            # sin medir
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Fase 7: barrido del peso de la pérdida de asociación. La prueba anterior (λ=10, 1 época) empeoró DetA y tracking y la causa no está
# aislada; λ=0 es el control (¿degrada el detector solo por el régimen de ajuste?). Desde la raíz del repo:
#   DET_CKPT=... ASSOC_CKPT=$OUTPUT_ROOT/assoc_hist_A_s1/best.pth BASE=joint_hist EXTRA="--samples-per-epoch 400 --epochs 15" \
#     sbatch slurm/joint_lambda_sweep.sbatch
# Cada tarea escribe en $OUTPUT_ROOT/<BASE>_l<λ>/ (detector_best/last.pth, assoc_best/last.pth, log.txt, train.log).
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
: "${BASE:?define BASE}"
LAMBDAS=(0 1 3 10); L="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086
RUN_NAME="${BASE}_l${L}" bash tracking/scripts/train_joint.sh hpc --lambda-assoc "$L" ${EXTRA:-}
