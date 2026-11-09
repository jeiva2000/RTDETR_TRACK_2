#!/bin/bash
#SBATCH --job-name=trackheads
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster (p. ej. gpu:a30:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00            # sin medir; ajustar tras la primera época
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrackHeads etapa 1 (detector congelado, en línea). Añadir --partition / --account según el clúster. Desde la raíz del repo (con mkdir -p logs):
#   sbatch slurm/trackheads.sbatch                                   # usa $OUTPUT_ROOT/detector_r50vd_person/best.pth
#   DET_CKPT=/ruta/best.pth RUN_NAME=th_noattn EXTRA="--w-attn 0" sbatch slurm/trackheads.sbatch
# Si se corta por tiempo, relanzar con el mismo RUN_NAME: reanuda desde last.pth (--resume).
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
DET_CKPT="${DET_CKPT:-$OUTPUT_ROOT/detector_r50vd_person/best.pth}"
RUN_NAME="${RUN_NAME:-trackheads_v1}"
[ -f "$DET_CKPT" ] || { echo "ERROR: no existe el checkpoint del detector $DET_CKPT (define DET_CKPT)" >&2; exit 1; }
# shellcheck disable=SC2086
python -m tracking.trackheads.train --det-ckpt "$DET_CKPT" --root "$DANCETRACK_DIR" --out "$OUTPUT_ROOT/trackheads/$RUN_NAME" \
  --seed "${SEED:-0}" --num-workers "${SLURM_CPUS_PER_TASK:-6}" --amp --resume ${EXTRA:-}
