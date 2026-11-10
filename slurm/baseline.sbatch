#!/bin/bash
#SBATCH --job-name=rtdetr_baseline
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00            # sin estimar; ajustar tras medir
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Baseline ByteTrack/OC-SORT con el MISMO detector (para la tabla comparativa). Desde la raíz del repo:
#   sbatch slurm/baseline.sbatch                          # detector: $OUTPUT_ROOT/detector_r50vd_person/best.pth; salida: $OUTPUT_ROOT/baseline/
#   DET_CKPT=/ruta/detector/best.pth RUN_NAME=baseline_v1 sbatch slurm/baseline.sbatch
# slurm/trackheads_track.sbatch busca $OUTPUT_ROOT/baseline/{bytetrack,ocsort}.json para añadirlos a su tabla.
# Variables opcionales: TRACK_THRESH (0.6), MAX_SEQS (0 = todas). Resultados en $OUTPUT_ROOT/$RUN_NAME/.
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
DET_CKPT="${DET_CKPT:-$OUTPUT_ROOT/detector_r50vd_person/best.pth}"
[ -f "$DET_CKPT" ] || { echo "ERROR: no existe el checkpoint del detector $DET_CKPT (define DET_CKPT)" >&2; exit 1; }
bash tracking/scripts/run_baseline.sh hpc "$DET_CKPT"
