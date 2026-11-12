#!/bin/bash
#SBATCH --job-name=trackheads_joint
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster (p. ej. gpu:a30:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00            # sin medir: forward+backward del detector completo sobre 2B frames por paso; 20 épocas de 8000 pares + DetA por época
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrackHeads con el RT-DETRv2 DESCONGELADO (tracking/trackheads/train_joint.py), inicializado desde la etapa congelada:
#   1) ajuste conjunto -> $OUTPUT_ROOT/trackheads/$RUN_NAME/{best,last,detector_best,detector_last}.pth
#   2) diagnóstico de best.pth en val (con su detector_best.pth)
#   3) tracker en línea + TrackEval + idsw (slurm/trackheads_track.sbatch como script, PRESET por defecto lifecycle) -> track_$TAG/table.md
# Añadir --partition / --account según el clúster. Desde la raíz del repo (con mkdir -p logs):
#   sbatch slurm/trackheads_joint.sbatch                                  # INIT = trackheads_v3/best.pth, RUN_NAME = trackheads_v3_joint
#   INIT=... RUN_NAME=th_joint_b2 EXTRA="--batch 2" sbatch slurm/trackheads_joint.sbatch     # si la GPU se queda sin memoria
#   EXTRA="--no-train-backbone" ...                                       # ablación: solo decoder + encoder (menos memoria y tiempo)
#   RUN_NAME=trackheads_v3_joint_l1 EXTRA="--lambda-th 1" ...             # con λ=10 el DetA cayó 1.8 puntos en la primera época (ver docs)
#   RUN_NAME=trackheads_v3_joint_dec EXTRA="--lambda-th 1 --no-train-encoder --no-train-backbone" ...   # solo decoder: solo cambia el emb
# Si se corta por tiempo, relanzar con el mismo RUN_NAME: reanuda desde last.pth (--resume).
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
DET_CKPT="${DET_CKPT:-$OUTPUT_ROOT/detector_r50vd_person/best.pth}"
INIT="${INIT:-$OUTPUT_ROOT/trackheads/trackheads_v3/best.pth}"
RUN_NAME="${RUN_NAME:-trackheads_v3_joint}"
OUT="$OUTPUT_ROOT/trackheads/$RUN_NAME"
[ -f "$DET_CKPT" ] || { echo "ERROR: no existe el detector $DET_CKPT (define DET_CKPT)" >&2; exit 1; }
[ -f "$INIT" ] || { echo "ERROR: no existe la inicialización $INIT (define INIT)" >&2; exit 1; }

echo "== [1/3] ajuste conjunto -> $OUT"
# shellcheck disable=SC2086
python -m tracking.trackheads.train_joint --det-ckpt "$DET_CKPT" --init "$INIT" --root "$DANCETRACK_DIR" --out "$OUT" \
  --seed "${SEED:-0}" --num-workers "${SLURM_CPUS_PER_TASK:-6}" --resume ${EXTRA:-}

echo "== [2/3] diagnóstico de $OUT/best.pth"
[ -f "$OUT/best.pth" ] || { echo "ERROR: ninguna época fue válida: no hay $OUT/best.pth" >&2; exit 1; }
# shellcheck disable=SC2086
python -m tracking.trackheads.diagnose --ckpt "$OUT/best.pth" --root "$DANCETRACK_DIR" --num-workers "${SLURM_CPUS_PER_TASK:-6}" ${DIAG_EXTRA:-}

echo "== [3/3] tracker en línea y evaluación"
RUN_NAME="$RUN_NAME" TAG="${TAG:-default}" PRESET="${PRESET:-lifecycle}" bash slurm/trackheads_track.sbatch
