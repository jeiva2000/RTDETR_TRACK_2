#!/bin/bash
#SBATCH --job-name=trackheads_track
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster (p. ej. gpu:a30:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00            # sin medir: un forward del detector por frame (~25k frames en val) + TrackEval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Tracker en línea con TrackHeads sobre DanceTrack val (tracking/trackheads/track.py), evaluado con TrackEval (HOTA, DetA, AssA, IDF1, IDSW;
# global y en cruces), frente al mismo tracker con IoU (mismo detector, nacimientos y muertes) y, si existen, ByteTrack y OC-SORT del baseline.
# Añadir --partition / --account según el clúster. Desde la raíz del repo (con mkdir -p logs):
#   sbatch slurm/trackheads_track.sbatch                                   # usa $OUTPUT_ROOT/trackheads/trackheads_v3/best.pth
#   RUN_NAME=trackheads_v3 TAG=nb1 TRACK_EXTRA="--null-bias 1" sbatch slurm/trackheads_track.sbatch
# Resultados en $OUTPUT_ROOT/trackheads/$RUN_NAME/track_$TAG/: <modo>/<seq>.txt, <modo>.json y table.md (también en el log).
set -euo pipefail
module load GCC/12.2.0 CUDA/11.8.0
eval "$(conda shell.bash hook)"
source setup_env/load.sh hpc
conda activate "$CONDA_ENV"
RUN_NAME="${RUN_NAME:-trackheads_v3}"
CKPT="$OUTPUT_ROOT/trackheads/$RUN_NAME/best.pth"
OUT="$OUTPUT_ROOT/trackheads/$RUN_NAME/track_${TAG:-default}"
BASE="$OUTPUT_ROOT/${BASELINE_RUN:-baseline}"              # salida de slurm/baseline.sbatch (opcional)
VAL="$DANCETRACK_DIR/val"
[ -f "$CKPT" ] || { echo "ERROR: no existe $CKPT" >&2; exit 1; }

echo "== [1/3] tracking -> $OUT"
# shellcheck disable=SC2086
python -m tracking.trackheads.track --ckpt "$CKPT" --out "$OUT" --root "$DANCETRACK_DIR" \
  --num-workers "${SLURM_CPUS_PER_TASK:-6}" ${TRACK_EXTRA:-}

echo "== [2/3] TrackEval"
SEQS=$(cd "$OUT/head" && ls *.txt | sed 's/.txt$//')
ARGS=()
for M in head iou; do
  # shellcheck disable=SC2086
  python -m tracking.baseline.evaluate --results "$OUT/$M" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
    --seqs $SEQS --out "$OUT/$M.json" > /dev/null
  ARGS+=("trackheads_$M=$OUT/$M.json")
done
for T in bytetrack ocsort; do
  [ -f "$BASE/$T.json" ] && ARGS+=("$T=$BASE/$T.json") || echo "(sin $BASE/$T.json: $T no entra en la tabla)"
done

echo "== [3/3] tabla"
python -m tracking.baseline.make_table "${ARGS[@]}" | tee "$OUT/table.md"
