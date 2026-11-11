#!/bin/bash
#SBATCH --job-name=trackheads_track
#SBATCH --gres=gpu:1               # ajustar a la sintaxis del clúster (p. ej. gpu:a30:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00            # sin medir: un forward del detector por frame (~25k frames en val) + TrackEval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Tracker en línea con TrackHeads sobre DanceTrack val (tracking/trackheads/track.py), evaluado con TrackEval (HOTA, DetA, AssA, IDF1, IDSW;
# global y en cruces). Por defecto prueba varias configuraciones en la misma pasada del detector (cabeza / IoU, confirmación min_hits,
# movimiento, umbrales; ver DEFAULT_CONFIGS en track.py) y añade ByteTrack y OC-SORT del baseline si existen.
# Añadir --partition / --account según el clúster. Desde la raíz del repo (con mkdir -p logs):
#   sbatch slurm/trackheads_track.sbatch                                   # usa $OUTPUT_ROOT/trackheads/trackheads_v3/best.pth
#   TAG=emb PRESET=emb sbatch slurm/trackheads_track.sbatch                 # variantes de actualización del emb (last/clean/ema)
#   TAG=nb1 TRACK_EXTRA="--configs a=head,min_hits=3,motion=1,null_bias=1 b=head,min_hits=3,motion=1,max_lost=60" sbatch slurm/trackheads_track.sbatch
# Resultados en $OUTPUT_ROOT/trackheads/$RUN_NAME/track_$TAG/: <config>/<seq>.txt, <config>.json, table.md e idsw.json (tablas también en el log).
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

echo "== [1/4] tracking -> $OUT"
# shellcheck disable=SC2086
python -m tracking.trackheads.track --ckpt "$CKPT" --out "$OUT" --root "$DANCETRACK_DIR" --preset "${PRESET:-lifecycle}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-6}" ${TRACK_EXTRA:-}

echo "== [2/4] TrackEval"
CFGS=$(cd "$OUT" && ls -d */ | sed 's#/$##')
FIRST=$(echo "$CFGS" | head -n1)
SEQS=$(cd "$OUT/$FIRST" && ls *.txt | sed 's/.txt$//')
ARGS=()
for M in $CFGS; do
  # shellcheck disable=SC2086
  python -m tracking.baseline.evaluate --results "$OUT/$M" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
    --seqs $SEQS --out "$OUT/$M.json" > /dev/null
  ARGS+=("$M=$OUT/$M.json")
done
for T in bytetrack ocsort; do
  [ -f "$BASE/$T.json" ] && ARGS+=("$T=$BASE/$T.json") || echo "(sin $BASE/$T.json: $T no entra en la tabla)"
done

echo "== [3/4] tabla"
python -m tracking.baseline.make_table "${ARGS[@]}" | tee "$OUT/table.md"

echo "== [4/4] cambios de identidad (idsw.py)"
IDSW=()
for M in $CFGS; do IDSW+=("$M=$OUT/$M"); done
for T in bytetrack ocsort; do [ -d "$BASE/$T" ] && IDSW+=("$T=$BASE/$T"); done
python -m tracking.trackheads.idsw --gt-root "$VAL" "${IDSW[@]}" --out "$OUT/idsw.json"
