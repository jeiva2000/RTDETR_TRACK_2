#!/usr/bin/env bash
# Baseline de seguimiento: detector (Fase 3) + ByteTrack y OC-SORT sobre DanceTrack val, evaluados con TrackEval.
# Uso: tracking/scripts/run_baseline.sh [local|hpc] <checkpoint best.pth>
# Variables opcionales: RUN_NAME (baseline), MAX_SEQS (0 = todas), MARGIN (10), CROSS_IOU (0.3), DEVICE (cuda:0),
#   TRACK_THRESH (0.6; los repos oficiales usan 0.6 con YOLOX, no está ajustado a RT-DETR), NUM_WORKERS (4)
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; CKPT="${2:?falta el checkpoint del detector}"
RUN="$OUTPUT_ROOT/${RUN_NAME:-baseline}"; VAL="$DANCETRACK_DIR/val"; cd "$PROJECT_ROOT"
python tracking/baseline/detect_sequences.py --ckpt "$CKPT" --seq-root "$VAL" --out "$RUN/dets" \
  --max-seqs "${MAX_SEQS:-0}" --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-4}"
ARGS=()
for T in bytetrack ocsort; do
  python tracking/baseline/run_tracker.py --tracker $T --dets "$RUN/dets" --seq-root "$VAL" --out "$RUN/$T" --track-thresh "${TRACK_THRESH:-0.6}"
  python -m tracking.baseline.evaluate --results "$RUN/$T" --gt-root "$VAL" --margin "${MARGIN:-10}" --cross-iou "${CROSS_IOU:-0.3}" \
    --seqs $(cd "$RUN/dets" && ls *.txt | sed 's/.txt$//') --out "$RUN/$T.json" > /dev/null
  ARGS+=("$T=$RUN/$T.json")
done
python -m tracking.baseline.make_table "${ARGS[@]}" | tee "$RUN/table.md"
