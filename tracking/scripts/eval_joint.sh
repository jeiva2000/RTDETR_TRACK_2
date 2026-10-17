#!/usr/bin/env bash
# Fase 7, evaluación: caché de val con el detector AJUSTADO, ByteTrack/OC-SORT con ese mismo detector y el módulo conjunto,
# con el mismo protocolo (TrackEval, global y cruces). Uso: tracking/scripts/eval_joint.sh [local|hpc]
# Variables: WHICH (best|last; last = estado tras la última época, útil si la parada por DetA dejó sin best), RUN_NAME (joint_graph), BIRTH_THRESH (0.6), TRACK_THRESH (0.6), EXTRA_JSONS ("assoc_graph=<eval.json> ..." para añadir filas)
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"
NAME="${RUN_NAME:-joint_graph}"; RUN="$OUTPUT_ROOT/$NAME"; W="${WHICH:-best}"; CK="$RUN/detector_$W.pth"; VAL="$DANCETRACK_DIR/val"; cd "$PROJECT_ROOT"
[ -f "$CK" ] || { echo "No existe $CK (¿ninguna época válida? ver train.log)"; exit 1; }
python -m tracking.video.cache_features --ckpt "$CK" --split val --min-iou 0.5 --out "$RUN/features/val"
RUN_NAME="${NAME}_baseline" TRACK_THRESH="${TRACK_THRESH:-0.6}" bash tracking/scripts/run_baseline.sh "${1:-local}" "$CK"
python -m tracking.assoc.track --ckpt "$RUN/assoc_$W.pth" --cache "$RUN/features/val" --out "$RUN/results" --birth-thresh "${BIRTH_THRESH:-0.6}"
python -m tracking.baseline.evaluate --results "$RUN/results" --gt-root "$VAL" \
  --seqs $(cd "$RUN/features/val" && ls *.pt | sed 's/.pt$//') --out "$RUN/eval.json" > /dev/null
B="$OUTPUT_ROOT/${NAME}_baseline"
python -m tracking.baseline.make_table "${NAME}=$RUN/eval.json" "bytetrack_ajustado=$B/bytetrack.json" "ocsort_ajustado=$B/ocsort.json" ${EXTRA_JSONS:-} | tee "$RUN/table.md"
