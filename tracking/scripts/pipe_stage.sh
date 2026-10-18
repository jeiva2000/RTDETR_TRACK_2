#!/usr/bin/env bash
# Etapas de la tubería e2e (una por invocación; idempotentes: si existe el marcador .done no se repiten salvo FORCE=1).
# Uso: tracking/scripts/pipe_stage.sh [local|hpc] <etapa> [argumentos]
#   cache                         features de train y val con el detector (DET_CKPT -> FEAT_DIR)
#   baseline                      ByteTrack y OC-SORT con el mismo detector + barrido de track_thresh
#   variant <nombre> <semilla>    entrena una variante e2e (caché, detector congelado) y evalúa con umbrales por defecto
#   sweep <nombre> <semillas...>  barrido de umbrales de inferencia de la variante y evaluación final con el mejor
#   select <nombres...>           elige la mejor variante (HOTA medio) -> $PIPE_DIR/best_variant.json
#   unfrozen <lambda>             ajuste conjunto con la variante elegida (decoder descongelado) + evaluación completa
#   final                         tabla final (baselines, variantes congeladas, descongelado)
# Variables obligatorias: DET_CKPT, FEAT_DIR. Opcionales: PIPE_NAME (e2e_pipe), NSEEDS (5), JOINT_ARGS, BASELINE_THRESHOLDS ("0.4 0.5 0.7 0.8 0.9"; el 0.6 siempre se ejecuta),
#   E2E_EXTRA_OV (overrides de config para todas las variantes, p. ej. "train.epochs=30"), SWEEP_ARGS (argumentos extra de tracking.e2e.sweep run),
#   UNFROZEN_GPUS (GPUs por entrenamiento descongelado con DDP; 1 = sin DDP) y CLIPS_PER_GPU (clips por GPU y paso; 1 si hay varias GPUs, 4 si hay una).
set -euo pipefail
ENVN="${1:-local}"; STAGE="${2:?etapa}"; shift 2
source "$(dirname "$0")/../../setup_env/load.sh" "$ENVN"
: "${DET_CKPT:?define DET_CKPT (best.pth del detector)}"; : "${FEAT_DIR:?define FEAT_DIR}"
cd "$PROJECT_ROOT"
PIPE="${PIPE_NAME:-e2e_pipe}"; PIPE_DIR="$OUTPUT_ROOT/$PIPE"; mkdir -p "$PIPE_DIR/markers"; VAL="$DANCETRACK_DIR/val"
source tracking/configs/e2e_variants.sh
done_marker() { echo "$PIPE_DIR/markers/$1.done"; }
skip_if_done() { if [ -f "$(done_marker "$1")" ] && [ "${FORCE:-0}" != "1" ]; then echo "[skip] $1 ya hecho"; exit 0; fi; }
mark() { date '+%Y-%m-%dT%H:%M:%S' > "$(done_marker "$1")"; }
overrides_of() {   # nombre -> overrides
  for v in "${E2E_VARIANTS[@]}"; do [ "${v%%|*}" = "$1" ] && { echo "${v#*|}" | sed 's/#.*//'; return; }; done
  echo "variante desconocida: $1" >&2; exit 1
}
case "$STAGE" in
  cache)
    skip_if_done cache
    [ -d "$FEAT_DIR/val" ] && [ -d "$FEAT_DIR/train" ] && [ -f "$FEAT_DIR/val/stats.json" ] && [ -f "$FEAT_DIR/train/stats.json" ] && { echo "[ok] caché ya completa en $FEAT_DIR"; mark cache; exit 0; }
    python -m tracking.video.cache_features --ckpt "$DET_CKPT" --split val   --min-iou 0.5 --out "$FEAT_DIR/val"
    python -m tracking.video.cache_features --ckpt "$DET_CKPT" --split train --min-iou 0.5 --out "$FEAT_DIR/train"
    mark cache ;;
  baseline)
    skip_if_done baseline
    B="$PIPE_DIR/baseline"
    RUN_NAME="${PIPE}/baseline" TRACK_THRESH=0.6 bash tracking/scripts/run_baseline.sh "$ENVN" "$DET_CKPT"
    mkdir -p "$B/sweep"; cp "$B/bytetrack.json" "$B/sweep/bytetrack_0.6.json"; cp "$B/ocsort.json" "$B/sweep/ocsort_0.6.json"   # el 0.6 también entra en la elección
    SEQS=$(cd "$B/dets" && ls *.txt | sed 's/.txt$//')
    for T in bytetrack ocsort; do for TH in ${BASELINE_THRESHOLDS:-0.4 0.5 0.7 0.8 0.9}; do
      python tracking/baseline/run_tracker.py --tracker $T --dets "$B/dets" --seq-root "$VAL" --out "$B/sweep/${T}_$TH" --track-thresh "$TH"
      python -m tracking.baseline.evaluate --results "$B/sweep/${T}_$TH" --gt-root "$VAL" --seqs $SEQS --out "$B/sweep/${T}_$TH.json" > /dev/null
    done; done
    mark baseline ;;
  variant)
    NAME="${1:?variante}"; SEED_N="${2:?semilla}"; skip_if_done "variant_${NAME}_s${SEED_N}"
    OV="$(overrides_of "$NAME") ${E2E_EXTRA_OV:-}"; OV=$(echo $OV)    # E2E_EXTRA_OV: overrides globales (p. ej. train.epochs=30)
    # shellcheck disable=SC2086
    if [ -n "$OV" ]; then bash tracking/scripts/e2e_one_seed.sh "$ENVN" "${PIPE}/${NAME}" "$SEED_N" -u $OV
    else bash tracking/scripts/e2e_one_seed.sh "$ENVN" "${PIPE}/${NAME}" "$SEED_N"; fi
    mark "variant_${NAME}_s${SEED_N}" ;;
  sweep)
    NAME="${1:?variante}"; shift; skip_if_done "sweep_${NAME}"
    RUNS=(); for S in "$@"; do RUNS+=("$OUTPUT_ROOT/${PIPE}/${NAME}_s${S}"); done
    # shellcheck disable=SC2086
    python -m tracking.e2e.sweep run --runs "${RUNS[@]}" --cache "$FEAT_DIR/val" --gt-root "$VAL" --out "$PIPE_DIR/sweep_${NAME}" ${SWEEP_ARGS:-}
    mark "sweep_${NAME}" ;;
  select)
    skip_if_done select
    ARGS=(); for N in "$@"; do ARGS+=("$N=$PIPE_DIR/sweep_$N"); done
    python -m tracking.e2e.sweep select --variants "${ARGS[@]}" --out "$PIPE_DIR/best_variant.json"
    mark select ;;
  unfrozen)
    LAM="${1:?lambda}"; skip_if_done "unfrozen_l${LAM}"
    [ -f "$PIPE_DIR/best_variant.json" ] || { echo "falta $PIPE_DIR/best_variant.json (etapa select)"; exit 1; }
    NAME=$(python3 -c "import json;print(json.load(open('$PIPE_DIR/best_variant.json'))['variant'])")
    OV="$(overrides_of "$NAME") ${E2E_EXTRA_OV:-}"; OV=$(echo $OV); OUT="$PIPE_DIR/unfrozen_${NAME}_l${LAM}"
    # varias GPUs por entrenamiento (DDP, una por proceso, cada GPU procesa sus propios clips): UNFROZEN_GPUS (1 = sin DDP). Lote efectivo = CLIPS_PER_GPU x UNFROZEN_GPUS
    UG="${UNFROZEN_GPUS:-1}"; CPG="${CLIPS_PER_GPU:-$([ "$UG" -gt 1 ] && echo 1 || echo 4)}"
    if [ "$UG" -gt 1 ]; then LAUNCH=(python -m torch.distributed.run --standalone --nproc_per_node="$UG" -m tracking.e2e.train_joint); else LAUNCH=(python -m tracking.e2e.train_joint); fi
    # shellcheck disable=SC2086
    "${LAUNCH[@]}" --det-ckpt "$DET_CKPT" --out "$OUT" --seed "$SEED" --lambda-e2e "$LAM" --clips-per-gpu "$CPG" ${JOINT_ARGS:-} ${OV:+-u $OV} 2>&1 | tee -a "$OUT.log"
    if [ ! -f "$OUT/detector_best.pth" ]; then echo "NINGUNA época válida en $OUT: el punto de partida sigue siendo lo mejor"; mark "unfrozen_l${LAM}"; exit 0; fi
    # evaluación completa: caché de val con el detector ajustado, baselines con ese detector y el módulo e2e con el umbral elegido
    python -m tracking.video.cache_features --ckpt "$OUT/detector_best.pth" --split val --min-iou 0.5 --out "$OUT/features/val"
    RUN_NAME="${PIPE}/unfrozen_${NAME}_l${LAM}_baseline" TRACK_THRESH=0.6 bash tracking/scripts/run_baseline.sh "$ENVN" "$OUT/detector_best.pth"
    mkdir -p "$OUT/seed_run"; cp "$OUT/e2e_best.pth" "$OUT/seed_run/best.pth"
    python -m tracking.e2e.sweep run --runs "$OUT/seed_run" --ckpt-name best.pth --cache "$OUT/features/val" --gt-root "$VAL" --out "$OUT/sweep" --sweep-seeds 1
    mark "unfrozen_l${LAM}" ;;
  final)
    ARGS=()
    for V in "${E2E_VARIANTS[@]}"; do N="${V%%|*}"; J=$(ls "$OUTPUT_ROOT/${PIPE}/${N}"_s*/eval.json 2>/dev/null | paste -sd, -) && [ -n "$J" ] && ARGS+=("${N}_congelado=$J"); done
    for D in "$PIPE_DIR"/unfrozen_*_l*; do [ -f "$D/seed_run/eval.json" ] && ARGS+=("$(basename "$D")=$D/seed_run/eval.json"); done
    for T in bytetrack ocsort; do [ -f "$PIPE_DIR/baseline/$T.json" ] && ARGS+=("${T}_0.6=$PIPE_DIR/baseline/$T.json"); done
    python3 - "$PIPE_DIR/baseline/sweep" <<'PY' >> "$PIPE_DIR/baseline_best.txt" 2>/dev/null || true
import glob, json, sys
for t in ('bytetrack', 'ocsort'):
    best = max(((json.load(open(f))['global_']['HOTA'], f) for f in glob.glob(f'{sys.argv[1]}/{t}_*.json')), default=None)
    if best: print(f'{t}_mejor_umbral={best[1]}')
PY
    while IFS='=' read -r K V; do [ -n "$K" ] && ARGS+=("$K=$V"); done < <(sort -u "$PIPE_DIR/baseline_best.txt" 2>/dev/null)
    python -m tracking.baseline.make_table "${ARGS[@]}" | tee "$PIPE_DIR/final_table.md" ;;
  *) echo "etapa desconocida: $STAGE"; exit 1 ;;
esac
