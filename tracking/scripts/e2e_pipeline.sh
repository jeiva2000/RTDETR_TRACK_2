#!/usr/bin/env bash
# TUBERÍA COMPLETA del módulo e2e en el servidor, con un solo comando:
#   1. caché de features con el detector congelado y baselines (ByteTrack/OC-SORT) con barrido de umbral
#   2. variantes e2e con el detector CONGELADO (NSEEDS semillas cada una)
#   3. barrido de umbrales de inferencia por variante y elección de la mejor (HOTA medio en val, misma regla para todas)
#   4. versión DESCONGELADA (decoder + módulo) con la variante elegida, para varios valores de lambda, y su evaluación completa
#   5. tabla final: baselines con su mejor umbral, variantes congeladas y versión descongelada
# Uso:
#   export DET_CKPT=/ruta/detector/best.pth FEAT_DIR=/ruta/features/detector_v1          # obligatorias
#   bash tracking/scripts/e2e_pipeline.sh                  # con SLURM (si existe sbatch): envía todos los trabajos con dependencias y termina
#   MODE=local bash tracking/scripts/e2e_pipeline.sh       # (ENV_NAME=hpc|local: archivo de rutas) sin SLURM: ejecuta aquí, repartiendo trabajos entre las GPUs visibles
#   DRY_RUN=1 bash tracking/scripts/e2e_pipeline.sh        # muestra lo que haría sin ejecutar nada
# Variables opcionales: PIPE_NAME (e2e_pipe), NSEEDS (5), LAMBDAS ("0 3 10"; 0 es el control), VARIANTS (por defecto todas las de
#   tracking/configs/e2e_variants.sh), JOINT_ARGS (argumentos extra de train_joint, p. ej. "--epochs 40 --clips-per-step 4"),
#   SBATCH_COMMON ("--partition=X --account=Y"), TIME_CACHE/TIME_BASELINE/TIME_VARIANT/TIME_SWEEP/TIME_UNFROZEN,
#   UNFROZEN_GPUS (GPUs por entrenamiento descongelado con DDP; 1 = sin DDP), CLIPS_PER_GPU, SBATCH_GPU_TEMPLATE ("--gres=gpu:a100:%N"),
#   NGPU (modo local: GPUs a usar), FORCE=1 (repite etapas ya hechas). Reanudable: las etapas hechas dejan un marcador y se saltan.
# Resultados en $OUTPUT_ROOT/<PIPE_NAME>/ ; tabla final en final_table.md
set -euo pipefail
cd "$(dirname "$0")/../.."
ENVN="${ENV_NAME:-hpc}"; export ENV_NAME="$ENVN"      # archivo de rutas setup_env/paths.<ENV_NAME>.env
source setup_env/load.sh "$ENVN"
: "${DET_CKPT:?define DET_CKPT}"; : "${FEAT_DIR:?define FEAT_DIR}"
source tracking/configs/e2e_variants.sh
PIPE="${PIPE_NAME:-e2e_pipe}"; NSEEDS="${NSEEDS:-5}"; LAMBDAS="${LAMBDAS-0 3 10}"; MODE="${MODE:-$(command -v sbatch >/dev/null 2>&1 && echo slurm || echo local)}"
NAMES=(); for v in "${E2E_VARIANTS[@]}"; do NAMES+=("${v%%|*}"); done
[ -n "${VARIANTS:-}" ] && read -r -a NAMES <<< "$VARIANTS"
SEEDS=$(seq -s ' ' 1 "$NSEEDS")
mkdir -p logs "$OUTPUT_ROOT/$PIPE"
echo "== tubería e2e | modo $MODE | variantes: ${NAMES[*]} | semillas: $SEEDS | lambdas: $LAMBDAS | salida: $OUTPUT_ROOT/$PIPE =="
UG="${UNFROZEN_GPUS:-1}"; export UNFROZEN_GPUS="$UG" CLIPS_PER_GPU="${CLIPS_PER_GPU:-}"
export PROJECT_ROOT DET_CKPT FEAT_DIR PIPE_NAME="$PIPE" NSEEDS JOINT_ARGS="${JOINT_ARGS:-}" FORCE="${FORCE:-0}" BASELINE_THRESHOLDS="${BASELINE_THRESHOLDS:-0.4 0.5 0.7 0.8 0.9}" E2E_EXTRA_OV="${E2E_EXTRA_OV:-}" SWEEP_ARGS="${SWEEP_ARGS:-}"
run() { if [ "${DRY_RUN:-0}" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi; }

if [ "$MODE" = "slurm" ]; then
  FAKE_FILE=$(mktemp); echo 100 > "$FAKE_FILE"
  submit() {   # submit <nombre> <tiempo> <gpu:1|0> <dependencias> <etapa y args...> -> imprime el id del trabajo
    local name="$1" tm="$2" gpu="$3" dep="$4"; shift 4
    local opts=(--parsable --job-name="$name" --time="$tm" --nodes=1 --ntasks-per-node=1 --cpus-per-task=$((8 * (gpu > 1 ? gpu : 1))) --mem=$((48 * (gpu > 1 ? gpu : 1)))G --output="logs/%x_%j.out" --error="logs/%x_%j.err" --export=ALL)
    # gpu = nº de GPUs del trabajo (0 = ninguna). SBATCH_GPU_TEMPLATE con %N (p. ej. "--gres=gpu:a100:%N") o, por defecto, --gres=gpu:N
    [ "$gpu" -ge 1 ] && opts+=($(echo "${SBATCH_GPU_TEMPLATE:---gres=gpu:%N}" | sed "s/%N/$gpu/"))
    [ -n "${SBATCH_COMMON:-}" ] && opts+=($SBATCH_COMMON)
    [ -n "$dep" ] && opts+=(--dependency="$dep" --kill-on-invalid-dep=yes)
    if [ "${DRY_RUN:-0}" = "1" ]; then local n=$(( $(cat "$FAKE_FILE") + 1 )); echo $n > "$FAKE_FILE"; echo "[dry-run] sbatch(id $n) ${opts[*]##*--export=ALL} ${dep:+[$dep]} :: $name" >&2; echo $n
    else sbatch "${opts[@]}" tracking/scripts/pipe_job.sh "$@"; fi
  }
  J_CACHE=$(submit e2e_cache "${TIME_CACHE:-06:00:00}" 1 "" cache)
  J_BASE=$(submit e2e_baseline "${TIME_BASELINE:-08:00:00}" 1 "" baseline)
  SWEEPS=()
  for N in "${NAMES[@]}"; do
    VJ=(); for S in $SEEDS; do VJ+=("$(submit "e2e_${N}_s$S" "${TIME_VARIANT:-04:00:00}" 1 "afterok:$J_CACHE" variant "$N" "$S")"); done
    DEP="afterok:$(IFS=:; echo "${VJ[*]}")"
    SWEEPS+=("$(submit "e2e_sweep_$N" "${TIME_SWEEP:-06:00:00}" 1 "$DEP" sweep "$N" $SEEDS)")
  done
  J_SEL=$(submit e2e_select 00:20:00 0 "afterok:$(IFS=:; echo "${SWEEPS[*]}")" select "${NAMES[@]}")
  UJ=(); for L in $LAMBDAS; do UJ+=("$(submit "e2e_unfrozen_l$L" "${TIME_UNFROZEN:-24:00:00}" "$UG" "afterok:$J_SEL" unfrozen "$L")"); done
  J_FIN=$(submit e2e_final 00:20:00 0 "afterok:$J_BASE,afterany:$(IFS=:; echo "${UJ[*]}")" final)
  echo "enviados: cache=$J_CACHE baseline=$J_BASE select=$J_SEL final=$J_FIN (sigue con: squeue -u \$USER)"
  echo "tabla final: $OUTPUT_ROOT/$PIPE/final_table.md  (cuando termine e2e_final)"
else
  NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; [ "$NGPU" -ge 1 ] || NGPU=1
  echo "(modo local: $NGPU GPU(s))"
  RC_DIR=$(mktemp -d); NJ=0
  GPJ=1      # GPUs por trabajo (1, salvo en la etapa descongelada con DDP)
  slot_run() {   # slot_run <slot> <etapa y args...>: ejecuta una etapa en las GPUs del hueco, en segundo plano; su código de salida queda en $RC_DIR
    local slot="$1"; shift; local devs="" k
    for ((k=0; k<GPJ; k++)); do devs="${devs:+$devs,}$(( (slot * GPJ + k) % NGPU ))"; done
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "[dry-run] (GPU $devs) pipe_stage.sh $ENVN $*"; return; fi
    NJ=$((NJ+1)); local rcf="$RC_DIR/$NJ" lg="logs/e2e_${PIPE}_$(echo "$*" | tr ' /' '__').log"
    ( CUDA_VISIBLE_DEVICES=$devs bash tracking/scripts/pipe_stage.sh "$ENVN" "$@" > "$lg" 2>&1; echo $? > "$rcf" ) &
  }
  wait_all() {   # espera a todos los trabajos y devuelve 1 si alguno terminó con error (independiente de la versión de bash)
    wait; local rc=0 f
    for f in "$RC_DIR"/*; do [ -e "$f" ] || continue; [ "$(cat "$f")" = "0" ] || rc=1; rm -f "$f"; done
    return $rc
  }
  # Los trabajos se lanzan en OLEADAS de W = NGPU / GPJ a la vez, cada uno con sus GPUs propias (sin solaparse); al llenarse la oleada se espera a que terminen.
  launch() {   # launch <contador_en_grupo> <etapa y args...>
    local n="$1"; shift; local W=$(( NGPU / GPJ > 0 ? NGPU / GPJ : 1 ))
    slot_run $(( n % W )) "$@"
    if [ "${DRY_RUN:-0}" != "1" ] && [ $(( (n + 1) % W )) -eq 0 ]; then wait_all || FAILED=1; fi
  }
  FAILED=0
  # 1) caché y baselines (en paralelo si hay >= 2 GPUs)
  GPJ=1; slot_run 0 cache; slot_run $(( NGPU > 1 ? 1 : 0 )) baseline; wait_all || { echo "falló caché o baseline (ver logs/)"; exit 1; }
  # 2) variantes congeladas: NAMES x SEEDS, una GPU cada una
  GPJ=1; i=0; for N in "${NAMES[@]}"; do for S in $SEEDS; do launch $i variant "$N" "$S"; i=$((i+1)); done; done; wait_all || FAILED=1
  [ "$FAILED" = "1" ] && echo "AVISO: alguna semilla falló (ver logs/); se continúa con las que existan"; FAILED=0
  # 3) barridos por variante y selección
  GPJ=1; i=0; for N in "${NAMES[@]}"; do launch $i sweep "$N" $SEEDS; i=$((i+1)); done; wait_all || FAILED=1
  [ "$FAILED" = "1" ] && { echo "falló un barrido (ver logs/)"; exit 1; }
  run bash tracking/scripts/pipe_stage.sh "$ENVN" select "${NAMES[@]}"
  # 4) descongelado: un lambda por grupo de UNFROZEN_GPUS GPUs (DDP con varias GPUs si UNFROZEN_GPUS > 1)
  GPJ=$UG; i=0; for L in $LAMBDAS; do launch $i unfrozen "$L"; i=$((i+1)); done; wait_all || echo "AVISO: algún entrenamiento descongelado falló o paró (ver logs/)"
  # 5) tabla final
  run bash tracking/scripts/pipe_stage.sh "$ENVN" final
fi
