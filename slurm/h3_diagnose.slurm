#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=h3_diagnose
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # una GPU basta (no hay entrenamiento ni DDP); ajusta el tipo si hace falta (p. ej. gpu:a40:1). Comprueba los nombres con: sinfo -o "%N %G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 06:00:00                  # el diagnóstico recorre la caché de val dos veces y el re-barrido, una por configuración (sin medir)
#
# DIAGNÓSTICO DEL H3: dónde falla la asociación. NO reentrena nada: usa lo que ya dejó slurm/e2e.slurm en $OUTPUT_ROOT/e2e_unfrozen_<variante>/
# (e2e_best.pth, detector_best.pth y, si existe, features/val).
#   0. Resumen de los log.txt del entrenamiento de las variantes (val_acc, épocas, parada por DetA): ¿aprendió H3 a asociar tanto como H1/H2?
#   1. Caché de val con el detector ajustado de la variante (solo si falta).
#   2. tracking.e2e.h3.diagnose sobre la caché de val, con el ID GT que guarda cada query, en dos modos:
#        closed  = el modelo y su memoria como en inferencia: acierto de las decisiones, top-1 de cada señal (apariencia, movimiento, relacional, gates, Sinkhorn), gates,
#                  curva de la probabilidad frente al umbral h3.min_prob, caja predicha frente a la última caja, causas de los cambios de ID y nacimientos, IDs creados;
#        teacher = la memoria se actualiza con la asignación GT (estado siempre limpio): calidad pura de la decisión, sin acumulación de errores.
#      Comparar closed con teacher separa «el modelo decide mal» de «los errores se acumulan». Al final del informe van pistas heurísticas (orientan, no demuestran).
#   2b. Segundo diagnóstico, solo closed, con los mecanismos contra el bucle de duplicados activos (DIAG_LB_OVERRIDES): compara IDs creados, cambios de ID y acierto con y sin ellos.
#   3. Re-barrido con métricas reales (HOTA, AssA, IDF1, IDSW) variando h3.min_prob, h3.new_thr, h3.birth.thr (cabeza de nacimiento) y los mecanismos contra duplicados (guarda de nacimientos h3.birth_guard_iou y
#      eliminación del track más joven h3.dedupe_iou, tracking/e2e/h3/loop_breakers.py) sobre una COPIA de la evaluación (no pisa eval/seed_run); DIAG_RESWEEP=0 lo omite.
# Salidas: $OUTPUT_ROOT/e2e_unfrozen_<variante>/diagnose/{report.md,report.json,sweep/...} y una copia del informe en $OUTPUT_ROOT/h3_diagnose_report.md (también en el log del trabajo).
#
# USO, desde la RAÍZ del repo:   cd ~/work/RTDETR_TRACK_2 && git pull && sbatch slurm/h3_diagnose.slurm
#   Requisitos: haber ejecutado antes slurm/e2e.slurm con la variante (h3 por defecto) y que exista el entorno conda de paths.<ENV_NAME>.env (aquí no se crea).
# Con --export=NONE el trabajo no hereda tu entorno. Variables con: sbatch --export=ALL,DIAG_VARIANT=h3,DIAG_MIN_PROB="0.1 0.3" slurm/h3_diagnose.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?ejecuta con sbatch desde la raíz del repo}"
mkdir -p logs

ENV_NAME="${ENV_NAME:-hpc}"                  # archivo de rutas: setup_env/paths.${ENV_NAME}.env (OUTPUT_ROOT, DANCETRACK_DIR, CONDA_ENV...)
source setup_env/load.sh "$ENV_NAME"

# ============================== CONFIGURACIÓN (EDITA AQUÍ) ==============================
DIAG_VARIANT="${DIAG_VARIANT:-h3}"                      # variante de H3 a diagnosticar (e2e_unfrozen_<variante>); debe ser una variante con h3.enabled=true
DIAG_COMPARE="${DIAG_COMPARE:-h1 h2 h3}"                # variantes cuyos log.txt se resumen en el paso 0 para comparar el entrenamiento
E2E_RUN_PREFIX="${E2E_RUN_PREFIX:-e2e_unfrozen}"        # prefijo de las carpetas de e2e.slurm (el mismo que se usó allí)
DIAG_MODES="${DIAG_MODES:-closed teacher}"              # modos del diagnóstico (closed teacher)
DIAG_SEQS="${DIAG_SEQS:-}"                              # secuencias de val a usar (vacío = todas); p. ej. "dancetrack0004 dancetrack0005" para una prueba rápida
DIAG_MAX_SEQS="${DIAG_MAX_SEQS:-0}"                     # 0 = todas; N = solo las N primeras (prueba rápida)
DIAG_OVERRIDES="${DIAG_OVERRIDES:-}"                    # sobrescribe claves de la configuración del checkpoint en el diagnóstico, p. ej. "h3.min_prob=0.3 h3.new_thr=0.5"
DIAG_RESWEEP="${DIAG_RESWEEP:-1}"                       # 1 = re-barrido con métricas reales (paso 3); 0 = solo el diagnóstico
DIAG_MIN_PROB="${DIAG_MIN_PROB:-0.5}"                   # umbrales de probabilidad h3.min_prob del re-barrido (el evaluado en e2e.slurm fue 0.5)
DIAG_SCORES="${DIAG_SCORES:-0.6}"                       # umbrales de score de nacimiento h3.new_thr del re-barrido
DIAG_BIRTH_THR="${DIAG_BIRTH_THR:-0.1 0.3 0.5}"           # h3.birth.thr del re-barrido: probabilidad mínima de la cabeza de nacimiento para crear un track (solo si el modelo la tiene)
DIAG_GUARD="${DIAG_GUARD:-0 0.3}"                   # h3.birth_guard_iou del re-barrido: no nace una detección cubierta por un track vivo (IoU >= valor); 0 = apagado
DIAG_DEDUPE="${DIAG_DEDUPE:-0}"                    # h3.dedupe_iou del re-barrido: se elimina el track más joven de dos solapados (IoU >= valor); 0 = apagado. Solo se prueba junto con la guarda
DIAG_DET_THR="${DIAG_DET_THR:-}"                        # h3.det_thr de la inferencia en el re-barrido (vacío = el del entrenamiento, 0.3); p. ej. "0.3 0.5"
DIAG_LB_OVERRIDES="${DIAG_LB_OVERRIDES-h3.birth_guard_iou=0.3 h3.dedupe_iou=0.85}"   # paso 2b: configuración con los mecanismos activos; vacío (DIAG_LB_OVERRIDES=) omite el paso
# --- Entorno del clúster y de conda (aquí NO se crea el entorno: lo crea slurm/e2e.slurm) ---
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CLUSTER_MODULES="${CLUSTER_MODULES:-GCC/12.2.0 CUDA/11.8.0}"
# ==========================================================================================

if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    [ -f "$f" ] && { source "$f"; break; }
  done
fi
module purge
for m in $CLUSTER_MODULES; do module load "$m"; done
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")"); export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; else eval "$(conda shell.bash hook)"; fi
: "${CONDA_ENV:?define CONDA_ENV en setup_env/paths.${ENV_NAME}.env}"
conda activate "$CONDA_ENV" || { echo "ERROR: no existe el entorno conda '$CONDA_ENV': lánzalo antes con slurm/e2e.slurm (lo crea) o con setup_env/setup_hpc.sh" >&2; exit 1; }
echo "===== ENV ====="; hostname; date; which python; python --version
python -c "import torch; print('torch:', torch.__version__, '| cuda disponible:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
echo "================"

# ============================== PASOS ==============================
RUN="$OUTPUT_ROOT/${E2E_RUN_PREFIX}_$DIAG_VARIANT"; DIAG="$RUN/diagnose"; VAL="$DANCETRACK_DIR/val"
[ -f "$RUN/e2e_best.pth" ] || { echo "ERROR: falta $RUN/e2e_best.pth: entrena antes la variante '$DIAG_VARIANT' con slurm/e2e.slurm (E2E_RUN_VARIANTS=$DIAG_VARIANT)" >&2; exit 1; }
[ -d "$VAL" ] || { echo "ERROR: falta $VAL (DANCETRACK_DIR en paths.${ENV_NAME}.env)" >&2; exit 1; }
mkdir -p "$DIAG"
echo "[diagnóstico] variante $DIAG_VARIANT en $RUN"

echo; echo "[paso 0] resumen de los entrenamientos (mejor época válida por val_acc)"
RUNS=(); for V in $DIAG_COMPARE; do RUNS+=("$OUTPUT_ROOT/${E2E_RUN_PREFIX}_$V"); done
python -m tracking.e2e.h3.diagnose logs --runs "${RUNS[@]}" | tee "$DIAG/training_logs.md"

echo; echo "[paso 1] caché de val con el detector ajustado"
FEAT="$RUN/features/val"
if [ -f "$FEAT/stats.json" ]; then
  echo "[ok] ya existe $FEAT"
else
  [ -f "$RUN/detector_best.pth" ] || { echo "ERROR: falta $RUN/detector_best.pth para crear la caché de val" >&2; exit 1; }
  python -m tracking.video.cache_features --ckpt "$RUN/detector_best.pth" --split val --min-iou 0.5 --out "$FEAT"
fi

echo; echo "[paso 2] diagnóstico (modos: $DIAG_MODES)"
SEQARGS=(); [ -n "$DIAG_SEQS" ] && SEQARGS+=(--seqs $DIAG_SEQS); [ "$DIAG_MAX_SEQS" != "0" ] && SEQARGS+=(--max-seqs "$DIAG_MAX_SEQS")
python -m tracking.e2e.h3.diagnose run --ckpt "$RUN/e2e_best.pth" --cache "$FEAT" --out "$DIAG" --modes $DIAG_MODES ${SEQARGS[@]+"${SEQARGS[@]}"} ${DIAG_OVERRIDES:+-u $DIAG_OVERRIDES}
{ echo; echo "## Entrenamiento (log.txt)"; echo; cat "$DIAG/training_logs.md"; } >> "$DIAG/report.md"

if [ -n "$DIAG_LB_OVERRIDES" ]; then
  echo; echo "[paso 2b] diagnóstico en closed con los mecanismos contra duplicados: $DIAG_LB_OVERRIDES"
  python -m tracking.e2e.h3.diagnose run --ckpt "$RUN/e2e_best.pth" --cache "$FEAT" --out "$DIAG/with_loop_breakers" --modes closed ${SEQARGS[@]+"${SEQARGS[@]}"} -u ${DIAG_OVERRIDES:-} $DIAG_LB_OVERRIDES > /dev/null
  python - "$DIAG/report.json" "$DIAG/with_loop_breakers/report.json" "$DIAG_LB_OVERRIDES" <<'PYEOF' | tee -a "$DIAG/report.md"
import json, sys
a, b = (json.load(open(f)).get('closed') for f in sys.argv[1:3])
if a is None:
    sys.exit(0)
def row(name, r):
    d, ic, ids, bi, lb = r['decisions_present'], r['id_changes'], r['ids'], r['births'], r['loop_breakers']
    return (f"| {name} | {ids['predicted']} ({ids['ratio']:.1f}x) | {ic['total']} | {d['correct']:.1%} | {d['missed']:.1%} | {d['swap_to_other_id']:.1%} | {bi.get('lost_no_slot', 0)} | "
            f"{r['detections']['tracks_alive_per_frame']:.1f} | {lb['births_guarded']} / {lb['dedupe_dropped']} |")
print('\n## Efecto de los mecanismos contra duplicados (modo closed, mismas decisiones del modelo)\n')
print(f'Configuración añadida: `{sys.argv[3]}`\n')
print('| | IDs predichos (frente al GT) | cambios de ID | tracks correctos | rechazados | enlace a otra identidad | nacimientos perdidos (sin hueco) | tracks vivos por frame | guardados / eliminados |')
print('|---|---|---|---|---|---|---|---|---|')
print(row('como se entrenó/evaluó', a)); print(row('con mecanismos', b))
PYEOF
  cp "$DIAG/report.md" "$OUTPUT_ROOT/h3_diagnose_report.md"
fi
cp "$DIAG/report.md" "$OUTPUT_ROOT/h3_diagnose_report.md"

if [ "$DIAG_RESWEEP" = "1" ]; then
  echo; echo "[paso 3] re-barrido con métricas reales: h3.min_prob = $DIAG_MIN_PROB × h3.new_thr = $DIAG_SCORES × guarda = $DIAG_GUARD × dedupe = $DIAG_DEDUPE × birth.thr = $DIAG_BIRTH_THR (sobre una copia: no se pisa eval/seed_run)"
  SR="$RUN/eval/seed_run_diag"; rm -rf "$SR"; mkdir -p "$SR"; cp "$RUN/e2e_best.pth" "$SR/best.pth"
  python -m tracking.e2e.sweep run --runs "$SR" --ckpt-name best.pth --cache "$FEAT" --gt-root "$VAL" --out "$DIAG/sweep" --sweep-seeds 1 --min-prob $DIAG_MIN_PROB --score $DIAG_SCORES --guard-iou $DIAG_GUARD --dedupe-iou $DIAG_DEDUPE --birth-thr $DIAG_BIRTH_THR ${DIAG_DET_THR:+--det-thr $DIAG_DET_THR}
  python - "$DIAG/sweep/sweep.json" "$RUN/eval/seed_run/best_infer.json" <<'PYEOF' | tee -a "$DIAG/report.md" | tee -a "$OUTPUT_ROOT/h3_diagnose_report.md"
import json, os, sys
res = json.load(open(sys.argv[1]))
print('\n## Re-barrido con métricas reales (ordenado por HOTA; una semilla; umbrales elegidos y reportados sobre val)\n')
print('| h3.min_prob | h3.new_thr | birth.thr | guarda IoU | dedupe IoU | det_thr | HOTA | DetA | AssA | IDF1 | IDSW |\n|---|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(res, key=lambda r: -r['mean']['HOTA']):
    c, m = r['cfg'], r['mean']
    print(f"| {c['birth_thr']} | {c['min_score']} | {c['bthr'] if c.get('bthr') is not None else '—'} | {c.get('guard') or 'off'} | {c.get('dedupe') or 'off'} | {c['det_thr'] if c.get('det_thr') is not None else 'entrenado'} | "
          f"{m['HOTA']:.1f} | {m['DetA']:.1f} | {m['AssA']:.1f} | {m['IDF1']:.1f} | {m['IDSW']:.0f} |")
if os.path.isfile(sys.argv[2]):
    print('\nConfiguración elegida en la evaluación de e2e.slurm (referencia):', open(sys.argv[2]).read())
PYEOF
else
  echo "[paso 3] DIAG_RESWEEP=0: no se hace el re-barrido"
fi

echo; echo "================ INFORME ================"
cat "$OUTPUT_ROOT/h3_diagnose_report.md"
echo "[diagnóstico] listo: $DIAG/report.md  (copia: $OUTPUT_ROOT/h3_diagnose_report.md)"
