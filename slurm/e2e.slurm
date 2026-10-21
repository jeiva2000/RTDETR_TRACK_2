#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=e2e_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a30:5             # TODAS las GPUs del trabajo en este nodo (p. ej. gpu:a100:5); las dos etapas usan DDP con todas. Comprueba los nombres con: sinfo -o "%N %G"
##SBATCH --nodelist=gpu02            # opcional: fija el nodo (comprueba antes con: scontrol show node gpu02)
#SBATCH --cpus-per-task=40           # unos 8 por GPU
#SBATCH --mem=160G                   # unos 32G por GPU
#SBATCH -t 72:00:00                  # ajusta al máximo de la partición (la duración de las dos etapas no está medida)
#
# DOS ETAPAS EN UN MISMO TRABAJO, ambas con DDP (torchrun, una GPU por proceso):
#   1. ENTRENAMIENTO DE RT-DETR (detección de personas, 1 clase; train = DanceTrack + CrowdHuman, val = DanceTrack val; solver con early stopping sobre el AP de val)
#      -> $OUTPUT_ROOT/$DET_RUN_NAME/best.pth     (checkpoint de detección). Se salta si ya existe el checkpoint final (marcador, best.pth sin last.pth, o DET_CKPT_EXTERNAL=<ruta>); si se cortó por tiempo, se REANUDA desde last.pth.
#   2. cuando la 1 termina: ENTRENAMIENTO DESCONGELADO del módulo e2e (detector + módulo, tracking/e2e/train_joint.py) partiendo de ese best.pth; cada GPU procesa sus propios clips.
#      El detector se entrena COMPLETO por defecto (decoder, encoder y backbone salvo su primera capa y sus normalizaciones, como RT-DETRv2); --no-train-backbone / --no-train-encoder en JOINT_ARGS los congelan.
#      Se entrenan SECUENCIALMENTE las variantes de E2E_RUN_VARIANTS (por defecto "h1 h2 h3": H1 = solo apariencia, H2 = + GRU de movimiento, H3 = asociación relacional), cada una desde el mismo detector
#      -> $OUTPUT_ROOT/e2e_unfrozen_<variante>/   (detector_best/last.pth, e2e_best/last.pth, log.txt, marcador TRAINING_DONE). Una variante terminada se salta al relanzar si se entrenó con la receta actual (el marcador TRAINING_DONE guarda su firma: E2E_TRAIN_REV, overrides, LAMBDA, CLIPS_PER_GPU y JOINT_ARGS); con otra receta se reentrena sola, y FORCE_E2E=1 la repite siempre;
#         si una falla, las demás se ejecutan igualmente y el trabajo termina con error al final. Otras variantes: tracking/configs/e2e_variants.sh (h2_w01, h2_m64, ...).
#   3. EVALUACIÓN de cada variante entrenada (RUN_EVAL=0 la desactiva): caché de val con SU detector ajustado, ByteTrack y OC-SORT con ese mismo detector (a 0.6 y a su mejor umbral), barrido de
#      los umbrales de nacimiento del módulo e2e y evaluación completa con TrackEval (HOTA, DetA, AssA, IDF1, IDSW; global y en cruces). Una semilla por ejecución.
#      -> $OUTPUT_ROOT/e2e_unfrozen_<variante>/eval/   y la tabla comparativa de todas las variantes en $OUTPUT_ROOT/e2e_unfrozen_comparacion.md
# POR AHORA no se ejecutan las variantes con detector congelado (siguen en tracking/scripts/e2e_pipeline.sh).
#
# USO (sin variables ni pasos previos), desde la RAÍZ del repo (logs/ ya viene en el repo; Slurm la necesita antes de arrancar):
#        cd ~/work/RTDETR_TRACK_2 && git pull && sbatch slurm/e2e.slurm
#   Requisitos: haber editado el bloque CONFIGURACIÓN de abajo y esta cabecera #SBATCH; tener setup_env/paths.<ENV_NAME>.env (copia de la plantilla .example);
#   y los DATOS preparados (Fase 2): COCO de DanceTrack+CrowdHuman convertido y mezclado en $COCO_DIR y el checkpoint COCO en $CKPT_DIR (el script lo comprueba al empezar).
# Con --export=NONE el trabajo no hereda tu entorno: todo se configura aquí. (Alternativa: sbatch --export=ALL,VAR=valor slurm/e2e.slurm; FORCE_DET=1 reentrena el detector.)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?ejecuta con sbatch desde la raíz del repo}"
mkdir -p logs

ENV_NAME="${ENV_NAME:-hpc}"                  # archivo de rutas: setup_env/paths.${ENV_NAME}.env (PROJECT_ROOT, DATA_ROOT, CKPT_DIR, OUTPUT_ROOT, SEED, CONDA_ENV...)
source setup_env/load.sh "$ENV_NAME"
SEED="${E2E_SEED:-${SEED:-42}}"                         # E2E_SEED sustituye a SEED de paths.<env>.env (con --export=ALL,E2E_SEED=2 y otro E2E_RUN_PREFIX se repite todo con otra semilla)

# ============================== CONFIGURACIÓN (EDITA AQUÍ) ==============================
# --- Etapa 0: preparación de datos (solo se ejecuta lo que FALTA; todo se salta si ya está) ---
PREPARE_DATA="${PREPARE_DATA:-1}"                       # 0 = no preparar nada (los datos ya están; solo se comprueba que existan)
USE_CROWDHUMAN="${USE_CROWDHUMAN:-1}"                   # 1 = el detector entrena con DanceTrack + CrowdHuman (como en el plan); 0 = solo DanceTrack (se aparta del plan)
DATASETS_DIR="${DATASETS_DIR:-/home/serperzar/datasets}"   # carpeta con datasets ya descargados (DanceTrack/, CrowdHuman/): si están completos se enlazan en $DATA_ROOT y NO se descarga ('' = no buscar)
DT_STRIDE_TRAIN="${DT_STRIDE_TRAIN:-2}"                 # 1 de cada N frames de train de DanceTrack en el COCO del detector
DT_STRIDE_VAL="${DT_STRIDE_VAL:-1}"                     # idem para val (la validación del detector se hace en cada época: 2 la acorta)
DT_EXPECT_TRAIN1="${DT_EXPECT_TRAIN1:-20}"; DT_EXPECT_TRAIN2="${DT_EXPECT_TRAIN2:-20}"; DT_EXPECT_VAL="${DT_EXPECT_VAL:-25}"   # nº de secuencias de DanceTrack
# --- Etapa 1: entrenamiento de RT-DETR ---
DET_RUN_NAME="${DET_RUN_NAME:-detector_r50vd_person}"   # carpeta de salida en $OUTPUT_ROOT; su best.pth es el checkpoint de detección
DET_EPOCHS="${DET_EPOCHS:-24}"                          # épocas máximas (el early stopping puede parar antes: patience/min_delta en tracking/configs/detector_r50vd_person.yml)
DET_BATCH="${DET_BATCH:-}"                              # lote TOTAL (suma de todas las GPUs); vacío = el menor múltiplo del nº de GPUs que sea >= 16 (el lr no se reescala; sin validar)
DET_NUM_WORKERS="${DET_NUM_WORKERS:-4}"                 # workers del dataloader por GPU
FORCE_DET="${FORCE_DET:-0}"                             # 1 = reentrenar el detector aunque ya exista
DET_CKPT_EXTERNAL="${DET_CKPT_EXTERNAL:-}"              # ruta a un best.pth de detección ya entrenado (de otra carpeta/ejecución): se usa tal cual y NO se entrena el detector
# --- Etapa 2: entrenamiento descongelado del módulo e2e (DDP) ---
E2E_RUN_VARIANTS="${E2E_RUN_VARIANTS:-${E2E_VARIANT:-h1 h2 h3}}"   # variantes de tracking/configs/e2e_variants.sh que se entrenan, EN ESTE ORDEN y una tras otra (h1 = solo apariencia; h2 = + GRU de movimiento; h3 = asociación relacional). E2E_VARIANT=<una> sigue valiendo para una sola
E2E_RUN_PREFIX="${E2E_RUN_PREFIX:-e2e_unfrozen}"        # carpeta de cada variante en $OUTPUT_ROOT: <prefijo>_<variante>
FORCE_E2E="${FORCE_E2E:-0}"                             # 1 = reentrenar también las variantes que ya terminaron (marcador TRAINING_DONE)
E2E_TRAIN_REV="${E2E_TRAIN_REV:-2}"                     # versión de la receta de entrenamiento e2e: 1 = solo decoder; 2 = detector completo (backbone + encoder + decoder). Súbela si cambia la receta: las variantes entrenadas con otra se reentrenan solas
RUN_EVAL="${RUN_EVAL:-1}"                               # 1 = tras entrenar, evaluar cada variante (etapa 3); 0 = solo entrenar
FORCE_EVAL="${FORCE_EVAL:-0}"                           # 1 = repetir la evaluación de las variantes ya evaluadas (marcador eval/EVAL_DONE)
LAMBDA="${LAMBDA:-10}"                                  # peso de la pérdida e2e frente a la de detección (sin ajustar; 0 = solo se ajusta el detector)
CLIPS_PER_GPU="${CLIPS_PER_GPU:-1}"                     # clips por GPU y paso (lote efectivo = clips por GPU x nº de GPUs)
JOINT_ARGS="${JOINT_ARGS:---epochs 40 --samples-per-epoch 1000 --train-backbone --train-encoder}"   # argumentos extra de tracking.e2e.train_joint (--samples-per-epoch cuenta los clips de TODAS las GPUs; --no-train-backbone / --no-train-encoder los congelan)
# --- Entorno del clúster y de conda ---
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CLUSTER_MODULES="${CLUSTER_MODULES:-GCC/12.2.0 CUDA/11.8.0}"
CREATE_ENV="${CREATE_ENV:-1}"                           # 1 = si el entorno conda $CONDA_ENV (de paths.<ENV_NAME>.env) NO existe, crearlo; si existe, solo se activa. 0 = no crearlo nunca
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"                # versiones usadas SOLO al crear el entorno (coinciden con setup_env/setup_hpc.sh)
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"; TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"   # debe corresponder a la versión de CUDA del módulo (11.8 -> cu118)
# ==========================================================================================

# --- Módulos y entorno CUDA/GCC ---
if ! type module >/dev/null 2>&1; then       # `module` no siempre existe en un shell de lotes: se carga su inicialización
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    [ -f "$f" ] && { source "$f"; break; }
  done
fi
module purge
for m in ${CLUSTER_MODULES:-GCC/12.2.0 CUDA/11.8.0}; do module load "$m"; done

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc no está en PATH tras cargar los módulos. Revisa 'module avail CUDA' en este nodo ($(hostname))." >&2
    module avail CUDA 2>&1 >&2 || true
    exit 1
fi

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_PATH=$CUDA_HOME
export CC=$(which gcc)
export CXX=$(which g++)
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS:-8.0;8.6}"   # cubre A30/A100 (8.0) y A40 (8.6); este código no compila extensiones CUDA, es solo por coherencia
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_ASYNC_ERROR_HANDLING=1

net_ok() {   # net_ok <host>: ¿hay acceso a internet? (los nodos de cómputo pueden no tenerlo)
  [ "${NET_CHECK:-1}" = "0" ] || timeout 15 bash -c "exec 3<>/dev/tcp/$1/443" 2>/dev/null
}

# --- Entorno conda: si no existe se crea (con candado, por si coinciden varios trabajos); si existe, solo se activa ---
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; else eval "$(conda shell.bash hook)"; fi
: "${CONDA_ENV:?define CONDA_ENV en setup_env/paths.${ENV_NAME}.env}"
env_exists() {   # CONDA_ENV puede ser un nombre o una ruta absoluta (conda create -p)
  if [[ "$CONDA_ENV" == /* ]]; then [ -d "$CONDA_ENV/conda-meta" ]; else conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; fi
}
create_env() {
  echo "[conda] el entorno '$CONDA_ENV' no existe: se crea (python $PYTHON_VERSION, torch $TORCH_VERSION, requisitos del proyecto)"
  # los nodos de cómputo pueden no tener internet: se comprueba antes de empezar (CREATE_ENV_NET_CHECK=0 lo desactiva)
  if [ "${CREATE_ENV_NET_CHECK:-1}" = "1" ] && ! net_ok pypi.org; then
    echo "ERROR: este nodo no tiene acceso a internet para crear el entorno '$CONDA_ENV'. Créalo antes desde el login (bash setup_env/setup_hpc.sh) y relanza." >&2; exit 1
  fi
  local TARGET=(-n "$CONDA_ENV"); [[ "$CONDA_ENV" == /* ]] && TARGET=(-p "$CONDA_ENV")
  # si algo falla (o el trabajo se cancela) a mitad de la creación, se borra el entorno a medias para no tomarlo luego por bueno
  trap 'echo "[conda] creación interrumpida: se borra el entorno incompleto" >&2; conda env remove -y "${TARGET[@]}" >/dev/null 2>&1 || true; exit 1' ERR TERM INT
  conda create -y "${TARGET[@]}" -c conda-forge --override-channels python="$PYTHON_VERSION" pip   # conda-forge: evita las condiciones de uso de los canales 'defaults'
  conda activate "$CONDA_ENV"
  pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
  pip install -r rtdetrv2_pytorch/requirements.txt "numpy<2"          # numpy<2: torch 2.1.2 no es compatible con NumPy 2
  pip install -r tracking/requirements.txt "numpy<2"                  # filterpy, lap, scipy
  python -c "import torch, torchvision, yaml, scipy, filterpy, lap, pycocotools, faster_coco_eval; print('[conda] paquetes verificados: torch', torch.__version__)"
  trap - ERR TERM INT
  conda deactivate
}
LOCK="${CONDA_ENV_LOCK:-$HOME/.cache/rtdetr_track_env.lock}"; mkdir -p "$(dirname "$LOCK")"
if command -v flock >/dev/null 2>&1; then exec 200>"$LOCK"; flock 200; fi
if env_exists; then
  echo "[conda] el entorno '$CONDA_ENV' ya existe: solo se activa"
elif [ "$CREATE_ENV" = "1" ]; then
  create_env
else
  echo "ERROR: no existe el entorno conda '$CONDA_ENV' y CREATE_ENV=0" >&2; exit 1
fi
conda activate "$CONDA_ENV"
if command -v flock >/dev/null 2>&1; then flock -u 200; fi

# --- Diagnóstico (queda en el log) ---
echo "===== ENV ====="
hostname
date
which python; python --version
which nvcc; nvcc --version
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi || echo "(nvidia-smi no disponible en este nodo)"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
echo "================"

# --- GPUs y comprobaciones de datos ---
NGPU="${SLURM_GPUS_ON_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"; NGPU="${NGPU##*:}"
[ "$NGPU" -ge 1 ] || { echo "ERROR: no hay GPUs visibles" >&2; exit 1; }
[ -n "$DET_BATCH" ] || DET_BATCH=$(( (16 + NGPU - 1) / NGPU * NGPU ))
DET_OUT="$OUTPUT_ROOT/$DET_RUN_NAME"; DET_CKPT="$DET_OUT/best.pth"; DET_DONE="$DET_OUT/TRAINING_DONE"

# --- ¿Hay ya un detector entrenado? (decide si se salta la etapa 1; con DET_SKIP=1 la etapa 0 solo prepara DanceTrack) ---
det_finished() {   # el entrenamiento de $DET_OUT terminó: llegó a DET_EPOCHS o paró por early stopping (log.txt + early_stop.json del solver)
  python - "$DET_OUT" "$DET_EPOCHS" <<'PYEOF'
import json, sys, yaml
out, n_ep = sys.argv[1], int(sys.argv[2])
try:
    last = [json.loads(l) for l in open(out + '/log.txt') if l.strip()][-1]['epoch']
    wait = json.load(open(out + '/early_stop.json'))['wait']
except Exception:
    sys.exit(1)
es = dict(patience=5, min_epochs=6)
es.update((yaml.safe_load(open('tracking/configs/detector_r50vd_person.yml')) or {}).get('early_stopping', {}) or {})
sys.exit(0 if last + 1 >= n_ep or (wait >= es['patience'] and last + 1 >= es['min_epochs']) else 1)
PYEOF
}
DET_SKIP=0
if [ "$FORCE_DET" = "1" ]; then
  echo "[detector] FORCE_DET=1: se reentrena aunque exista"
elif [ -n "$DET_CKPT_EXTERNAL" ]; then
  [ -f "$DET_CKPT_EXTERNAL" ] || { echo "ERROR: DET_CKPT_EXTERNAL=$DET_CKPT_EXTERNAL no existe" >&2; exit 1; }
  DET_CKPT="$DET_CKPT_EXTERNAL"; DET_SKIP=1; echo "[detector] checkpoint externo $DET_CKPT: no se entrena el detector"
elif [ -f "$DET_CKPT" ]; then
  if [ -f "$DET_DONE" ]; then DET_SKIP=1; echo "[detector] $DET_CKPT y marcador $DET_DONE: ya entrenado"
  elif [ ! -f "$DET_OUT/last.pth" ]; then DET_SKIP=1; echo "[detector] $DET_CKPT sin last.pth (un entrenamiento en curso siempre lo tiene): se toma como checkpoint final"
  elif det_finished; then DET_SKIP=1; date '+%Y-%m-%dT%H:%M:%S' > "$DET_DONE"; echo "[detector] $DET_OUT terminó (épocas completas o early stopping) sin marcador: se crea y se salta"
  else echo "[detector] existe $DET_CKPT pero el entrenamiento NO terminó: se reanuda desde last.pth"; fi
elif [ -f "$DET_DONE" ]; then
  echo "[detector] AVISO: existe el marcador $DET_DONE pero no $DET_CKPT: se reentrena"
fi
echo "Job $SLURM_JOB_ID en $(hostname): $NGPU GPU(s) | detector: $DET_CKPT (lote total $DET_BATCH, saltar entrenamiento: $DET_SKIP) | e2e: $E2E_RUN_VARIANTS en $OUTPUT_ROOT/${E2E_RUN_PREFIX}_<variante>"
#export OMP_NUM_THREADS=1

need() { [ -e "$1" ] || { echo "ERROR: falta $1 ($2)" >&2; exit 1; }; }

# ============================== ETAPA 0: preparación de datos (solo lo que falta) ==============================
# Idempotente y con candado: si ya existen (y están completos) no se ejecuta nada. Descargar necesita internet en el nodo (se comprueba).
# Archivos que prepara: DanceTrack en $DANCETRACK_DIR, su COCO (train/val + IDs de track) en $COCO_DIR/dancetrack, CrowdHuman en $CROWDHUMAN_DIR, su COCO en
# $COCO_DIR/crowdhuman y la mezcla $COCO_DIR/detector_train/annotations/train.json (entrenamiento del detector).
count_dirs() { if [ -d "$1" ]; then find "$1" -mindepth 1 -maxdepth 1 -type d | wc -l; else echo 0; fi; }
json_ok() { [ -s "$1" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null; }   # existe y es JSON completo (una escritura cortada no vale)
dancetrack_ok() {   # dancetrack_ok [dir]
  local D="${1:-$DANCETRACK_DIR}"
  [ "$(count_dirs "$D/train1")" -ge "$DT_EXPECT_TRAIN1" ] && [ "$(count_dirs "$D/train2")" -ge "$DT_EXPECT_TRAIN2" ] && \
  [ "$(count_dirs "$D/val")" -ge "$DT_EXPECT_VAL" ] && \
  [ "$(ls "$D"/train1/*/gt/gt.txt "$D"/train2/*/gt/gt.txt "$D"/val/*/gt/gt.txt 2>/dev/null | wc -l)" -ge $((DT_EXPECT_TRAIN1 + DT_EXPECT_TRAIN2 + DT_EXPECT_VAL)) ]
}
crowdhuman_ok() {   # crowdhuman_ok [dir]
  local D="${1:-$CROWDHUMAN_DIR}"
  [ -s "$D/annotation_train.odgt" ] && [ -d "$D/Images" ] && \
  [ "$(find "$D/Images" -maxdepth 1 -name '*.jpg' | wc -l)" -ge $(( $(wc -l < "$D/annotation_train.odgt") * 99 / 100 )) ]   # >= 99 % de las imágenes del .odgt
}
link_existing() {   # link_existing <ok_fn> <destino> <nombre...>: si hay una copia completa en $DATASETS_DIR y el destino no existe, la enlaza
  local ok="$1" dest="$2"; shift 2
  [ -n "$DATASETS_DIR" ] && [ ! -e "$dest" ] && [ ! -L "$dest" ] || return 0
  for n in "$@"; do
    if [ -d "$DATASETS_DIR/$n" ] && "$ok" "$DATASETS_DIR/$n"; then
      echo "[datos] $n ya existe y está completo en $DATASETS_DIR/$n: se enlaza en $dest (sin descargar)"
      mkdir -p "$(dirname "$dest")"; ln -s "$DATASETS_DIR/$n" "$dest"; return 0
    fi
  done
}
prepare_data() {
  link_existing dancetrack_ok "$DANCETRACK_DIR" DanceTrack dancetrack
  [ "$USE_CROWDHUMAN" = "1" ] && [ "$DET_SKIP" != "1" ] && link_existing crowdhuman_ok "$CROWDHUMAN_DIR" CrowdHuman crowdhuman
  if ! dancetrack_ok; then
    echo "[datos] DanceTrack ausente o incompleto en $DANCETRACK_DIR: se descarga"
    net_ok huggingface.co || { echo "ERROR: sin internet en este nodo para descargar DanceTrack. Descárgalo antes desde el login: bash setup_env/download_dancetrack.sh $ENV_NAME" >&2; exit 1; }
    eval "${DT_DOWNLOAD_CMD:-bash setup_env/download_dancetrack.sh $ENV_NAME}"
    dancetrack_ok || { echo "ERROR: DanceTrack sigue incompleto tras la descarga (¿extracción interrumpida?). Borra la carpeta incompleta de $DANCETRACK_DIR y relanza." >&2; exit 1; }
  fi
  [ "$DET_SKIP" = "1" ] && { echo "[datos] detector ya entrenado: solo se necesita DanceTrack (sin COCO, CrowdHuman ni mezcla)"; return 0; }
  if ! { json_ok "$COCO_DIR/dancetrack/annotations/train.json" && json_ok "$COCO_DIR/dancetrack/annotations/val.json"; }; then
    echo "[datos] convirtiendo DanceTrack a COCO (stride train $DT_STRIDE_TRAIN, val $DT_STRIDE_VAL)"
    python tracking/data/convert_dancetrack.py --stride-train "$DT_STRIDE_TRAIN" --stride-val "$DT_STRIDE_VAL"
  fi
  INPUTS=("$COCO_DIR/dancetrack/annotations/train.json:$DANCETRACK_DIR")
  if [ "$USE_CROWDHUMAN" = "1" ]; then
    if ! crowdhuman_ok; then
      echo "[datos] CrowdHuman ausente o incompleto en $CROWDHUMAN_DIR: se descarga (train completo, ~8 GB)"
      net_ok huggingface.co || { echo "ERROR: sin internet en este nodo para descargar CrowdHuman. Descárgalo antes desde el login: bash setup_env/download_crowdhuman.sh $ENV_NAME full (o USE_CROWDHUMAN=0)" >&2; exit 1; }
      eval "${CH_DOWNLOAD_CMD:-bash setup_env/download_crowdhuman.sh $ENV_NAME full}"
      crowdhuman_ok || { echo "ERROR: CrowdHuman sigue incompleto tras la descarga. Revisa $CROWDHUMAN_DIR (o USE_CROWDHUMAN=0)." >&2; exit 1; }
    fi
    if ! json_ok "$COCO_DIR/crowdhuman/annotations/train.json"; then
      echo "[datos] convirtiendo CrowdHuman a COCO (las imágenes muy grandes se reducen)"
      python tracking/data/convert_crowdhuman.py
    fi
    INPUTS+=("$COCO_DIR/crowdhuman/annotations/train.json:$CROWDHUMAN_DIR")
  fi
  MERGED="$COCO_DIR/detector_train/annotations/train.json"; REBUILD=0
  json_ok "$MERGED" || REBUILD=1
  for i in "${INPUTS[@]}"; do [ "${i%%:*}" -nt "$MERGED" ] && REBUILD=1; done          # si una entrada es más nueva que la mezcla, se rehace
  if [ "$REBUILD" = "1" ]; then
    echo "[datos] mezclando los COCO de entrenamiento del detector (${#INPUTS[@]} fuente(s)) -> $MERGED"
    python tracking/data/merge_coco.py --base "$DATA_ROOT" --out "$MERGED" --inputs "${INPUTS[@]}"
  fi
}
if [ "$PREPARE_DATA" = "1" ]; then
  mkdir -p "$DATA_ROOT"; exec 201>"$DATA_ROOT/.data_prep.lock"
  command -v flock >/dev/null 2>&1 && flock 201                                            # un solo trabajo prepara los datos a la vez
  prepare_data
  command -v flock >/dev/null 2>&1 && flock -u 201
else
  echo "[datos] PREPARE_DATA=0: no se prepara nada, solo se comprueba"
fi
if [ "$DET_SKIP" != "1" ]; then
  need "$COCO_DIR/detector_train/annotations/train.json" "COCO de train = DanceTrack + CrowdHuman mezclado: python tracking/data/merge_coco.py (Fase 2)"
  need "$COCO_DIR/dancetrack/annotations/val.json" "COCO de val de DanceTrack: python tracking/data/convert_dancetrack.py (Fase 2)"
  if [ ! -f "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" ]; then   # checkpoint COCO de partida; los nodos de cómputo pueden no tener internet
    bash setup_env/download_ckpt.sh "$ENV_NAME" || true
  fi
  need "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" "checkpoint COCO: bash setup_env/download_ckpt.sh $ENV_NAME (desde el login, con internet)"
fi
need "$DANCETRACK_DIR/train1" "DanceTrack (train1, train2, val) para la etapa 2"; need "$DANCETRACK_DIR/val" "DanceTrack val"

# ============================== ETAPA 1: entrenamiento de RT-DETR ==============================
if [ "$DET_SKIP" = "1" ]; then
  need "$DET_CKPT" "checkpoint de detección"
  echo "[etapa 1] detector ya entrenado ($DET_CKPT): se salta"
else
  echo "[etapa 1] entrenando RT-DETR con $NGPU GPU(s)"
  RESUME_FLAG=0; [ -f "$DET_OUT/last.pth" ] && [ "$FORCE_DET" != "1" ] && RESUME_FLAG=1   # sin FORCE_DET, un last.pth sin terminar se reanuda
  # PResNet.pretrained=False: el backbone se carga del checkpoint COCO, no hace falta descargar pesos ImageNet (los nodos pueden no tener internet)
  NGPU=$NGPU RUN_NAME="$DET_RUN_NAME" EPOCHS="$DET_EPOCHS" BATCH="$DET_BATCH" NUM_WORKERS="$DET_NUM_WORKERS" RESUME=$RESUME_FLAG \
    bash tracking/scripts/train_detector.sh "$ENV_NAME" PResNet.pretrained=False
  [ -f "$DET_CKPT" ] || { echo "ERROR: el entrenamiento del detector terminó pero no existe $DET_CKPT" >&2; exit 1; }
  date '+%Y-%m-%dT%H:%M:%S' > "$DET_DONE"
  echo "[etapa 1] terminada: $DET_CKPT"
fi

# ============================== ETAPA 2: entrenamiento descongelado del módulo e2e (DDP), una variante tras otra ==============================
source tracking/configs/e2e_variants.sh
overrides_of() {   # overrides de una variante de e2e_variants.sh; falla si no existe
  local v
  for v in "${E2E_VARIANTS[@]}"; do [ "${v%%|*}" = "$1" ] && { echo "${v#*|}" | sed 's/#.*//'; return 0; }; done
  echo "ERROR: variante desconocida: $1 (ver tracking/configs/e2e_variants.sh)" >&2; return 1
}
for V in $E2E_RUN_VARIANTS; do overrides_of "$V" >/dev/null || exit 1; done     # se comprueban todas antes de empezar a entrenar
echo "[etapa 2] variantes en orden: $E2E_RUN_VARIANTS | $NGPU GPU(s), $CLIPS_PER_GPU clip(s) por GPU (lote efectivo $((CLIPS_PER_GPU * NGPU))), lambda $LAMBDA"
FAILED=(); DONE_NOW=(); SKIPPED=()
for V in $E2E_RUN_VARIANTS; do
  OV=$(overrides_of "$V"); E2E_OUT="$OUTPUT_ROOT/${E2E_RUN_PREFIX}_$V"
  SIG="rev=$E2E_TRAIN_REV variant=$V overrides=[$OV] lambda=$LAMBDA clips_per_gpu=$CLIPS_PER_GPU joint_args=[${JOINT_ARGS:-}]"   # receta de entrenamiento (sin nº de GPUs)
  OLD_SIG=$(sed -n 2p "$E2E_OUT/TRAINING_DONE" 2>/dev/null || true)                                                              # los marcadores antiguos no tienen firma: receta 1
  if [ -f "$E2E_OUT/TRAINING_DONE" ] && [ "$FORCE_E2E" != "1" ] && [ "$OLD_SIG" = "$SIG" ]; then
    echo "[etapa 2:$V] ya entrenada con esta receta ($E2E_OUT/TRAINING_DONE): se salta"; SKIPPED+=("$V"); continue
  fi
  if [ -f "$E2E_OUT/TRAINING_DONE" ] && [ "$FORCE_E2E" != "1" ]; then
    echo "[etapa 2:$V] entrenada con OTRA receta (marcador: ${OLD_SIG:-sin firma = receta 1, solo decoder}; ahora: $SIG): se reentrena"
  fi
  if [ -e "$E2E_OUT" ]; then   # el entrenamiento e2e no tiene reanudación: una ejecución previa (a medias o con otra receta) se aparta (no se borra) y se empieza de cero
    mv "$E2E_OUT" "${E2E_OUT}_previo_$SLURM_JOB_ID"; echo "[etapa 2:$V] AVISO: había una ejecución previa; se aparta en ${E2E_OUT}_previo_$SLURM_JOB_ID"
  fi
  mkdir -p "$E2E_OUT"
  MASTER_PORT=$((10000 + RANDOM % 20000))
  echo "[etapa 2:$V] $(date '+%F %T') entrenando (overrides: ${OV:-ninguno}) -> $E2E_OUT"
  # shellcheck disable=SC2086
  if torchrun --master_port="$MASTER_PORT" --nproc_per_node="$NGPU" \
      -m tracking.e2e.train_joint --det-ckpt "$DET_CKPT" --out "$E2E_OUT" --seed "$SEED" --lambda-e2e "$LAMBDA" \
      --clips-per-gpu "$CLIPS_PER_GPU" ${JOINT_ARGS:-} ${OV:+-u $OV}; then
    { date '+%Y-%m-%dT%H:%M:%S'; echo "$SIG"; } > "$E2E_OUT/TRAINING_DONE"; DONE_NOW+=("$V"); echo "[etapa 2:$V] terminada: $E2E_OUT"
  else
    RC=$?; FAILED+=("$V"); echo "[etapa 2:$V] ERROR: el entrenamiento falló (código $RC); se continúa con la siguiente variante" >&2
  fi
done
echo "[etapa 2] resumen: entrenadas=[${DONE_NOW[*]:-}] saltadas=[${SKIPPED[*]:-}] fallidas=[${FAILED[*]:-}]"

# ============================== ETAPA 3: evaluación de cada variante (HOTA, DetA, AssA, IDF1, IDSW; global y cruces) ==============================
# Por variante: (1) caché de val con su detector ajustado (detector_best.pth), (2) ByteTrack y OC-SORT con ESE detector (a 0.6 y barrido de track_thresh),
# (3) barrido de umbrales de nacimiento del módulo e2e (e2e_best.pth) y evaluación completa con el mejor. Los umbrales se eligen sobre val y se reportan sobre val
# (optimista), con la misma regla para todos los métodos. Cada paso es idempotente; un fallo en una variante no impide evaluar las demás.
eval_variant() {   # eval_variant <variante>; con set -e desactivado dentro de funciones llamadas con ||, cada paso lleva "|| return 1"
  local V="$1" OUT EV RUN VAL
  OUT="$OUTPUT_ROOT/${E2E_RUN_PREFIX}_$V"; EV="$OUT/eval"; RUN="$EV/seed_run"; VAL="$DANCETRACK_DIR/val"
  if [ -f "$EV/EVAL_DONE" ] && [ "$FORCE_EVAL" != "1" ]; then echo "[etapa 3:$V] ya evaluada ($EV/EVAL_DONE): se salta"; return 0; fi
  if [ ! -f "$OUT/detector_best.pth" ] || [ ! -f "$OUT/e2e_best.pth" ]; then
    echo "[etapa 3:$V] ERROR: faltan detector_best.pth / e2e_best.pth en $OUT (¿ninguna época válida o no entrenada?)" >&2; return 1
  fi
  mkdir -p "$EV"
  echo "[etapa 3:$V] $(date '+%F %T') caché de val con el detector ajustado"
  if [ ! -f "$OUT/features/val/stats.json" ]; then
    python -m tracking.video.cache_features --ckpt "$OUT/detector_best.pth" --split val --min-iou 0.5 --out "$OUT/features/val" || return 1
  fi
  echo "[etapa 3:$V] $(date '+%F %T') ByteTrack y OC-SORT con el mismo detector"
  DET_CKPT="$OUT/detector_best.pth" FEAT_DIR="$OUT/features" PIPE_NAME="${E2E_RUN_PREFIX}_$V/eval" FORCE="$FORCE_EVAL" \
    bash tracking/scripts/pipe_stage.sh "$ENV_NAME" baseline || return 1
  echo "[etapa 3:$V] $(date '+%F %T') barrido de umbrales de nacimiento y evaluación del módulo e2e"
  mkdir -p "$RUN"; cp "$OUT/e2e_best.pth" "$RUN/best.pth" || return 1
  python -m tracking.e2e.sweep run --runs "$RUN" --ckpt-name best.pth --cache "$OUT/features/val" --gt-root "$VAL" --out "$EV/sweep" --sweep-seeds 1 || return 1
  date '+%Y-%m-%dT%H:%M:%S' > "$EV/EVAL_DONE"
  echo "[etapa 3:$V] terminada: $RUN/eval.json"
}
best_baseline() {   # best_baseline <dir sweep> <tracker>: ruta del JSON con mayor HOTA global entre los umbrales probados
  python3 - "$1" "$2" <<'PYEOF'
import glob, json, sys
best = max(((json.load(open(f))['global_']['HOTA'], f) for f in glob.glob(f'{sys.argv[1]}/{sys.argv[2]}_*.json')), default=None)
print(best[1] if best else '')
PYEOF
}
comparison_table() {
  local ARGS=() V EV T F N TABLE="$OUTPUT_ROOT/${E2E_RUN_PREFIX}_comparacion.md"
  for V in $E2E_RUN_VARIANTS; do
    EV="$OUTPUT_ROOT/${E2E_RUN_PREFIX}_$V/eval"
    [ -f "$EV/seed_run/eval.json" ] || continue
    ARGS+=("${V}_e2e=$EV/seed_run/eval.json")
    for T in bytetrack ocsort; do
      [ -f "$EV/baseline/$T.json" ] && ARGS+=("${V}_${T}_0.6=$EV/baseline/$T.json")
      F=$(best_baseline "$EV/baseline/sweep" "$T")                                         # mejor umbral por HOTA; si es el 0.6 ya está en la fila anterior (no se duplica)
      if [ -n "$F" ] && [ "$(basename "$F" .json)" != "${T}_0.6" ]; then N=$(basename "$F" .json); ARGS+=("${V}_${T}_mejor_${N#${T}_}=$F"); fi   # p. ej. h1_bytetrack_mejor_0.7
    done
  done
  [ "${#ARGS[@]}" -gt 0 ] || { echo "[etapa 3] no hay evaluaciones que tabular"; return 1; }
  { python -m tracking.baseline.make_table "${ARGS[@]}"
    echo; echo "Configuración de inferencia elegida para el módulo e2e (barrido sobre val; una semilla por variante):"
    for V in $E2E_RUN_VARIANTS; do
      F="$OUTPUT_ROOT/${E2E_RUN_PREFIX}_$V/eval/seed_run/best_infer.json"
      [ -f "$F" ] && echo "- $V: $(cat "$F")"
    done; } | tee "$TABLE"
  echo "[etapa 3] tabla comparativa: $TABLE"
}
EVAL_FAILED=()
if [ "$RUN_EVAL" = "1" ]; then
  echo "[etapa 3] evaluación de: $E2E_RUN_VARIANTS"
  for V in $E2E_RUN_VARIANTS; do
    eval_variant "$V" || { EVAL_FAILED+=("$V"); echo "[etapa 3:$V] ERROR: la evaluación falló; se continúa con la siguiente" >&2; }
  done
  comparison_table || true
  echo "[etapa 3] resumen: evaluaciones fallidas=[${EVAL_FAILED[*]:-}]"
else
  echo "[etapa 3] RUN_EVAL=0: no se evalúa"
fi
[ "${#FAILED[@]}" -eq 0 ] && [ "${#EVAL_FAILED[@]}" -eq 0 ] || exit 1
