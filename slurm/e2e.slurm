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
#      -> $OUTPUT_ROOT/$DET_RUN_NAME/best.pth     (checkpoint de detección). Se salta si ya está hecho (marcador); si se cortó por tiempo, se REANUDA desde last.pth.
#   2. cuando la 1 termina: ENTRENAMIENTO DESCONGELADO del módulo e2e (decoder + módulo, tracking/e2e/train_joint.py) partiendo de ese best.pth; cada GPU procesa sus propios clips.
#      -> $OUTPUT_ROOT/$E2E_RUN_NAME/   (detector_best/last.pth, e2e_best/last.pth, log.txt).
# POR AHORA no se ejecutan las variantes con detector congelado, el barrido de umbrales ni las evaluaciones (siguen en tracking/scripts/e2e_pipeline.sh).
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

# ============================== CONFIGURACIÓN (EDITA AQUÍ) ==============================
# --- Etapa 1: entrenamiento de RT-DETR ---
DET_RUN_NAME="${DET_RUN_NAME:-detector_r50vd_person}"   # carpeta de salida en $OUTPUT_ROOT; su best.pth es el checkpoint de detección
DET_EPOCHS="${DET_EPOCHS:-24}"                          # épocas máximas (el early stopping puede parar antes: patience/min_delta en tracking/configs/detector_r50vd_person.yml)
DET_BATCH="${DET_BATCH:-}"                              # lote TOTAL (suma de todas las GPUs); vacío = el menor múltiplo del nº de GPUs que sea >= 16 (el lr no se reescala; sin validar)
DET_NUM_WORKERS="${DET_NUM_WORKERS:-4}"                 # workers del dataloader por GPU
FORCE_DET="${FORCE_DET:-0}"                             # 1 = reentrenar el detector aunque ya exista
# --- Etapa 2: entrenamiento descongelado del módulo e2e (DDP) ---
E2E_RUN_NAME="${E2E_RUN_NAME:-e2e_unfrozen}"            # carpeta de salida en $OUTPUT_ROOT
E2E_VARIANT="${E2E_VARIANT:-h1}"                        # variante de tracking/configs/e2e_variants.sh
LAMBDA="${LAMBDA:-10}"                                  # peso de la pérdida e2e frente a la de detección (sin ajustar; 0 = solo se ajusta el detector)
CLIPS_PER_GPU="${CLIPS_PER_GPU:-1}"                     # clips por GPU y paso (lote efectivo = clips por GPU x nº de GPUs)
JOINT_ARGS="${JOINT_ARGS:---epochs 40 --samples-per-epoch 1000}"   # argumentos extra de tracking.e2e.train_joint (--samples-per-epoch cuenta los clips de TODAS las GPUs)
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

# --- Entorno conda: si no existe se crea (con candado, por si coinciden varios trabajos); si existe, solo se activa ---
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; else eval "$(conda shell.bash hook)"; fi
: "${CONDA_ENV:?define CONDA_ENV en setup_env/paths.${ENV_NAME}.env}"
env_exists() {   # CONDA_ENV puede ser un nombre o una ruta absoluta (conda create -p)
  if [[ "$CONDA_ENV" == /* ]]; then [ -d "$CONDA_ENV/conda-meta" ]; else conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; fi
}
create_env() {
  echo "[conda] el entorno '$CONDA_ENV' no existe: se crea (python $PYTHON_VERSION, torch $TORCH_VERSION, requisitos del proyecto)"
  # los nodos de cómputo pueden no tener internet: se comprueba antes de empezar (CREATE_ENV_NET_CHECK=0 lo desactiva)
  if [ "${CREATE_ENV_NET_CHECK:-1}" = "1" ] && ! timeout 15 bash -c 'exec 3<>/dev/tcp/pypi.org/443' 2>/dev/null; then
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
E2E_OUT="$OUTPUT_ROOT/$E2E_RUN_NAME"
echo "Job $SLURM_JOB_ID en $(hostname): $NGPU GPU(s) | detector: $DET_OUT (lote total $DET_BATCH) | e2e: $E2E_OUT"
#export OMP_NUM_THREADS=1

need() { [ -e "$1" ] || { echo "ERROR: falta $1 ($2)" >&2; exit 1; }; }
if [ ! -f "$DET_DONE" ] || [ "$FORCE_DET" = "1" ]; then
  need "$COCO_DIR/detector_train/annotations/train.json" "COCO de train = DanceTrack + CrowdHuman mezclado: python tracking/data/merge_coco.py (Fase 2)"
  need "$COCO_DIR/dancetrack/annotations/val.json" "COCO de val de DanceTrack: python tracking/data/convert_dancetrack.py (Fase 2)"
  if [ ! -f "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" ]; then   # checkpoint COCO de partida; los nodos de cómputo pueden no tener internet
    bash setup_env/download_ckpt.sh "$ENV_NAME" || true
  fi
  need "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" "checkpoint COCO: bash setup_env/download_ckpt.sh $ENV_NAME (desde el login, con internet)"
fi
need "$DANCETRACK_DIR/train1" "DanceTrack (train1, train2, val) para la etapa 2"; need "$DANCETRACK_DIR/val" "DanceTrack val"

# ============================== ETAPA 1: entrenamiento de RT-DETR ==============================
if [ -f "$DET_DONE" ] && [ "$FORCE_DET" != "1" ]; then
  echo "[etapa 1] detector ya entrenado ($DET_DONE): se salta"
else
  echo "[etapa 1] entrenando RT-DETR con $NGPU GPU(s)"
  RESUME_FLAG=0; [ -f "$DET_OUT/last.pth" ] && [ "$FORCE_DET" != "1" ] && RESUME_FLAG=1
  # PResNet.pretrained=False: el backbone se carga del checkpoint COCO, no hace falta descargar pesos ImageNet (los nodos pueden no tener internet)
  NGPU=$NGPU RUN_NAME="$DET_RUN_NAME" EPOCHS="$DET_EPOCHS" BATCH="$DET_BATCH" NUM_WORKERS="$DET_NUM_WORKERS" RESUME=$RESUME_FLAG \
    bash tracking/scripts/train_detector.sh "$ENV_NAME" PResNet.pretrained=False
  [ -f "$DET_CKPT" ] || { echo "ERROR: el entrenamiento del detector terminó pero no existe $DET_CKPT" >&2; exit 1; }
  date '+%Y-%m-%dT%H:%M:%S' > "$DET_DONE"
  echo "[etapa 1] terminada: $DET_CKPT"
fi

# ============================== ETAPA 2: entrenamiento descongelado del módulo e2e (DDP) ==============================
source tracking/configs/e2e_variants.sh
OV=""; FOUND=0
for v in "${E2E_VARIANTS[@]}"; do [ "${v%%|*}" = "$E2E_VARIANT" ] && { OV=$(echo "${v#*|}" | sed 's/#.*//'); FOUND=1; }; done
[ "$FOUND" = "1" ] || { echo "ERROR: variante desconocida: $E2E_VARIANT (ver tracking/configs/e2e_variants.sh)" >&2; exit 1; }
mkdir -p "$E2E_OUT"
MASTER_PORT=$((10000 + RANDOM % 20000))
echo "[etapa 2] e2e descongelado con $NGPU GPU(s), $CLIPS_PER_GPU clip(s) por GPU (lote efectivo $((CLIPS_PER_GPU * NGPU))); variante $E2E_VARIANT, lambda $LAMBDA"
# shellcheck disable=SC2086
torchrun --master_port="$MASTER_PORT" --nproc_per_node="$NGPU" \
    -m tracking.e2e.train_joint --det-ckpt "$DET_CKPT" --out "$E2E_OUT" --seed "$SEED" --lambda-e2e "$LAMBDA" \
    --clips-per-gpu "$CLIPS_PER_GPU" ${JOINT_ARGS:-} ${OV:+-u $OV}
echo "[etapa 2] terminada: $E2E_OUT"
