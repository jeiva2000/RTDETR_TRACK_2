#!/usr/bin/env bash
# Entrena el detector (1 clase) desde el checkpoint COCO. Train = DanceTrack train + CrowdHuman train; val = DanceTrack val.
# Uso: tracking/scripts/train_detector.sh [local|hpc] [-u clave=valor ...]
# Usa el solver con early stopping (config: early_stopping en tracking/configs/detector_r50vd_person.yml).
# Reanudar: RESUME=1 (con el mismo RUN_NAME) continúa desde $OUTPUT_ROOT/$RUN_NAME/last.pth si existe (RT-DETR no admite -t y -r a la vez, así que entonces no se pasa el checkpoint COCO).
# Con varias GPUs (NGPU>1) se lanza con torchrun --standalone; BATCH (lote total) debe ser divisible por NGPU. Argumentos extra al final = más claves de -u (p. ej. PResNet.pretrained=False).
# Variables opcionales: NGPU (1), EPOCHS, BATCH, RUN_NAME (detector_r50vd_person), NUM_WORKERS (4),
#   TRAIN_ANN (por defecto $COCO_DIR/detector_train/annotations/train.json), VAL_ANN (por defecto $COCO_DIR/dancetrack/annotations/val.json)
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"; shift || true
NGPU="${NGPU:-1}"; RUN_NAME="${RUN_NAME:-detector_r50vd_person}"
TRAIN_ANN="${TRAIN_ANN:-$COCO_DIR/detector_train/annotations/train.json}"
VAL_ANN="${VAL_ANN:-$COCO_DIR/dancetrack/annotations/val.json}"
OV=(sync_bn=$([ "$NGPU" -gt 1 ] && echo True || echo False)
    train_dataloader.dataset.img_folder="$DATA_ROOT" train_dataloader.dataset.ann_file="$TRAIN_ANN"
    val_dataloader.dataset.img_folder="$DANCETRACK_DIR" val_dataloader.dataset.ann_file="$VAL_ANN"
    train_dataloader.num_workers="${NUM_WORKERS:-4}" val_dataloader.num_workers="${NUM_WORKERS:-4}")
[ -n "${EPOCHS:-}" ] && OV+=(epoches="$EPOCHS")
[ -n "${BATCH:-}" ]  && OV+=(train_dataloader.total_batch_size="$BATCH" val_dataloader.total_batch_size="$BATCH")
cd "$PROJECT_ROOT/rtdetrv2_pytorch"
[ -z "${BATCH:-}" ] || [ $(( BATCH % NGPU )) -eq 0 ] || { echo "ERROR: BATCH=$BATCH no es divisible por NGPU=$NGPU" >&2; exit 1; }
OUT="$OUTPUT_ROOT/$RUN_NAME"
if [ "${RESUME:-0}" = "1" ] && [ -f "$OUT/last.pth" ]; then INIT=(-r "$OUT/last.pth"); echo "reanudando desde $OUT/last.pth"
else INIT=(-t "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth"); fi
if [ "$NGPU" -gt 1 ]; then LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU"); else LAUNCH=(python); fi
"${LAUNCH[@]}" ../tracking/detector/train.py -c ../tracking/configs/detector_r50vd_person.yml --seed "$SEED" --use-amp \
  "${INIT[@]}" --output-dir "$OUT" -u "${OV[@]}" "$@"
