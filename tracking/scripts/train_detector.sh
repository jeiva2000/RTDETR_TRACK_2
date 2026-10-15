#!/usr/bin/env bash
# Entrena el detector (1 clase) desde el checkpoint COCO. Train = DanceTrack train + CrowdHuman train; val = DanceTrack val.
# Uso: tracking/scripts/train_detector.sh [local|hpc] [-u clave=valor ...]
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
if [ "$NGPU" -gt 1 ]; then LAUNCH=(torchrun --nproc_per_node="$NGPU"); else LAUNCH=(python); fi
"${LAUNCH[@]}" tools/train.py -c ../tracking/configs/detector_r50vd_person.yml --seed "$SEED" --use-amp \
  -t "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" --output-dir "$OUTPUT_ROOT/$RUN_NAME" -u "${OV[@]}" "$@"
