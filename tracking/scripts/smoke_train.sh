#!/usr/bin/env bash
# Entrenamiento de humo (1 clase, dataset sintético, init desde COCO, 2 épocas). Uso: tracking/scripts/smoke_train.sh [local|hpc]
set -euo pipefail
source "$(dirname "$0")/../../setup_env/load.sh" "${1:-local}"
cd "$PROJECT_ROOT/rtdetrv2_pytorch"
S="$DATA_ROOT/smoke_coco"
python "$PROJECT_ROOT/tracking/scripts/make_smoke_coco.py" --out "$S"
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml --seed "$SEED" --use-amp \
  -t "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" --output-dir "$OUTPUT_ROOT/smoke" \
  -u epoches=2 sync_bn=False num_classes=1 remap_mscoco_category=False \
     train_dataloader.total_batch_size=4 val_dataloader.total_batch_size=4 \
     train_dataloader.num_workers=2 val_dataloader.num_workers=2 \
     train_dataloader.dataset.img_folder="$S/train" train_dataloader.dataset.ann_file="$S/annotations/train.json" \
     val_dataloader.dataset.img_folder="$S/val" val_dataloader.dataset.ann_file="$S/annotations/val.json" \
     print_freq=5
