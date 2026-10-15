#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/load.sh" "${1:-local}"
mkdir -p "$CKPT_DIR"
curl -L -o "$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth" \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth
