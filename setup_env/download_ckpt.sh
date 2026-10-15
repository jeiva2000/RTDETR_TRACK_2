#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/load.sh" "${1:-local}"
mkdir -p "$CKPT_DIR"
URL=https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth
OUT="$CKPT_DIR/rtdetrv2_r50vd_6x_coco_ema.pth"
if   command -v curl >/dev/null; then curl -L -o "$OUT" "$URL"
elif command -v wget >/dev/null; then wget -O "$OUT" "$URL"
else python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$URL" "$OUT"
fi
ls -lh "$OUT"
