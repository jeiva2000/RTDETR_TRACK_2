#!/usr/bin/env bash
# Descarga DanceTrack (train1, train2, val) desde Hugging Face, un archivo por llamada
# (el --include con varios archivos omitió train1.zip), verifica y extrae.
# Uso: bash setup_env/download_dancetrack.sh [local|hpc]
# Destino: $DANCETRACK_DIR si está definido; si no, $DATA_ROOT/dancetrack
set -euo pipefail
source "$(dirname "$0")/load.sh" "${1:-local}"
DEST="${DANCETRACK_DIR:-$DATA_ROOT/dancetrack}"
mkdir -p "$DEST"; cd "$DEST"
python -m pip install -q -U huggingface_hub
for z in train1 train2 val; do
  if [ -d "$z" ] && [ "$(ls "$z" | wc -l)" -gt 0 ]; then echo "[skip] $z ya extraído"; continue; fi
  [ -f "$z.zip" ] || python -c "from huggingface_hub import hf_hub_download as d; d('noahcao/dancetrack', '$z.zip', repo_type='dataset', local_dir='.')"
  unzip -tq "$z.zip" && unzip -q "$z.zip"
done
echo "train1: $(ls train1 | wc -l)  train2: $(ls train2 | wc -l)  val: $(ls val | wc -l)  (esperado 20/20/25)"
