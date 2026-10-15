#!/usr/bin/env bash
# Descarga CrowdHuman train (anotaciones .odgt + zips) desde el espejo de Hugging Face sshao0516/CrowdHuman.
# Un archivo por llamada. Uso:
#   bash setup_env/download_crowdhuman.sh local sample 300   # solo train01.zip, extrae N imágenes (muestra local)
#   bash setup_env/download_crowdhuman.sh hpc full           # train01..03 completos
# Destino: $CROWDHUMAN_DIR (por defecto $DATA_ROOT/CrowdHuman). Licencia: solo investigación no comercial.
set -euo pipefail
source "$(dirname "$0")/load.sh" "${1:-local}"
MODE="${2:-sample}"; N="${3:-300}"
mkdir -p "$CROWDHUMAN_DIR/Images"; cd "$CROWDHUMAN_DIR"
python -m pip install -q -U huggingface_hub
dl() { python -c "from huggingface_hub import hf_hub_download as d; d('sshao0516/CrowdHuman', '$1', repo_type='dataset', local_dir='.')"; }
[ -f annotation_train.odgt ] || dl annotation_train.odgt
if [ "$MODE" = "sample" ]; then ZIPS="CrowdHuman_train01.zip"; else ZIPS="CrowdHuman_train01.zip CrowdHuman_train02.zip CrowdHuman_train03.zip"; fi
for z in $ZIPS; do
  [ -f "$z" ] || dl "$z"
  unzip -tq "$z"
  if [ "$MODE" = "sample" ]; then
    python - "$z" "$N" <<'PY'
import sys, zipfile, os
z, n = sys.argv[1], int(sys.argv[2])
with zipfile.ZipFile(z) as f:
    names = [x for x in f.namelist() if x.lower().endswith('.jpg')][:n]
    for x in names:
        open(os.path.join('Images', os.path.basename(x)), 'wb').write(f.read(x))
print(f'extraídas {len(names)} imágenes en Images/')
PY
  else
    unzip -qjo "$z" -d Images
  fi
done
echo "imágenes en Images/: $(ls Images | wc -l)"
