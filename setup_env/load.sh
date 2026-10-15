#!/usr/bin/env bash
# Uso: source setup_env/load.sh [local|hpc]
_P="${1:-local}"; _D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_D/paths.$_P.env" || { echo "Falta $_D/paths.$_P.env"; return 1; }
# Derivadas de DATA_ROOT salvo que paths.*.env las defina
export DANCETRACK_DIR="${DANCETRACK_DIR:-$DATA_ROOT/DanceTrack}"
export COCO_DIR="${COCO_DIR:-$DATA_ROOT/coco_person}"
for _v in PROJECT_ROOT DATA_ROOT CKPT_DIR OUTPUT_ROOT DANCETRACK_DIR COCO_DIR; do
  case "${!_v}" in /*) ;; *) echo "ERROR: $_v debe ser una ruta absoluta (es '${!_v}')"; return 1 2>/dev/null || exit 1;; esac
done
