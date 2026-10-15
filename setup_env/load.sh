#!/usr/bin/env bash
# Uso: source setup_env/load.sh [local|hpc]
_P="${1:-local}"; _D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_D/paths.$_P.env" || { echo "Falta $_D/paths.$_P.env"; return 1; }
for _v in PROJECT_ROOT DATA_ROOT CKPT_DIR OUTPUT_ROOT; do
  case "${!_v}" in /*) ;; *) echo "ERROR: $_v debe ser una ruta absoluta (es '${!_v}')"; return 1 2>/dev/null || exit 1;; esac
done
