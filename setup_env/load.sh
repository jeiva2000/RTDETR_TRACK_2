#!/usr/bin/env bash
# Uso: source setup_env/load.sh [local|hpc]
_P="${1:-local}"; _D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_D/paths.$_P.env" || { echo "Falta $_D/paths.$_P.env"; return 1; }
