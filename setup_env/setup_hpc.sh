#!/usr/bin/env bash
# Clúster: módulos GCC/12.2.0 y CUDA/11.8.0. Ejecutar en nodo de login/interactivo.
set -euo pipefail
source "$(dirname "$0")/load.sh" hpc
module load GCC/12.2.0 CUDA/11.8.0
conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels python=3.10 pip
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r "$PROJECT_ROOT/rtdetrv2_pytorch/requirements.txt"
pip install "numpy<2"   # torch 2.1.2 no es compatible con NumPy 2
pip install -r "$PROJECT_ROOT/tracking/requirements.txt" "numpy<2"   # filterpy, lap, scipy
python -c "import torch;print(torch.__version__, torch.version.cuda)"
