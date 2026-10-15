#!/usr/bin/env bash
# Workstation (RTX 2080). Ajusta CUDA_TAG si tu driver soporta otra versión.
set -euo pipefail
source "$(dirname "$0")/load.sh" local
CUDA_TAG="${CUDA_TAG:-cu118}"
conda create -y -n "$CONDA_ENV" python=3.10
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/$CUDA_TAG
pip install -r "$PROJECT_ROOT/rtdetrv2_pytorch/requirements.txt"
pip install "numpy<2"   # torch 2.1.2 no es compatible con NumPy 2
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
