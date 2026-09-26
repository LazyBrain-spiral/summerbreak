#!/usr/bin/env bash
# Creates the v3 runtime environment on Linux / WSL2 without sudo.
#   bash scripts/setup_env.sh            # full env (CUDA torch, colmap, python deps)
#   SKIP_TORCH=1 bash scripts/setup_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-drone3d}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ ! -x "$PREFIX/bin/conda" ]; then
  echo "[setup] installing Miniforge into $PREFIX"
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$PREFIX"
fi
# shellcheck disable=SC1091
source "$PREFIX/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] creating env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

echo "[setup] installing COLMAP from conda-forge (CUDA build is picked when a GPU is visible)"
# colmap 4.0.4 is linked against libfaiss 1.10; newer libfaiss (1.14) breaks it with
# "undefined symbol: faiss::IndexIVFFlat::IndexIVFFlat(...)". Pin until conda-forge rebuilds.
conda install -y -c conda-forge "colmap=4.0.4" "libfaiss=1.10.0"
if colmap help 2>/dev/null | grep -q global_mapper; then
  echo "[setup] COLMAP has a built-in global_mapper; GLOMAP is not needed"
else
  # GLOMAP from conda-forge can pull a libfaiss that breaks the colmap binary; verify and roll back.
  conda install -y -c conda-forge glomap || true
  if ! colmap help >/dev/null 2>&1; then
    echo "[setup] glomap broke colmap (library conflict); removing glomap, incremental mapping will be used"
    conda remove -y glomap || true
    conda install -y -c conda-forge --force-reinstall "colmap=4.0.4" "libfaiss=1.10.0"
  fi
fi
colmap help >/dev/null 2>&1 || { echo "[setup] ERROR: colmap does not run"; exit 1; }

echo "[setup] installing python dependencies"
pip install --upgrade pip
pip install -r "$HERE/requirements-v3.txt"
if [ -z "${SKIP_TORCH:-}" ]; then
  pip install torch torchvision --index-url "$TORCH_INDEX"
fi

python "$HERE/scripts/check_env.py" --profile live || true
echo "[setup] done. Activate with: source $PREFIX/etc/profile.d/conda.sh && conda activate $ENV_NAME"
