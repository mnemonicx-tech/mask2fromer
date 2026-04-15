#!/usr/bin/env bash
# setup.sh — Install Mask2Former training environment
# Ubuntu | Python 3.10 | CUDA 12.2 | RTX A4000
# Run once: bash setup.sh

set -euo pipefail

PYTHON=${PYTHON:-python3.10}
TORCH_VERSION="2.2.2"
CUDA_VERSION="cu121"
VENV_DIR=${VENV_DIR:-.venv}

echo "==> 0/7  Setting up virtual environment"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    ${PYTHON} -m venv "${VENV_DIR}"
else
    echo "Using existing virtual environment at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==>      Python: $(python --version)"
echo "==>      Pip:    $(pip --version)"

echo "==>      Upgrading pip tooling"
pip install --upgrade pip setuptools wheel

echo "==> 1/7  Installing PyTorch ${TORCH_VERSION}+${CUDA_VERSION}"
pip install \
    torch==${TORCH_VERSION} \
    torchvision==0.17.2 \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

echo "==> 2/7  Installing Python dependencies"
pip install -r requirements.txt

echo "==> 3/7  Installing Detectron2 from source"
pip install \
    "git+https://github.com/facebookresearch/detectron2.git" \
    --no-build-isolation

echo "==> 4/7  Cloning Mask2Former"
if [ ! -d "Mask2Former" ]; then
    git clone https://github.com/facebookresearch/Mask2Former.git
fi
cd Mask2Former

echo "==> 5/7  Installing Mask2Former"
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Detected Python package metadata; installing editable package."
    pip install -e .
else
    echo "No setup.py/pyproject.toml found; using repo-path install mode."

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo "Enforcing compatible numeric stack for torch/detectron2"
        pip install "numpy>=1.26,<2.0" "iopath>=0.1.7,<0.1.10"
    fi

    MASK2FORMER_ROOT="$(pwd)"

    # Register repo paths in venv site-packages so `import mask2former` works.
    # Skips any site-packages directory that doesn't actually exist on disk.
    python - <<PYEOF
import os, site

root = os.path.abspath("${MASK2FORMER_ROOT}")
paths = [root, os.path.join(root, "mask2former")]

for sp in site.getsitepackages():
    if not os.path.exists(sp):
        print(f"Skipping non-existent site-packages: {sp}")
        continue
    pth = os.path.join(sp, "mask2former_local.pth")
    with open(pth, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Wrote {pth}")
PYEOF

    # Persist PYTHONPATH for future shell sessions when venv is activated.
    ACTIVATE_FILE="${VIRTUAL_ENV}/bin/activate"
    if ! grep -q "MASK2FORMER_ROOT" "${ACTIVATE_FILE}"; then
        {
            echo ""
            echo "# Added by setup.sh for local Mask2Former repo"
            echo "export MASK2FORMER_ROOT=\"${MASK2FORMER_ROOT}\""
            echo "export PYTHONPATH=\"\${MASK2FORMER_ROOT}\${PYTHONPATH:+:\$PYTHONPATH}\""
        } >> "${ACTIVATE_FILE}"
    fi
fi

# Build custom CUDA ops (MSDeformAttn)
if [ -d "mask2former/modeling/pixel_decoder/ops" ]; then
    cd mask2former/modeling/pixel_decoder/ops
    sh make.sh
    cd ../../../../..
else
    echo "WARNING: CUDA ops directory not found, skipping custom ops build."
    cd ..
fi

echo "==> 6/7  Downloading R50 + Mask2Former COCO instance-seg config"
# Pre-download the official config weights so training starts immediately
python - <<'EOF'
from detectron2.utils.file_io import PathManager
import detectron2
url = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl"
PathManager.get_local_path(url)
print("R-50 backbone weights cached.")
EOF

echo "==> 7/7  Final environment info"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo "======================================================"
echo " Setup complete!  Start training with:"
echo "   source ${VENV_DIR}/bin/activate"
echo "   python training.py --output-dir ./output"
echo "======================================================"
