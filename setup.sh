#!/usr/bin/env bash
# setup.sh — Install Mask2Former training environment
# Ubuntu | Python 3.10 | CUDA 12.2 | RTX A4000
# Run once: bash setup.sh

set -euo pipefail

PYTHON=${PYTHON:-python3.10}
TORCH_VERSION="2.2.2"
CUDA_VERSION="cu121"

echo "==> 1/6  Installing PyTorch ${TORCH_VERSION}+${CUDA_VERSION}"
pip install \
    torch==${TORCH_VERSION} \
    torchvision==0.17.2 \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

echo "==> 2/6  Installing Python dependencies"
pip install -r requirements.txt

echo "==> 3/6  Installing Detectron2 from source"
pip install \
    "git+https://github.com/facebookresearch/detectron2.git" \
    --no-build-isolation

echo "==> 4/6  Cloning Mask2Former"
if [ ! -d "Mask2Former" ]; then
    git clone https://github.com/facebookresearch/Mask2Former.git
fi
cd Mask2Former

echo "==> 5/6  Installing Mask2Former"
pip install -e .

# Build custom CUDA ops (MSDeformAttn)
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..

echo "==> 6/6  Downloading R50 + Mask2Former COCO instance-seg config"
# Pre-download the official config weights so training starts immediately
${PYTHON} - <<'EOF'
from detectron2.utils.file_io import PathManager
import detectron2
url = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl"
PathManager.get_local_path(url)
print("R-50 backbone weights cached.")
EOF

echo ""
echo "======================================================"
echo " Setup complete!  Start training with:"
echo "   python training.py --output-dir ./output"
echo "======================================================"
