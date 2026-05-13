#!/bin/bash
# ============================================================================
# RadGraph — Automated Dependency Installation
# CMC Vellore | QIRAIL Laboratory | Radiation Oncology
# ============================================================================
# Usage:
#   conda create -n GNNvenv python=3.8 -y
#   conda activate GNNvenv
#   bash install_deps.sh
# ============================================================================

set -e  # Exit immediately on any error

echo ""
echo "============================================"
echo "  RadGraph Dependency Installation Script"
echo "============================================"
echo ""

# ── Guard: conda environment must be active ──────────────────────────────────
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "ERROR: No conda environment is active."
    echo "Please run:"
    echo "  conda create -n GNNvenv python=3.8 -y"
    echo "  conda activate GNNvenv"
    echo "  bash install_deps.sh"
    exit 1
fi
echo "Active environment : ${CONDA_DEFAULT_ENV}"

# ── Guard: Python 3.8+ required ──────────────────────────────────────────────
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
echo "Python version     : ${PYTHON_VERSION}"

if [[ "${PYTHON_MAJOR}" -lt 3 ]] || [[ "${PYTHON_MAJOR}" -eq 3 && "${PYTHON_MINOR}" -lt 8 ]]; then
    echo ""
    echo "ERROR: Python 3.8+ is required. Found ${PYTHON_VERSION}."
    echo "Recreate the environment:"
    echo "  conda create -n GNNvenv python=3.8 -y"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# ── Step 1: NumPy + setuptools ────────────────────────────────────────────────
echo "[1/8] Installing NumPy and setuptools..."
pip install numpy==1.24.3 "setuptools>=65.0" || {
    echo "ERROR: Failed to install NumPy / setuptools"; exit 1
}
echo "✓ Done"
echo ""

# ── Step 2: PyTorch ───────────────────────────────────────────────────────────
echo "[2/8] Installing PyTorch..."
pip install torch==2.0.1 || {
    echo "ERROR: Failed to install PyTorch"; exit 1
}
echo "✓ Done"
echo ""

# ── Step 3: PyTorch Geometric extensions ─────────────────────────────────────
# torch-scatter, torch-sparse, torch-cluster cannot be installed via plain
# PyPI — they require the --find-links flag pointing to the PyG wheel server.
echo "[3/8] Installing PyTorch Geometric extensions (torch-scatter etc.)..."
echo "      This step may take a few minutes..."
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html || {
    echo ""
    echo "WARNING: CPU wheel install failed. Trying generic PyPI fallback..."
    pip install torch-scatter torch-sparse torch-cluster || {
        echo ""
        echo "ERROR: Could not install PyTorch Geometric extensions."
        echo "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        exit 1
    }
}
echo "✓ Done"
echo ""

# ── Step 4: Medical imaging ───────────────────────────────────────────────────
echo "[4/8] Installing medical imaging packages..."
pip install SimpleITK==2.2.1 pydicom==2.3.1 rt-utils==1.2.7 || {
    echo "ERROR: Failed to install medical imaging packages"; exit 1
}
echo "✓ Done"
echo ""

# ── Step 5: PyRadiomics ───────────────────────────────────────────────────────
# Must come after NumPy (Step 1) — will fail if NumPy is not already installed.
echo "[5/8] Installing PyRadiomics..."
pip install pyradiomics==3.0.1 || {
    echo "ERROR: Failed to install PyRadiomics"; exit 1
}
echo "✓ Done"
echo ""

# ── Step 6: PyTorch Geometric ─────────────────────────────────────────────────
# Must come after torch-scatter / torch-sparse / torch-cluster (Step 3).
echo "[6/8] Installing PyTorch Geometric..."
pip install torch-geometric==2.3.1 || {
    echo "ERROR: Failed to install PyTorch Geometric"; exit 1
}
echo "✓ Done"
echo ""

# ── Step 7: ML + image processing ────────────────────────────────────────────
echo "[7/8] Installing ML and image processing packages..."
pip install \
    scikit-learn==1.3.0 \
    mrmr-selection==0.2.6 \
    joblib==1.3.2 \
    scikit-image==0.21.0 \
    scipy==1.10.1 \
    opencv-python==4.8.0.76 || {
    echo "ERROR: Failed to install ML / image processing packages"; exit 1
}
echo "✓ Done"
echo ""

# ── Step 8: Data + visualisation ─────────────────────────────────────────────
echo "[8/8] Installing data handling and visualisation packages..."
pip install \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    tqdm==4.65.0 \
    pyyaml==6.0 || {
    echo "ERROR: Failed to install data / visualisation packages"; exit 1
}
echo "✓ Done"
echo ""

# ── Verification ──────────────────────────────────────────────────────────────
echo "============================================"
echo "  Verifying installation..."
echo "============================================"
echo ""

python3 -c "import torch;          print('torch           :', torch.__version__)"
python3 -c "import torch_geometric; print('torch_geometric :', torch_geometric.__version__)"
python3 -c "import SimpleITK;      print('SimpleITK       :', SimpleITK.__version__)"
python3 -c "import radiomics;      print('pyradiomics     :', radiomics.__version__)"
python3 -c "import sklearn;        print('scikit-learn    :', sklearn.__version__)"
python3 -c "import skimage;        print('scikit-image    :', skimage.__version__)"
python3 -c "import pandas;         print('pandas          :', pandas.__version__)"
python3 -c "import cv2;            print('opencv          :', cv2.__version__)"

echo ""
echo "============================================"
echo "  All packages installed successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit config.py — update your data paths"
echo "  2. python3 setup.py — verify configuration"
echo "  3. python3 main_simple.py --task LR --split_data --train --evaluate --fast"
echo ""