#!/bin/bash
# Automated installation script for RadGraph dependencies
# Handles dependency ordering issues automatically

set -e  # Exit on error

echo "=========================================="
echo "RadGraph Dependencies Installation Script"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "ERROR: No conda environment is activated!"
    echo "Please run: conda activate GNNvenv"
    exit 1
fi

echo "Active environment: ${CONDA_DEFAULT_ENV}"
echo ""

# Step 1: Core dependencies
echo "[Step 1/8] Installing core dependencies (numpy, setuptools)..."
pip install numpy==1.24.3 setuptools>=65.0 || {
    echo "ERROR: Failed to install core dependencies"
    exit 1
}
echo "✓ Core dependencies installed"
echo ""

# Step 2: PyTorch
echo "[Step 2/8] Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 || {
    echo "ERROR: Failed to install PyTorch"
    exit 1
}
echo "✓ PyTorch installed"
echo ""

# Step 3: PyTorch Geometric extensions (special installation)
echo "[Step 3/8] Installing PyTorch Geometric extensions..."
echo "  This may take a few minutes..."
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html || {
    echo "WARNING: PyTorch Geometric extensions installation failed"
    echo "Trying alternative installation method..."
    pip install torch-scatter torch-sparse torch-cluster || {
        echo "ERROR: Failed to install PyTorch Geometric extensions"
        echo "You may need to install manually. See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        exit 1
    }
}
echo "✓ PyTorch Geometric extensions installed"
echo ""

# Step 4: Medical imaging packages
echo "[Step 4/8] Installing medical imaging packages..."
pip install SimpleITK==2.2.1 pydicom==2.3.1 rt-utils==1.2.7 || {
    echo "ERROR: Failed to install medical imaging packages"
    exit 1
}
echo "✓ Medical imaging packages installed"
echo ""

# Step 5: PyRadiomics
echo "[Step 5/8] Installing PyRadiomics..."
pip install pyradiomics==3.0.1 || {
    echo "ERROR: Failed to install PyRadiomics"
    exit 1
}
echo "✓ PyRadiomics installed"
echo ""

# Step 6: PyTorch Geometric
echo "[Step 6/8] Installing PyTorch Geometric..."
pip install torch-geometric==2.3.1 || {
    echo "ERROR: Failed to install PyTorch Geometric"
    exit 1
}
echo "✓ PyTorch Geometric installed"
echo ""

# Step 7: ML and image processing packages
echo "[Step 7/8] Installing ML and image processing packages..."
pip install mrmr-selection==0.2.6 scikit-learn==1.3.0 scikit-image==0.21.0 scipy==1.11.1 || {
    echo "ERROR: Failed to install ML/image processing packages"
    exit 1
}
echo "✓ ML and image processing packages installed"
echo ""

# Step 8: Data handling and visualization
echo "[Step 8/8] Installing data handling and visualization packages..."
pip install pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 tqdm==4.65.0 pyyaml==6.0 joblib==1.3.2 || {
    echo "ERROR: Failed to install data/visualization packages"
    exit 1
}
echo "✓ Data handling and visualization packages installed"
echo ""

# Verification
echo "=========================================="
echo "Installation Complete! Verifying..."
echo "=========================================="
echo ""

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"
python -c "import SimpleITK; print(f'SimpleITK version: {SimpleITK.__version__}')"
python -c "import radiomics; print(f'PyRadiomics version: {radiomics.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')"

echo ""
echo "=========================================="
echo "✓ All packages installed successfully!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python setup.py          # Check your setup"
echo "  python data_loader.py    # Test data loading"
echo "  python main_simple.py --task LR --split_data --train --evaluate"
echo ""
