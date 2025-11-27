# 🔧 Installation Troubleshooting Guide

## Your Current Error (SOLVED)

### ❌ Error 1: pyradiomics needs numpy
```
ModuleNotFoundError: No module named 'numpy'
```

### ❌ Error 2: torch-scatter needs torch
```
ModuleNotFoundError: No module named 'torch'
```

### ✅ Solution: Install in correct order

---

## 🚀 **RECOMMENDED INSTALLATION (Choose ONE method)**

### **Method 1: Automated Script (EASIEST)** ⭐

```bash
conda activate GNNvenv
chmod +x install_deps.sh
bash install_deps.sh
```

This handles everything automatically!

---

### **Method 2: Step-by-Step Manual Installation**

```bash
conda activate GNNvenv

# Step 1: Core
pip install numpy==1.24.3 setuptools

# Step 2: PyTorch
pip install torch==2.0.1 torchvision==0.15.2

# Step 3: PyTorch Geometric extensions (CRITICAL - special installation)
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Step 4: Medical imaging
pip install SimpleITK==2.2.1 pydicom==2.3.1 rt-utils==1.2.7

# Step 5: PyRadiomics
pip install pyradiomics==3.0.1

# Step 6: PyTorch Geometric
pip install torch-geometric==2.3.1

# Step 7: ML packages
pip install scikit-learn==1.3.0 scikit-image==0.21.0 scipy==1.11.1
pip install mrmr-selection==0.2.6 opencv-python==4.8.0

# Step 8: Data/visualization
pip install pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2
pip install tqdm==4.65.0 pyyaml==6.0 joblib==1.3.2
```

---

### **Method 3: Minimal Installation (If still having issues)**

Just install what you need right now:

```bash
conda activate GNNvenv

# Core packages only
pip install numpy pandas scikit-learn matplotlib seaborn tqdm joblib

# For baseline model (no graphs yet)
python main_simple.py --task LR --split_data --train --evaluate
```

Later, when you need graphs, install PyTorch Geometric.

---

## 🐛 **Common Installation Issues**

### Issue 1: torch-scatter/torch-sparse won't install

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch-scatter
```

**Solutions:**

**A) Use the special PyG wheel repository:**
```bash
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**B) If A fails, try without version pinning:**
```bash
pip install torch-scatter torch-sparse torch-cluster
```

**C) If B fails, try building from source (slower):**
```bash
pip install torch-scatter torch-sparse torch-cluster --no-binary :all:
```

**D) Skip for now - you don't need these for baseline!**
```bash
# Run baseline without graphs
python main_simple.py --task LR --split_data --train --evaluate
```

---

### Issue 2: pyradiomics compilation errors

**Symptoms:**
```
error: command 'gcc' failed
Building wheel for pyradiomics (setup.py) ... error
```

**Solution A - Install build tools:**
```bash
# On Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# Then retry
pip install pyradiomics==3.0.1
```

**Solution B - Use pre-built wheel (if available):**
```bash
pip install pyradiomics --prefer-binary
```

**Solution C - Skip for now (if you already have features):**
Since you mentioned you already have extracted radiomics features, you can skip pyradiomics installation!

```bash
# Just comment out or skip pyradiomics in requirements
# You already have your features, so you don't need to extract them again
```

---

### Issue 3: rt-utils errors

**Symptoms:**
```
ERROR: rt-utils requires specific package versions
```

**Solution:**
```bash
# Install compatible version
pip install rt-utils==1.2.7

# Or try latest
pip install rt-utils

# If still failing, you can load RT structures manually
# (code already handles this in data_loader.py)
```

---

### Issue 4: CUDA vs CPU confusion

**Symptoms:**
```
RuntimeError: CUDA not available
torch.cuda.is_available() returns False
```

**Solution:**

**If you have GPU:**
```bash
# Install CUDA version
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**If you DON'T have GPU (most likely your case):**
```bash
# Install CPU version (what we're using)
pip install torch==2.0.1 torchvision==0.15.2
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

Then in `config.py`:
```python
USE_CUDA = False
DEVICE = 'cpu'
```

---

### Issue 5: Package version conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solution A - Install with --no-deps for problematic packages:**
```bash
pip install <problematic-package> --no-deps
```

**Solution B - Use conda for some packages:**
```bash
conda install pytorch torchvision -c pytorch
conda install scikit-learn pandas matplotlib seaborn -c conda-forge
```

**Solution C - Fresh environment:**
```bash
conda deactivate
conda env remove -n GNNvenv
conda create -n GNNvenv python=3.9
conda activate GNNvenv
# Try installation again
```

---

## ✅ **Verify Installation**

After installation, check everything works:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
```

**For full verification:**
```bash
python setup.py
```

---

## 🎯 **Minimal Working Setup**

If you just want to START NOW with minimal packages:

```bash
conda activate GNNvenv

# Absolute minimum (for baseline model only)
pip install numpy pandas scikit-learn matplotlib seaborn joblib tqdm

# Test it works
python -c "import numpy, pandas, sklearn; print('Basic setup OK!')"
```

Then in your code, you can skip the graph-related features and just use Random Forest baseline (which you already have in `main_simple.py`).

---

## 📊 **What You Can Do Without Full Installation**

### With JUST numpy, pandas, scikit-learn:
✅ Load clinical data  
✅ Load your radiomics features  
✅ Feature selection  
✅ Train Random Forest baseline  
✅ Evaluate (AUC, ROC curves)  
✅ Generate predictions  

### What you CANNOT do (needs PyTorch Geometric):
❌ Graph construction  
❌ GAT model training  
❌ Attention visualization  

**But you can do these later!** Start with baseline first.

---

## 🔍 **Debugging Commands**

```bash
# Check Python version
python --version  # Should be 3.9.x

# Check pip
pip --version

# Check conda environment
conda env list
echo $CONDA_DEFAULT_ENV

# List installed packages
pip list

# Check specific package
pip show torch
pip show torch-geometric

# Test imports one by one
python -c "import numpy; print('numpy OK')"
python -c "import pandas; print('pandas OK')"
python -c "import sklearn; print('sklearn OK')"
python -c "import torch; print('torch OK')"
python -c "import torch_geometric; print('PyG OK')"
```

---

## 🆘 **Still Having Issues?**

### Quick Diagnostic:

```bash
# Run this to see what's missing
python << EOF
import sys
packages = ['numpy', 'pandas', 'sklearn', 'torch', 'torch_geometric', 
            'SimpleITK', 'radiomics', 'matplotlib']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except:
        print(f"✗ {pkg} - MISSING")
EOF
```

### Nuclear Option (Fresh Start):

```bash
# Remove everything and start fresh
conda deactivate
conda env remove -n GNNvenv -y
conda clean --all -y

# Create new environment
conda create -n GNNvenv python=3.9 -y
conda activate GNNvenv

# Use the automated script
bash install_deps.sh
```

---

## 📝 **Installation Order Matters!**

Remember this hierarchy:

```
1. numpy, setuptools         (needed by everything)
   ↓
2. torch, torchvision        (needed by torch-geometric)
   ↓
3. torch-scatter, torch-sparse  (needed by torch-geometric)
   ↓
4. torch-geometric           (graph neural networks)
   ↓
5. Other packages            (medical imaging, ML, viz)
```

**Never install all at once from requirements.txt** - it causes dependency resolution failures!

---

## ✅ **Success Checklist**

After successful installation, you should be able to:

- [ ] `python -c "import torch"`  ✓
- [ ] `python -c "import torch_geometric"`  ✓
- [ ] `python -c "import sklearn"`  ✓
- [ ] `python -c "import pandas"`  ✓
- [ ] `python setup.py`  ✓ (runs without errors)
- [ ] `python data_loader.py`  ✓ (test succeeds)

---

## 📧 **Common Error Messages Decoded**

| Error | Meaning | Fix |
|-------|---------|-----|
| `No module named 'numpy'` | numpy not installed | `pip install numpy` first |
| `No module named 'torch'` | torch not installed | `pip install torch` before torch-geometric |
| `Could not find torch-scatter` | Wrong installation method | Use PyG wheel repo |
| `gcc failed` | Missing build tools | `sudo apt install build-essential` |
| `CUDA not available` | No GPU or wrong version | Use CPU version |
| `Metadata generation failed` | Dependency missing | Install dependencies first |

---

## 🎉 **You're Almost There!**

The installation issues are **just ordering problems** - not actual bugs.

**Use the automated script and you'll be fine! 🚀**

```bash
bash install_deps.sh
```
