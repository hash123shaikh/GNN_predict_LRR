# 🚀 QUICK INSTALLATION COMMANDS - Copy & Paste

## ✅ **Option 1: Automated (RECOMMENDED)** ⭐

```bash
# Download and run the installation script
conda activate GNNvenv
chmod +x install_deps.sh
bash install_deps.sh
```

**That's it!** Everything installs automatically in correct order.

---

## ✅ **Option 2: Manual Step-by-Step**

Copy and paste these commands **one at a time**:

```bash
# Activate environment
conda activate GNNvenv

# Step 1
pip install numpy==1.24.3

# Step 2
pip install torch==2.0.1 torchvision==0.15.2

# Step 3 (IMPORTANT - special URL)
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Step 4
pip install SimpleITK==2.2.1 pydicom==2.3.1 rt-utils==1.2.7

# Step 5
pip install pyradiomics==3.0.1

# Step 6
pip install torch-geometric==2.3.1

# Step 7
pip install scikit-learn==1.3.0 scikit-image==0.21.0 scipy==1.11.1 opencv-python==4.8.0

# Step 8
pip install mrmr-selection==0.2.6 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2

# Step 9
pip install tqdm==4.65.0 pyyaml==6.0 joblib==1.3.2
```

---

## ✅ **Option 3: Minimal (For Baseline Only)**

If you just want to run the baseline model NOW:

```bash
conda activate GNNvenv
pip install numpy pandas scikit-learn matplotlib seaborn joblib tqdm
```

Then run:
```bash
python main_simple.py --task LR --split_data --train --evaluate
```

(You can install the rest later when needed)

---

## ✅ **Verify Installation**

```bash
python -c "import torch; print('PyTorch OK')"
python -c "import sklearn; print('scikit-learn OK')"
python -c "import pandas; print('pandas OK')"
python setup.py
```

---

## ✅ **Test Your Setup**

```bash
python data_loader.py       # Test data loading
python preprocessing.py     # Test preprocessing
python utils.py            # Test utilities
```

---

## ✅ **Run the Pipeline**

```bash
# For Locoregional Recurrence
python main_simple.py --task LR --split_data --train --evaluate

# For Distant Metastasis  
python main_simple.py --task DM --split_data --train --evaluate
```

---

## 🐛 **If Something Fails**

```bash
# Check what's installed
pip list | grep -E "torch|numpy|pandas|sklearn"

# Check environment
conda env list

# Fresh start
conda deactivate
conda env remove -n GNNvenv
conda create -n GNNvenv python=3.9
conda activate GNNvenv
bash install_deps.sh
```

---

## 📥 **Files You Need**

Download these from the outputs folder:
1. `install_deps.sh` - Automated installer
2. `requirements_fixed.txt` - Fixed requirements
3. `INSTALLATION_TROUBLESHOOTING.md` - Detailed help
4. All Python files (config.py, utils.py, etc.)

---

## 🎯 **Your Next Steps**

```bash
# 1. Fix installation (choose one option above)
bash install_deps.sh

# 2. Verify it worked
python setup.py

# 3. Configure your paths
nano config.py  # Edit lines 18-21

# 4. Run baseline model
python main_simple.py --task LR --split_data --train --evaluate

# 5. Check results
ls outputs/
```

**Done! 🎉**
