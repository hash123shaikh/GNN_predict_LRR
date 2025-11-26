# RadGraph Implementation Guide for CMC Vellore

## 📋 Overview

This guide will help you implement the complete RadGraph pipeline step-by-step using your 176-patient dataset.

## 📁 Files Created (So Far)

✅ **Core Files:**
1. `requirements.txt` - All dependencies
2. `config.py` - Configuration and hyperparameters  
3. `utils.py` - Utility functions
4. `data_loader.py` - Load CT scans, RT structures, clinical data

🔄 **Files Still Needed:**
5. `preprocessing.py` - CT preprocessing and resampling
6. `supervoxel_generator.py` - SLIC supervoxel generation
7. `feature_extractor.py` - PyRadiomics feature extraction (you may have this)
8. `feature_selector.py` - mRMR feature selection
9. `graph_builder.py` - Build graphs from features
10. `dataset.py` - PyTorch dataset class
11. `model.py` - GAT model architecture
12. `train.py` - Training script
13. `evaluate.py` - Evaluation script
14. `visualize_attention.py` - Attention map visualization
15. `main.py` - Main orchestrator

---

## 🚀 Step-by-Step Implementation

### STEP 1: Setup Environment (15 minutes)

```bash
# Create virtual environment
python -m venv radgraph_env
source radgraph_env/bin/activate  # On Windows: radgraph_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For CPU only:
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### STEP 2: Organize Your Data (30 minutes)

Your data structure should look like:

```
data/
├── ct_scans/
│   ├── patient_001/
│   │   ├── CT.001.dcm
│   │   ├── CT.002.dcm
│   │   └── ...
│   ├── patient_002/
│   └── ...
├── rt_structs/
│   ├── patient_001.dcm
│   ├── patient_002.dcm
│   └── ...
├── clinical_data.csv
└── radiomics_features.csv  # Your extracted features
```

**clinical_data.csv format:**
```csv
patient_id,age,sex,hpv_status,ajcc_stage,ecog_status,concurrent_chemo,tumor_subsite,tumor_volume,locoregional_recurrence,distant_metastasis,followup_months
patient_001,62,1,1,4,0,1,0,15234.5,0,0,36
patient_002,58,0,0,3,1,1,0,8932.1,1,0,28
...
```

### STEP 3: Configure Paths (5 minutes)

Edit `config.py`:

```python
# Line 14-20: Update these paths to match your data
CT_SCANS_DIR = DATA_DIR / "ct_scans"  # Your CT folder
RTSTRUCT_DIR = DATA_DIR / "rt_structs"  # Your RT structures
CLINICAL_DATA_FILE = DATA_DIR / "clinical_data.csv"  # Your clinical data
RADIOMICS_FEATURES_FILE = DATA_DIR / "radiomics_features.csv"  # Your features

# Line 30-40: Verify clinical feature names match your CSV columns
CLINICAL_FEATURES = [
    'age',
    'sex',  # Make sure these match your CSV column names
    ...
]
```

### STEP 4: Test Data Loading (10 minutes)

```bash
python data_loader.py
```

**Expected output:**
```
Loaded clinical data for 176 patients
Total patients: 176
Patients with >=24 months follow-up: XXX/176

Testing with patient: patient_001
CT image size: (512, 512, 120)
CT spacing: (0.976, 0.976, 3.0)
GTV mask size: (512, 512, 120)
GTV contour: GTV
Clinical features: {'age': 62, 'sex': 1, ...}

Data loader test successful!
```

**If you get errors:**
- Check paths in config.py
- Verify CSV column names match config.py
- Ensure rt-utils is installed: `pip install rt-utils`

---

## 🔧 Remaining Implementation Steps

I will now create the remaining files. Here's what each does:

### Files 5-15: Complete Pipeline

**preprocessing.py**
- Resamples CT to 1mm isotropic
- Defines 5cm peritumoral region
- Prepares images for feature extraction

**supervoxel_generator.py**
- Generates ~100 supervoxels using SLIC
- Creates 3D segmentation of peritumoral region

**feature_extractor.py** (you may skip if you have features)
- Extracts 93 PyRadiomics features
- From GTV + all supervoxels

**feature_selector.py**
- Selects 4 features for LR, 6 for DM
- Uses mRMR + Random Forest

**graph_builder.py**
- Selects top 20 most tumor-like supervoxels
- Builds star-topology graph

**dataset.py**
- PyTorch Dataset class
- Loads graphs for training

**model.py**
- GAT architecture
- Custom read-out layer for GTV node

**train.py**
- Training loop
- Handles class imbalance
- Early stopping

**evaluate.py**
- Computes AUC, sensitivity, specificity
- Bootstrap confidence intervals
- Generates ROC curves

**visualize_attention.py**
- Extracts attention weights
- Creates attention heatmaps
- Overlays on CT images

**main.py**
- Orchestrates entire pipeline
- Run everything with one command

---

## 📊 Expected Timeline

| Phase | Time | Description |
|-------|------|-------------|
| Setup | 1 day | Install packages, organize data |
| Data Loading | 1 day | Test loading, fix issues |
| Preprocessing | 2 days | Supervoxels, feature extraction |
| Feature Selection | 1 day | Select optimal features |
| Graph Building | 2 days | Build and test graphs |
| Model Training | 3 days | Implement GAT, train |
| Evaluation | 2 days | Test, visualize results |
| **Total** | **~2 weeks** | **With your existing features** |

---

## 🎯 Quick Start (After All Files Created)

```python
# Run complete pipeline
python main.py --task LR --mode train

# Evaluate on test set
python main.py --task LR --mode evaluate

# Generate attention maps
python visualize_attention.py --patient patient_001
```

---

## ⚠️ Common Issues and Solutions

### Issue 1: RT Structure Not Loading
**Solution:** Install rt-utils properly
```bash
pip uninstall rt-utils
pip install rt-utils --no-cache-dir
```

### Issue 2: PyTorch Geometric Installation
**Solution:** Follow official guide for your system
```bash
# Check your PyTorch version
python -c "import torch; print(torch.__version__)"
# Then install matching torch-geometric version
```

### Issue 3: Out of Memory
**Solution:** Reduce batch size in config.py
```python
BATCH_SIZE = 4  # Instead of 8
```

### Issue 4: CUDA Not Available
**Solution:** Use CPU (slower but works)
```python
# In config.py
USE_CUDA = False
DEVICE = 'cpu'
```

---

## 📈 What Results to Expect

With 176 patients:
- **Training set:** ~123 patients
- **Validation set:** ~26 patients  
- **Test set:** ~27 patients

**Expected Performance (ballpark):**
- AUC: 0.65-0.80 (depends on your data quality)
- Training time: 2-4 hours (with GPU)
- Inference: <1 second per patient

---

## 🆘 Getting Help

If you encounter issues:

1. **Check the error message** - most errors are self-explanatory
2. **Verify paths** - 90% of errors are wrong paths
3. **Check data format** - ensure CSV columns match config
4. **Test step-by-step** - don't run everything at once
5. **Read code comments** - extensive documentation in each file

---

## 📝 Next Steps

I will now create the remaining 11 Python files (files 5-15).

**Shall I proceed?**

Just confirm and I'll create:
✅ All remaining core files
✅ Complete implementations
✅ Ready to run
✅ Well-documented

The complete system will be ready to:
1. Load your 176 patients
2. Use your extracted radiomics features (or extract new ones)
3. Train RadGraph model
4. Evaluate performance
5. Visualize attention maps

**Ready to continue?** 🚀
