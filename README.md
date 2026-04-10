# RadGraph GAT — Head & Neck Cancer Recurrence Prediction
**CMC Vellore | QIRAIL Lab | Radiation Oncology**

Graph Attention Network for predicting locoregional recurrence in head and neck cancer, based on the RadGraph supervoxel approach (Bae et al.).

---

## Table of Contents

1. [What This Code Does](#what-this-code-does)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Environment Verification](#environment-verification)
5. [Data Setup](#data-setup)
6. [Configuration](#configuration)
7. [How to Run](#how-to-run)
8. [Code Flow](#code-flow)
9. [Expected Performance & Timeline](#expected-performance--timeline)
10. [Troubleshooting](#troubleshooting)
11. [Common Error Messages](#common-error-messages)

---

## What This Code Does

Each patient's CT scan is divided into small 3D regions called **supervoxels**. Radiomic features are extracted from each supervoxel and the primary tumour (GTV). These are connected into a **graph** where the GTV is the central node. A **Graph Attention Network (GAT)** learns which supervoxels are most important for predicting recurrence.

```
CT Scan + GTV Mask
       ↓
  Supervoxels (SLIC ~100 per patient)
       ↓
  Radiomic Features per supervoxel (PyRadiomics ~93 features)
       ↓
  Patient Graph (GTV hub + supervoxel nodes, star topology)
       ↓
  GAT Model (attention-weighted message passing)
       ↓
  P(Locoregional Recurrence)
```

---

## Repository Structure

```
radgraph-hnscc/
│
├── config.py                ← ⭐ EDIT YOUR PATHS HERE FIRST
├── utils.py                 ← Metrics, plots, early stopping, checkpointing
├── data_loader.py           ← Loads DICOM CT scans + RT structures (GTV mask)
├── preprocessing.py         ← CT resampling to 1mm isotropic, peritumoral region
├── supervoxel_generator.py  ← SLIC supervoxel generation (~100 per patient)
├── feature_extractor.py     ← PyRadiomics features per supervoxel → .npz cache
├── graph_builder.py         ← Builds PyTorch Geometric star graphs
├── dataset.py               ← PyTorch Dataset, DataLoader, train/val/test splits
├── model.py                 ← GAT layers + GTV readout + classifier
├── train.py                 ← Training loop, early stopping, K-fold CV
├── evaluate.py              ← AUC + CI, ROC curves, attention extraction
├── main_simple.py           ← Baseline Random Forest (no graphs, works immediately)
├── main.py                  ← ⭐ MASTER PIPELINE — run this
│
├── requirements.txt         ← All Python dependencies
└── install_deps.sh          ← Automated installer (handles dependency ordering)
```

---

## Installation

> **Important:** Never run `pip install -r requirements.txt` all at once.
> Packages must be installed in a specific order or they will fail.

---

### Option 1 — Automated (Recommended) ⭐

```bash
conda create -n GNNvenv python=3.9
conda activate GNNvenv
chmod +x install_deps.sh
bash install_deps.sh
```

That is it. The script handles everything in the correct order automatically.

---

### Option 2 — Manual Step by Step

If the automated script fails, copy and paste these commands one at a time:

```bash
conda activate GNNvenv

# Step 1: Core (must be first)
pip install numpy==1.24.3 setuptools

# Step 2: PyTorch (must be before torch-geometric)
pip install torch==2.0.1 torchvision==0.15.2

# Step 3: PyG extensions (CRITICAL — special URL required)
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Step 4: Medical imaging
pip install SimpleITK==2.2.1 pydicom==2.3.1 rt-utils==1.2.7

# Step 5: PyRadiomics (must come after numpy)
pip install pyradiomics==3.0.1

# Step 6: PyTorch Geometric (must come after Step 3)
pip install torch-geometric==2.3.1

# Step 7: ML and image processing
pip install scikit-learn==1.3.0 scikit-image==0.21.0 scipy==1.11.1 \
    mrmr-selection==0.2.6 opencv-python==4.8.0

# Step 8: Data and visualisation
pip install pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2

# Step 9: Utilities
pip install tqdm==4.65.0 pyyaml==6.0 joblib==1.3.2
```

**Why this order matters:**
```
numpy, setuptools       ← needed by everything else
      ↓
torch, torchvision      ← needed by torch-geometric
      ↓
torch-scatter/sparse    ← needed by torch-geometric
      ↓
torch-geometric         ← graph neural networks
      ↓
everything else         ← medical imaging, ML, visualisation
```

---

### Option 3 — Minimal (Baseline Only, Start Immediately)

If you just want to run the Random Forest baseline right now using your existing radiomics CSV:

```bash
conda activate GNNvenv
pip install numpy pandas scikit-learn matplotlib seaborn joblib tqdm
python main_simple.py --task LR --split_data --train --evaluate
```

Install the rest later when you are ready for the GNN pipeline.

---

### What You Can Do Without Full Installation

| Packages Installed | What Works |
|---|---|
| numpy, pandas, scikit-learn only | ✅ Load CSV, train baseline RF, evaluate, ROC curves |
| + PyTorch + torch-geometric | ✅ Build graphs, train GAT |
| + PyRadiomics | ✅ Extract supervoxel features from scratch |
| + SimpleITK + rt-utils | ✅ Load CT scans and RT structures |

---

## Environment Verification

After installation, run these checks to confirm everything works:

```bash
# Quick individual checks
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import radiomics; print(f'PyRadiomics: {radiomics.__version__}')"
python -c "import SimpleITK; print(f'SimpleITK: {SimpleITK.__version__}')"
```

**Full diagnostic** — shows exactly what is and is not installed:

```bash
python << EOF
packages = ['numpy', 'pandas', 'sklearn', 'torch', 'torch_geometric',
            'SimpleITK', 'radiomics', 'matplotlib', 'skimage', 'cv2']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  OK  {pkg}")
    except:
        print(f"  MISSING  {pkg}")
EOF
```

**Test each module individually:**

```bash
python data_loader.py           # Tests CT + RT loading
python preprocessing.py         # Tests CT resampling
python supervoxel_generator.py  # Tests SLIC supervoxel generation
python utils.py                 # Tests metrics and plotting
```

Each should print `...test successful!` at the end.

**Success checklist before proceeding:**

- [ ] `import torch` works
- [ ] `import torch_geometric` works
- [ ] `import sklearn` works
- [ ] `import pandas` works
- [ ] `python data_loader.py` runs without errors
- [ ] `python preprocessing.py` runs without errors

---

## Data Setup

Organise your data folder like this:

```
data/
├── ct_scans/
│   ├── P001/
│   │   ├── CT.001.dcm
│   │   ├── CT.002.dcm
│   │   └── ...
│   ├── P002/
│   └── ...
├── rt_structs/
│   ├── P001.dcm
│   ├── P002.dcm
│   └── ...
├── clinical_data.csv
└── radiomics_features.csv      ← your already-extracted features
```

**clinical_data.csv** — required columns:

```csv
patient_id,age,sex,hpv_status,ajcc_stage,ecog_status,concurrent_chemo,
tumor_subsite,tumor_volume,locoregional_recurrence,distant_metastasis,followup_months

P001,62,1,1,4,0,1,0,15234.5,0,0,36
P002,58,0,0,3,1,1,0,8932.1,1,0,28
```

**radiomics_features.csv** — your existing extracted features:

```csv
patient_id,original_firstorder_Mean,original_glcm_Contrast,...
P001,45.2,125.6,...
P002,38.9,145.2,...
```

---

## Configuration

Open `config.py` and update **lines 21–24** with your actual paths:

```python
CT_SCANS_DIR            = DATA_DIR / "ct_scans"
RTSTRUCT_DIR            = DATA_DIR / "rt_structs"
CLINICAL_DATA_FILE      = DATA_DIR / "clinical_data.csv"
RADIOMICS_FEATURES_FILE = DATA_DIR / "radiomics_features.csv"
```

Verify your **column names** match what is in `config.py`:

```python
# Lines 38–47: must match your CSV column names exactly
CLINICAL_FEATURES = [
    'age', 'sex', 'hpv_status', 'ajcc_stage',
    'ecog_status', 'concurrent_chemo', 'tumor_subsite', 'tumor_volume'
]
OUTCOME_LR    = 'locoregional_recurrence'
OUTCOME_DM    = 'distant_metastasis'
FOLLOWUP_TIME = 'followup_months'
```

Add your GTV contour name if it differs from the defaults:

```python
# Line 72
GTV_NAMES = ['GTV', 'GTVp', 'GTV_Primary', 'GTV_T', 'YOUR_CONTOUR_NAME']
```

---

## How to Run

### Step 0 — Baseline (Run This First, No CT Needed)

Uses your existing radiomics CSV directly. Gets you a working AUC immediately.

```bash
python main.py --task LR --stage baseline
```

Or equivalently:

```bash
python main_simple.py --task LR --split_data --train --evaluate
```

Expected output:
```
Train: 123 patients  |  Val: 26 patients  |  Test: 27 patients

Test Set Metrics:
AUC:         0.7189
Sensitivity: 0.6000
Specificity: 0.7895
```

---

### Step 1 — Preprocess CT Scans + Generate Supervoxels

Resamples CT to 1mm isotropic spacing, defines the 5cm peritumoral region, and runs SLIC to generate ~100 supervoxels per patient.

```bash
python main.py --task LR --stage preprocess
```

Output: `outputs/preprocessed/<patient_id>_preprocessed.npz`

Expected output per patient:
```
Original CT size: (512, 512, 120)
Resampled CT size: (498, 498, 360)
Generated 94 supervoxels
```

---

### Step 2 — Extract Radiomic Features

Extracts ~93 PyRadiomics features from each supervoxel and the whole GTV. Results are cached so you never need to re-run this step.

```bash
python main.py --task LR --stage extract
```

Output: `outputs/features_cache/<patient_id>_features.npz`

To process a single patient:

```bash
python feature_extractor.py --patient_id P001
```

---

### Step 3 — Build Patient Graphs

Selects the top 20 supervoxels most similar to the GTV and builds a star-topology graph for each patient.

```bash
python main.py --task LR --stage graph
```

Output: `outputs/graphs/<patient_id>_LR.pt`

To build a single graph:

```bash
python graph_builder.py --patient_id P001 --task LR
```

---

### Step 4 — Train GAT Model

Trains the Graph Attention Network with early stopping and best model checkpointing.

```bash
python main.py --task LR --stage train
```

With K-fold cross-validation (recommended for small datasets):

```bash
python main.py --task LR --stage train --use_kfold --n_folds 5
```

Output: `models/best_model_LR.pth`

Expected output:
```
Epoch  10/100 | Train Loss: 0.4821 | Val Loss: 0.5102 | Val AUC: 0.7234
Epoch  20/100 | Train Loss: 0.3904 | Val Loss: 0.4823 | Val AUC: 0.7511
...
Best val AUC: 0.7734 at epoch 67
```

---

### Step 5 — Evaluate + Attention Maps

Full evaluation with bootstrap 95% CI, ROC curve, confusion matrix, and attention weight extraction.

```bash
python main.py --task LR --stage evaluate --attention --compare
```

---

### Run Everything at Once

```bash
python main.py --task LR --all
```

Start from a specific stage if earlier stages are already complete:

```bash
python main.py --task LR --from_stage graph      # skip preprocessing
python main.py --task LR --from_stage train      # skip graph building
python main.py --task LR --from_stage evaluate --attention --compare
```

---

### Output Files After Full Pipeline

```
outputs/
├── preprocessed/               ← CT arrays + supervoxel label maps (.npz)
├── features_cache/             ← Per-supervoxel radiomic features (.npz)
├── graphs/                     ← Patient graph objects (.pt)
├── splits/                     ← Train/val/test patient ID lists (.npy)
├── attention_maps/             ← Per-patient attention weights
├── metrics_LR.json             ← AUC, sensitivity, specificity + 95% CI
├── roc_curve_LR.png            ← ROC curve
├── confusion_matrix_LR.png     ← Confusion matrix
├── prob_histogram_LR.png       ← Prediction probability distribution
├── training_history_LR.png     ← Loss and AUC per epoch
├── test_predictions_LR.csv     ← Per-patient predictions
├── model_comparison_LR.csv     ← GAT vs Baseline comparison table
└── kfold_results_LR.json       ← K-fold results (if used)

models/
└── best_model_LR.pth           ← Best model weights
```

---

## Code Flow

How all files connect to each other:

```
config.py  ←  imported by every file (paths, hyperparameters)
    │
    ├── data_loader.py
    │       Reads DICOM CT series + extracts GTV mask from RT structure
    │       ↓
    ├── preprocessing.py
    │       Resamples CT to 1mm isotropic
    │       Defines 5cm peritumoral region around GTV
    │       ↓
    ├── supervoxel_generator.py
    │       SLIC algorithm: divides peritumoral region into ~100 supervoxels
    │       ↓
    ├── feature_extractor.py
    │       PyRadiomics: extracts ~93 features from GTV + each supervoxel
    │       Saves to .npz cache (skips patients already processed)
    │       ↓
    ├── graph_builder.py
    │       Selects top-20 supervoxels most similar to GTV
    │       Builds star graph: GTV node (hub) + supervoxel nodes (spokes)
    │       Edge features = 3D Euclidean distance between centroids
    │       ↓
    ├── dataset.py
    │       Wraps graphs in PyTorch Dataset
    │       Attaches clinical features to each graph
    │       Handles train/val/test splits and K-fold
    │       ↓
    ├── model.py
    │       Input projection → GAT layers → GTV node readout
    │       Concatenate clinical features → FC classifier → sigmoid
    │       ↓
    ├── train.py
    │       Training loop + early stopping + LR scheduling
    │       Saves best model checkpoint
    │       ↓
    ├── evaluate.py
    │       AUC + bootstrap 95% CI
    │       ROC curve, confusion matrix, probability histogram
    │       Attention weight extraction per patient
    │
    └── main.py   ←  orchestrates all of the above in sequence
```

Parallel path — baseline model (no graphs needed):

```
config.py
    └── main_simple.py
            Loads your existing radiomics CSV directly
            Feature selection using Random Forest importance
            Trains Random Forest with class balancing
            Outputs AUC, ROC curve, predictions CSV
```

---

## Expected Performance & Timeline

### Performance

| Model | AUC | Notes |
|---|---|---|
| Clinical only | 0.60 – 0.65 | TNM staging alone |
| Baseline Random Forest | 0.65 – 0.75 | Your existing CSV features |
| RadGraph GAT | 0.75 – 0.83 | Full pipeline |

With ~200 patients the approximate data split will be:
- Training: ~140 patients
- Validation: ~30 patients
- Test: ~30 patients

Training time: 1–2 hours on CPU, 20–40 minutes on GPU. Inference: less than 1 second per patient.

### Timeline

| Phase | Time | What Happens |
|---|---|---|
| Setup + data organisation | 1 day | Install, organise folders, edit config |
| Baseline model | 30 min | Random Forest on existing CSV |
| Test data loader | 1 day | Verify CT and RT loading, fix issues |
| Preprocessing + supervoxels | 2 days | Resampling + SLIC for all patients |
| Feature extraction | 2 days | PyRadiomics per supervoxel (slow, cached) |
| Graph building | 1 day | Build and verify patient graphs |
| GAT training | 3 days | Train, tune, evaluate |
| **Total** | **~10 days** | Full pipeline from scratch |

---

## Troubleshooting

### torch-scatter / torch-sparse will not install

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement torch-scatter
```

**Fix A — use the PyG wheel repo (try this first):**
```bash
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Fix B — without version pinning:**
```bash
pip install torch-scatter torch-sparse torch-cluster
```

**Fix C — build from source:**
```bash
pip install torch-scatter torch-sparse torch-cluster --no-binary :all:
```

**Fix D — skip and run baseline only:**
```bash
python main_simple.py --task LR --split_data --train --evaluate
```

---

### pyradiomics compilation error

**Symptom:**
```
error: command 'gcc' failed
Building wheel for pyradiomics (setup.py) ... error
```

**Fix A — install build tools:**
```bash
sudo apt-get install build-essential python3-dev
pip install pyradiomics==3.0.1
```

**Fix B — use pre-built wheel:**
```bash
pip install pyradiomics --prefer-binary
```

**Fix C — skip (you already have features in your CSV):**
Your existing `radiomics_features.csv` is sufficient to run the baseline. You only need pyradiomics if you want to extract per-supervoxel features for the GNN.

---

### rt-utils error

**Symptom:**
```
ERROR: rt-utils requires specific package versions
```

**Fix:**
```bash
pip install rt-utils==1.2.7
```

---

### CUDA not available

**Symptom:**
```
RuntimeError: CUDA not available
```

**If you have a GPU:**
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**If you do not have a GPU:**
```python
# In config.py
USE_CUDA = False
DEVICE   = 'cpu'
```

---

### GTV contour not found

**Symptom:**
```
Warning: No GTV contour found for patient P001
Available ROIs: ['PTV', 'Spinal_Cord', ...]
```

**Fix:** Add your contour name to `config.py`:
```python
GTV_NAMES = ['GTV', 'GTVp', 'GTV_Primary', 'GTV_T', 'YOUR_ACTUAL_NAME']
```

---

### Package version conflicts

**Fix A:**
```bash
pip install <package> --no-deps
```

**Fix B — use conda:**
```bash
conda install pytorch torchvision -c pytorch
conda install scikit-learn pandas matplotlib seaborn -c conda-forge
```

**Fix C — nuclear option (fresh environment):**
```bash
conda deactivate
conda env remove -n GNNvenv -y
conda clean --all -y
conda create -n GNNvenv python=3.9 -y
conda activate GNNvenv
bash install_deps.sh
```

---

### Out of memory

**Fix:**
```python
# In config.py
BATCH_SIZE = 4   # default is 8
```

---

### CSV column name mismatch

**Symptom:**
```
KeyError: 'age'
```

**Fix:** Print your actual column names and update `config.py`:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/clinical_data.csv'); print(df.columns.tolist())"
```

---

## Common Error Messages

| Error | Meaning | Fix |
|---|---|---|
| `No module named 'numpy'` | numpy not installed | `pip install numpy` first |
| `No module named 'torch'` | torch not installed | install torch before torch-geometric |
| `Could not find torch-scatter` | wrong install method | use PyG wheel URL |
| `gcc failed` | missing build tools | `sudo apt install build-essential` |
| `CUDA not available` | no GPU or wrong version | set `USE_CUDA = False` in config.py |
| `Metadata generation failed` | dependency missing | install in correct order |
| `KeyError: 'patient_id'` | wrong column name | check CSV column names in config.py |
| `FileNotFoundError` | wrong path | update paths in config.py lines 21–24 |
| `ValueError: incompatible dimensions` | feature count mismatch | check N_FEATURES_TOTAL in config.py |
| `No GTV contour found` | wrong contour name | add contour name to GTV_NAMES |
