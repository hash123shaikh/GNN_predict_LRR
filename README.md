# RadGraph GAT — Head & Neck Cancer Recurrence Prediction
**CMC Vellore | QIRAIL Lab | Radiation Oncology**

Graph Attention Network for predicting locoregional recurrence in head and neck cancer patients, based on the RadGraph supervoxel approach (Bae et al.).

---

## What This Code Does

Each patient's CT scan is divided into small 3D regions called **supervoxels**. Radiomic features are extracted from each supervoxel and the primary tumour (GTV). These are connected into a **graph** where the GTV is the central node. A **Graph Attention Network (GAT)** learns which supervoxels are most important for predicting recurrence.

```
CT Scan + GTV Mask
       ↓
  Supervoxels (SLIC)
       ↓
  Radiomic Features (PyRadiomics)
       ↓
  Patient Graph (GTV node + supervoxel nodes)
       ↓
  GAT Model
       ↓
  P(Locoregional Recurrence)
```

---

## Repository Structure

```
radgraph-hnscc/
│
├── config.py                ← ⭐ Edit your data paths here first
├── utils.py                 ← Metrics, plots, checkpointing
├── data_loader.py           ← Loads CT scans + RT structures (DICOM)
├── preprocessing.py         ← CT resampling to 1mm isotropic
├── supervoxel_generator.py  ← SLIC supervoxel generation (~100 per patient)
├── feature_extractor.py     ← PyRadiomics features per supervoxel
├── graph_builder.py         ← Builds PyTorch Geometric graphs
├── dataset.py               ← PyTorch Dataset + DataLoader + splits
├── model.py                 ← GAT architecture + loss functions
├── train.py                 ← Training loop + early stopping
├── evaluate.py              ← AUC, ROC curves, attention weights
├── main_simple.py           ← Baseline Random Forest (no graphs)
├── main.py                  ← ⭐ Master pipeline — run this
│
├── requirements.txt         ← Python dependencies
└── install_deps.sh          ← Automated installer
```

---

## Installation

```bash
# Step 1: Create conda environment
conda create -n radgraph python=3.9
conda activate radgraph

# Step 2: Install all dependencies (handles ordering automatically)
chmod +x install_deps.sh
bash install_deps.sh

# Step 3: Verify
python -c "import torch, torch_geometric, radiomics; print('All OK')"
```

---

## Data Setup

Your data folder should look like this:

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
└── radiomics_features.csv   ← your already-extracted features
```

**clinical_data.csv** format:
```
patient_id, age, sex, hpv_status, ajcc_stage, ecog_status,
concurrent_chemo, tumor_subsite, locoregional_recurrence,
distant_metastasis, followup_months
```

Then open `config.py` and update **lines 21–24**:
```python
CT_SCANS_DIR           = DATA_DIR / "ct_scans"
RTSTRUCT_DIR           = DATA_DIR / "rt_structs"
CLINICAL_DATA_FILE     = DATA_DIR / "clinical_data.csv"
RADIOMICS_FEATURES_FILE= DATA_DIR / "radiomics_features.csv"
```

---

## How to Run

### Option A — Quick Baseline (Start Here, No CT Processing Needed)
Uses your existing radiomics CSV directly. Run this first to get a baseline AUC.

```bash
python main.py --task LR --stage baseline
```
Expected AUC: **0.65 – 0.75**

---

### Option B — Full GNN Pipeline (Stage by Stage)

Run each stage separately so you can check outputs before proceeding.

**Stage 1 — Preprocess CT scans + generate supervoxels**
```bash
python main.py --task LR --stage preprocess
```
Output: `outputs/preprocessed/<patient_id>_preprocessed.npz`

---

**Stage 2 — Extract radiomic features from supervoxels**
```bash
python main.py --task LR --stage extract
```
Output: `outputs/features_cache/<patient_id>_features.npz`

---

**Stage 3 — Build patient graphs**
```bash
python main.py --task LR --stage graph
```
Output: `outputs/graphs/<patient_id>_LR.pt`

---

**Stage 4 — Train GAT model**
```bash
python main.py --task LR --stage train
```
Output: `models/best_model_LR.pth`

For K-fold cross-validation:
```bash
python main.py --task LR --stage train --use_kfold --n_folds 5
```

---

**Stage 5 — Evaluate + attention maps**
```bash
python main.py --task LR --stage evaluate --attention --compare
```
Output: `outputs/metrics_LR.json`, `outputs/roc_curve_LR.png`,
`outputs/attention_weights_LR.csv`, `outputs/model_comparison_LR.csv`

---

### Option C — Run Everything at Once
```bash
python main.py --task LR --all
```

---

## Output Files

After running the full pipeline you will have:

```
outputs/
├── preprocessed/              ← CT arrays + supervoxel label maps
├── features_cache/            ← Per-supervoxel radiomic features
├── graphs/                    ← Patient graph .pt files
├── splits/                    ← Train/val/test patient ID lists
├── attention_maps/            ← Per-patient attention weights
├── metrics_LR.json            ← AUC, sensitivity, specificity + 95% CI
├── roc_curve_LR.png           ← ROC curve plot
├── confusion_matrix_LR.png    ← Confusion matrix
├── prob_histogram_LR.png      ← Prediction distribution
├── training_history_LR.png    ← Loss + AUC per epoch
├── test_predictions_LR.csv    ← Per-patient predictions
└── model_comparison_LR.csv    ← GAT vs Baseline table

models/
└── best_model_LR.pth          ← Best model weights
```

---

## Code Flow (How Files Connect)

```
config.py          ← All settings, imported by every file
    │
    ├── data_loader.py         ← Reads DICOM CT + extracts GTV mask
    │       ↓
    ├── preprocessing.py       ← Resamples CT, defines peritumoral region
    │       ↓
    ├── supervoxel_generator.py← SLIC: divides region into ~100 supervoxels
    │       ↓
    ├── feature_extractor.py   ← PyRadiomics per supervoxel → .npz cache
    │       ↓
    ├── graph_builder.py       ← Star graph: GTV node + K supervoxel nodes
    │       ↓
    ├── dataset.py             ← PyTorch Dataset, splits, clinical fusion
    │       ↓
    ├── model.py               ← GAT layers → GTV readout → classifier
    │       ↓
    ├── train.py               ← Training loop, early stopping, checkpoints
    │       ↓
    ├── evaluate.py            ← AUC + CI, ROC, attention extraction
    │
    └── main.py                ← Calls all of the above in sequence
```

---

## Expected Performance

| Model | AUC |
|---|---|
| Clinical only | 0.60 – 0.65 |
| Baseline Random Forest (your CSV) | 0.65 – 0.75 |
| RadGraph GAT (full pipeline) | 0.75 – 0.83 |

---

## Troubleshooting

**torch-scatter install fails**
```bash
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**GTV contour not found**
Open `config.py` and add your contour name to `GTV_NAMES`:
```python
GTV_NAMES = ['GTV', 'GTVp', 'GTV_Primary', 'YOUR_CONTOUR_NAME']
```

**Out of memory**
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 4
```

**Column name mismatch**
Check your CSV column names match `CLINICAL_FEATURES` in `config.py`.
