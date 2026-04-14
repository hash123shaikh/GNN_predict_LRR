"""
Configuration file for RadGraph implementation
Adjust paths and parameters according to your data and requirements
"""

import os
import torch
from pathlib import Path

# ============================================================================
# PATHS - MODIFY THESE TO MATCH YOUR DATA STRUCTURE
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Input data paths - MODIFY THESE
CT_SCANS_DIR = DATA_DIR / "ct_scans"  # Folder containing patient CT folders
RTSTRUCT_DIR = DATA_DIR / "rt_structs"  # Folder containing RT structure files
CLINICAL_DATA_FILE = DATA_DIR / "clinical_data.csv"  # CSV with clinical features
RADIOMICS_FEATURES_FILE = DATA_DIR / "radiomics_features.csv"  # Your extracted features

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# NOTE: Patient count is determined dynamically at runtime from clinical_data.csv
# Do not hardcode N_PATIENTS here — it will silently diverge from actual data.

# Clinical features to use (column names in your CSV)
CLINICAL_FEATURES = [
    'age',
    'sex',  # 0=Female, 1=Male
    'hpv_status',  # 0=Negative, 1=Positive, 2=Unknown
    'ajcc_stage',  # 0, 1, 2, 3, 4
    'ecog_status',  # 0, 1, 2, 3, 4
    'concurrent_chemo',  # 0=No, 1=Yes
    'tumor_subsite',  # 0=Oropharynx, 1=Larynx, 2=Nasopharynx, 3=Other
    'tumor_volume'  # In mm³
]

# Outcome variables (column names in your CSV)
OUTCOME_LR = 'locoregional_recurrence'  # 0=No, 1=Yes
OUTCOME_DM = 'distant_metastasis'  # 0=No, 1=Yes
FOLLOWUP_TIME = 'followup_months'  # Follow-up duration in months

# Minimum follow-up required (in months)
MIN_FOLLOWUP_MONTHS = 24

# Patient ID column name
PATIENT_ID_COL = 'patient_id'

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# CT preprocessing
TARGET_SPACING = [1.0, 1.0, 1.0]  # Isotropic 1mm spacing
INTERPOLATION = 'linear'  # 'linear' or 'nearest'

# Peritumoral region definition
PERITUMORAL_MARGIN_MM = 50  # 5cm as in the paper

# GTV contour name patterns (will try these in order)
GTV_NAMES = ['GTV', 'GTV_Primary', 'GTVp', 'GTV_T', 'CTV', 'GTV1']

# ============================================================================
# SUPERVOXEL GENERATION (SLIC)
# ============================================================================

# Target number of supervoxels
N_SUPERVOXELS_TARGET = 100

# SLIC parameters (tune these on validation set)
SLIC_COMPACTNESS = 10  # Balance between spatial and intensity similarity
                       # Higher = more compact/spherical supervoxels
                       # Lower = follows intensity boundaries more
                       # Typical range: 1-50

SLIC_SIGMA = 1  # Gaussian smoothing before segmentation
SLIC_MAX_ITER = 10  # Maximum iterations for SLIC
SLIC_ENFORCE_CONNECTIVITY = True

# ============================================================================
# RADIOMIC FEATURE EXTRACTION
# ============================================================================

# PyRadiomics settings
PYRADIOMICS_SETTINGS = {
    'binWidth': 25,
    'resampledPixelSpacing': TARGET_SPACING,
    'interpolator': 'sitkBSpline',
    'normalize': True,
    'normalizeScale': 100,
    'removeOutliers': None,
}

# Feature classes to extract (all by default)
FEATURE_CLASSES = [
    'firstorder',
    'glcm',
    'glrlm',
    'glszm',
    'gldm',
    'ngtdm'
]

# Total expected features per region
N_FEATURES_TOTAL = 93

# ============================================================================
# FEATURE SELECTION
# Table S3 — Bae et al. RadGraph paper
# Exact radiomic features selected via mRMR for each task
# ============================================================================

# Number of features to select for each task
N_FEATURES_LR = 4   # Table S3: 4 features for LR
N_FEATURES_DM = 6   # Table S3: 6 features for DM

# ── Table S3: Exact selected features (PyRadiomics naming convention) ────────
#
# LR features (4):
#   GLSZM Gray Level Non-Uniformity      → original_glszm_GrayLevelNonUniformity
#   GLCM Inverse Difference              → original_glcm_Id
#   GLCM Inverse Difference Normalized   → original_glcm_Idn
#   GLSZM Zone Entropy                   → original_glszm_ZoneEntropy
#
# DM features (6):
#   NGTDM Coarseness                               → original_ngtdm_Coarseness
#   First Order Mean Intensity                     → original_firstorder_Mean
#   GLDM Large Dependence Low Gray Level Emphasis  → original_gldm_LargeDependenceLowGrayLevelEmphasis
#   GLCM Cluster Prominence                        → original_glcm_ClusterProminence
#   GLSZM Low Gray Level Zone Emphasis             → original_glszm_LowGrayLevelZoneEmphasis
#   GLSZM Small Area Emphasis                      → original_glszm_SmallAreaEmphasis

SELECTED_FEATURES = {
    'LR': [
        'original_glszm_GrayLevelNonUniformity',
        'original_glcm_Id',
        'original_glcm_Idn',
        'original_glszm_ZoneEntropy',
    ],
    'DM': [
        'original_ngtdm_Coarseness',
        'original_firstorder_Mean',
        'original_gldm_LargeDependenceLowGrayLevelEmphasis',
        'original_glcm_ClusterProminence',
        'original_glszm_LowGrayLevelZoneEmphasis',
        'original_glszm_SmallAreaEmphasis',
    ],
}

def get_selected_features(task='LR'):
    """Return Table S3 feature names for a given task."""
    return SELECTED_FEATURES.get(task, SELECTED_FEATURES['LR'])

# mRMR parameters (used for custom feature selection experiments)
MRMR_METHOD       = 'MIQ'   # 'MIQ' or 'MID'
MRMR_N_CANDIDATES = 20

# Random Forest parameters for feature refinement
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 10
RF_RANDOM_STATE = 42

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Number of supervoxels to select for graph (top-k most similar to GTV)
N_SUPERVOXELS_SELECTED = 20

# Total nodes in graph
N_NODES = N_SUPERVOXELS_SELECTED + 1  # +1 for GTV node

# Distance metric for supervoxel selection
DISTANCE_METRIC = 'euclidean'  # 'euclidean' or 'cosine'

# Edge weight calculation
EDGE_WEIGHT_METHOD = 'inverse_distance'  # 'inverse_distance', 'gaussian', or 'uniform'
EDGE_WEIGHT_SIGMA = 1.0  # For Gaussian edge weights

# ============================================================================
# MODEL ARCHITECTURE (GAT)
# Values from Table S2 — Bae et al. RadGraph paper
# ============================================================================

# Per-task GAT hyperparameters (Table S2)
# Use get_gat_config(task) to retrieve the correct values at runtime.
#
#  Hyperparameter       LR      DM
#  -------------------  ------  ------
#  GAT Layers           5       3
#  Hidden Feature Dim   16      8
#  Attention Heads      4       4
#  Dropout              0.67    0.89
#  Learning Rate        4e-4    3e-4
#  Weight Decay         7e-4    8e-4
#  LR Warmup Steps      10      5
#  Batch Size           64      64
#  Neighbors (K)        20      20
#  Sampling Strategy    interm. over

GAT_CONFIG = {
    'LR': {
        'n_layers'        : 5,
        'hidden_dim'      : 16,
        'n_heads'         : 4,
        'dropout'         : 0.67,
        'learning_rate'   : 4e-4,
        'weight_decay'    : 7e-4,
        'warmup_steps'    : 10,
        'batch_size'      : 64,
        'n_neighbors'     : 20,
        'sampling_strategy': 'intermediate',  # under/over/intermediate
    },
    'DM': {
        'n_layers'        : 3,
        'hidden_dim'      : 8,
        'n_heads'         : 4,
        'dropout'         : 0.89,
        'learning_rate'   : 3e-4,
        'weight_decay'    : 8e-4,
        'warmup_steps'    : 5,
        'batch_size'      : 64,
        'n_neighbors'     : 20,
        'sampling_strategy': 'over',
    },
}

def get_gat_config(task='LR'):
    """Return Table S2 hyperparameters for a given task."""
    return GAT_CONFIG.get(task, GAT_CONFIG['LR'])

# Convenience accessors (default to LR values; overridden at runtime via get_gat_config)
_gat = GAT_CONFIG['LR']
GAT_N_LAYERS      = _gat['n_layers']       # 5  (LR)
GAT_HIDDEN_DIM    = _gat['hidden_dim']     # 16 (LR)
GAT_N_HEADS       = _gat['n_heads']        # 4  (LR)
GAT_DROPOUT       = _gat['dropout']        # 0.67 (LR)
GAT_NEGATIVE_SLOPE= 0.2                    # LeakyReLU slope (not tuned in paper)
GAT_CONCAT_HEADS  = True                   # Concat heads in hidden layers
GAT_FINAL_HEADS   = 1                      # Average in final layer

# Output dimension after GAT
GAT_OUTPUT_DIM = GAT_HIDDEN_DIM * GAT_N_HEADS if GAT_CONCAT_HEADS else GAT_HIDDEN_DIM

# Clinical feature integration
N_CLINICAL_FEATURES = len(CLINICAL_FEATURES)
COMBINE_METHOD      = 'concatenate'

# Final classification layer
FC_HIDDEN_DIM = 32
USE_FC_HIDDEN = False

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Task to train ('LR' or 'DM')
TASK = 'LR'  # Change to 'DM' for distant metastasis

# Data split
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15  # 15% for testing
RANDOM_SEED = 42

# Training hyperparameters
# Note: BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY are task-specific (Table S2).
# The values below are defaults for LR task. Use get_gat_config(task) at runtime.
BATCH_SIZE            = 64      # Table S2: 64 for both LR and DM
LEARNING_RATE         = 4e-4    # Table S2: 4e-4 (LR), 3e-4 (DM)
WEIGHT_DECAY          = 7e-4    # Table S2: 7e-4 (LR), 8e-4 (DM)
N_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 15

# Optimizer — Table S2: Adam for both tasks
OPTIMIZER         = 'Adam'
SCHEDULER         = 'WarmupReduceLROnPlateau'  # warmup + ReduceLROnPlateau
SCHEDULER_PATIENCE= 5
SCHEDULER_FACTOR  = 0.5
LR_WARMUP_STEPS   = 10    # Table S2: 10 (LR), 5 (DM)

# Class imbalance — Table S2 sampling strategies
# LR: 'intermediate' (mix of under + oversampling)
# DM: 'over'         (oversample minority class)
HANDLE_IMBALANCE   = True
IMBALANCE_METHOD   = 'intermediate'  # 'weighted_loss', 'over', 'under', 'intermediate'

# Focal loss parameters (used when IMBALANCE_METHOD = 'weighted_loss')
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Loss function
LOSS_FUNCTION = 'BCE'  # 'BCE' or 'Focal'

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Metrics to compute
METRICS = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1']

# Threshold for binary classification
CLASSIFICATION_THRESHOLD = 0.5
OPTIMIZE_THRESHOLD = True  # Find optimal threshold on validation set

# Bootstrap parameters for confidence intervals
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Attention map visualization
ATTENTION_COLORMAP = 'hot'
ATTENTION_ALPHA = 0.6  # Transparency for overlay
ATTENTION_THRESHOLD_LOW = 0.3
ATTENTION_THRESHOLD_HIGH = 0.7

# Figures
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'  # 'png', 'pdf', or 'svg'

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================

# Device — checked at runtime so CPU machines don't silently fail
USE_CUDA = True
DEVICE   = 'cuda' if (USE_CUDA and torch.cuda.is_available()) else 'cpu'

if DEVICE == 'cuda':
    print(f"[config] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("[config] No GPU detected — using CPU")

# Multi-processing
NUM_WORKERS = 4  # For data loading
PIN_MEMORY = True

# Reproducibility
TORCH_SEED = 42
NUMPY_SEED = 42

# ============================================================================
# LOGGING AND CHECKPOINTING
# ============================================================================

# Experiment tracking
USE_WANDB = False  # Set to True if you want to use Weights & Biases
WANDB_PROJECT = 'radgraph-hnscc'
WANDB_ENTITY = None  # Your W&B username

# Logging
LOG_INTERVAL          = 10   # Log every N batches
SAVE_BEST_MODEL       = True
SAVE_CHECKPOINT_EVERY = 10   # Save checkpoint every N epochs

# NOTE: Model and result paths are task-specific.
# Use get_model_paths(task) below instead of hardcoded paths.
# The lines below were removed because they always pointed to LR
# regardless of the actual task being run:
#   CHECKPOINT_PATH = MODEL_DIR / f'checkpoint_{TASK}.pth'  ← always LR
#   BEST_MODEL_PATH = MODEL_DIR / f'best_model_{TASK}.pth'  ← always LR
#   RESULTS_FILE    = OUTPUT_DIR / f'results_{TASK}.csv'    ← always LR
#   PREDICTIONS_FILE= OUTPUT_DIR / f'predictions_{TASK}.csv'← always LR
ATTENTION_MAPS_DIR = OUTPUT_DIR / 'attention_maps'

# Create attention maps directory
ATTENTION_MAPS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_n_features_for_task(task):
    """Get number of features for specific task."""
    return N_FEATURES_LR if task == 'LR' else N_FEATURES_DM


def get_outcome_column(task):
    """Get outcome column name for specific task."""
    return OUTCOME_LR if task == 'LR' else OUTCOME_DM


def get_model_paths(task):
    """
    Return task-specific file paths for checkpoints and results.

    Replaces the old hardcoded CHECKPOINT_PATH / BEST_MODEL_PATH which
    always pointed to LR regardless of the actual task being run.

    Parameters
    ----------
    task : str  'LR' or 'DM'

    Returns
    -------
    paths : dict with keys:
        checkpoint   — latest checkpoint (saved every N epochs)
        best_model   — best validation AUC model
        results      — metrics JSON
        predictions  — per-patient predictions CSV
    """
    return {
        'checkpoint'  : MODEL_DIR  / f'checkpoint_{task}.pth',
        'best_model'  : MODEL_DIR  / f'best_model_{task}.pth',
        'results'     : OUTPUT_DIR / f'metrics_{task}.json',
        'predictions' : OUTPUT_DIR / f'test_predictions_{task}.csv',
    }


def print_config(task=None):
    """Print current configuration."""
    task    = task or TASK
    gat_cfg = get_gat_config(task)
    print("=" * 80)
    print("RadGraph Configuration")
    print("=" * 80)
    print(f"Task                : {task}")
    print(f"Device              : {DEVICE}")
    print(f"Selected features   : {get_n_features_for_task(task)}")
    print(f"GAT layers          : {gat_cfg['n_layers']}")
    print(f"GAT hidden dim      : {gat_cfg['hidden_dim']}")
    print(f"GAT attention heads : {gat_cfg['n_heads']}")
    print(f"GAT dropout         : {gat_cfg['dropout']}")
    print(f"Learning rate       : {gat_cfg['learning_rate']}")
    print(f"Batch size          : {gat_cfg['batch_size']}")
    print(f"Sampling strategy   : {gat_cfg['sampling_strategy']}")
    print(f"Epochs              : {N_EPOCHS}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print(f"Model directory     : {MODEL_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    print_config()
