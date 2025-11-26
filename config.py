"""
Configuration file for RadGraph implementation
Adjust paths and parameters according to your data and requirements
"""

import os
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

# Total number of patients
N_PATIENTS = 176

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
# ============================================================================

# Number of features to select for each task
N_FEATURES_LR = 4  # For locoregional recurrence
N_FEATURES_DM = 6  # For distant metastasis

# mRMR parameters
MRMR_METHOD = 'MIQ'  # 'MIQ' or 'MID'
MRMR_N_CANDIDATES = 20  # Top candidates to consider

# Random Forest parameters for feature refinement
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
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
# ============================================================================

# GAT hyperparameters - TUNE THESE ON VALIDATION SET
GAT_HIDDEN_DIM = 64  # Hidden dimension size
GAT_N_HEADS = 4  # Number of attention heads
GAT_N_LAYERS = 2  # Number of GAT layers
GAT_DROPOUT = 0.3  # Dropout rate
GAT_NEGATIVE_SLOPE = 0.2  # LeakyReLU negative slope
GAT_CONCAT_HEADS = True  # Concatenate attention heads (True for hidden layers)

# Final layer configuration
GAT_FINAL_HEADS = 1  # Typically 1 for final layer (average instead of concat)

# Output dimension after GAT
GAT_OUTPUT_DIM = GAT_HIDDEN_DIM * GAT_N_HEADS if GAT_CONCAT_HEADS else GAT_HIDDEN_DIM

# Clinical feature integration
N_CLINICAL_FEATURES = len(CLINICAL_FEATURES)
COMBINE_METHOD = 'concatenate'  # How to combine GAT output with clinical features

# Final classification layer
FC_HIDDEN_DIM = 32  # Hidden dimension for final FC layer (optional)
USE_FC_HIDDEN = False  # Whether to use hidden layer before output

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
BATCH_SIZE = 8  # Small batch size for small graphs
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Optimizer
OPTIMIZER = 'Adam'  # 'Adam', 'SGD', or 'AdamW'
SCHEDULER = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau', 'StepLR', or None
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# Class imbalance handling
HANDLE_IMBALANCE = True
IMBALANCE_METHOD = 'weighted_loss'  # 'weighted_loss', 'focal_loss', or 'oversample'

# Focal loss parameters (if using focal loss)
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

# Device
USE_CUDA = True
DEVICE = 'cuda' if USE_CUDA else 'cpu'

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
LOG_INTERVAL = 10  # Log every N batches
SAVE_BEST_MODEL = True
SAVE_CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs

# Model checkpoint
CHECKPOINT_PATH = MODEL_DIR / f'checkpoint_{TASK}.pth'
BEST_MODEL_PATH = MODEL_DIR / f'best_model_{TASK}.pth'

# Results
RESULTS_FILE = OUTPUT_DIR / f'results_{TASK}.csv'
PREDICTIONS_FILE = OUTPUT_DIR / f'predictions_{TASK}.csv'
ATTENTION_MAPS_DIR = OUTPUT_DIR / 'attention_maps'

# Create attention maps directory
ATTENTION_MAPS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_n_features_for_task(task):
    """Get number of features for specific task"""
    return N_FEATURES_LR if task == 'LR' else N_FEATURES_DM

def get_outcome_column(task):
    """Get outcome column name for specific task"""
    return OUTCOME_LR if task == 'LR' else OUTCOME_DM

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("RadGraph Configuration")
    print("=" * 80)
    print(f"Task: {TASK}")
    print(f"Number of patients: {N_PATIENTS}")
    print(f"Number of features: {get_n_features_for_task(TASK)}")
    print(f"GAT Architecture: {GAT_N_LAYERS} layers, {GAT_HIDDEN_DIM} hidden dim, {GAT_N_HEADS} heads")
    print(f"Training: {N_EPOCHS} epochs, batch size {BATCH_SIZE}, LR {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    print_config()
