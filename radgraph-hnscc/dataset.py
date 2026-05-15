"""
PyTorch Geometric Dataset for RadGraph
=======================================
Wraps pre-built patient graphs with optional clinical feature fusion.

Two dataset classes:
  RadGraphDataset       — loads graphs from a list or directory of .pt files
  RadGraphDatasetWithClinical — adds clinical features to each graph

Usage:
    from dataset import RadGraphDataset, RadGraphDatasetWithClinical, split_dataset

    dataset = RadGraphDatasetWithClinical(graphs, clinical_df)
    train_set, val_set, test_set = split_dataset(dataset, train_r=0.7, val_r=0.15)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import config


# ─── Core Dataset ─────────────────────────────────────────────────────────────

class RadGraphDataset(Dataset):
    """
    Simple dataset that wraps a list of PyTorch Geometric Data objects.

    Parameters
    ----------
    graphs : list[Data]
    transform : callable, optional
    """

    def __init__(self, graphs, transform=None):
        self.graphs    = graphs
        self.transform = transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.transform:
            graph = self.transform(graph)
        return graph

    @property
    def labels(self):
        """Return all labels as a numpy array."""
        return np.array([g.y.item() for g in self.graphs])

    @property
    def patient_ids(self):
        """Return all patient IDs."""
        return [getattr(g, 'patient_id', str(i)) for i, g in enumerate(self.graphs)]

    def get_class_weights(self):
        """
        Compute positive-class weight for BCE loss.

        Returns
        -------
        pos_weight : float
        """
        y     = self.labels
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0:
            return 1.0
        pos_weight = n_neg / n_pos
        print(f"Class distribution: {int(n_neg)} negative, {int(n_pos)} positive")
        print(f"Positive class weight: {pos_weight:.2f}")
        return float(pos_weight)


# ─── Dataset with Clinical Features ───────────────────────────────────────────

class RadGraphDatasetWithClinical(RadGraphDataset):
    """
    Extends RadGraphDataset by attaching a clinical feature vector to each graph.

    Clinical features are standardised using training set statistics (call
    fit_scaler() on the training split before creating val/test splits).

    Parameters
    ----------
    graphs       : list[Data]
    clinical_df  : pd.DataFrame  — must have config.PATIENT_ID_COL column
    feature_cols : list[str] or None  — clinical columns to use.
                   Defaults to config.CLINICAL_FEATURES.
    scaler       : MinMaxScaler or None  — if None, no scaling applied
    """

    def __init__(self, graphs, clinical_df, feature_cols=None, scaler=None):
        super().__init__(graphs)

        self.feature_cols = feature_cols or config.CLINICAL_FEATURES
        self.scaler       = scaler

        # ── Build encoded clinical feature matrix ─────────────────────────
        # Paper (Appendix S1):
        #   "Categorical variables were one-hot encoded"
        #   "Quantitative variables normalised so that values remained between 0 and 1"
        #
        # Binary categorical  → 0/1  (one-hot for 2-class is just one binary column)
        # Multi-class cat.    → proper one-hot using fixed category lists from config.py
        # Quantitative        → min-max normalised to [0, 1] using training scaler

        raw_df = clinical_df.set_index(config.PATIENT_ID_COL).copy()

        # ── Step 1: Binary categorical columns (0/1) ─────────────────────
        BINARY_MAP = {
            'sex':             {'male': 1, 'female': 0},
            'hpv_status':      {'positive': 1, 'negative': 0},
            'concurrent_chemo':{'concomitant': 1, 'concurrent': 1, 'none': 0},
        }
        for col, mapping in BINARY_MAP.items():
            if col in raw_df.columns:
                raw_df[col] = (
                    raw_df[col].astype(str).str.lower().str.strip()
                    .map(mapping).fillna(0).astype(np.float32)
                )

        # ── Step 2: Multi-class categorical → one-hot ────────────────────
        # Fixed category lists from config ensure consistent column count
        # across train/val/test splits regardless of which values appear.
        for col, categories in config.CATEGORICAL_CLINICAL_ONEHOT.items():
            if col not in raw_df.columns:
                continue
            col_lower = raw_df[col].astype(str).str.lower().str.strip()
            for cat in categories:
                raw_df[f'{col}_{cat}'] = (col_lower == cat).astype(np.float32)
            raw_df.drop(columns=[col], inplace=True)

        # ── Step 3: Quantitative columns — min-max normalised ─────────────
        quant_cols = [c for c in config.QUANTITATIVE_CLINICAL if c in raw_df.columns]
        for col in quant_cols:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)
        if self.scaler is not None and quant_cols:
            raw_df[quant_cols] = self.scaler.transform(raw_df[quant_cols].values)

        # ── Step 4: Assemble final column order and fill any remaining NaN ─
        keep_cols = (
            [c for c in BINARY_MAP if c in raw_df.columns]
            + [f'{col}_{cat}'
               for col, cats in config.CATEGORICAL_CLINICAL_ONEHOT.items()
               for cat in cats
               if f'{col}_{cat}' in raw_df.columns]
            + [c for c in quant_cols if c in raw_df.columns]
        )
        lookup_df = raw_df[keep_cols].fillna(0).astype(np.float32)

        self.n_clinical      = lookup_df.shape[1]
        self.clinical_cols   = keep_cols
        self.clinical_lookup = {
            str(pid): row.values
            for pid, row in lookup_df.iterrows()
        }

    def fit_scaler(self):
        """
        Fit a MinMaxScaler on quantitative clinical features only.

        Per Appendix S1:
          - Categorical variables → one-hot encoded (already 0/1, no scaling)
          - Quantitative variables → min-max normalised to [0, 1]

        Fits only on quantitative columns (age, ecog_status, tumor_volume)
        from the training patients, then applies normalisation in-place to
        the training clinical_lookup so __getitem__ always returns scaled values.

        Returns
        -------
        scaler : MinMaxScaler or None
        """
        quant_cols = getattr(self, 'clinical_cols', [])
        quant_idx  = [i for i, c in enumerate(quant_cols)
                      if c in config.QUANTITATIVE_CLINICAL]

        if not quant_idx:
            print("No quantitative clinical columns — skipping scaler.")
            return None

        all_clinical = np.vstack([
            self.clinical_lookup[pid]
            for pid in self.patient_ids
            if pid in self.clinical_lookup
        ])                                      # (N_train, N_features)

        quant_data = all_clinical[:, quant_idx] # (N_train, N_quant)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(quant_data)
        self.scaler    = scaler
        self.quant_idx = quant_idx

        # Apply normalisation in-place to training lookup
        for pid in list(self.clinical_lookup.keys()):
            vec = self.clinical_lookup[pid].copy()
            vec[quant_idx] = scaler.transform(vec[np.newaxis, quant_idx])[0]
            self.clinical_lookup[pid] = vec

        n_train = len([p for p in self.patient_ids if p in self.clinical_lookup])
        print(f"Clinical MinMaxScaler fitted on {n_train} patients — "
              f"{len(quant_idx)} quantitative features normalised to [0, 1]")
        return scaler

    def apply_scaler(self, scaler, quant_idx=None):
        """
        Apply a scaler fitted on the training set to this dataset.
        Only quantitative columns (identified by quant_idx) are scaled.
        """
        self.scaler    = scaler
        self.quant_idx = quant_idx or []
        if scaler is not None and self.quant_idx:
            for pid in list(self.clinical_lookup.keys()):
                vec = self.clinical_lookup[pid].copy()
                vec[self.quant_idx] = scaler.transform(
                    vec[np.newaxis, self.quant_idx]
                )[0]
                self.clinical_lookup[pid] = vec

    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        pid   = getattr(graph, 'patient_id', str(idx))

        # Attach clinical features
        # Scaling is applied in-place during fit_scaler()/apply_scaler() so
        # clinical_lookup already contains normalised values — just retrieve.
        if pid in self.clinical_lookup:
            clinical = self.clinical_lookup[pid].copy()
            graph.clinical = torch.tensor(clinical, dtype=torch.float).unsqueeze(0)
        else:
            graph.clinical = torch.zeros(1, self.n_clinical, dtype=torch.float)

        return graph


# ─── Data Splitting ───────────────────────────────────────────────────────────

def split_dataset(graphs, train_ratio=None, val_ratio=None, test_ratio=None,
                  random_seed=None):
    """
    Stratified train / val / test split.

    Parameters
    ----------
    graphs      : list[Data]
    train_ratio : float  default config.TRAIN_RATIO
    val_ratio   : float  default config.VAL_RATIO
    test_ratio  : float  default config.TEST_RATIO
    random_seed : int    default config.RANDOM_SEED

    Returns
    -------
    train_graphs, val_graphs, test_graphs : list[Data]
    """
    train_r = config.TRAIN_RATIO if train_ratio is None else train_ratio
    val_r   = config.VAL_RATIO   if val_ratio   is None else val_ratio
    test_r  = config.TEST_RATIO  if test_ratio  is None else test_ratio
    seed    = config.RANDOM_SEED if random_seed  is None else random_seed

    labels  = np.array([g.y.item() for g in graphs])
    indices = np.arange(len(graphs))

    # First split: train+val vs test
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size    = test_r,
        stratify     = labels,
        random_state = seed
    )

    # Second split: train vs val
    val_ratio_adj = val_r / (train_r + val_r)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size    = val_ratio_adj,
        stratify     = labels[idx_trainval],
        random_state = seed
    )

    train_graphs = [graphs[i] for i in idx_train]
    val_graphs   = [graphs[i] for i in idx_val]
    test_graphs  = [graphs[i] for i in idx_test]

    print(f"\nData split:")
    print(f"  Train : {len(train_graphs)} patients "
          f"({sum(g.y.item() for g in train_graphs)} positive)")
    print(f"  Val   : {len(val_graphs)} patients "
          f"({sum(g.y.item() for g in val_graphs)} positive)")
    print(f"  Test  : {len(test_graphs)} patients "
          f"({sum(g.y.item() for g in test_graphs)} positive)")

    return train_graphs, val_graphs, test_graphs


def kfold_split(graphs, n_splits=5, random_seed=None):
    """
    Stratified K-fold split generator.

    Yields
    ------
    fold_idx : int
    train_graphs : list[Data]
    val_graphs   : list[Data]
    """
    seed   = config.RANDOM_SEED if random_seed is None else random_seed
    labels = np.array([g.y.item() for g in graphs])
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(graphs, labels)):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs   = [graphs[i] for i in val_idx]
        yield fold_idx, train_graphs, val_graphs


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_data_loaders(train_dataset, val_dataset, test_dataset,
                     batch_size=None, num_workers=None):
    """
    Create PyTorch Geometric DataLoaders.

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset : Dataset
    batch_size   : int   default config.BATCH_SIZE
    num_workers  : int   default config.NUM_WORKERS

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    bs         = batch_size  or config.BATCH_SIZE
    nw         = num_workers or config.NUM_WORKERS
    # pin_memory is only valid (and beneficial) when using CUDA
    pin_memory = config.PIN_MEMORY and (config.DEVICE == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size  = bs,
        shuffle     = True,
        num_workers = nw,
        pin_memory  = pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = bs,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = bs,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = pin_memory
    )

    return train_loader, val_loader, test_loader


# ─── Graph loading utilities ──────────────────────────────────────────────────

def load_graphs_from_directory(graph_dir, task='LR', patient_ids=None):
    """
    Load all pre-built .pt graph files from a directory.

    Parameters
    ----------
    graph_dir   : Path or str
    task        : 'LR' or 'DM'
    patient_ids : list[str] or None  — load only specific patients

    Returns
    -------
    graphs : list[Data]
    """
    graph_dir = Path(graph_dir)
    graphs    = []

    if patient_ids:
        files = [graph_dir / f'{pid}_{task}.pt' for pid in patient_ids]
    else:
        files = sorted(graph_dir.glob(f'*_{task}.pt'))

    for f in files:
        if f.exists():
            # weights_only=False required for PyTorch Geometric Data objects
            graphs.append(torch.load(f, weights_only=False))
        else:
            print(f"  WARNING: Graph file not found: {f}")

    print(f"Loaded {len(graphs)} graphs from {graph_dir}")
    return graphs


def save_split_indices(train_graphs, val_graphs, test_graphs, save_dir, task='LR'):
    """Save train/val/test patient ID splits to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_graphs in [('train', train_graphs),
                                     ('val',   val_graphs),
                                     ('test',  test_graphs)]:
        ids = [getattr(g, 'patient_id', '') for g in split_graphs]
        np.save(save_dir / f'{split_name}_ids_{task}.npy', np.array(ids))

    print(f"Split indices saved to {save_dir}")


def load_split_indices(save_dir, task='LR'):
    """Load previously saved train/val/test patient ID splits."""
    save_dir = Path(save_dir)
    splits   = {}
    for split_name in ('train', 'val', 'test'):
        f = save_dir / f'{split_name}_ids_{task}.npy'
        if f.exists():
            splits[split_name] = list(np.load(f, allow_pickle=True))
        else:
            splits[split_name] = []
    return splits


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing dataset module with synthetic graphs...")

    torch.manual_seed(42)

    # Create 20 synthetic graphs
    synthetic_graphs = []
    for i in range(20):
        n_nodes = np.random.randint(5, 22)
        n_feats = 93
        n_edges = (n_nodes - 1) * 2   # star topology

        src = [0] * (n_nodes - 1) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + [0] * (n_nodes - 1)

        g = Data(
            x          = torch.randn(n_nodes, n_feats),
            edge_index = torch.tensor([src, dst], dtype=torch.long),
            edge_attr  = torch.rand(n_edges, 1),
            y          = torch.tensor([i % 2], dtype=torch.long),
        )
        g.patient_id = f'P{i:03d}'
        synthetic_graphs.append(g)

    # Test split
    train_g, val_g, test_g = split_dataset(synthetic_graphs)

    # Test dataset
    train_ds = RadGraphDataset(train_g)
    print(f"\nTrain dataset size  : {len(train_ds)}")
    print(f"Labels distribution : {train_ds.labels.sum()} positive / {len(train_ds)} total")
    print(f"Pos weight          : {train_ds.get_class_weights():.2f}")

    # Test DataLoader
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch  = next(iter(loader))
    print(f"\nSample batch:")
    print(f"  x shape     : {batch.x.shape}")
    print(f"  edge_index  : {batch.edge_index.shape}")
    print(f"  y           : {batch.y}")

    print("\nDataset module test passed!")