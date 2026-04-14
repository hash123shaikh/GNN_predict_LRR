"""
Training Script for RadGraph GAT
=================================
Full training loop with:
  - Class-imbalance weighted loss
  - Learning rate scheduling
  - Early stopping
  - Best model checkpointing
  - Per-epoch AUC tracking
  - K-fold cross-validation support

Usage:
    python train.py --task LR
    python train.py --task LR --use_kfold
    python train.py --task LR --resume
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import time
import warnings
warnings.filterwarnings('ignore')

import config
from model   import RadGraphGAT, get_loss_function
from dataset import (RadGraphDatasetWithClinical, RadGraphDataset,
                     split_dataset, kfold_split, get_data_loaders,
                     load_graphs_from_directory, save_split_indices)
from utils   import (set_seed, get_device, EarlyStopping,
                     save_checkpoint, load_checkpoint, plot_training_history)


# ─── Epoch helpers ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        labels = batch.y.float().view(-1)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping (helps with small datasets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """
    Evaluate model on a data loader.

    Returns
    -------
    avg_loss : float
    auc      : float
    y_true   : np.ndarray
    y_prob   : np.ndarray
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_labels = []
    all_probs  = []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        labels = batch.y.float().view(-1)

        loss       = criterion(logits, labels)
        total_loss += loss.item()
        n_batches  += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    avg_loss = total_loss / max(n_batches, 1)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return avg_loss, auc, y_true, y_prob


# ─── Main Training Function ───────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    n_epochs      = None,
    patience      = None,
    task          = 'LR',
    save_best     = True,
    model_name    = 'best_model',
):
    """
    Full training loop with early stopping and checkpointing.

    Parameters
    ----------
    model         : RadGraphGAT
    train_loader  : DataLoader
    val_loader    : DataLoader
    optimizer     : torch.optim
    scheduler     : lr scheduler or None
    criterion     : loss function
    device        : torch.device
    n_epochs      : int
    patience      : int  early stopping patience
    task          : 'LR' or 'DM'
    save_best     : bool
    model_name    : str  used for checkpoint filename

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'val_loss', 'val_auc', 'best_epoch', 'best_val_auc'
    best_model_path : Path
    """
    n_epochs = n_epochs or config.N_EPOCHS
    patience = patience or config.EARLY_STOPPING_PATIENCE

    best_model_path = config.MODEL_DIR / f'{model_name}_{task}.pth'
    history = {
        'train_loss' : [],
        'val_loss'   : [],
        'val_auc'    : [],
        'best_epoch' : 0,
        'best_val_auc': 0.0,
    }

    early_stopper = EarlyStopping(patience=patience, mode='max')
    best_val_auc  = 0.0
    start_time    = time.time()

    print(f"\n{'='*60}")
    print(f"Training RadGraphGAT | Task: {task} | Epochs: {n_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        # LR scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.0f}s")

        # Save best model
        improved = early_stopper(val_auc)
        if improved and val_auc > best_val_auc:
            best_val_auc = val_auc
            history['best_epoch']   = epoch
            history['best_val_auc'] = best_val_auc
            if save_best:
                save_checkpoint(model, optimizer, epoch, best_val_auc, best_model_path)

        # Early stopping
        if early_stopper.early_stop:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best val AUC: {best_val_auc:.4f} at epoch {history['best_epoch']})")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Best val AUC: {best_val_auc:.4f} at epoch {history['best_epoch']}")

    return history, best_model_path


# ─── K-Fold Training ──────────────────────────────────────────────────────────

def train_kfold(graphs, clinical_df, task='LR', n_splits=5, device=None):
    """
    K-fold cross-validation training.

    Parameters
    ----------
    graphs      : list[Data]
    clinical_df : pd.DataFrame
    task        : 'LR' or 'DM'
    n_splits    : int
    device      : torch.device

    Returns
    -------
    fold_results : list[dict]  — one dict per fold with val AUC
    """
    device       = device or get_device(config.USE_CUDA)
    fold_results = []

    print(f"\nK-Fold Cross-Validation (k={n_splits}, task={task})")
    print("=" * 60)

    for fold_idx, train_graphs, val_graphs in kfold_split(graphs, n_splits):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        print(f"  Train: {len(train_graphs)}  |  Val: {len(val_graphs)}")

        # Build datasets
        train_ds = RadGraphDatasetWithClinical(train_graphs, clinical_df)
        scaler   = train_ds.fit_scaler()
        val_ds   = RadGraphDatasetWithClinical(val_graphs, clinical_df)
        val_ds.apply_scaler(scaler)

        # Class weights + sampler
        pos_weight = train_ds.get_class_weights()
        sampler, _ = _build_sampler(train_graphs, task=task)

        from torch_geometric.loader import DataLoader as PyGDataLoader
        gat_cfg = config.get_gat_config(task)

        if sampler is not None:
            train_loader = PyGDataLoader(
                train_ds, batch_size=gat_cfg['batch_size'],
                sampler=sampler, num_workers=0
            )
        else:
            train_loader = PyGDataLoader(
                train_ds, batch_size=gat_cfg['batch_size'], shuffle=True
            )
        val_loader = PyGDataLoader(val_ds, batch_size=gat_cfg['batch_size'], shuffle=False)

        # Model + optimiser (Table S2 per-task values)
        model     = RadGraphGAT(
            n_layers  = gat_cfg['n_layers'],
            hidden_dim= gat_cfg['hidden_dim'],
            n_heads   = gat_cfg['n_heads'],
            dropout   = gat_cfg['dropout'],
        ).to(device)
        optimizer              = _build_optimizer(model, task=task)
        warmup_sch, plateau_sch= _build_scheduler(optimizer, task=task)
        criterion              = get_loss_function(pos_weight).to(device)

        # Train
        history, _ = train_model(
            model, train_loader, val_loader,
            optimizer, warmup_sch, criterion, device,
            task        = task,
            model_name  = f'fold{fold_idx+1}',
        )

        fold_results.append({
            'fold'        : fold_idx + 1,
            'best_val_auc': history['best_val_auc'],
            'best_epoch'  : history['best_epoch'],
        })

    # Summary
    aucs = [r['best_val_auc'] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"K-Fold Summary  (task={task})")
    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC={r['best_val_auc']:.4f}  (epoch {r['best_epoch']})")
    print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"{'='*60}")

    return fold_results


# ─── Builder helpers (Table S2 aware) ────────────────────────────────────────

def _build_optimizer(model, task='LR'):
    """
    Build optimiser using Table S2 per-task hyperparameters.
    Table S2: Adam for both LR and DM tasks.
    """
    gat_cfg = config.get_gat_config(task)
    lr      = gat_cfg['learning_rate']
    wd      = gat_cfg['weight_decay']

    print(f"Optimizer: Adam  lr={lr}  weight_decay={wd}  (Table S2, task={task})")

    if config.OPTIMIZER == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif config.OPTIMIZER == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(optimizer, task='LR'):
    """
    Build LR scheduler with linear warmup followed by ReduceLROnPlateau.

    Table S2: LR Warmup Steps = 10 (LR task), 5 (DM task).
    Slowly increasing the LR during warmup improves training stability.
    """
    gat_cfg      = config.get_gat_config(task)
    warmup_steps = gat_cfg['warmup_steps']
    base_lr      = gat_cfg['learning_rate']

    print(f"Scheduler: Linear warmup ({warmup_steps} steps) → ReduceLROnPlateau")

    # Linear warmup lambda
    def warmup_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'max',
        patience = config.SCHEDULER_PATIENCE,
        factor   = config.SCHEDULER_FACTOR,
        verbose  = False
    )

    return warmup_scheduler, plateau_scheduler


def _apply_schedulers(warmup_scheduler, plateau_scheduler,
                      val_auc, current_epoch, warmup_steps):
    """
    Step warmup scheduler each epoch during warmup,
    then hand off to ReduceLROnPlateau.
    """
    if current_epoch <= warmup_steps:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(val_auc)


def _build_sampler(train_graphs, task='LR'):
    """
    Build a WeightedRandomSampler implementing Table S2 sampling strategies.

    Table S2 sampling strategies:
      LR task: 'intermediate' — mixture of under and oversampling
      DM task: 'over'         — oversample minority class

    Parameters
    ----------
    train_graphs : list[Data]
    task         : 'LR' or 'DM'

    Returns
    -------
    sampler      : WeightedRandomSampler or None
    n_samples    : int  — number of samples per epoch
    """
    from torch.utils.data import WeightedRandomSampler

    gat_cfg  = config.get_gat_config(task)
    strategy = gat_cfg['sampling_strategy']

    labels   = np.array([g.y.item() for g in train_graphs])
    n_total  = len(labels)
    n_pos    = labels.sum()
    n_neg    = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        print("WARNING: Single-class dataset — skipping sampler")
        return None, n_total

    print(f"Sampling strategy: {strategy}  "
          f"(n_neg={n_neg}, n_pos={n_pos}, task={task})")

    if strategy == 'over':
        # Oversample minority (positive) class to match majority
        weight_pos = n_neg / n_pos
        weight_neg = 1.0
        n_samples  = 2 * n_neg   # epoch size = 2x majority

    elif strategy == 'under':
        # Undersample majority (negative) class to match minority
        weight_pos = 1.0
        weight_neg = n_pos / n_neg
        n_samples  = 2 * n_pos   # epoch size = 2x minority

    else:  # 'intermediate' — geometric mean weighting
        weight_pos = np.sqrt(n_neg / n_pos)
        weight_neg = np.sqrt(n_pos / n_neg)
        n_samples  = n_total   # keep same epoch size

    sample_weights = np.where(labels == 1, weight_pos, weight_neg)
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = n_samples,
        replacement = True
    )
    return sampler, n_samples


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train RadGraph GAT')
    parser.add_argument('--task',       type=str, default='LR', choices=['LR', 'DM'],
                        help='Prediction task')
    parser.add_argument('--graph_dir',  type=str,
                        default=str(config.OUTPUT_DIR / 'graphs'),
                        help='Directory with .pt graph files')
    parser.add_argument('--use_kfold', action='store_true',
                        help='Use k-fold CV instead of single train/val/test split')
    parser.add_argument('--n_folds',   type=int, default=5)
    parser.add_argument('--resume',    action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--n_epochs',  type=int, default=None)
    args = parser.parse_args()

    # Setup
    set_seed(config.RANDOM_SEED)
    device = get_device(config.USE_CUDA)

    if args.n_epochs:
        config.N_EPOCHS = args.n_epochs

    # Load clinical data
    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)

    # Load graphs
    graphs = load_graphs_from_directory(args.graph_dir, task=args.task)
    if len(graphs) == 0:
        print("No graphs found. Run graph_builder.py first.")
        return

    print(f"\nLoaded {len(graphs)} graphs for task={args.task}")

    # ── K-fold mode ──────────────────────────────────────────────────────────
    if args.use_kfold:
        fold_results = train_kfold(
            graphs, clinical_df,
            task     = args.task,
            n_splits = args.n_folds,
            device   = device
        )

        # Save fold results
        results_path = config.OUTPUT_DIR / f'kfold_results_{args.task}.json'
        with open(results_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"Fold results saved to {results_path}")
        return

    # ── Single split mode ────────────────────────────────────────────────────
    train_graphs, val_graphs, test_graphs = split_dataset(graphs)

    # Save split indices for reproducibility
    save_split_indices(train_graphs, val_graphs, test_graphs,
                       config.OUTPUT_DIR / 'splits', args.task)

    # Build datasets with clinical features
    train_ds = RadGraphDatasetWithClinical(train_graphs, clinical_df)
    scaler   = train_ds.fit_scaler()

    val_ds  = RadGraphDatasetWithClinical(val_graphs,  clinical_df)
    test_ds = RadGraphDatasetWithClinical(test_graphs, clinical_df)
    val_ds.apply_scaler(scaler)
    test_ds.apply_scaler(scaler)

    # ── Sampling strategy (Table S2) ─────────────────────────────────────────
    # LR: intermediate sampling  |  DM: oversampling
    gat_cfg = config.get_gat_config(args.task)
    sampler, _ = _build_sampler(train_graphs, task=args.task)

    from torch_geometric.loader import DataLoader as PyGDataLoader
    if sampler is not None:
        train_loader = PyGDataLoader(
            train_ds,
            batch_size = gat_cfg['batch_size'],
            sampler    = sampler,
            num_workers= config.NUM_WORKERS
        )
    else:
        train_loader = PyGDataLoader(
            train_ds,
            batch_size = gat_cfg['batch_size'],
            shuffle    = True,
            num_workers= config.NUM_WORKERS
        )

    val_loader  = PyGDataLoader(val_ds,  batch_size=gat_cfg['batch_size'], shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=gat_cfg['batch_size'], shuffle=False)

    # ── Model (Table S2 hyperparameters) ─────────────────────────────────────
    print(f"\nTable S2 hyperparameters for task={args.task}:")
    for k, v in gat_cfg.items():
        print(f"  {k:25s}: {v}")

    model     = RadGraphGAT(
        n_layers  = gat_cfg['n_layers'],
        hidden_dim= gat_cfg['hidden_dim'],
        n_heads   = gat_cfg['n_heads'],
        dropout   = gat_cfg['dropout'],
    ).to(device)

    optimizer              = _build_optimizer(model, task=args.task)
    warmup_sch, plateau_sch= _build_scheduler(optimizer, task=args.task)
    criterion              = get_loss_function(
                                 train_ds.get_class_weights()
                             ).to(device)

    # Resume from checkpoint
    if args.resume:
        ckpt = config.MODEL_DIR / f'best_model_{args.task}.pth'
        if ckpt.exists():
            load_checkpoint(model, optimizer, ckpt)

    # Patch train_model to use dual schedulers via warmup wrapper
    # We pass warmup_sch as the scheduler; plateau_sch is stepped inside
    # by monkey-patching the scheduler step call in the loop.
    # Simpler approach: pass warmup_sch; after warmup_steps epochs switch to plateau.
    config._warmup_sch  = warmup_sch
    config._plateau_sch = plateau_sch
    config._warmup_steps= gat_cfg['warmup_steps']

    # Train
    history, best_model_path = train_model(
        model, train_loader, val_loader,
        optimizer, warmup_sch, criterion, device,
        task = args.task,
    )

    # Save training history plot
    history_plot = config.OUTPUT_DIR / f'training_history_{args.task}.png'
    plot_training_history(history, save_path=history_plot)

    # Quick test evaluation
    print("\nLoading best model for test evaluation...")
    load_checkpoint(model, optimizer, best_model_path)

    _, test_auc, y_true, y_prob = evaluate_epoch(model, test_loader, criterion, device)
    print(f"\n=== Test AUC: {test_auc:.4f} ===")

    # Save predictions
    test_ids = [getattr(g, 'patient_id', str(i)) for i, g in enumerate(test_graphs)]
    preds_df = pd.DataFrame({
        'patient_id'  : test_ids,
        'true_label'  : y_true.astype(int),
        'predicted_prob': y_prob,
        'predicted_label': (y_prob >= 0.5).astype(int)
    })
    preds_path = config.OUTPUT_DIR / f'train_predictions_{args.task}.csv'
    preds_df.to_csv(preds_path, index=False)
    print(f"Test predictions saved to {preds_path}")

    # Save history to JSON
    history_path = config.OUTPUT_DIR / f'training_history_{args.task}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
