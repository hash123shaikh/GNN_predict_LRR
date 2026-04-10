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

        # Data loaders
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)

        # Model + optimiser
        pos_weight = train_ds.get_class_weights()
        model      = RadGraphGAT().to(device)
        optimizer  = _build_optimizer(model)
        scheduler  = _build_scheduler(optimizer)
        criterion  = get_loss_function(pos_weight).to(device)

        # Train
        history, _ = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion, device,
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


# ─── Builder helpers ──────────────────────────────────────────────────────────

def _build_optimizer(model):
    """Build optimiser from config."""
    if config.OPTIMIZER == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr           = config.LEARNING_RATE,
            weight_decay = config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr           = config.LEARNING_RATE,
            momentum     = 0.9,
            weight_decay = config.WEIGHT_DECAY
        )
    else:   # Adam (default)
        return torch.optim.Adam(
            model.parameters(),
            lr           = config.LEARNING_RATE,
            weight_decay = config.WEIGHT_DECAY
        )


def _build_scheduler(optimizer):
    """Build LR scheduler from config."""
    if config.SCHEDULER == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = 'max',
            patience = config.SCHEDULER_PATIENCE,
            factor   = config.SCHEDULER_FACTOR,
            verbose  = False
        )
    elif config.SCHEDULER == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
    return None


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

    # Data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_ds, val_ds, test_ds
    )

    # Class weights
    pos_weight = train_ds.get_class_weights()

    # Model components
    model     = RadGraphGAT().to(device)
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    criterion = get_loss_function(pos_weight).to(device)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = config.MODEL_DIR / f'best_model_{args.task}.pth'
        if ckpt.exists():
            start_epoch, _ = load_checkpoint(model, optimizer, ckpt)

    # Train
    history, best_model_path = train_model(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion, device,
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
