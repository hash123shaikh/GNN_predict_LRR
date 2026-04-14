"""
Evaluation Script for RadGraph GAT
=====================================
Comprehensive evaluation including:
  - AUC with bootstrap 95% CI
  - Sensitivity, Specificity, Accuracy, F1
  - ROC curve + Confusion matrix plots
  - Attention weight extraction per patient
  - Comparison table: GAT vs Baseline

Usage:
    python evaluate.py --task LR
    python evaluate.py --task LR --attention
    python evaluate.py --task LR --compare_baseline
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import DataLoader
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import config
from model   import RadGraphGAT
from dataset import (RadGraphDatasetWithClinical, load_graphs_from_directory,
                     load_split_indices)
from utils   import (get_device, set_seed, bootstrap_auc, find_optimal_threshold,
                     calculate_metrics, print_metrics, plot_roc_curve,
                     plot_confusion_matrix, load_checkpoint, save_predictions)
from train   import evaluate_epoch, _build_optimizer


# ─── Full Evaluation ──────────────────────────────────────────────────────────

def full_evaluation(model, test_loader, device, task='LR',
                    threshold=None, save_dir=None):
    """
    Run complete evaluation on test set.

    Parameters
    ----------
    model      : RadGraphGAT  (loaded with best weights)
    test_loader: DataLoader
    device     : torch.device
    task       : 'LR' or 'DM'
    threshold  : float or None  — if None, uses Youden's J
    save_dir   : Path or None

    Returns
    -------
    metrics  : dict
    y_true   : np.ndarray
    y_prob   : np.ndarray
    """
    save_dir = Path(save_dir) if save_dir else config.OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    from model import get_loss_function
    criterion = get_loss_function()

    _, auc, y_true, y_prob = evaluate_epoch(model, test_loader, criterion, device)

    # Find optimal threshold
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)
        print(f"Optimal threshold (Youden's J): {threshold:.3f}")

    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob, threshold=threshold)

    # Bootstrap CI for AUC
    auc_mean, ci_low, ci_high = bootstrap_auc(
        y_true, y_prob,
        n_bootstrap     = config.N_BOOTSTRAP,
        confidence_level= config.CONFIDENCE_LEVEL
    )
    metrics['auc_ci_low']  = ci_low
    metrics['auc_ci_high'] = ci_high
    metrics['threshold']   = threshold

    # Print
    print_metrics(metrics, prefix=f"Test Set ({task})")
    print(f"AUC 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_roc_curve(y_true, y_prob,
                   save_path=save_dir / f'roc_curve_{task}.png')

    plot_confusion_matrix(y_true, y_pred,
                          save_path=save_dir / f'confusion_matrix_{task}.png')

    _plot_probability_histogram(y_true, y_prob, task, save_dir)

    # ── Save results ─────────────────────────────────────────────────────────
    metrics_path = save_dir / f'metrics_{task}.json'
    with open(metrics_path, 'w') as f:
        serialisable = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                        for k, v in metrics.items()}
        json.dump(serialisable, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics, y_true, y_prob


# ─── Attention extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_weights(model, test_loader, device, task='LR', save_dir=None):
    """
    Extract per-patient attention weights from the FINAL GAT layer.

    Paper (Methods — Graph attention atlas creation):
      "attention values from the GTV readout node to all other graph nodes
       were extracted from the final GAT layer of each model studied.
       These attention values were then matched to corresponding CT
       supervoxels for each patient."

    Only GTV→SV directed edges are recorded (source = GTV node index 0
    within each graph), matching the paper's description of attention
    from the "GTV readout node."

    Parameters
    ----------
    model      : RadGraphGAT
    test_loader: DataLoader
    device     : torch.device
    task       : str
    save_dir   : Path or None

    Returns
    -------
    attention_df : pd.DataFrame
        Columns: patient_id, sv_node_local_idx, attention_weight (mean over heads)
    """
    save_dir = Path(save_dir) if save_dir else config.ATTENTION_MAPS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    records = []

    for batch in test_loader:
        batch = batch.to(device)

        try:
            # Returns attention from FINAL layer (fixed in model.py)
            alpha, edge_index_out = model.get_attention_weights(batch)
        except Exception as e:
            print(f"  WARNING: Attention extraction failed: {e}")
            continue

        if alpha is None or edge_index_out is None:
            continue

        # Average over attention heads: (E, n_heads) → (E,)
        alpha_mean = alpha.mean(dim=-1).cpu().numpy()
        src_nodes  = edge_index_out[0].cpu().numpy()
        dst_nodes  = edge_index_out[1].cpu().numpy()
        batch_vec  = batch.batch.cpu().numpy()

        for g_idx in range(batch.num_graphs):
            # Find global node indices for this graph
            node_mask = (batch_vec == g_idx)
            node_ids  = np.where(node_mask)[0]

            if len(node_ids) == 0:
                continue

            # GTV node is always the first node of this graph
            gtv_global_idx = node_ids[0]

            # Extract only GTV → SV edges (source = GTV node)
            # Paper: "attention values from the GTV readout node to all
            #         other graph nodes"
            gtv_src_mask = (src_nodes == gtv_global_idx)

            # Map destination (SV) global indices to local 1-based indices
            dst_global = dst_nodes[gtv_src_mask]
            weights    = alpha_mean[gtv_src_mask]

            # Local index within this patient's graph (1 = first SV, etc.)
            sv_local_indices = dst_global - gtv_global_idx

            # Patient ID
            pid = f'graph_{g_idx}'
            if hasattr(batch, 'patient_id'):
                try:
                    pid = batch.patient_id[g_idx]
                except Exception:
                    pass

            for sv_local, w in zip(sv_local_indices, weights):
                if sv_local > 0:   # Skip self-loops (GTV→GTV)
                    records.append({
                        'patient_id'       : pid,
                        'sv_node_local_idx': int(sv_local),
                        'attention_weight' : float(w),
                        'task'             : task,
                    })

    attention_df = pd.DataFrame(records)

    if len(attention_df) > 0:
        attn_csv = save_dir / f'attention_weights_{task}.csv'
        attention_df.to_csv(attn_csv, index=False)
        print(f"Attention weights saved to {attn_csv}")
        print(f"  Patients: {attention_df['patient_id'].nunique()}")
        print(f"  Total GTV→SV attention records: {len(attention_df)}")

        _plot_attention_distribution(attention_df, task, save_dir)
    else:
        print("WARNING: No attention weights extracted.")

    return attention_df


# ─── Baseline comparison ──────────────────────────────────────────────────────

def compare_with_baseline(task='LR', save_dir=None):
    """
    Build a comparison table: GAT vs Random Forest baseline.

    Reads from:
        outputs/metrics_{task}.json           (GAT)
        outputs/test_results_{task}.csv       (Baseline)

    Returns
    -------
    comparison_df : pd.DataFrame
    """
    save_dir = Path(save_dir) if save_dir else config.OUTPUT_DIR

    rows = []

    # GAT results
    gat_metrics_file = save_dir / f'metrics_{task}.json'
    if gat_metrics_file.exists():
        with open(gat_metrics_file) as f:
            gat_m = json.load(f)
        rows.append({
            'Model'      : 'RadGraph GAT',
            'AUC'        : f"{gat_m.get('auc', 0):.3f} "
                           f"[{gat_m.get('auc_ci_low', 0):.3f}–"
                           f"{gat_m.get('auc_ci_high', 0):.3f}]",
            'Sensitivity': f"{gat_m.get('sensitivity', 0):.3f}",
            'Specificity': f"{gat_m.get('specificity', 0):.3f}",
            'F1'         : f"{gat_m.get('f1', 0):.3f}",
        })

    # Baseline results
    baseline_file = save_dir / f'test_results_{task}.csv'
    if baseline_file.exists():
        baseline_df = pd.read_csv(baseline_file)
        y_true = baseline_df['true_label'].values
        y_prob = baseline_df['predicted_prob'].values
        y_pred = baseline_df['predicted_label'].values

        base_auc_mean, base_ci_low, base_ci_high = bootstrap_auc(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        rows.append({
            'Model'      : 'Baseline (Random Forest)',
            'AUC'        : f"{base_auc_mean:.3f} [{base_ci_low:.3f}–{base_ci_high:.3f}]",
            'Sensitivity': f"{sens:.3f}",
            'Specificity': f"{spec:.3f}",
            'F1'         : f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
        })

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) > 0:
        print(f"\n{'='*70}")
        print(f"Model Comparison — Task: {task}")
        print(f"{'='*70}")
        print(comparison_df.to_string(index=False))
        print(f"{'='*70}")

        comp_path = save_dir / f'model_comparison_{task}.csv'
        comparison_df.to_csv(comp_path, index=False)
        print(f"Comparison table saved to {comp_path}")
    else:
        print("No results found. Run training and baseline first.")

    return comparison_df


# ─── Private plot helpers ─────────────────────────────────────────────────────

def _plot_probability_histogram(y_true, y_prob, task, save_dir):
    """Plot predicted probability distributions for positive and negative classes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, colour, name in [(0, '#2196F3', 'No Recurrence'),
                                  (1, '#F44336', 'Recurrence')]:
        probs_cls = y_prob[y_true == label]
        if len(probs_cls) > 0:
            ax.hist(probs_cls, bins=20, alpha=0.6, color=colour, label=name,
                    density=True, edgecolor='white')

    ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold 0.5')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density',               fontsize=12)
    ax.set_title(f'Predicted Probability Distribution — {task}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'prob_histogram_{task}.png', dpi=config.FIGURE_DPI,
                bbox_inches='tight')
    plt.close()


def _plot_attention_distribution(attention_df, task, save_dir):
    """Bar chart of mean attention weight per supervoxel position."""
    if attention_df.empty:
        return

    mean_attn = (attention_df.groupby('sv_node_local_idx')['attention_weight']
                              .mean()
                              .sort_values(ascending=False)
                              .head(20))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(mean_attn)), mean_attn.values,
                  color=cm.hot(mean_attn.values / mean_attn.values.max()))
    ax.set_xlabel('Supervoxel Index (ranked by mean attention)', fontsize=11)
    ax.set_ylabel('Mean Attention Weight', fontsize=11)
    ax.set_title(f'Top-20 Supervoxel Attention Weights — {task}', fontsize=13)
    ax.set_xticks(range(len(mean_attn)))
    ax.set_xticklabels(mean_attn.index, rotation=45)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / f'attention_distribution_{task}.png',
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Attention distribution plot saved.")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate RadGraph GAT')
    parser.add_argument('--task',             type=str, default='LR', choices=['LR', 'DM'])
    parser.add_argument('--graph_dir',        type=str,
                        default=str(config.OUTPUT_DIR / 'graphs'))
    parser.add_argument('--attention',        action='store_true',
                        help='Extract and save attention weights')
    parser.add_argument('--compare_baseline', action='store_true',
                        help='Compare GAT vs baseline model')
    parser.add_argument('--threshold',        type=float, default=None,
                        help='Fixed classification threshold (default: Youden J)')
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device(config.USE_CUDA)

    # Load graphs
    graphs = load_graphs_from_directory(args.graph_dir, task=args.task)
    if len(graphs) == 0:
        print("No graphs found. Run graph_builder.py first.")
        return

    # Load saved split (to use same test set as training)
    splits   = load_split_indices(config.OUTPUT_DIR / 'splits', args.task)
    test_ids = splits.get('test', [])

    if len(test_ids) > 0:
        test_graphs = [g for g in graphs
                       if getattr(g, 'patient_id', '') in test_ids]
        print(f"Using saved test split: {len(test_graphs)} patients")
    else:
        # Fallback: use all graphs
        test_graphs = graphs
        print(f"No saved split found — evaluating all {len(test_graphs)} graphs")

    # Clinical data
    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)
    test_ds     = RadGraphDatasetWithClinical(test_graphs, clinical_df)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Load best model
    best_model_path = config.MODEL_DIR / f'best_model_{args.task}.pth'
    if not best_model_path.exists():
        print(f"Model not found: {best_model_path}")
        print("Train the model first with: python train.py --task {args.task}")
        return

    model     = RadGraphGAT().to(device)
    optimizer = _build_optimizer(model)
    load_checkpoint(model, optimizer, best_model_path)

    # Full evaluation
    print(f"\nEvaluating on {len(test_graphs)} test patients...")
    metrics, y_true, y_prob = full_evaluation(
        model, test_loader, device,
        task      = args.task,
        threshold = args.threshold,
        save_dir  = config.OUTPUT_DIR
    )

    # Save predictions CSV
    test_pids  = [getattr(g, 'patient_id', str(i)) for i, g in enumerate(test_graphs)]
    threshold  = metrics.get('threshold', 0.5)
    y_pred     = (y_prob >= threshold).astype(int)
    save_predictions(test_pids, y_true, y_prob, y_pred,
                     config.OUTPUT_DIR / f'test_predictions_{args.task}.csv')

    # Attention
    if args.attention:
        print("\nExtracting attention weights...")
        extract_attention_weights(model, test_loader, device,
                                  task=args.task,
                                  save_dir=config.ATTENTION_MAPS_DIR)

    # Comparison table
    if args.compare_baseline:
        compare_with_baseline(task=args.task, save_dir=config.OUTPUT_DIR)


if __name__ == '__main__':
    main()
