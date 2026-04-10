"""
Main Pipeline Orchestrator — RadGraph GAT
==========================================
Single entry point that runs the entire pipeline end-to-end.

Pipeline stages:
  Stage 0: Baseline   — Random Forest on existing CSV features (immediate)
  Stage 1: Preprocess — CT resampling + supervoxel generation
  Stage 2: Extract    — PyRadiomics feature extraction per supervoxel
  Stage 3: Graph      — Build PyTorch Geometric graphs
  Stage 4: Train      — Train GAT model
  Stage 5: Evaluate   — Full evaluation + attention maps
  Stage 6: Compare    — GAT vs Baseline comparison table

Usage examples:
    # Run full pipeline (all stages)
    python main.py --task LR --all

    # Run baseline model only (uses your existing CSV — works immediately)
    python main.py --task LR --stage baseline

    # Run from preprocessing onwards
    python main.py --task LR --from_stage preprocess

    # Run from graph building (skip preprocessing if already done)
    python main.py --task LR --from_stage graph

    # Just train (graphs already built)
    python main.py --task LR --stage train

    # Evaluate only (model already trained)
    python main.py --task LR --stage evaluate --attention --compare
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import set_seed, get_device


# ─── Stage 0: Baseline ────────────────────────────────────────────────────────

def run_baseline(task='LR'):
    """
    Train and evaluate a Random Forest baseline using existing radiomics CSV.
    No CT processing needed — uses config.RADIOMICS_FEATURES_FILE directly.
    """
    print_banner("Stage 0: Baseline Random Forest")

    try:
        import subprocess, sys
        cmd = [
            sys.executable, 'main_simple.py',
            '--task', task,
            '--split_data', '--train', '--evaluate'
        ]
        result = subprocess.run(cmd, check=True)
        print("\n✓ Baseline stage complete")
    except FileNotFoundError:
        print("main_simple.py not found — running inline baseline...")
        _inline_baseline(task)


def _inline_baseline(task='LR'):
    """Inline baseline fallback (same logic as main_simple.py)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from utils import compute_class_weights, calculate_metrics, print_metrics

    print("Loading features...")
    radiomics_df = pd.read_csv(config.RADIOMICS_FEATURES_FILE)
    clinical_df  = pd.read_csv(config.CLINICAL_DATA_FILE)

    merged = pd.merge(radiomics_df, clinical_df,
                      on=config.PATIENT_ID_COL, how='inner')

    outcome_col  = config.get_outcome_column(task)
    exclude_cols = ([config.PATIENT_ID_COL, config.OUTCOME_LR,
                     config.OUTCOME_DM, config.FOLLOWUP_TIME]
                    + config.CLINICAL_FEATURES)
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    X  = merged[feature_cols].values
    y  = merged[outcome_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )

    pos_w = compute_class_weights(y_tr)
    rf    = RandomForestClassifier(
        n_estimators = 200,
        class_weight = {0: 1.0, 1: pos_w},
        random_state = config.RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = calculate_metrics(y_te, y_pred, y_prob)
    print_metrics(metrics, prefix=f"Baseline Test ({task})")


# ─── Stage 1: Preprocess ─────────────────────────────────────────────────────

def run_preprocess(patient_ids):
    """CT resampling + supervoxel generation for all patients."""
    print_banner("Stage 1: CT Preprocessing + Supervoxel Generation")

    from data_loader          import HNSCCDataLoader
    from preprocessing        import CTPreprocessor
    from supervoxel_generator import SupervoxelGenerator
    from feature_extractor    import preprocess_and_save_all

    loader       = HNSCCDataLoader(config.CT_SCANS_DIR,
                                   config.RTSTRUCT_DIR,
                                   config.CLINICAL_DATA_FILE)
    preprocessor = CTPreprocessor(target_spacing=config.TARGET_SPACING)
    sv_gen       = SupervoxelGenerator(
        n_segments  = config.N_SUPERVOXELS_TARGET,
        compactness = config.SLIC_COMPACTNESS,
        sigma       = config.SLIC_SIGMA
    )

    preprocessed_dir = config.OUTPUT_DIR / 'preprocessed'
    failed = preprocess_and_save_all(
        patient_ids, loader, preprocessor, sv_gen, preprocessed_dir
    )

    print(f"\n✓ Preprocessing complete — "
          f"{len(patient_ids) - len(failed)}/{len(patient_ids)} succeeded")
    return failed


# ─── Stage 2: Feature Extraction ─────────────────────────────────────────────

def run_feature_extraction(patient_ids, skip_existing=True):
    """Extract PyRadiomics features from supervoxels."""
    print_banner("Stage 2: Radiomic Feature Extraction")

    try:
        from feature_extractor import SupervoxelFeatureExtractor
    except ImportError as e:
        print(f"Feature extraction unavailable: {e}")
        return

    preprocessed_dir = config.OUTPUT_DIR / 'preprocessed'
    extractor        = SupervoxelFeatureExtractor()

    results, failed = extractor.extract_all_patients(
        patient_ids           = patient_ids,
        preprocessed_data_dir = preprocessed_dir,
        skip_existing         = skip_existing
    )

    if results:
        extractor.save_gtv_features_csv(
            results,
            config.OUTPUT_DIR / 'gtv_features_extracted.csv'
        )

    print(f"\n✓ Feature extraction complete — "
          f"{len(results)}/{len(patient_ids)} succeeded")
    return results


# ─── Stage 3: Graph Building ──────────────────────────────────────────────────

def run_graph_building(task='LR'):
    """Build patient-level graphs from supervoxel features."""
    print_banner("Stage 3: Graph Construction")

    from graph_builder import GraphBuilder

    clinical_df      = pd.read_csv(config.CLINICAL_DATA_FILE)
    feature_cache_dir= config.OUTPUT_DIR / 'features_cache'
    graph_save_dir   = config.OUTPUT_DIR / 'graphs'

    builder = GraphBuilder()
    graphs, failed = builder.build_all_graphs(
        feature_cache_dir = feature_cache_dir,
        clinical_df       = clinical_df,
        task              = task,
        save_dir          = graph_save_dir
    )

    builder.get_graph_statistics(graphs)

    print(f"\n✓ Graph building complete — "
          f"{len(graphs)} graphs built, {len(failed)} failed")
    return graphs


# ─── Stage 4: Training ────────────────────────────────────────────────────────

def run_training(task='LR', use_kfold=False, n_folds=5):
    """Train the GAT model."""
    print_banner("Stage 4: GAT Model Training")

    import torch
    from model   import RadGraphGAT, get_loss_function
    from dataset import (RadGraphDatasetWithClinical, split_dataset,
                         get_data_loaders, save_split_indices,
                         load_graphs_from_directory)
    from train   import (train_model, train_kfold,
                         _build_optimizer, _build_scheduler)
    from utils   import plot_training_history

    device      = get_device(config.USE_CUDA)
    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)
    graph_dir   = config.OUTPUT_DIR / 'graphs'
    graphs      = load_graphs_from_directory(graph_dir, task=task)

    if len(graphs) == 0:
        print("No graphs found. Run graph building first.")
        return

    if use_kfold:
        fold_results = train_kfold(
            graphs, clinical_df,
            task=task, n_splits=n_folds, device=device
        )
        import json
        results_path = config.OUTPUT_DIR / f'kfold_results_{task}.json'
        with open(results_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"K-fold results saved to {results_path}")
        return

    # Single split
    train_g, val_g, test_g = split_dataset(graphs)
    save_split_indices(train_g, val_g, test_g,
                       config.OUTPUT_DIR / 'splits', task)

    train_ds = RadGraphDatasetWithClinical(train_g, clinical_df)
    scaler   = train_ds.fit_scaler()

    val_ds  = RadGraphDatasetWithClinical(val_g,  clinical_df)
    test_ds = RadGraphDatasetWithClinical(test_g, clinical_df)
    val_ds.apply_scaler(scaler)
    test_ds.apply_scaler(scaler)

    train_loader, val_loader, test_loader = get_data_loaders(
        train_ds, val_ds, test_ds
    )

    pos_weight = train_ds.get_class_weights()
    model      = RadGraphGAT().to(device)
    optimizer  = _build_optimizer(model)
    scheduler  = _build_scheduler(optimizer)
    criterion  = get_loss_function(pos_weight).to(device)

    history, best_path = train_model(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion, device,
        task=task
    )

    plot_training_history(history,
                          save_path=config.OUTPUT_DIR / f'training_history_{task}.png')

    print(f"\n✓ Training complete — Best val AUC: {history['best_val_auc']:.4f}")
    return history


# ─── Stage 5: Evaluation ─────────────────────────────────────────────────────

def run_evaluation(task='LR', extract_attention=False):
    """Full evaluation of the trained model."""
    print_banner("Stage 5: Model Evaluation")

    import torch
    from model    import RadGraphGAT
    from dataset  import (RadGraphDatasetWithClinical,
                          load_graphs_from_directory,
                          load_split_indices)
    from torch_geometric.loader import DataLoader
    from evaluate import full_evaluation, extract_attention_weights
    from train    import _build_optimizer
    from utils    import load_checkpoint, save_predictions

    device      = get_device(config.USE_CUDA)
    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)

    # Load graphs + test split
    graphs    = load_graphs_from_directory(config.OUTPUT_DIR / 'graphs', task=task)
    splits    = load_split_indices(config.OUTPUT_DIR / 'splits', task)
    test_ids  = splits.get('test', [])

    test_graphs = ([g for g in graphs if getattr(g, 'patient_id', '') in test_ids]
                   if test_ids else graphs)

    test_ds  = RadGraphDatasetWithClinical(test_graphs, clinical_df)
    test_ldr = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Load model
    best_model_path = config.MODEL_DIR / f'best_model_{task}.pth'
    if not best_model_path.exists():
        print(f"Trained model not found at {best_model_path}. Train first.")
        return

    model     = RadGraphGAT().to(device)
    optimizer = _build_optimizer(model)
    load_checkpoint(model, optimizer, best_model_path)

    metrics, y_true, y_prob = full_evaluation(
        model, test_ldr, device,
        task=task, save_dir=config.OUTPUT_DIR
    )

    test_pids = [getattr(g, 'patient_id', str(i)) for i, g in enumerate(test_graphs)]
    threshold = metrics.get('threshold', 0.5)
    y_pred    = (y_prob >= threshold).astype(int)
    save_predictions(test_pids, y_true, y_prob, y_pred,
                     config.OUTPUT_DIR / f'test_predictions_{task}.csv')

    if extract_attention:
        print("\nExtracting attention weights...")
        extract_attention_weights(model, test_ldr, device,
                                  task=task, save_dir=config.ATTENTION_MAPS_DIR)

    print(f"\n✓ Evaluation complete — AUC: {metrics['auc']:.4f}")
    return metrics


# ─── Stage 6: Compare ────────────────────────────────────────────────────────

def run_comparison(task='LR'):
    """Build comparison table: GAT vs Baseline."""
    print_banner("Stage 6: Model Comparison")

    from evaluate import compare_with_baseline
    comparison_df = compare_with_baseline(task=task, save_dir=config.OUTPUT_DIR)
    print("\n✓ Comparison complete")
    return comparison_df


# ─── Utility ──────────────────────────────────────────────────────────────────

def print_banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def get_patient_ids():
    """Load and filter patient IDs from clinical CSV."""
    from data_loader import HNSCCDataLoader
    loader = HNSCCDataLoader(config.CT_SCANS_DIR,
                             config.RTSTRUCT_DIR,
                             config.CLINICAL_DATA_FILE)
    return loader.filter_patients_by_followup(config.MIN_FOLLOWUP_MONTHS)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='RadGraph GAT — Full Pipeline Orchestrator',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py --task LR --stage baseline          # Quick baseline (no CT needed)
  python main.py --task LR --all                     # Full pipeline
  python main.py --task LR --from_stage graph        # Skip preprocessing
  python main.py --task LR --stage train             # Train only
  python main.py --task LR --stage evaluate --attention --compare
        """
    )

    parser.add_argument('--task',        type=str, default='LR', choices=['LR', 'DM'],
                        help='Prediction task: LR (locoregional) or DM (distant metastasis)')
    parser.add_argument('--stage',       type=str, default=None,
                        choices=['baseline', 'preprocess', 'extract',
                                 'graph', 'train', 'evaluate', 'compare'],
                        help='Run a single specific stage')
    parser.add_argument('--from_stage',  type=str, default=None,
                        choices=['baseline', 'preprocess', 'extract',
                                 'graph', 'train', 'evaluate', 'compare'],
                        help='Run from this stage to the end')
    parser.add_argument('--all',         action='store_true',
                        help='Run all stages in sequence')
    parser.add_argument('--use_kfold',   action='store_true',
                        help='Use K-fold cross-validation in training stage')
    parser.add_argument('--n_folds',     type=int, default=5,
                        help='Number of folds for K-fold CV')
    parser.add_argument('--attention',   action='store_true',
                        help='Extract attention weights during evaluation')
    parser.add_argument('--compare',     action='store_true',
                        help='Generate GAT vs Baseline comparison table')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip patients with existing cached files')

    args = parser.parse_args()

    # Determine which stages to run
    STAGE_ORDER = ['baseline', 'preprocess', 'extract', 'graph', 'train', 'evaluate', 'compare']

    if args.all:
        stages_to_run = STAGE_ORDER
    elif args.from_stage:
        start_idx     = STAGE_ORDER.index(args.from_stage)
        stages_to_run = STAGE_ORDER[start_idx:]
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        print("No stage specified. Use --stage, --from_stage, or --all.")
        parser.print_help()
        return

    # Setup
    set_seed(config.RANDOM_SEED)
    config.TASK = args.task

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           RadGraph GAT — CMC Vellore Implementation              ║
║           Task: {args.task:<6}  |  Stages: {', '.join(stages_to_run):<30}║
╚══════════════════════════════════════════════════════════════════╝
    """)

    start_time  = time.time()
    patient_ids = None

    for stage in stages_to_run:

        stage_start = time.time()

        if stage == 'baseline':
            run_baseline(task=args.task)

        elif stage == 'preprocess':
            if patient_ids is None:
                patient_ids = get_patient_ids()
            run_preprocess(patient_ids)

        elif stage == 'extract':
            if patient_ids is None:
                patient_ids = get_patient_ids()
            run_feature_extraction(patient_ids, skip_existing=args.skip_existing)

        elif stage == 'graph':
            run_graph_building(task=args.task)

        elif stage == 'train':
            run_training(task=args.task,
                         use_kfold=args.use_kfold,
                         n_folds=args.n_folds)

        elif stage == 'evaluate':
            run_evaluation(task=args.task,
                           extract_attention=args.attention)

        elif stage == 'compare':
            run_comparison(task=args.task)

        stage_time = time.time() - stage_start
        print(f"\n  [Stage '{stage}' completed in {stage_time:.0f}s]")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {total_time:.0f}s  ({total_time/60:.1f} min)")
    print(f"  Results saved to: {config.OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
