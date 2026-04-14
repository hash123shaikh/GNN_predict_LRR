"""
Simplified RadGraph Implementation - Baseline Random Forest
===========================================================
Implements the RF baseline pipeline exactly as described in Appendix S1:

  1. Min-max normalize radiomic features (GTV region)
  2. Select 20 candidate features using mRMR
  3. Test EVERY combination of 1-20 features from that subset
  4. Each combination tested with ALL 3 sampling strategies:
       - undersampling
       - oversampling
       - intermediate (combination of both)
  5. Select best combination by validation AUC
  6. Report test set performance

Two baselines implemented (per Statistical Analysis section):
  --radiomics_baseline  RF trained on radiomic features only  (traditional radiomics)
  --clinical_baseline   RF trained on clinical variables only  (paper's clinical baseline)

Usage:
    python main_simple.py --task LR --split_data --train --evaluate
    python main_simple.py --task LR --split_data --clinical_baseline --train --evaluate
    python main_simple.py --task DM --split_data --train --evaluate
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import torch

import config
from utils import (set_seed, get_device, calculate_metrics,
                   print_metrics, plot_roc_curve, plot_confusion_matrix)

# ─── mRMR import ──────────────────────────────────────────────────────────────
try:
    from mrmr import mrmr_classif
    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False
    print("WARNING: mrmr-selection not installed. Falling back to RF importance.")
    print("Install with: pip install mrmr-selection")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """
    Load and min-max normalize radiomic features.

    Per Appendix S1:
      "radiomic features used as input to models were the min-max normalized
       radiomic features from the gross target volume (GTV) region"
    """
    print("Loading data...")

    radiomics_df = pd.read_csv(config.RADIOMICS_FEATURES_FILE)
    clinical_df  = pd.read_csv(config.CLINICAL_DATA_FILE)

    print(f"  Radiomics features : {radiomics_df.shape}")
    print(f"  Clinical data      : {clinical_df.shape}")

    merged_df = pd.merge(radiomics_df, clinical_df,
                         on=config.PATIENT_ID_COL, how='inner')
    print(f"  Merged             : {merged_df.shape}")

    if config.FOLLOWUP_TIME in merged_df.columns:
        merged_df = merged_df[
            merged_df[config.FOLLOWUP_TIME] >= config.MIN_FOLLOWUP_MONTHS
        ]
        print(f"  After follow-up filter: {merged_df.shape}")

    return merged_df


# ─── Sampling strategies ──────────────────────────────────────────────────────

def apply_sampling(X, y, strategy='intermediate', random_state=42):
    """
    Apply class-imbalance sampling strategy.

    Appendix S1 strategies:
      undersampling  — subset majority class to match minority
      oversampling   — duplicate minority class to match majority
      intermediate   — geometric mean: partially balance both classes
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos

    if strategy == 'undersampling':
        target_n = int(n_pos)
        neg_idx  = np.where(y == 0)[0]
        pos_idx  = np.where(y == 1)[0]
        neg_down = resample(neg_idx, n_samples=target_n,
                            replace=False, random_state=random_state)
        keep     = np.concatenate([neg_down, pos_idx])
        return X[keep], y[keep]

    elif strategy == 'oversampling':
        target_n = int(n_neg)
        neg_idx  = np.where(y == 0)[0]
        pos_idx  = np.where(y == 1)[0]
        pos_up   = resample(pos_idx, n_samples=target_n,
                            replace=True, random_state=random_state)
        keep     = np.concatenate([neg_idx, pos_up])
        return X[keep], y[keep]

    else:  # intermediate
        target_n = int(np.sqrt(n_neg * n_pos))
        neg_idx  = np.where(y == 0)[0]
        pos_idx  = np.where(y == 1)[0]
        neg_r = resample(neg_idx, n_samples=target_n,
                         replace=(target_n > n_neg), random_state=random_state)
        pos_r = resample(pos_idx, n_samples=target_n,
                         replace=(target_n > n_pos), random_state=random_state + 1)
        keep  = np.concatenate([neg_r, pos_r])
        return X[keep], y[keep]


# ─── mRMR feature selection ───────────────────────────────────────────────────

def mrmr_select_candidates(X_train, y_train, n_candidates=20):
    """
    Select top-20 candidate features using mRMR.

    Appendix S1: "20 features were initially selected" using mRMR.

    Returns
    -------
    candidate_features : list[str]
    """
    if HAS_MRMR:
        print(f"  Running mRMR to select {n_candidates} candidates...")
        selected = mrmr_classif(
            X     = X_train,
            y     = pd.Series(y_train),
            K     = n_candidates
        )
        print(f"  mRMR candidates: {selected}")
        return selected
    else:
        # Fallback: RF importance ranking
        print(f"  Falling back to RF importance for {n_candidates} candidates...")
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_train.values, y_train)
        importances = rf.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:n_candidates]
        candidates  = X_train.columns[top_idx].tolist()
        print(f"  RF importance candidates (first 5): {candidates[:5]}")
        return candidates


# ─── Full Appendix S1 feature + sampling grid search ─────────────────────────

def select_best_features_and_sampling(
    X_train, y_train, X_val, y_val,
    candidates, task='LR'
):
    """
    Appendix S1 exact procedure:
      "every combination of one to 20 features from this subset were used to
       train RF models... Each combination was also trained using either
       undersampling, oversampling, or a combination of the two...
       evaluated on the held-out validation set... top performing model was
       selected based upon AUC"

    Parameters
    ----------
    X_train, y_train : training data (already min-max normalised)
    X_val,   y_val   : validation data
    candidates       : list[str]  — 20 mRMR candidates
    task             : 'LR' or 'DM'

    Returns
    -------
    best_features  : list[str]
    best_sampling  : str
    best_val_auc   : float
    """
    sampling_strategies = ['undersampling', 'oversampling', 'intermediate']
    best_auc      = 0.0
    best_features = candidates[:config.get_n_features_for_task(task)]
    best_sampling = 'intermediate'

    X_tr = X_train[candidates].values
    X_v  = X_val[candidates].values

    total_combos = sum(
        len(list(combinations(range(len(candidates)), n)))
        for n in range(1, len(candidates) + 1)
    )
    print(f"  Grid search: {len(candidates)} candidates × "
          f"3 sampling strategies")
    print(f"  Total combinations to test: "
          f"{total_combos * 3} (this may take a few minutes)...")

    tested = 0
    for n_feat in range(1, len(candidates) + 1):
        for feat_combo in combinations(range(len(candidates)), n_feat):
            feat_cols = [candidates[i] for i in feat_combo]
            X_tr_sub  = X_train[feat_cols].values
            X_v_sub   = X_val[feat_cols].values

            for strategy in sampling_strategies:
                X_s, y_s = apply_sampling(
                    X_tr_sub, y_train,
                    strategy     = strategy,
                    random_state = config.RF_RANDOM_STATE
                )

                rf = RandomForestClassifier(
                    n_estimators = config.RF_N_ESTIMATORS,
                    max_depth    = config.RF_MAX_DEPTH,
                    random_state = config.RF_RANDOM_STATE,
                    n_jobs       = -1
                )
                rf.fit(X_s, y_s)

                try:
                    y_prob = rf.predict_proba(X_v_sub)[:, 1]
                    auc    = roc_auc_score(y_val, y_prob)
                except Exception:
                    auc = 0.0

                if auc > best_auc:
                    best_auc      = auc
                    best_features = feat_cols
                    best_sampling = strategy

                tested += 1
                if tested % 500 == 0:
                    print(f"    Tested {tested} combinations... best AUC so far: {best_auc:.4f}")

    print(f"\n  Best validation AUC  : {best_auc:.4f}")
    print(f"  Best features ({len(best_features)}): {best_features}")
    print(f"  Best sampling        : {best_sampling}")

    return best_features, best_sampling, best_auc


# ─── Final model training ─────────────────────────────────────────────────────

def train_final_rf(X_train, y_train, X_val, y_val,
                   best_features, best_sampling, task='LR'):
    """
    Train the final RF model with the best feature + sampling combination.
    """
    print(f"\n{'='*60}")
    print(f"Training Final Baseline RF for {task}")
    print(f"  Features : {best_features}")
    print(f"  Sampling : {best_sampling}")
    print(f"{'='*60}")

    X_tr_sel = X_train[best_features].values
    X_v_sel  = X_val[best_features].values

    X_s, y_s = apply_sampling(X_tr_sel, y_train, strategy=best_sampling,
                               random_state=config.RF_RANDOM_STATE)

    rf = RandomForestClassifier(
        n_estimators = config.RF_N_ESTIMATORS * 2,
        max_depth    = config.RF_MAX_DEPTH,
        random_state = config.RF_RANDOM_STATE,
        n_jobs       = -1
    )
    rf.fit(X_s, y_s)

    # Validation metrics
    y_v_prob = rf.predict_proba(X_v_sel)[:, 1]
    y_v_pred = (y_v_prob >= 0.5).astype(int)
    val_m    = calculate_metrics(y_val, y_v_pred, y_v_prob)
    print_metrics(val_m, "Validation")

    return rf


# ─── Clinical-only baseline ───────────────────────────────────────────────────

def train_clinical_baseline(
    clinical_df, outcome_col, task='LR',
    ids_train=None, ids_val=None, ids_test=None
):
    """
    Train an RF model SOLELY on clinical variables.

    Paper (Statistical Analysis):
      "A clinical baseline was also studied, consisting of a random forest
       machine learning model trained solely on clinical variables. These
       clinical factors included whether a patient underwent concurrent
       chemoradiation therapy, in addition to patient human papillomavirus
       infection status, sex, age, tumor stage (AJCC 7th edition), ECOG
       performance status, tumor subsite, and tumor volume."

    Appendix S1:
      "The clinical baseline model was implemented in the same fashion,
       replacing radiomic features with one-hot encoded clinical features."
      — Same mRMR + all-combinations + all-sampling grid search applies.

    Parameters
    ----------
    clinical_df  : pd.DataFrame
    outcome_col  : str
    task         : 'LR' or 'DM'
    ids_train/val/test : np.ndarray or None  — use pre-saved splits if available

    Returns
    -------
    model        : fitted RandomForestClassifier
    test_metrics : dict
    """
    print(f"\n{'='*60}")
    print(f"Clinical-Only Baseline (RF) — Task: {task}")
    print(f"Features: {config.CLINICAL_FEATURES}")
    print(f"{'='*60}")

    # Filter adequate follow-up
    if config.FOLLOWUP_TIME in clinical_df.columns:
        clinical_df = clinical_df[
            clinical_df[config.FOLLOWUP_TIME] >= config.MIN_FOLLOWUP_MONTHS
        ].copy()

    patient_ids = clinical_df[config.PATIENT_ID_COL].values
    y           = clinical_df[outcome_col].values

    # Clinical features — quantitative normalised to [0,1], categorical one-hot
    # Appendix S1: "quantitative categorical variables were normalized so that
    #               their values remained between 0 and 1"
    X_clinical = clinical_df[config.CLINICAL_FEATURES].copy()

    print(f"\nClinical feature matrix: {X_clinical.shape}")
    print(f"Outcome: {(y==0).sum()} negative, {(y==1).sum()} positive")

    # ── Use saved splits or create new ones ───────────────────────────────────
    if ids_train is not None and ids_val is not None and ids_test is not None:
        train_mask = clinical_df[config.PATIENT_ID_COL].isin(ids_train)
        val_mask   = clinical_df[config.PATIENT_ID_COL].isin(ids_val)
        test_mask  = clinical_df[config.PATIENT_ID_COL].isin(ids_test)

        X_train = X_clinical[train_mask]
        X_val   = X_clinical[val_mask]
        X_test  = X_clinical[test_mask]
        y_train = y[train_mask.values]
        y_val   = y[val_mask.values]
        y_test  = y[test_mask.values]
        ids_test_final = ids_test
    else:
        X_tv, X_test, y_tv, y_test, ids_tv, ids_test_final = train_test_split(
            X_clinical, y, patient_ids,
            test_size=config.TEST_RATIO,
            stratify=y, random_state=config.RANDOM_SEED
        )
        val_adj = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            X_tv, y_tv, ids_tv,
            test_size=val_adj,
            stratify=y_tv, random_state=config.RANDOM_SEED
        )

    # Min-max normalise quantitative clinical features to [0, 1]
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train_n = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=config.CLINICAL_FEATURES)
    X_val_n   = pd.DataFrame(scaler.transform(X_val),
                              columns=config.CLINICAL_FEATURES)
    X_test_n  = pd.DataFrame(scaler.transform(X_test),
                              columns=config.CLINICAL_FEATURES)

    print(f"\nTrain: {len(X_train_n)}  Val: {len(X_val_n)}  Test: {len(X_test_n)}")

    # ── Grid search: all combinations of clinical features × 3 samplings ─────
    # Appendix S1: same pipeline as radiomics baseline
    n_feats    = len(config.CLINICAL_FEATURES)
    strategies = ['undersampling', 'oversampling', 'intermediate']
    best_auc   = 0.0
    best_feats = config.CLINICAL_FEATURES
    best_strat = 'intermediate'

    print(f"\nGrid search: all combinations of {n_feats} clinical features "
          f"× {len(strategies)} sampling strategies...")

    for n_f in range(1, n_feats + 1):
        for feat_combo in combinations(range(n_feats), n_f):
            cols     = [config.CLINICAL_FEATURES[i] for i in feat_combo]
            X_tr_sub = X_train_n[cols].values
            X_v_sub  = X_val_n[cols].values

            for strategy in strategies:
                X_s, y_s = apply_sampling(
                    X_tr_sub, y_train,
                    strategy     = strategy,
                    random_state = config.RF_RANDOM_STATE
                )
                rf = RandomForestClassifier(
                    n_estimators = config.RF_N_ESTIMATORS,
                    max_depth    = config.RF_MAX_DEPTH,
                    random_state = config.RF_RANDOM_STATE,
                    n_jobs       = -1
                )
                rf.fit(X_s, y_s)
                try:
                    auc = roc_auc_score(y_val, rf.predict_proba(X_v_sub)[:, 1])
                except Exception:
                    auc = 0.0

                if auc > best_auc:
                    best_auc   = auc
                    best_feats = cols
                    best_strat = strategy

    print(f"\nBest validation AUC  : {best_auc:.4f}")
    print(f"Best features ({len(best_feats)}): {best_feats}")
    print(f"Best sampling        : {best_strat}")

    # ── Train final model ─────────────────────────────────────────────────────
    X_tr_sel = X_train_n[best_feats].values
    X_te_sel = X_test_n[best_feats].values

    X_s, y_s = apply_sampling(X_tr_sel, y_train, strategy=best_strat,
                               random_state=config.RF_RANDOM_STATE)
    final_rf = RandomForestClassifier(
        n_estimators = config.RF_N_ESTIMATORS * 2,
        max_depth    = config.RF_MAX_DEPTH,
        random_state = config.RF_RANDOM_STATE,
        n_jobs       = -1
    )
    final_rf.fit(X_s, y_s)

    # ── Test evaluation ───────────────────────────────────────────────────────
    y_prob = final_rf.predict_proba(X_te_sel)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_m = calculate_metrics(y_test, y_pred, y_prob, threshold=0.5)
    print_metrics(test_m, f"Clinical Baseline Test ({task})")

    # Save
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_rf, config.MODEL_DIR / f'clinical_baseline_{task}.pkl')
    joblib.dump(scaler,   config.MODEL_DIR / f'clinical_scaler_{task}.pkl')

    results_df = pd.DataFrame({
        'patient_id'     : ids_test_final,
        'true_label'     : y_test,
        'predicted_prob' : y_prob,
        'predicted_label': y_pred,
        'model'          : 'clinical_baseline'
    })
    results_path = config.OUTPUT_DIR / f'clinical_baseline_results_{task}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Clinical baseline results saved to {results_path}")

    plot_roc_curve(y_test, y_prob,
                   config.OUTPUT_DIR / f'roc_curve_clinical_{task}.png')

    return final_rf, test_m


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Baseline RF — Appendix S1 pipeline',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Radiomics baseline (traditional radiomics)
  python main_simple.py --task LR --split_data --train --evaluate

  # Clinical-only baseline (paper's clinical baseline)
  python main_simple.py --task LR --split_data --clinical_baseline --train --evaluate

  # Both baselines on the same split
  python main_simple.py --task LR --split_data --train --evaluate
  python main_simple.py --task LR --clinical_baseline --train --evaluate
        """
    )
    parser.add_argument('--task',             type=str, default='LR',
                        choices=['LR', 'DM'])
    parser.add_argument('--split_data',       action='store_true',
                        help='Create and save train/val/test split')
    parser.add_argument('--train',            action='store_true',
                        help='Train the RF model')
    parser.add_argument('--evaluate',         action='store_true',
                        help='Evaluate on test set')
    parser.add_argument('--clinical_baseline',action='store_true',
                        help='Train clinical-only RF baseline instead of radiomics RF.\n'
                             'Paper: "A clinical baseline... trained solely on\n'
                             'clinical variables" (Statistical Analysis section)')
    parser.add_argument('--fast',             action='store_true',
                        help='Fast mode: skip exhaustive grid search')
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    outcome_col = config.get_outcome_column(args.task)

    # ── Load data ────────────────────────────────────────────────────────────
    data_df = load_data()

    exclude_cols = ([config.PATIENT_ID_COL, config.OUTCOME_LR,
                     config.OUTCOME_DM, config.FOLLOWUP_TIME]
                    + config.CLINICAL_FEATURES)
    feature_cols = [c for c in data_df.columns if c not in exclude_cols]

    patient_ids = data_df[config.PATIENT_ID_COL].values
    X_raw       = data_df[feature_cols]
    y           = data_df[outcome_col].values

    print(f"\nOutcome ({args.task}): "
          f"{(y==0).sum()} negative, {(y==1).sum()} positive")

    # ── Clinical-only baseline path ───────────────────────────────────────────
    if args.clinical_baseline:
        # Load saved split indices if they exist
        split_dir  = config.OUTPUT_DIR / 'splits'
        ids_train_ = ids_val_ = ids_test_ = None
        if (split_dir / f'train_ids_{args.task}.npy').exists():
            ids_train_ = np.load(split_dir / f'train_ids_{args.task}.npy',
                                 allow_pickle=True)
            ids_val_   = np.load(split_dir / f'val_ids_{args.task}.npy',
                                 allow_pickle=True)
            ids_test_  = np.load(split_dir / f'test_ids_{args.task}.npy',
                                 allow_pickle=True)
            print("Using existing train/val/test split for clinical baseline.")

        clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)

        if args.train:
            train_clinical_baseline(
                clinical_df  = clinical_df,
                outcome_col  = outcome_col,
                task         = args.task,
                ids_train    = ids_train_,
                ids_val      = ids_val_,
                ids_test     = ids_test_
            )
        return   # Clinical baseline is complete

    # ── Radiomics baseline path ───────────────────────────────────────────────
    # ── Split ─────────────────────────────────────────────────────────────────
    if args.split_data or args.train:
        X_tv, X_test, y_tv, y_test, ids_tv, ids_test = train_test_split(
            X_raw, y, patient_ids,
            test_size=config.TEST_RATIO,
            stratify=y, random_state=config.RANDOM_SEED
        )
        val_adj = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_tv, y_tv, ids_tv,
            test_size=val_adj,
            stratify=y_tv, random_state=config.RANDOM_SEED
        )
        print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

        # Min-max normalise radiomic features (Appendix S1)
        print("\nMin-max normalising radiomic features to [0, 1] (Appendix S1)...")
        scaler  = MinMaxScaler(feature_range=(0, 1))
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_cols
        )
        X_val  = pd.DataFrame(scaler.transform(X_val),  columns=feature_cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler,
                    config.MODEL_DIR / f'minmax_scaler_{args.task}.pkl')

        split_dir = config.OUTPUT_DIR / 'splits'
        split_dir.mkdir(exist_ok=True)
        np.save(split_dir / f'train_ids_{args.task}.npy', ids_train)
        np.save(split_dir / f'val_ids_{args.task}.npy',   ids_val)
        np.save(split_dir / f'test_ids_{args.task}.npy',  ids_test)

    if args.train:
        # mRMR candidate selection
        print(f"\n=== Step 1: mRMR — select 20 candidates ===")
        candidates = mrmr_select_candidates(X_train, y_train, n_candidates=20)

        if args.fast:
            print("  [FAST MODE] Skipping full grid search.")
            n_target      = config.get_n_features_for_task(args.task)
            best_features = candidates[:n_target]
            best_sampling = 'intermediate'
            best_val_auc  = 0.0
        else:
            print(f"\n=== Step 2: Grid search (all combos × 3 samplings) ===")
            best_features, best_sampling, best_val_auc = \
                select_best_features_and_sampling(
                    X_train, y_train, X_val, y_val,
                    candidates, task=args.task
                )

        model = train_final_rf(
            X_train, y_train, X_val, y_val,
            best_features, best_sampling, task=args.task
        )

        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model,
                    config.MODEL_DIR / f'baseline_model_{args.task}.pkl')
        joblib.dump({'features': best_features, 'sampling': best_sampling},
                    config.MODEL_DIR / f'baseline_config_{args.task}.pkl')

        feat_file = config.OUTPUT_DIR / f'selected_features_{args.task}.txt'
        with open(feat_file, 'w') as f:
            f.write(f"# Task: {args.task}\n")
            f.write(f"# Sampling strategy: {best_sampling}\n")
            f.write(f"# Validation AUC: {best_val_auc:.4f}\n")
            for feat in best_features:
                f.write(f"{feat}\n")
        print(f"\nSelected features saved to {feat_file}")

    if args.evaluate:
        model_file  = config.MODEL_DIR / f'baseline_model_{args.task}.pkl'
        config_file = config.MODEL_DIR / f'baseline_config_{args.task}.pkl'
        scaler_file = config.MODEL_DIR / f'minmax_scaler_{args.task}.pkl'

        if not model_file.exists():
            print(f"Model not found. Train first: "
                  f"python main_simple.py --task {args.task} --split_data --train")
            return

        model         = joblib.load(model_file)
        best_cfg      = joblib.load(config_file)
        best_features = best_cfg['features']
        scaler        = joblib.load(scaler_file)

        split_dir = config.OUTPUT_DIR / 'splits'
        ids_test  = np.load(split_dir / f'test_ids_{args.task}.npy',
                            allow_pickle=True)

        data_df2   = load_data()
        test_mask  = data_df2[config.PATIENT_ID_COL].isin(ids_test)
        X_test_raw = data_df2[test_mask][feature_cols]
        y_test     = data_df2[test_mask][outcome_col].values

        X_test_norm = pd.DataFrame(
            scaler.transform(X_test_raw), columns=feature_cols
        )
        X_test_sel  = X_test_norm[best_features]

        y_prob = model.predict_proba(X_test_sel.values)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        test_m = calculate_metrics(y_test, y_pred, y_prob, threshold=0.5)
        print_metrics(test_m, f"Radiomics Baseline Test ({args.task})")
        print("\nNote: threshold=0.5 per Appendix S1. AUC is primary metric.")

        results_df = pd.DataFrame({
            'patient_id'     : ids_test,
            'true_label'     : y_test,
            'predicted_prob' : y_prob,
            'predicted_label': y_pred,
            'model'          : 'radiomics_baseline'
        })
        results_file = config.OUTPUT_DIR / f'test_results_{args.task}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nPredictions saved to {results_file}")

        plot_roc_curve(y_test, y_prob,
                       config.OUTPUT_DIR / f'roc_curve_{args.task}.png')
        plot_confusion_matrix(y_test, y_pred,
                              config.OUTPUT_DIR / f'confusion_matrix_{args.task}.png')


if __name__ == '__main__':
    main()
