"""
Simplified RadGraph Implementation - Using Your Existing Features
This version assumes you already have extracted radiomics features

Usage:
    python main_simple.py --task LR --split_data
    python main_simple.py --task LR --train
    python main_simple.py --task LR --evaluate
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import config
from utils import set_seed, get_device, compute_class_weights, calculate_metrics, print_metrics


def load_data():
    """
    Load your existing radiomics features and clinical data
    
    Returns:
    --------
    data : dict
        Dictionary with features, labels, and patient IDs
    """
    print("Loading data...")
    
    # Load radiomics features (YOUR EXTRACTED FEATURES)
    radiomics_df = pd.read_csv(config.RADIOMICS_FEATURES_FILE)
    print(f"Loaded radiomics features: {radiomics_df.shape}")
    
    # Load clinical data
    clinical_df = pd.read_csv(config.CLINICAL_DATA_FILE)
    print(f"Loaded clinical data: {clinical_df.shape}")
    
    # Merge on patient ID
    merged_df = pd.merge(
        radiomics_df,
        clinical_df,
        on=config.PATIENT_ID_COL,
        how='inner'
    )
    
    print(f"Merged data: {merged_df.shape}")
    
    # Filter by follow-up time
    if config.FOLLOWUP_TIME in merged_df.columns:
        merged_df = merged_df[merged_df[config.FOLLOWUP_TIME] >= config.MIN_FOLLOWUP_MONTHS]
        print(f"After follow-up filter: {merged_df.shape}")
    
    return merged_df


def select_features_simple(X_train, y_train, X_val, y_val, task='LR'):
    """
    Simple feature selection using Random Forest feature importance
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_val, y_val : validation data
    task : str ('LR' or 'DM')
    
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    print(f"\nFeature selection for {task}...")
    
    n_features_target = config.get_n_features_for_task(task)
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X_train.columns.tolist()
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Try different numbers of features
    best_auc = 0
    best_n_features = n_features_target
    
    for n_feat in range(2, min(21, len(feature_names))):
        selected_idx = indices[:n_feat]
        selected_cols = [feature_names[i] for i in selected_idx]
        
        # Train and evaluate
        rf_temp = RandomForestClassifier(
            n_estimators=50,
            random_state=config.RF_RANDOM_STATE
        )
        rf_temp.fit(X_train[selected_cols], y_train)
        y_pred = rf_temp.predict_proba(X_val[selected_cols])[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        if auc > best_auc:
            best_auc = auc
            best_n_features = n_feat
    
    # Select best features
    selected_idx = indices[:best_n_features]
    selected_features = [feature_names[i] for i in selected_idx]
    
    print(f"Selected {len(selected_features)} features (target: {n_features_target})")
    print(f"Validation AUC: {best_auc:.4f}")
    print(f"Features: {selected_features}")
    
    return selected_features


def build_simple_graph(patient_features_dict, selected_features):
    """
    Build a simplified graph for one patient
    Since you don't have supervoxels yet, we'll create a simple baseline
    
    Parameters:
    -----------
    patient_features_dict : dict
        Dictionary with GTV features (from your radiomics file)
    selected_features : list
        List of selected feature names
    
    Returns:
    --------
    graph_data : dict
        Simple representation with just GTV features (to be extended later)
    """
    # For now, just return GTV features
    # Later you'll add supervoxel features when available
    
    gtv_features = np.array([patient_features_dict[feat] for feat in selected_features])
    
    return {
        'gtv_features': gtv_features,
        'n_features': len(selected_features)
    }


def train_baseline_model(X_train, y_train, X_val, y_val, selected_features, task='LR'):
    """
    Train a baseline Random Forest model
    
    This is a simplified version without graphs
    You can extend this to use graphs later
    """
    print(f"\n{'='*60}")
    print(f"Training Baseline Model for {task}")
    print(f"{'='*60}")
    
    # Filter to selected features
    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    
    # Handle class imbalance
    pos_weight = compute_class_weights(y_train)
    
    # Train Random Forest with class weights
    class_weights = {0: 1.0, 1: pos_weight}
    
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS * 2,  # More trees for better performance
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RF_RANDOM_STATE,
        class_weight=class_weights,
        n_jobs=-1
    )
    
    print("Training...")
    rf.fit(X_train_sel, y_train)
    
    # Evaluate on training set
    y_train_pred = rf.predict(X_train_sel)
    y_train_prob = rf.predict_proba(X_train_sel)[:, 1]
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_prob)
    print_metrics(train_metrics, "Training")
    
    # Evaluate on validation set
    y_val_pred = rf.predict(X_val_sel)
    y_val_prob = rf.predict_proba(X_val_sel)[:, 1]
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
    print_metrics(val_metrics, "Validation")
    
    return rf, val_metrics


def main():
    parser = argparse.ArgumentParser(description='Simplified RadGraph - Using Existing Features')
    parser.add_argument('--task', type=str, default='LR', choices=['LR', 'DM'],
                       help='Prediction task')
    parser.add_argument('--split_data', action='store_true',
                       help='Create train/val/test split')
    parser.add_argument('--train', action='store_true',
                       help='Train model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Update config for task
    config.TASK = args.task
    outcome_col = config.get_outcome_column(args.task)
    
    # Load data
    data_df = load_data()
    
    # Get patient IDs
    patient_ids = data_df[config.PATIENT_ID_COL].values
    
    # Separate features and labels
    # Exclude patient ID, outcomes, and clinical features
    exclude_cols = [config.PATIENT_ID_COL, config.OUTCOME_LR, config.OUTCOME_DM, 
                   config.FOLLOWUP_TIME] + config.CLINICAL_FEATURES
    
    feature_cols = [col for col in data_df.columns if col not in exclude_cols]
    
    print(f"\nNumber of radiomic features: {len(feature_cols)}")
    
    X = data_df[feature_cols]
    y = data_df[outcome_col].values
    clinical = data_df[config.CLINICAL_FEATURES]
    
    print(f"\nOutcome distribution for {args.task}:")
    print(f"  Negative: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Positive: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Data splitting
    if args.split_data or args.train:
        print("\nSplitting data...")
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test, ids_trainval, ids_test = train_test_split(
            X, y, patient_ids,
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=y
        )
        
        # Second split: train vs val
        val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_trainval, y_trainval, ids_trainval,
            test_size=val_ratio_adjusted,
            random_state=config.RANDOM_SEED,
            stratify=y_trainval
        )
        
        print(f"Train: {len(X_train)} patients")
        print(f"Val:   {len(X_val)} patients")
        print(f"Test:  {len(X_test)} patients")
        
        # Save splits
        split_dir = config.OUTPUT_DIR / 'splits'
        split_dir.mkdir(exist_ok=True)
        
        np.save(split_dir / f'train_ids_{args.task}.npy', ids_train)
        np.save(split_dir / f'val_ids_{args.task}.npy', ids_val)
        np.save(split_dir / f'test_ids_{args.task}.npy', ids_test)
        
        print(f"Splits saved to {split_dir}")
    
    if args.train:
        # Feature selection
        selected_features = select_features_simple(X_train, y_train, X_val, y_val, args.task)
        
        # Save selected features
        feature_file = config.OUTPUT_DIR / f'selected_features_{args.task}.txt'
        with open(feature_file, 'w') as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        print(f"\nSelected features saved to {feature_file}")
        
        # Train model
        model, val_metrics = train_baseline_model(
            X_train, y_train, X_val, y_val, 
            selected_features, args.task
        )
        
        # Save model
        import joblib
        model_file = config.MODEL_DIR / f'baseline_model_{args.task}.pkl'
        joblib.dump(model, model_file)
        print(f"\nModel saved to {model_file}")
    
    if args.evaluate:
        import joblib
        
        # Load model
        model_file = config.MODEL_DIR / f'baseline_model_{args.task}.pkl'
        if not model_file.exists():
            print(f"Model not found: {model_file}")
            print("Please train first with: python main_simple.py --task {args.task} --train")
            return
        
        model = joblib.load(model_file)
        print(f"Loaded model from {model_file}")
        
        # Load selected features
        feature_file = config.OUTPUT_DIR / f'selected_features_{args.task}.txt'
        with open(feature_file, 'r') as f:
            selected_features = [line.strip() for line in f]
        
        # Load test split
        split_dir = config.OUTPUT_DIR / 'splits'
        ids_test = np.load(split_dir / f'test_ids_{args.task}.npy')
        
        # Get test data
        test_mask = data_df[config.PATIENT_ID_COL].isin(ids_test)
        X_test = data_df[test_mask][selected_features]
        y_test = data_df[test_mask][outcome_col].values
        
        print(f"\nEvaluating on {len(X_test)} test patients...")
        
        # Predict
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
        print_metrics(test_metrics, "Test Set")
        
        # Save results
        results_df = pd.DataFrame({
            'patient_id': ids_test,
            'true_label': y_test,
            'predicted_prob': y_test_prob,
            'predicted_label': y_test_pred
        })
        
        results_file = config.OUTPUT_DIR / f'test_results_{args.task}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        # Plot ROC curve
        from utils import plot_roc_curve, plot_confusion_matrix
        
        roc_file = config.OUTPUT_DIR / f'roc_curve_{args.task}.png'
        plot_roc_curve(y_test, y_test_prob, save_path=roc_file)
        
        cm_file = config.OUTPUT_DIR / f'confusion_matrix_{args.task}.png'
        plot_confusion_matrix(y_test, y_test_pred, save_path=cm_file)


if __name__ == '__main__':
    main()
