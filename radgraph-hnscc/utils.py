"""
Utility functions for RadGraph implementation
"""

import numpy as np
import pandas as pd
import torch
import random
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import config

def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Parameters:
    -----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def get_device(use_cuda=True):
    """
    Get PyTorch device
    
    Parameters:
    -----------
    use_cuda : bool
        Whether to use CUDA if available
        
    Returns:
    --------
    device : torch.device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
        
    Returns:
    --------
    n_params : int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced dataset
    
    Parameters:
    -----------
    labels : array-like
        Binary labels (0 or 1)
        
    Returns:
    --------
    pos_weight : float
        Weight for positive class
    """
    n_samples = len(labels)
    n_positive = np.sum(labels)
    n_negative = n_samples - n_positive
    
    # Weight for positive class
    pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    print(f"Class distribution: {n_negative} negative, {n_positive} positive")
    print(f"Positive class weight: {pos_weight:.2f}")
    
    return pos_weight


def calculate_metrics(y_true, y_pred, y_prob, threshold=0.5):
    """
    Calculate classification metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like
        Predicted probabilities
    threshold : float
        Classification threshold
        
    Returns:
    --------
    metrics : dict
        Dictionary of metrics
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Raised when only one class present in y_true
        auc = 0.0
    
    # labels=[0,1] forces a 2×2 matrix even if model predicts only one class
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'auc': auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics


def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal classification threshold using Youden's J statistic
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
        
    Returns:
    --------
    optimal_threshold : float
        Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden's J statistic = sensitivity + specificity - 1
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def bootstrap_auc(y_true, y_prob, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence interval for AUC
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
    --------
    auc_mean : float
        Mean AUC
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    n_samples = len(y_true)
    aucs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Skip if all same class
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            auc = roc_auc_score(y_true[indices], y_prob[indices])
            aucs.append(auc)
        except ValueError:
            # Single-class bootstrap sample — skip
            continue
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(aucs, alpha/2 * 100)
    ci_upper = np.percentile(aucs, (1 - alpha/2) * 100)
    auc_mean = np.mean(aucs)
    
    return auc_mean, ci_lower, ci_upper


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    save_path : str, optional
        Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Dictionary with 'train_loss', 'val_loss', 'val_auc' lists
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', lw=2)
    ax1.plot(history['val_loss'], label='Validation Loss', lw=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # AUC plot
    ax2.plot(history['val_auc'], label='Validation AUC', lw=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Validation AUC', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.close()


def save_predictions(patient_ids, y_true, y_prob, y_pred, save_path):
    """
    Save predictions to CSV
    
    Parameters:
    -----------
    patient_ids : list
        Patient IDs
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    y_pred : array-like
        Predicted labels
    save_path : str
        Path to save CSV
    """
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': y_true,
        'predicted_probability': y_prob,
        'predicted_label': y_pred
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def print_metrics(metrics, prefix=""):
    """
    Print metrics in a formatted way
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    prefix : str
        Prefix for printing (e.g., "Train", "Val", "Test")
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print("-" * 50)


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint
    
    Parameters:
    -----------
    model : nn.Module
        Model to load weights into
    optimizer : torch.optim.Optimizer
        Optimizer to load state into
    checkpoint_path : str
        Path to checkpoint file
        
    Returns:
    --------
    epoch : int
        Epoch number
    best_val_auc : float
        Best validation AUC
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_auc = checkpoint.get('best_val_auc', 0.0)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, best val AUC: {best_val_auc:.4f}")
    
    return epoch, best_val_auc


def save_checkpoint(model, optimizer, epoch, val_auc, checkpoint_path):
    """
    Save model checkpoint
    
    Parameters:
    -----------
    model : nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer to save
    epoch : int
        Current epoch
    val_auc : float
        Current validation AUC
    checkpoint_path : str
        Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_auc': val_auc
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Parameters:
        -----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum change to qualify as improvement
        mode : str
            'max' for metrics to maximize (e.g., AUC), 'min' for minimize (e.g., loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if should stop
        
        Parameters:
        -----------
        score : float
            Current metric value
            
        Returns:
        --------
        improved : bool
            Whether metric improved
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    set_seed(42)
    device = get_device()
    
    # Test metrics
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7, 0.4, 0.6])
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Test")
    
    print("\nUtilities tested successfully!")
