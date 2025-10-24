"""Evaluation utilities for deepfake audio detection."""
import json
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

from src.training.train import validate_model


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate (EER) for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_scores : array-like
        Predicted scores or probabilities for the positive class.

    Returns
    -------
    eer : float
        Equal Error Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # EER is the point where FPR == FNR
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2.0
    return eer


def evaluate_model(
    model: nn.Module,
    test_loader,
    checkpoint_path: str,
    device: str = 'cpu',
    output_dir: str = 'results',
    plot: bool = True,
) -> Dict[str, Any]:
    """Comprehensive evaluation of a trained model.

    Loads checkpoint, evaluates on test set, computes metrics, plots confusion matrix,
    saves results to JSON, prints classification report, and computes EER.

    Parameters
    ----------
    model : nn.Module
        Model architecture (state will be loaded from checkpoint).
    test_loader : DataLoader
        Test data loader.
    checkpoint_path : str
        Path to the saved model checkpoint (.pt file).
    device : str
        Device to run on ('cuda' or 'cpu').
    output_dir : str
        Directory to save evaluation outputs.
    plot : bool
        Whether to plot and save confusion matrix.

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Run validation
    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy, all_preds, all_labels = validate_model(model, test_loader, criterion, device)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    # For AUC-ROC, we need probability scores; run inference again to get softmax outputs
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch[0], (list, tuple)):
                inputs = [b.to(device) for b in batch[0]]
                outputs = model(*inputs)
            else:
                inputs = batch[0].to(device)
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of positive class
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    auc_roc = roc_auc_score(all_labels, all_probs)
    eer = compute_eer(all_labels, all_probs)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Collect metrics
    metrics = {
        'test_loss': val_loss,
        'accuracy': val_accuracy,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'eer': float(eer),
        'confusion_matrix': cm.tolist(),
    }

    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], zero_division=0))
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"EER: {eer:.4f}")

    # Save metrics to JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {json_path}")

    # Plot confusion matrix
    if plot and plt is not None and sns is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {plot_path}")
        plt.close()

    return metrics
