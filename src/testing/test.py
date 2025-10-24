"""
Model Testing and Evaluation

Provides functions for testing trained models on test datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Test model on test dataset and return metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run on (default: auto-detect)
        
    Returns:
        Dictionary with test metrics:
        - accuracy: Classification accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - confusion_matrix: Confusion matrix
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    print("\nTesting model...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    
    # Per-class metrics (assuming binary classification)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confusion matrix
    confusion_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
    
    avg_loss = total_loss / len(test_loader)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'loss': float(avg_loss),
        'confusion_matrix': confusion_matrix,
        'num_samples': len(all_labels),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }
    
    return results


def evaluate_predictions(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        predictions: Model predictions (numpy array)
        labels: Ground truth labels (numpy array)
        
    Returns:
        Dictionary with evaluation metrics
    """
    accuracy = np.mean(predictions == labels)
    
    # Binary classification metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
