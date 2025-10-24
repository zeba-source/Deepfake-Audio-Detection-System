"""Visualization utilities for deepfake audio detection."""
import os
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import torch

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_history(history: Dict[str, List[float]], save_dir: str = 'results/plots') -> None:
    """Plot training and validation loss and accuracy curves.

    Parameters
    ----------
    history : dict
        Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        containing lists of values per epoch.
    save_dir : str
        Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str = 'results/plots') -> None:
    """Plot confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    save_dir : str
        Directory to save plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_dir: str = 'results/plots') -> None:
    """Plot ROC curve with AUC score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_scores : array-like
        Predicted scores or probabilities for positive class.
    save_dir : str
        Directory to save plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ROC curve to {save_path}")
    plt.close()


def visualize_features(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cpu',
    num_samples: int = 100,
    save_dir: str = 'results/plots'
) -> None:
    """Visualize feature embeddings using t-SNE.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with get_embedding() method.
    dataloader : DataLoader
        Data loader to extract features from.
    device : str
        Device to run on.
    num_samples : int
        Maximum number of samples to visualize.
    save_dir : str
        Directory to save plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    embeddings_list = []
    labels_list = []
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            
            # Handle single or multi-input batches
            if isinstance(batch[0], (list, tuple)):
                inputs = [b.to(device) for b in batch[0]]
                labels = batch[1]
                # For multi-input models, use first input or call model differently
                if hasattr(model, 'get_embedding'):
                    emb = model.get_embedding(inputs[0])
                else:
                    outputs = model(*inputs)
                    emb = outputs  # fallback
            else:
                inputs = batch[0].to(device)
                labels = batch[1]
                if hasattr(model, 'get_embedding'):
                    emb = model.get_embedding(inputs)
                else:
                    emb = model(inputs)
            
            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(labels.numpy())
            count += len(labels)
    
    embeddings = np.concatenate(embeddings_list, axis=0)[:num_samples]
    labels = np.concatenate(labels_list, axis=0)[:num_samples]
    
    # Apply t-SNE
    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_samples - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    labels_names = ['Real', 'Fake']
    
    for label_id in [0, 1]:
        mask = labels == label_id
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[label_id],
            label=labels_names[label_id],
            alpha=0.6,
            s=50
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title(f't-SNE Visualization of Feature Embeddings (n={num_samples})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'tsne_embeddings.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {save_path}")
    plt.close()
