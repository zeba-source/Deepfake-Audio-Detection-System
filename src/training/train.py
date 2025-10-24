import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def validate_model(model: nn.Module, val_loader, criterion, device: str = 'cpu'):
    """Validate a model on a validation dataset.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : str
        Device to run on ('cuda' or 'cpu').

    Returns
    -------
    val_loss : float
        Average validation loss.
    val_accuracy : float
        Validation accuracy.
    all_preds : list
        All predicted labels.
    all_labels : list
        All ground-truth labels.
    """
    model.eval()
    model = model.to(device)
    
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Handle single or multi-input batches
            if isinstance(batch[0], (list, tuple)):
                inputs = [b.to(device) for b in batch[0]]
                labels = batch[1].to(device)
                outputs = model(*inputs)
            else:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total = len(all_labels)
    val_loss = val_loss / total if total > 0 else 0.0
    val_accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / total if total > 0 else 0.0
    
    return val_loss, val_accuracy, all_preds, all_labels


def train_model(model: nn.Module, train_loader, val_loader, config) -> Dict[str, Any]:
    """Train a PyTorch model with validation, checkpointing, early stopping and TensorBoard logging.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : TrainingConfig-like
        Configuration object with attributes used below.

    Returns
    -------
    history : dict
        Dictionary containing training and validation loss/accuracy history.
    """

    device = getattr(config, 'device', 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Loss (with optional class weights)
    if getattr(config, 'class_weights', None) is not None:
        weights = torch.tensor(list(config.class_weights.values()), dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler.patience, factor=config.scheduler.factor)

    # TensorBoard
    writer = None
    if SummaryWriter is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.log_dir)

    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs} - Train', leave=False)
        for batch in pbar:
            # batch may be (inputs, labels) or ((in1, in2), labels)
            optimizer.zero_grad()

            if isinstance(batch[0], (list, tuple)):
                inputs = [b.to(device) for b in batch[0]]
                labels = batch[1].to(device)
                outputs = model(*inputs)
            else:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': float(running_corrects) / total})

        epoch_train_loss = running_loss / total
        epoch_train_acc = running_corrects / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch}/{config.num_epochs} - Val', leave=False)
            for batch in pbar_val:
                if isinstance(batch[0], (list, tuple)):
                    inputs = [b.to(device) for b in batch[0]]
                    labels = batch[1].to(device)
                    outputs = model(*inputs)
                else:
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    outputs = model(inputs)

                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * labels.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = val_corrects / val_total if val_total > 0 else 0.0

        # Scheduler step (ReduceLROnPlateau expects metric)
        scheduler.step(epoch_val_loss)

        # Logging
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        if writer is not None:
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('Acc/train', epoch_train_acc, epoch)
            writer.add_scalar('Acc/val', epoch_val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        epoch_time = time.time() - epoch_start
        tqdm.write(f"Epoch {epoch} | Time: {epoch_time:.1f}s | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # Checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            best_path = os.path.join(config.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"Saved best model to {best_path}")

        # Early stopping
        if epoch - best_epoch >= config.early_stopping_patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

    if writer is not None:
        writer.close()

    return history
