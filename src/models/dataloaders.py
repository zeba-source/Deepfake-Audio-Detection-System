from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from src.models.dataset import DeepfakeAudioDataset


def create_dataloaders(
    features_path: str,
    labels_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from precomputed features and labels.

    Splits data into train/val/test with proportions 80/10/10.
    """
    # Load arrays
    features = np.load(features_path)
    labels = np.load(labels_path)

    # First split train vs temp (train_split vs rest)
    feat_train, feat_temp, lab_train, lab_temp = train_test_split(
        features, labels, train_size=train_split, random_state=random_seed, stratify=labels
    )

    # Split temp into val and test equally
    val_size = 0.5
    feat_val, feat_test, lab_val, lab_test = train_test_split(
        feat_temp, lab_temp, train_size=val_size, random_state=random_seed, stratify=lab_temp
    )

    # Create datasets
    train_ds = DeepfakeAudioDataset(feat_train, lab_train, train=True)
    val_ds = DeepfakeAudioDataset(feat_val, lab_val, train=False)
    test_ds = DeepfakeAudioDataset(feat_test, lab_test, train=False)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    return train_loader, val_loader, test_loader
