from typing import Optional, Tuple
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _time_mask(spec: np.ndarray, max_mask_pct: float = 0.1) -> np.ndarray:
    """Apply a single time mask to the spectrogram (SpecAugment-style)."""
    spec = spec.copy()
    num_frames = spec.shape[0]
    max_mask = int(num_frames * max_mask_pct)
    if max_mask < 1:
        return spec
    t = np.random.randint(0, max_mask)
    t0 = np.random.randint(0, num_frames - t + 1)
    spec[t0:t0 + t, :] = 0
    return spec


def _freq_mask(spec: np.ndarray, max_mask_pct: float = 0.1) -> np.ndarray:
    """Apply a single frequency mask to the spectrogram (SpecAugment-style)."""
    spec = spec.copy()
    num_mels = spec.shape[1]
    max_mask = int(num_mels * max_mask_pct)
    if max_mask < 1:
        return spec
    f = np.random.randint(0, max_mask)
    f0 = np.random.randint(0, num_mels - f + 1)
    spec[:, f0:f0 + f] = 0
    return spec


class DeepfakeAudioDataset(Dataset):
    """PyTorch Dataset for precomputed deepfake audio features.

    Expects features.npy and labels.npy produced by the FeatureExtractor.
    """

    def __init__(
        self,
        features_path_or_array,
        labels_path_or_array,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        train: bool = True,
        time_mask_pct: float = 0.05,
        freq_mask_pct: float = 0.1,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        features_path_or_array : str or np.ndarray
            Path to a .npy features file, or an in-memory numpy array of features.
        labels_path_or_array : str or np.ndarray
            Path to a .npy labels file, or an in-memory numpy array of labels.
        """
        # Load features (accept path or array)
        if isinstance(features_path_or_array, str):
            if not os.path.isfile(features_path_or_array):
                raise FileNotFoundError(f"Features file not found: {features_path_or_array}")
            self.features = np.load(features_path_or_array)
        else:
            self.features = np.asarray(features_path_or_array)

        # Load labels (accept path or array)
        if isinstance(labels_path_or_array, str):
            if not os.path.isfile(labels_path_or_array):
                raise FileNotFoundError(f"Labels file not found: {labels_path_or_array}")
            self.labels = np.load(labels_path_or_array)
        else:
            self.labels = np.asarray(labels_path_or_array)
        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels must have the same length")

        # If mean/std not provided, compute from dataset
        if mean is None or std is None:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0) + 1e-9
        else:
            self.mean = mean
            self.std = std

        self.train = train
        self.time_mask_pct = time_mask_pct
        self.freq_mask_pct = freq_mask_pct

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feat = self.features[idx]
        label = int(self.labels[idx])

        # If feature is 1D (already pooled), reshape to (time, freq) with a single frame
        if feat.ndim == 1:
            spec = feat[np.newaxis, :]
        else:
            spec = feat

        # Data augmentation (SpecAugment) only in training
        if self.train:
            # Time and frequency masking expect 2D array (time x freq)
            spec = _time_mask(spec, max_mask_pct=self.time_mask_pct)
            spec = _freq_mask(spec, max_mask_pct=self.freq_mask_pct)

        # Normalize
        spec = (spec - self.mean) / self.std

        # Convert to tensor (channels first) — shape: (1, time, freq)
        tensor = torch.tensor(spec, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)

        return tensor, label
