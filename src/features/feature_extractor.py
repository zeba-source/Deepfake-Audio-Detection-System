import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.extract import extract_cqcc_features, extract_mel_spectrogram


class FeatureExtractor:
    """Batch feature extraction helper.

    Parameters
    ----------
    feature_type : str
        Either 'cqcc' or 'mel' to choose the extractor.
    sample_rate : int
        Target sampling rate for audio loading.
    """

    def __init__(self, feature_type: str = 'cqcc', sample_rate: int = 16000):
        self.feature_type = feature_type
        self.sample_rate = sample_rate

    def process_audio_file(self, file_path: str):
        """Extract features for a single audio file.

        Returns the feature array or None on failure.
        """
        try:
            if self.feature_type == 'cqcc':
                feat = extract_cqcc_features(file_path, sr=self.sample_rate)
            else:
                feat = extract_mel_spectrogram(file_path, sr=self.sample_rate)
            return feat
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_dataset(self, data_folder: str, output_folder: str, labels_csv: str = None):
        """Process all audio files in data_folder and save features and labels.

        Expects `data_folder` to contain two subfolders: 'real' and 'fake'.
        Creates a DataFrame with file paths and labels and saves it to labels_csv if provided.

        Returns:
            features_list, labels_list, df
        """
        real_dir = os.path.join(data_folder, 'real')
        fake_dir = os.path.join(data_folder, 'fake')

        file_paths: List[str] = []
        labels: List[int] = []

        # Collect files
        if os.path.isdir(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith(('.wav', '.flac', '.mp3')):
                    file_paths.append(os.path.join(real_dir, f))
                    labels.append(0)
        if os.path.isdir(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith(('.wav', '.flac', '.mp3')):
                    file_paths.append(os.path.join(fake_dir, f))
                    labels.append(1)

        # Create DataFrame
        df = pd.DataFrame({'file_path': file_paths, 'label': labels})
        if labels_csv:
            df.to_csv(labels_csv, index=False)

        features_list = []
        labels_list = []

        # Process files with progress bar
        for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc='Extracting features')):
            file_path = row.file_path
            label = int(row.label)
            feat = self.process_audio_file(file_path)
            if feat is None:
                continue
            # Flatten temporal features to a fixed-size vector by mean-pooling over time
            if feat.ndim == 2:
                feat_vec = np.mean(feat, axis=0)
            else:
                feat_vec = np.ravel(feat)

            features_list.append(feat_vec)
            labels_list.append(label)

        # Stack into arrays
        if len(features_list) == 0:
            raise RuntimeError('No features extracted from dataset.')

        features_arr = np.stack(features_list)
        labels_arr = np.array(labels_list, dtype=np.int64)

        os.makedirs(output_folder, exist_ok=True)
        features_path = os.path.join(output_folder, 'features.npy')
        labels_path = os.path.join(output_folder, 'labels.npy')

        self.save_features(features_arr, labels_arr, output_folder)

        return features_arr, labels_arr, df

    def save_features(self, features: np.ndarray, labels: np.ndarray, output_folder: str):
        """Save features and labels as .npy files in output_folder.

        Files saved: features.npy, labels.npy
        """
        os.makedirs(output_folder, exist_ok=True)
        features_path = os.path.join(output_folder, 'features.npy')
        labels_path = os.path.join(output_folder, 'labels.npy')
        np.save(features_path, features)
        np.save(labels_path, labels)
