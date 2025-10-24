"""Inference utilities for deepfake audio detection."""
import os
from typing import Tuple, Dict, Any
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.features.extract import extract_cqcc_features


def predict_audio(
    audio_path: str,
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu',
    feature_type: str = 'cqcc',
    sample_rate: int = 16000,
) -> Tuple[str, float]:
    """Predict whether an audio file is real or fake.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to predict.
    model : nn.Module
        Model architecture (state will be loaded from checkpoint).
    checkpoint_path : str
        Path to the saved model checkpoint (.pt file).
    device : str
        Device to run on ('cuda' or 'cpu').
    feature_type : str
        Feature type to extract ('cqcc' or 'mel').
    sample_rate : int
        Target sampling rate for audio loading.

    Returns
    -------
    prediction : str
        'Real' or 'Fake'
    confidence : float
        Confidence score (probability of predicted class, 0-1).

    Raises
    ------
    FileNotFoundError
        If audio file or checkpoint does not exist.
    RuntimeError
        If feature extraction or inference fails.
    """
    # Validate inputs
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        # Load model checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✓ Loaded model from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {e}") from e

    try:
        # Extract features
        print(f"Extracting {feature_type.upper()} features from {audio_path}...")
        if feature_type == 'cqcc':
            features = extract_cqcc_features(audio_path, sr=sample_rate)
        else:
            from src.features.extract import extract_mel_spectrogram
            features = extract_mel_spectrogram(audio_path, sr=sample_rate)

        if features is None or features.size == 0:
            raise RuntimeError("Feature extraction returned empty result")

        # Handle variable-length features: mean-pool over time if 2D
        if features.ndim == 2:
            features = np.mean(features, axis=0)
        features = features.ravel()

        # Simple normalization (ideally use training set statistics)
        # For now, standardize the features
        feat_mean = np.mean(features)
        feat_std = np.std(features) + 1e-9
        features = (features - feat_mean) / feat_std

        # Convert to tensor and add batch + channel dims: (1, 1, 1, num_features)
        # Reshape to match expected input: (batch, channels, height, width)
        # For a 1D feature vector, we'll treat it as a single "pixel" with many channels
        # or reshape to (1, 1, 1, num_features) if model expects 4D input
        # Since ResNet expects (B, C, H, W), we'll create a dummy spatial dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, num_features)
        # Reshape to (1, 1, 1, num_features) then permute or adapt as needed
        # For simplicity, we'll reshape to a square-ish 2D grid
        num_features = features_tensor.shape[1]
        # Create a "square" image: find closest square dimensions
        side = int(np.ceil(np.sqrt(num_features)))
        pad_size = side * side - num_features
        if pad_size > 0:
            features_tensor = torch.nn.functional.pad(features_tensor, (0, pad_size))
        features_tensor = features_tensor.view(1, 1, side, side).to(device)

        print(f"✓ Features extracted, shape: {features_tensor.shape}")
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}") from e

    try:
        # Run inference
        with torch.no_grad():
            outputs = model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence_scores = probs.cpu().numpy()[0]
            pred_class = int(torch.argmax(probs, dim=1).item())

        # Map to labels
        label = 'Real' if pred_class == 0 else 'Fake'
        confidence = float(confidence_scores[pred_class])

        # Print result
        print("\n" + "=" * 50)
        print(f"  🎤 Audio File: {os.path.basename(audio_path)}")
        print(f"  📊 Prediction: {label}")
        print(f"  🎯 Confidence: {confidence:.2%}")
        print(f"  📈 Real probability: {confidence_scores[0]:.2%}")
        print(f"  📉 Fake probability: {confidence_scores[1]:.2%}")
        print("=" * 50 + "\n")

        return label, confidence
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}") from e


def batch_predict(
    folder_path: str,
    model: nn.Module,
    checkpoint_path: str,
    output_csv: str = 'results/batch_predictions.csv',
    device: str = 'cpu',
    feature_type: str = 'cqcc',
    sample_rate: int = 16000,
) -> pd.DataFrame:
    """Run batch inference on all audio files in a folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing audio files.
    model : nn.Module
        Model architecture.
    checkpoint_path : str
        Path to checkpoint file.
    output_csv : str
        Path to save results CSV.
    device : str
        Device to run on.
    feature_type : str
        Feature type ('cqcc' or 'mel').
    sample_rate : int
        Target sampling rate.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: filename, prediction, confidence, status
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all audio files
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if len(audio_files) == 0:
        print(f"⚠️  No audio files found in {folder_path}")
        return pd.DataFrame(columns=['filename', 'prediction', 'confidence', 'status'])

    print(f"Found {len(audio_files)} audio file(s) in {folder_path}")

    # Load model once
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✓ Loaded model from {checkpoint_path}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    results = []

    # Process each file
    for audio_path in tqdm(audio_files, desc='Processing audio files'):
        filename = os.path.basename(audio_path)
        try:
            # Extract features (simplified version without reloading model)
            if feature_type == 'cqcc':
                features = extract_cqcc_features(audio_path, sr=sample_rate)
            else:
                from src.features.extract import extract_mel_spectrogram
                features = extract_mel_spectrogram(audio_path, sr=sample_rate)

            if features is None or features.size == 0:
                results.append({
                    'filename': filename,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'status': 'Feature extraction failed'
                })
                continue

            # Preprocess
            if features.ndim == 2:
                features = np.mean(features, axis=0)
            features = features.ravel()

            feat_mean = np.mean(features)
            feat_std = np.std(features) + 1e-9
            features = (features - feat_mean) / feat_std

            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            num_features = features_tensor.shape[1]
            side = int(np.ceil(np.sqrt(num_features)))
            pad_size = side * side - num_features
            if pad_size > 0:
                features_tensor = torch.nn.functional.pad(features_tensor, (0, pad_size))
            features_tensor = features_tensor.view(1, 1, side, side).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(features_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence_scores = probs.cpu().numpy()[0]
                pred_class = int(torch.argmax(probs, dim=1).item())

            label = 'Real' if pred_class == 0 else 'Fake'
            confidence = float(confidence_scores[pred_class])

            results.append({
                'filename': filename,
                'prediction': label,
                'confidence': confidence,
                'status': 'Success'
            })

        except Exception as e:
            results.append({
                'filename': filename,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'status': str(e)[:100]
            })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")

    # Print summary statistics
    successful = results_df[results_df['status'] == 'Success']
    if len(successful) > 0:
        real_count = (successful['prediction'] == 'Real').sum()
        fake_count = (successful['prediction'] == 'Fake').sum()
        total = len(successful)

        print("\n" + "=" * 50)
        print("  📊 BATCH PREDICTION SUMMARY")
        print("=" * 50)
        print(f"  Total files processed: {len(results_df)}")
        print(f"  Successful predictions: {total}")
        print(f"  Failed predictions: {len(results_df) - total}")
        print(f"\n  🟢 Real: {real_count} ({real_count/total*100:.1f}%)")
        print(f"  🔴 Fake: {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"\n  Average confidence: {successful['confidence'].mean():.2%}")
        print("=" * 50 + "\n")
    else:
        print("\n⚠️  No successful predictions to summarize\n")

    return results_df

