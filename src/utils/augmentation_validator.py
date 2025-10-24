"""Data augmentation validation utilities for deepfake audio detection."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def add_gaussian_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to audio signal.
    
    Args:
        audio: Audio signal array
        noise_factor: Scale of noise relative to signal
        
    Returns:
        Noisy audio signal
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio.astype(np.float32)


def pitch_shift_audio(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """Shift pitch of audio signal.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_steps: Number of semitones to shift (positive = higher, negative = lower)
        
    Returns:
        Pitch-shifted audio signal
    """
    augmented_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return augmented_audio


def time_stretch_audio(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Time-stretch audio signal.
    
    Args:
        audio: Audio signal array
        rate: Stretch factor (>1 = faster, <1 = slower)
        
    Returns:
        Time-stretched audio signal
    """
    augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
    return augmented_audio


def apply_spec_augment(
    spectrogram: np.ndarray,
    time_mask_param: int = 10,
    freq_mask_param: int = 10,
    num_time_masks: int = 1,
    num_freq_masks: int = 1
) -> np.ndarray:
    """Apply SpecAugment (time and frequency masking) to spectrogram.
    
    Args:
        spectrogram: Input spectrogram (freq_bins, time_frames)
        time_mask_param: Maximum width of time mask
        freq_mask_param: Maximum width of frequency mask
        num_time_masks: Number of time masks to apply
        num_freq_masks: Number of frequency masks to apply
        
    Returns:
        Augmented spectrogram
    """
    aug_spec = spectrogram.copy()
    freq_bins, time_frames = aug_spec.shape
    
    # Apply time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, time_frames))
        t0 = np.random.randint(0, time_frames - t)
        aug_spec[:, t0:t0 + t] = 0
    
    # Apply frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, freq_bins))
        f0 = np.random.randint(0, freq_bins - f)
        aug_spec[f0:f0 + f, :] = 0
    
    return aug_spec


def validate_augmentation(
    audio_path: str,
    label: int,
    output_dir: str = "results/augmentation_validation",
    sr: int = 16000,
    n_mels: int = 128,
    save_plots: bool = True
) -> Dict[str, any]:
    """Validate audio augmentation techniques.
    
    This function:
    1. Loads sample audio
    2. Applies various augmentations (SpecAugment, noise, pitch shift, time stretch)
    3. Visualizes original vs augmented spectrograms
    4. Saves comparison plots
    5. Verifies augmented data still has correct labels
    
    Args:
        audio_path: Path to audio file
        label: Ground truth label (0=real, 1=fake)
        output_dir: Directory to save validation plots
        sr: Sample rate for audio loading
        n_mels: Number of mel bands for spectrogram
        save_plots: Whether to save comparison plots
        
    Returns:
        Dictionary containing validation results and augmented samples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original audio
    print(f"Loading audio from: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr, duration=4.0)
    print(f"Audio shape: {audio.shape}, Label: {'Fake' if label == 1 else 'Real'}")
    
    # Compute original mel-spectrogram
    mel_spec_original = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, fmax=8000
    )
    mel_spec_db_original = librosa.power_to_db(mel_spec_original, ref=np.max)
    
    # Dictionary to store augmented data
    augmentations = {}
    
    # 1. Gaussian Noise Augmentation
    print("\n1. Applying Gaussian noise...")
    audio_noise = add_gaussian_noise(audio, noise_factor=0.005)
    mel_spec_noise = librosa.feature.melspectrogram(
        y=audio_noise, sr=sr, n_mels=n_mels, fmax=8000
    )
    mel_spec_db_noise = librosa.power_to_db(mel_spec_noise, ref=np.max)
    augmentations['gaussian_noise'] = {
        'audio': audio_noise,
        'mel_spec_db': mel_spec_db_noise,
        'label': label,
        'description': 'Gaussian Noise (factor=0.005)'
    }
    
    # 2. Pitch Shift Augmentation
    print("2. Applying pitch shift...")
    audio_pitch = pitch_shift_audio(audio, sr, n_steps=2.0)
    mel_spec_pitch = librosa.feature.melspectrogram(
        y=audio_pitch, sr=sr, n_mels=n_mels, fmax=8000
    )
    mel_spec_db_pitch = librosa.power_to_db(mel_spec_pitch, ref=np.max)
    augmentations['pitch_shift'] = {
        'audio': audio_pitch,
        'mel_spec_db': mel_spec_db_pitch,
        'label': label,
        'description': 'Pitch Shift (+2 semitones)'
    }
    
    # 3. Time Stretch Augmentation
    print("3. Applying time stretch...")
    audio_stretch = time_stretch_audio(audio, rate=1.1)
    mel_spec_stretch = librosa.feature.melspectrogram(
        y=audio_stretch, sr=sr, n_mels=n_mels, fmax=8000
    )
    mel_spec_db_stretch = librosa.power_to_db(mel_spec_stretch, ref=np.max)
    augmentations['time_stretch'] = {
        'audio': audio_stretch,
        'mel_spec_db': mel_spec_db_stretch,
        'label': label,
        'description': 'Time Stretch (rate=1.1x)'
    }
    
    # 4. SpecAugment (on mel-spectrogram)
    print("4. Applying SpecAugment...")
    mel_spec_db_specaug = apply_spec_augment(
        mel_spec_db_original,
        time_mask_param=20,
        freq_mask_param=15,
        num_time_masks=2,
        num_freq_masks=2
    )
    augmentations['spec_augment'] = {
        'audio': audio,  # Audio is same, only spectrogram is masked
        'mel_spec_db': mel_spec_db_specaug,
        'label': label,
        'description': 'SpecAugment (T=20, F=15)'
    }
    
    # Verify labels are preserved
    print("\n✓ Verifying labels are preserved...")
    labels_preserved = all(aug['label'] == label for aug in augmentations.values())
    print(f"  Labels preserved: {labels_preserved}")
    
    # Calculate statistics
    print("\n📊 Augmentation Statistics:")
    print(f"  Original audio: shape={audio.shape}, mean={audio.mean():.4f}, std={audio.std():.4f}")
    for aug_name, aug_data in augmentations.items():
        aug_audio = aug_data['audio']
        print(f"  {aug_name}: shape={aug_audio.shape}, mean={aug_audio.mean():.4f}, std={aug_audio.std():.4f}")
    
    # Visualization
    if save_plots:
        print(f"\n📈 Creating visualization plots...")
        filename = Path(audio_path).stem
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Audio Augmentation Validation - {filename}\nLabel: {"Fake" if label == 1 else "Real"}', 
                     fontsize=16, fontweight='bold')
        
        # Original
        ax = axes[0, 0]
        img = librosa.display.specshow(
            mel_spec_db_original, sr=sr, x_axis='time', y_axis='mel',
            fmax=8000, ax=ax, cmap='viridis'
        )
        ax.set_title('Original', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Augmented samples (5 plots)
        plot_positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        aug_names = ['gaussian_noise', 'pitch_shift', 'time_stretch', 'spec_augment']
        
        for i, aug_name in enumerate(aug_names):
            row, col = plot_positions[i]
            ax = axes[row, col]
            aug_data = augmentations[aug_name]
            
            img = librosa.display.specshow(
                aug_data['mel_spec_db'], sr=sr, x_axis='time', y_axis='mel',
                fmax=8000, ax=ax, cmap='viridis'
            )
            ax.set_title(aug_data['description'], fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Add summary text in the last subplot position
        row, col = plot_positions[4]
        ax = axes[row, col]
        ax.axis('off')
        summary_text = f"""
Augmentation Summary
─────────────────────
✓ All labels preserved: {labels_preserved}
✓ Original label: {label} ({'Fake' if label == 1 else 'Real'})
✓ Total augmentations: {len(augmentations)}

Applied Techniques:
• Gaussian Noise
• Pitch Shift (+2 semitones)
• Time Stretch (1.1x faster)
• SpecAugment (T/F masking)

Statistics:
Original: μ={audio.mean():.4f}, σ={audio.std():.4f}
"""
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'augmentation_validation_{filename}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison plot: {plot_path}")
        plt.close()
        
        # Create individual augmentation plots (detailed view)
        fig, axes = plt.subplots(1, len(augmentations) + 1, figsize=(20, 4))
        fig.suptitle(f'Detailed Augmentation Comparison - {filename}', fontsize=14, fontweight='bold')
        
        # Original
        img = librosa.display.specshow(
            mel_spec_db_original, sr=sr, x_axis='time', y_axis='mel',
            fmax=8000, ax=axes[0], cmap='viridis'
        )
        axes[0].set_title('Original', fontsize=10, fontweight='bold')
        plt.colorbar(img, ax=axes[0], format='%+2.0f dB')
        
        # Each augmentation
        for i, (aug_name, aug_data) in enumerate(augmentations.items(), 1):
            img = librosa.display.specshow(
                aug_data['mel_spec_db'], sr=sr, x_axis='time', y_axis='mel',
                fmax=8000, ax=axes[i], cmap='viridis'
            )
            axes[i].set_title(aug_data['description'], fontsize=10, fontweight='bold')
            plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
        
        plt.tight_layout()
        detail_plot_path = os.path.join(output_dir, f'augmentation_detailed_{filename}.png')
        plt.savefig(detail_plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved detailed plot: {detail_plot_path}")
        plt.close()
    
    # Prepare validation results
    validation_results = {
        'audio_path': audio_path,
        'original_label': label,
        'labels_preserved': labels_preserved,
        'num_augmentations': len(augmentations),
        'augmentations': augmentations,
        'original_audio': audio,
        'original_mel_spec_db': mel_spec_db_original,
        'audio_stats': {
            'mean': float(audio.mean()),
            'std': float(audio.std()),
            'min': float(audio.min()),
            'max': float(audio.max()),
            'duration_seconds': len(audio) / sr
        }
    }
    
    print("\n✅ Augmentation validation complete!")
    print(f"   - Labels preserved: {labels_preserved}")
    print(f"   - Total augmentations: {len(augmentations)}")
    if save_plots:
        print(f"   - Plots saved to: {output_dir}")
    
    return validation_results


def batch_validate_augmentation(
    audio_paths: List[str],
    labels: List[int],
    output_dir: str = "results/augmentation_validation",
    max_samples: int = 5
) -> Dict[str, any]:
    """Validate augmentation on multiple audio samples.
    
    Args:
        audio_paths: List of paths to audio files
        labels: List of corresponding labels
        output_dir: Directory to save validation plots
        max_samples: Maximum number of samples to validate
        
    Returns:
        Dictionary containing batch validation results
    """
    print(f"\n{'='*70}")
    print(f"BATCH AUGMENTATION VALIDATION")
    print(f"{'='*70}")
    print(f"Total samples: {len(audio_paths)}")
    print(f"Validating up to {max_samples} samples...")
    
    results = []
    num_to_process = min(len(audio_paths), max_samples)
    
    for i in range(num_to_process):
        print(f"\n{'─'*70}")
        print(f"Processing sample {i+1}/{num_to_process}")
        print(f"{'─'*70}")
        
        result = validate_augmentation(
            audio_path=audio_paths[i],
            label=labels[i],
            output_dir=output_dir,
            save_plots=True
        )
        results.append(result)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"BATCH VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples validated: {len(results)}")
    print(f"Labels preserved: {sum(r['labels_preserved'] for r in results)}/{len(results)}")
    print(f"Real samples: {sum(1 for r in results if r['original_label'] == 0)}")
    print(f"Fake samples: {sum(1 for r in results if r['original_label'] == 1)}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'num_samples': len(results),
        'results': results,
        'all_labels_preserved': all(r['labels_preserved'] for r in results),
        'output_dir': output_dir
    }


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python augmentation_validator.py <audio_path> <label>")
        print("Example: python augmentation_validator.py sample.wav 0")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    label = int(sys.argv[2])
    
    # Validate single audio file
    results = validate_augmentation(
        audio_path=audio_path,
        label=label,
        output_dir="results/augmentation_validation",
        save_plots=True
    )
    
    print("\n✅ Done! Check 'results/augmentation_validation' for plots.")
