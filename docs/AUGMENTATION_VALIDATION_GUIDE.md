# Data Augmentation Validation - Usage Guide

## Overview
The augmentation validation module ensures that data augmentation techniques preserve labels and don't corrupt audio data while providing visual verification of augmentation effects.

## Features

### Augmentation Techniques
1. **Gaussian Noise** - Adds random noise to simulate recording conditions
2. **Pitch Shift** - Changes pitch by semitones (±2 semitones)
3. **Time Stretch** - Speeds up or slows down audio (1.1x rate)
4. **SpecAugment** - Masks time and frequency bands in spectrograms

### Validation Checks
- ✅ Labels preserved across all augmentations
- ✅ Audio statistics tracking (mean, std, min, max)
- ✅ Shape verification
- ✅ Visual comparison plots

## Quick Start

### Single Audio File Validation

```python
from src.utils.augmentation_validator import validate_augmentation

# Validate a single audio file
result = validate_augmentation(
    audio_path='path/to/audio.wav',
    label=0,  # 0=real, 1=fake
    output_dir='results/augmentation_validation',
    sr=16000,
    save_plots=True
)

# Access results
print(f"Labels preserved: {result['labels_preserved']}")
print(f"Number of augmentations: {result['num_augmentations']}")
print(f"Audio duration: {result['audio_stats']['duration_seconds']:.2f}s")
```

### Batch Validation

```python
from src.utils.augmentation_validator import batch_validate_augmentation

# Validate multiple files
audio_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
labels = [0, 1, 0]

batch_result = batch_validate_augmentation(
    audio_paths=audio_paths,
    labels=labels,
    output_dir='results/augmentation_validation',
    max_samples=5
)

print(f"Total samples: {batch_result['num_samples']}")
print(f"All labels preserved: {batch_result['all_labels_preserved']}")
```

### Using Individual Augmentation Functions

```python
from src.utils.augmentation_validator import (
    add_gaussian_noise,
    pitch_shift_audio,
    time_stretch_audio,
    apply_spec_augment
)
import librosa

# Load audio
audio, sr = librosa.load('audio.wav', sr=16000)

# Apply individual augmentations
audio_noisy = add_gaussian_noise(audio, noise_factor=0.005)
audio_pitched = pitch_shift_audio(audio, sr, n_steps=2.0)
audio_stretched = time_stretch_audio(audio, rate=1.1)

# For SpecAugment on spectrograms
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_spec_aug = apply_spec_augment(
    mel_spec,
    time_mask_param=20,
    freq_mask_param=15,
    num_time_masks=2,
    num_freq_masks=2
)
```

## Command Line Usage

```bash
# Validate single file
python src/utils/augmentation_validator.py audio.wav 0

# Run demo with synthetic data
python test_augmentation_validation.py
```

## Output Files

### Generated Plots
1. **augmentation_validation_*.png** - Comprehensive 3x2 grid comparison
   - Original spectrogram
   - Gaussian noise
   - Pitch shift
   - Time stretch
   - SpecAugment
   - Summary statistics

2. **augmentation_detailed_*.png** - Detailed side-by-side view
   - All augmentations in a single row
   - Easy visual comparison

### File Structure
```
results/augmentation_validation/
├── augmentation_validation_real_0.png
├── augmentation_detailed_real_0.png
├── augmentation_validation_fake_1.png
└── augmentation_detailed_fake_1.png
```

## Result Dictionary Structure

```python
{
    'audio_path': str,                    # Path to original audio
    'original_label': int,                # 0=real, 1=fake
    'labels_preserved': bool,             # True if all labels match
    'num_augmentations': int,             # Number of augmentations applied
    'augmentations': {
        'gaussian_noise': {
            'audio': np.ndarray,          # Augmented audio
            'mel_spec_db': np.ndarray,    # Mel-spectrogram (dB)
            'label': int,                 # Preserved label
            'description': str            # Human-readable description
        },
        # ... other augmentations
    },
    'original_audio': np.ndarray,         # Original audio array
    'original_mel_spec_db': np.ndarray,   # Original spectrogram
    'audio_stats': {
        'mean': float,
        'std': float,
        'min': float,
        'max': float,
        'duration_seconds': float
    }
}
```

## Integration with Training Pipeline

### Before Training
```python
from src.utils.augmentation_validator import validate_augmentation
from src.features.feature_extractor import FeatureExtractor

# 1. Validate augmentations on sample data
validate_augmentation(
    audio_path='samples/real_sample.wav',
    label=0,
    output_dir='results/augmentation_check'
)

# 2. Extract features with validated augmentation pipeline
extractor = FeatureExtractor(feature_type='mel', sample_rate=16000)
features, labels, df = extractor.process_dataset(
    data_folder='data/raw',
    output_folder='data/processed'
)
```

### During Training
```python
from src.models.dataset import DeepfakeAudioDataset

# Use SpecAugment in training dataset
train_dataset = DeepfakeAudioDataset(
    features, labels,
    train=True,  # Enables SpecAugment
    time_mask_param=20,
    freq_mask_param=15
)
```

## Best Practices

1. **Validate Before Training**
   - Run validation on representative samples
   - Check plots to ensure augmentations are reasonable
   - Verify labels are preserved

2. **Augmentation Parameters**
   - Start with conservative parameters
   - Gaussian noise: 0.005 - 0.01
   - Pitch shift: ±2 semitones
   - Time stretch: 0.9 - 1.1x
   - SpecAugment: T=10-20, F=10-15

3. **Visual Inspection**
   - Check that augmented spectrograms are realistic
   - Ensure augmentations don't completely destroy features
   - Verify no artifacts or distortions

4. **Statistical Monitoring**
   - Track mean/std changes
   - Monitor audio duration changes (time stretch)
   - Ensure values stay in reasonable ranges

## Troubleshooting

### Issue: Labels not preserved
```python
# Check result
if not result['labels_preserved']:
    for name, aug in result['augmentations'].items():
        if aug['label'] != result['original_label']:
            print(f"Label mismatch in {name}")
```

### Issue: Plots not generated
```python
# Ensure output directory exists and is writable
import os
output_dir = 'results/augmentation_validation'
os.makedirs(output_dir, exist_ok=True)

# Run with save_plots=True
validate_augmentation(..., save_plots=True)
```

### Issue: Audio too distorted
```python
# Reduce augmentation strength
audio_noise = add_gaussian_noise(audio, noise_factor=0.001)  # Reduced
audio_pitch = pitch_shift_audio(audio, sr, n_steps=1.0)      # Reduced
audio_stretch = time_stretch_audio(audio, rate=1.05)         # Reduced
```

## Testing

Run unit tests to verify functionality:
```bash
# Run all augmentation tests
python -m pytest tests/test_augmentation.py -v

# Run specific test
python -m pytest tests/test_augmentation.py::test_validate_augmentation -v
```

## Example: Complete Validation Workflow

```python
from src.utils.augmentation_validator import batch_validate_augmentation
import os

# Get sample files
real_files = [f for f in os.listdir('data/raw/real') if f.endswith('.wav')][:5]
fake_files = [f for f in os.listdir('data/raw/fake') if f.endswith('.wav')][:5]

audio_paths = (
    [os.path.join('data/raw/real', f) for f in real_files] +
    [os.path.join('data/raw/fake', f) for f in fake_files]
)
labels = [0] * len(real_files) + [1] * len(fake_files)

# Run batch validation
result = batch_validate_augmentation(
    audio_paths=audio_paths,
    labels=labels,
    output_dir='results/augmentation_validation',
    max_samples=10
)

# Check results
print(f"✅ Validated {result['num_samples']} samples")
print(f"✅ All labels preserved: {result['all_labels_preserved']}")
print(f"📊 Check plots in: {result['output_dir']}")
```

## Notes

- All augmentations preserve the original label
- Spectrograms use mel-scale with 128 bands, max frequency 8kHz
- Time-stretched audio has different length (faster = shorter)
- SpecAugment only affects the spectrogram, not raw audio
- Plots are saved as PNG at 150 DPI

## See Also

- `src/models/dataset.py` - SpecAugment in PyTorch Dataset
- `src/features/extract.py` - Feature extraction functions
- `tests/test_augmentation.py` - Unit tests for validation
