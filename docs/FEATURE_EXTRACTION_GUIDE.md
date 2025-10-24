# Feature Extraction Guide

Complete guide for extracting audio features for deepfake detection.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Command Line Usage](#command-line-usage)
- [Feature Types](#feature-types)
- [Examples](#examples)
- [Output Formats](#output-formats)
- [Advanced Usage](#advanced-usage)

---

## Overview

The feature extraction script processes audio files and extracts features suitable for deepfake detection training.

### Supported Features

1. **CQCC** (Constant-Q Cepstral Coefficients) - Best for deepfake detection
2. **MEL** (Mel Spectrogram) - General audio analysis
3. **MFCC** (Mel-Frequency Cepstral Coefficients) - Speech recognition

---

## Quick Start

### Basic Usage

```bash
# Extract CQCC features
python main_extract_features.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --feature_type cqcc
```

### With Labels

```bash
# Extract features with labels CSV
python main_extract_features.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --feature_type cqcc \
    --labels labels.csv
```

### Multiple Feature Types

```bash
# Extract multiple feature types
python main_extract_features.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --feature_type cqcc mel mfcc \
    --save_metadata \
    --create_manifest
```

---

## Command Line Usage

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--input_dir` | str | Input directory containing audio files |
| `--output_dir` | str | Output directory for extracted features |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--labels` | str | None | Path to labels CSV file |
| `--feature_type` | str(s) | `cqcc` | Feature type(s) to extract |
| `--sample_rate` | int | `16000` | Audio sample rate |
| `--n_mels` | int | `128` | Number of mel bands |
| `--n_mfcc` | int | `40` | Number of MFCC coefficients |
| `--hop_length` | int | `512` | Hop length for STFT |
| `--n_fft` | int | `2048` | FFT window size |
| `--extensions` | str(s) | `wav mp3 flac ogg m4a` | Audio extensions |
| `--recursive` | flag | False | Search recursively |
| `--max_duration` | float | None | Maximum duration (seconds) |
| `--output_format` | str | `npy` | Output format (npy/npz/pt) |
| `--save_metadata` | flag | False | Save metadata JSON |
| `--create_manifest` | flag | False | Create manifest CSV |
| `--verbose` | flag | False | Verbose output |
| `--quiet` | flag | False | Suppress progress |

---

## Feature Types

### 1. CQCC (Constant-Q Cepstral Coefficients)

**Best for:** Deepfake audio detection

**Description:** Captures fine-grained spectral details using constant-Q transform.

**Shape:** `(20, n_frames)`

**Usage:**
```bash
python main_extract_features.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --feature_type cqcc
```

**Advantages:**
- ✅ Excellent for detecting synthesized speech
- ✅ Captures harmonic structure
- ✅ Robust to compression artifacts

### 2. MEL (Mel Spectrogram)

**Best for:** General audio analysis, visualization

**Description:** Time-frequency representation using mel scale.

**Shape:** `(n_mels, n_frames)` - Default: `(128, n_frames)`

**Usage:**
```bash
python main_extract_features.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --feature_type mel \
    --n_mels 128
```

**Advantages:**
- ✅ Fast extraction
- ✅ Good for visualization
- ✅ Standard in audio processing

### 3. MFCC (Mel-Frequency Cepstral Coefficients)

**Best for:** Speech recognition, speaker identification

**Description:** Cepstral coefficients derived from mel spectrogram.

**Shape:** `(n_mfcc, n_frames)` - Default: `(40, n_frames)`

**Usage:**
```bash
python main_extract_features.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --feature_type mfcc \
    --n_mfcc 40
```

**Advantages:**
- ✅ Compact representation
- ✅ Captures spectral envelope
- ✅ Standard in speech processing

---

## Examples

### Example 1: Basic Feature Extraction

```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type cqcc
```

**Output Structure:**
```
data/features/
└── cqcc/
    ├── audio_001.npy
    ├── audio_002.npy
    └── audio_003.npy
```

### Example 2: Multiple Feature Types

```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type cqcc mel mfcc
```

**Output Structure:**
```
data/features/
├── cqcc/
│   ├── audio_001.npy
│   └── audio_002.npy
├── mel/
│   ├── audio_001.npy
│   └── audio_002.npy
└── mfcc/
    ├── audio_001.npy
    └── audio_002.npy
```

### Example 3: With Labels

**labels.csv:**
```csv
filename,label
audio_001.wav,real
audio_002.wav,fake
audio_003.wav,real
```

**Command:**
```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type cqcc \
    --labels labels.csv \
    --create_manifest
```

**Output:** Creates `data/features/manifest.csv`

### Example 4: Recursive Processing

```bash
# Process nested directories
python main_extract_features.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --feature_type cqcc \
    --recursive
```

**Input Structure:**
```
data/raw/
├── real/
│   ├── speaker1/
│   │   └── audio.wav
│   └── speaker2/
│       └── audio.wav
└── fake/
    └── audio.wav
```

**Output Structure:**
```
data/processed/cqcc/
├── real/
│   ├── speaker1/
│   │   └── audio.npy
│   └── speaker2/
│       └── audio.npy
└── fake/
    └── audio.npy
```

### Example 5: Custom Parameters

```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type mel \
    --sample_rate 22050 \
    --n_mels 256 \
    --n_fft 4096 \
    --hop_length 1024
```

### Example 6: Production Pipeline

```bash
python main_extract_features.py \
    --input_dir /data/production/audio \
    --output_dir /data/production/features \
    --feature_type cqcc mel \
    --labels /data/production/labels.csv \
    --output_format npz \
    --save_metadata \
    --create_manifest \
    --recursive \
    --max_duration 10.0
```

---

## Output Formats

### 1. NumPy (.npy)

**Default format** - Uncompressed NumPy array

```bash
--output_format npy
```

**Pros:**
- Fast loading
- Simple format
- Direct numpy compatibility

**Cons:**
- Larger file size

### 2. Compressed NumPy (.npz)

Compressed NumPy archive

```bash
--output_format npz
```

**Pros:**
- Smaller file size
- Good compression ratio
- NumPy compatible

**Cons:**
- Slightly slower loading

### 3. PyTorch (.pt)

PyTorch tensor format

```bash
--output_format pt
```

**Pros:**
- Direct PyTorch compatibility
- GPU loading support

**Cons:**
- Requires PyTorch

---

## Advanced Usage

### Metadata File

Enable metadata saving to track extraction parameters:

```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type cqcc \
    --save_metadata
```

**metadata.json:**
```json
{
  "timestamp": "2025-10-21T14:30:00",
  "input_dir": "data/audio",
  "output_dir": "data/features",
  "feature_types": ["cqcc"],
  "sample_rate": 16000,
  "n_mels": 128,
  "n_mfcc": 40,
  "hop_length": 512,
  "n_fft": 2048,
  "statistics": {
    "total_files": 100,
    "processed": 98,
    "failed": 2
  }
}
```

### Manifest File

Create a manifest CSV for easy dataset loading:

```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --labels labels.csv \
    --create_manifest
```

**manifest.csv:**
```csv
filename,relative_path,label
audio_001.wav,audio_001.wav,real
audio_002.wav,audio_002.wav,fake
audio_003.wav,subdir/audio_003.wav,real
```

### Maximum Duration

Limit audio duration (clips longer files):

```bash
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type cqcc \
    --max_duration 10.0  # 10 seconds max
```

---

## Performance Tips

### 1. Batch Processing

Process multiple feature types in one pass:

```bash
# Efficient: Extract all features in one pass
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --feature_type cqcc mel mfcc

# Inefficient: Separate runs
python main_extract_features.py --feature_type cqcc ...
python main_extract_features.py --feature_type mel ...
python main_extract_features.py --feature_type mfcc ...
```

### 2. Output Format Selection

| Format | Speed | Size | Use Case |
|--------|-------|------|----------|
| `.npy` | ⚡⚡⚡ | Large | Development |
| `.npz` | ⚡⚡ | Small | Production |
| `.pt` | ⚡⚡⚡ | Medium | PyTorch models |

### 3. Feature Type Selection

| Feature | Extraction Speed | File Size | Accuracy |
|---------|-----------------|-----------|----------|
| CQCC | Medium | Small | **Best** |
| MEL | **Fast** | Medium | Good |
| MFCC | Fast | **Small** | Medium |

---

## Troubleshooting

### Issue 1: No Audio Files Found

**Error:**
```
ValueError: No audio files found in data/audio
```

**Solution:**
```bash
# Check directory
ls data/audio

# Use correct extensions
python main_extract_features.py \
    --input_dir data/audio \
    --output_dir data/features \
    --extensions wav mp3 flac
```

### Issue 2: Labels CSV Format Error

**Error:**
```
ValueError: Labels CSV must have 'filename' and 'label' columns
```

**Solution:**
Ensure CSV has correct columns:
```csv
filename,label
audio_001.wav,real
audio_002.wav,fake
```

### Issue 3: Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Use compressed output
python main_extract_features.py \
    --output_format npz

# Or limit duration
python main_extract_features.py \
    --max_duration 10.0
```

### Issue 4: Feature Extraction Failed

**Error:**
```
RuntimeError: Failed to extract features from audio_001.wav
```

**Solutions:**
1. Check audio file integrity
2. Verify audio format is supported
3. Try converting to WAV:
   ```bash
   ffmpeg -i input.mp3 -ar 16000 output.wav
   ```

---

## Python API Usage

### Direct Function Call (Original Method)

```python
from src.features.extract import extract_cqcc_features, extract_mel_spectrogram

# Extract CQCC
cqcc = extract_cqcc_features('audio.wav', sr=16000)

# Extract MEL
mel = extract_mel_spectrogram('audio.wav', sr=16000, n_mels=128)
```

### Batch Processing Script

```python
import subprocess

# Run feature extraction
subprocess.run([
    'python', 'main_extract_features.py',
    '--input_dir', 'data/audio',
    '--output_dir', 'data/features',
    '--feature_type', 'cqcc',
    '--save_metadata',
    '--create_manifest'
])

# Load features
import numpy as np
features = np.load('data/features/cqcc/audio_001.npy')
```

---

## Integration with Training

### Step 1: Extract Features

```bash
python main_extract_features.py \
    --input_dir data/raw_audio \
    --output_dir data/features \
    --feature_type cqcc \
    --labels labels.csv \
    --create_manifest
```

### Step 2: Train Model

```bash
python main_train.py \
    --data_folder data/raw_audio \
    --feature_type cqcc \
    --model_type resnet18 \
    --epochs 50
```

**Note:** Training script extracts features on-the-fly, so pre-extraction is optional.

---

## Summary

### Quick Commands

```bash
# Basic extraction
python main_extract_features.py --input_dir ./data/raw --output_dir ./data/processed --feature_type cqcc

# With labels
python main_extract_features.py --input_dir ./data/raw --output_dir ./data/processed --labels labels.csv

# Multiple features
python main_extract_features.py --input_dir ./data/raw --output_dir ./data/processed --feature_type cqcc mel mfcc

# Production
python main_extract_features.py --input_dir ./data --output_dir ./features --feature_type cqcc --save_metadata --create_manifest
```

### Best Practices

✅ **DO:**
- Use CQCC for deepfake detection
- Save metadata for reproducibility
- Create manifest for dataset tracking
- Use compressed format (.npz) for large datasets

❌ **DON'T:**
- Mix different sample rates
- Skip error checking
- Ignore failed extractions
- Delete original audio files

---

## Related Documentation

- [Training Guide](MAIN_TRAIN_GUIDE.md)
- [Inference Guide](INFERENCE_GUIDE.md)
- [Complete Workflow](../COMPLETE_WORKFLOW.md)
- [Project Summary](PROJECT_SUMMARY.md)

---

**Last Updated:** October 21, 2025
**Version:** 1.0.0
