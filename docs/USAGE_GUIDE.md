# Deepfake Audio Detection - Usage Guide

Complete guide for training, testing, and deploying deepfake audio detection models.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Training Models](#training-models)
3. [Testing Models](#testing-models)
4. [Model Deployment](#model-deployment)
5. [Performance Profiling](#performance-profiling)
6. [Data Preparation](#data-preparation)
7. [Advanced Usage](#advanced-usage)

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your audio files:
```
data/
├── real/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── fake/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### 3. Train a Model

```bash
# Basic training
python main_train.py --data_folder ./data --feature_type mel --model_type resnet18

# With more epochs
python main_train.py --data_folder ./data --feature_type cqcc --model_type resnet34 --epochs 50 --batch_size 32
```

---

## Training Models

### Basic Training

```bash
python main_train.py \
    --data_folder ./data \
    --feature_type mel \
    --model_type resnet18 \
    --epochs 30 \
    --batch_size 32
```

### Available Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| `simple_cnn` | ~93K | Quick experiments, limited data |
| `resnet18` | ~11M | Balanced performance/speed |
| `resnet34` | ~21M | Higher accuracy |
| `resnet50` | ~23M | Maximum accuracy |
| `vgg16` | ~138M | Deep features |

### Feature Types

| Feature | Description | Shape | Best For |
|---------|-------------|-------|----------|
| `mel` | Mel spectrogram | 128×N | General purpose |
| `cqcc` | Constant-Q Cepstral Coefficients | 60×N | Deepfake detection |
| `mfcc` | Mel-Frequency Cepstral Coefficients | 40×N | Speech recognition |

### Training Examples

**1. ResNet18 with Mel Spectrograms**
```bash
python main_train.py \
    --data_folder ./data \
    --feature_type mel \
    --model_type resnet18 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001
```

**2. ResNet34 with CQCC Features**
```bash
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet34 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.0005
```

**3. VGG16 with Data Augmentation**
```bash
python main_train.py \
    --data_folder ./data \
    --feature_type mel \
    --model_type vgg16 \
    --epochs 40 \
    --batch_size 16 \
    --augmentation \
    --aug_prob 0.5
```

**4. Simple CNN for Quick Testing**
```bash
python main_train.py \
    --data_folder ./data \
    --feature_type mfcc \
    --model_type simple_cnn \
    --epochs 20 \
    --batch_size 64
```

### Command Line Arguments

#### Data Arguments
```bash
--data_folder PATH          # Path to audio data folder (required)
--train_split 0.7           # Training data proportion
--val_split 0.15            # Validation data proportion
--test_split 0.15           # Test data proportion
```

#### Feature Arguments
```bash
--feature_type {mel,cqcc,mfcc}  # Feature extraction method
--sample_rate 16000             # Audio sample rate (Hz)
--n_mels 128                    # Number of mel bands
--n_mfcc 40                     # Number of MFCC coefficients
```

#### Model Arguments
```bash
--model_type {resnet18,resnet34,resnet50,vgg16,simple_cnn}
--num_classes 2                 # Number of classes (default: 2)
```

#### Training Arguments
```bash
--epochs 30                     # Number of training epochs
--batch_size 32                 # Batch size
--learning_rate 0.001           # Initial learning rate
--weight_decay 1e-4             # L2 regularization
--optimizer {adam,sgd,adamw}    # Optimizer choice
--scheduler {step,cosine,plateau,none}  # LR scheduler
```

#### Augmentation Arguments
```bash
--augmentation                  # Enable data augmentation
--aug_prob 0.5                  # Augmentation probability
```

#### Device Arguments
```bash
--device {auto,cpu,cuda}        # Compute device
--num_workers 0                 # Data loader workers (0=main thread)
```

#### Output Arguments
```bash
--output_dir ./outputs          # Output directory
--model_save_dir ./models       # Model save directory
--experiment_name myexp         # Experiment name (auto-generated if not set)
```

#### Additional Features
```bash
--export_model                  # Export to TorchScript after training
--profile_model                 # Profile performance after training
--resume ./models/checkpoint.pth  # Resume from checkpoint
--seed 42                       # Random seed
```

### Training Output

The training script creates the following structure:

```
outputs/
└── {experiment_name}/
    ├── config.json              # Training configuration
    ├── training_history.json    # Loss and accuracy per epoch
    ├── test_results.json        # Final test metrics
    └── profiling/               # Performance metrics (if --profile_model)
        ├── {name}_profiling.json
        └── {name}_report.txt

models/
├── {experiment_name}_best.pth   # Best model checkpoint
└── {experiment_name}_scripted.pt # TorchScript model (if --export_model)
```

### Monitoring Training

Training progress is displayed in real-time:

```
================================================================================
[STEP 3/5] Training Model
================================================================================

Epoch 1/30
----------
  Training: 100%|████████| 50/50 [00:15<00:00, 3.21batch/s]
  Train Loss: 0.5234, Train Acc: 73.45%
  Val Loss: 0.4123, Val Acc: 81.23%
  ✓ New best model saved!

Epoch 2/30
----------
  Training: 100%|████████| 50/50 [00:14<00:00, 3.42batch/s]
  Train Loss: 0.3421, Train Acc: 85.67%
  Val Loss: 0.3012, Val Acc: 87.34%
  ✓ New best model saved!

...
```

---

## Testing Models

### Test Pre-trained Model

```bash
python -c "
from src.testing.test import test_model
from src.data.dataset import DeepfakeAudioDataset
import torch
from torch.utils.data import DataLoader

# Load test data
test_dataset = DeepfakeAudioDataset('./data', feature_type='mel')
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model
model = torch.load('models/best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test
results = test_model(model, test_loader, device)
print(f'Accuracy: {results[\"accuracy\"]*100:.2f}%')
"
```

### Evaluate on New Data

```bash
# Create test dataset and run evaluation
python test_evaluation.py --model models/best_model.pth --data ./new_data
```

---

## Model Deployment

### 1. Export Model to TorchScript

```bash
# During training
python main_train.py --data_folder ./data --feature_type mel --export_model

# Or export separately
python -c "
from src.deployment.export_model import export_model

export_info = export_model(
    model_path='models/best_model.pth',
    output_path='models/deployed_model.pt',
    model_type='resnet',
    input_shape=(1, 1, 128, 100)
)
print(f'Speedup: {export_info[\"performance\"][\"speedup\"]:.2f}x')
"
```

### 2. Start Flask API Server

```bash
# Start API server
python src/deployment/api.py --model models/deployed_model.pt

# With custom configuration
python src/deployment/api.py \
    --model models/deployed_model.pt \
    --feature-type mel \
    --sample-rate 16000 \
    --device cpu \
    --host 0.0.0.0 \
    --port 5000
```

### 3. Make Predictions

**Using cURL:**
```bash
# Predict single file
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict

# Health check
curl http://localhost:5000/health
```

**Using Python:**
```python
import requests

# Predict
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

---

## Performance Profiling

### Profile During Training

```bash
python main_train.py \
    --data_folder ./data \
    --feature_type mel \
    --model_type resnet18 \
    --profile_model
```

### Profile Existing Model

```bash
python test_profiling.py
```

### Custom Profiling

```python
from src.profiling import profile_model

# Profile your model
results = profile_model(
    model=my_model,
    input_shape=(1, 1, 128, 100),
    model_name='MyModel',
    devices=['cpu', 'cuda'],
    batch_sizes=[1, 4, 8, 16, 32],
    num_iterations=100,
    output_dir='profiling_results'
)

# View results
print(f"CPU: {results['profiles']['cpu']['single_inference']['mean_ms']:.2f} ms")
print(f"GPU: {results['profiles']['cuda']['single_inference']['mean_ms']:.2f} ms")
print(f"Speedup: {results['gpu_speedup']['speedup']:.2f}x")
```

---

## Data Preparation

### Audio Format Requirements

- **Supported formats**: WAV, MP3, FLAC, OGG, M4A
- **Recommended**: WAV (uncompressed)
- **Sample rate**: 16000 Hz (configurable)
- **Duration**: At least 0.5 seconds
- **Channels**: Mono or stereo (converted to mono)

### Directory Structure

```
data/
├── real/              # Real/genuine audio files
│   ├── real_001.wav
│   ├── real_002.wav
│   └── ...
└── fake/              # Fake/synthesized audio files
    ├── fake_001.wav
    ├── fake_002.wav
    └── ...
```

### Data Preprocessing

Audio files are automatically:
1. Loaded and resampled to target sample rate
2. Converted to mono if stereo
3. Feature extracted (mel/cqcc/mfcc)
4. Normalized
5. Augmented (if enabled)

### Data Augmentation

Enable augmentation for better generalization:

```bash
python main_train.py \
    --data_folder ./data \
    --augmentation \
    --aug_prob 0.5
```

**Available augmentations:**
- Time stretching
- Pitch shifting
- Noise injection
- Volume adjustment

---

## Advanced Usage

### Hyperparameter Tuning with Optuna

```python
from src.hyperparameter_tuning.tune import hyperparameter_tuning

# Run hyperparameter search
best_params, study = hyperparameter_tuning(
    train_loader=train_loader,
    val_loader=val_loader,
    model_type='resnet',
    n_trials=20,
    device='cuda'
)

print(f"Best learning rate: {best_params['learning_rate']}")
print(f"Best weight decay: {best_params['weight_decay']}")
```

### Resume Training

```bash
# Resume from checkpoint
python main_train.py \
    --data_folder ./data \
    --feature_type mel \
    --model_type resnet18 \
    --resume models/checkpoint.pth
```

### Custom Learning Rate Schedule

```bash
# Step scheduler (decrease every 10 epochs)
python main_train.py --scheduler step

# Cosine annealing
python main_train.py --scheduler cosine

# Reduce on plateau
python main_train.py --scheduler plateau

# No scheduler
python main_train.py --scheduler none
```

### Batch Size Tuning

Find optimal batch size for your GPU:

```bash
# Small GPU (4GB)
python main_train.py --batch_size 16

# Medium GPU (8GB)
python main_train.py --batch_size 32

# Large GPU (16GB+)
python main_train.py --batch_size 64
```

### Multi-GPU Training

```python
# Wrap model in DataParallel
model = nn.DataParallel(model)
model = model.to('cuda')
```

---

## Example Workflows

### Workflow 1: Quick Experiment

```bash
# Train simple model quickly
python main_train.py \
    --data_folder ./data \
    --model_type simple_cnn \
    --epochs 10 \
    --batch_size 64
```

### Workflow 2: Production Model

```bash
# Train production-ready model
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet34 \
    --epochs 50 \
    --batch_size 32 \
    --augmentation \
    --export_model \
    --profile_model \
    --experiment_name production_v1
```

### Workflow 3: Model Comparison

```bash
# Test different models
for model in simple_cnn resnet18 resnet34 vgg16; do
    python main_train.py \
        --data_folder ./data \
        --feature_type mel \
        --model_type $model \
        --epochs 30 \
        --experiment_name comparison_$model
done
```

### Workflow 4: Feature Comparison

```bash
# Test different features
for feat in mel cqcc mfcc; do
    python main_train.py \
        --data_folder ./data \
        --feature_type $feat \
        --model_type resnet18 \
        --epochs 30 \
        --experiment_name features_$feat
done
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python main_train.py --batch_size 16

# Use CPU
python main_train.py --device cpu

# Reduce workers
python main_train.py --num_workers 0
```

### Slow Training

```bash
# Use GPU
python main_train.py --device cuda

# Increase batch size
python main_train.py --batch_size 64

# Add workers (if CPU)
python main_train.py --num_workers 4
```

### Poor Accuracy

```bash
# Add augmentation
python main_train.py --augmentation

# Train longer
python main_train.py --epochs 100

# Try different features
python main_train.py --feature_type cqcc

# Use larger model
python main_train.py --model_type resnet34
```

---

## Performance Benchmarks

### Model Comparison (CPU)

| Model | Parameters | Inference Time | Accuracy |
|-------|-----------|----------------|----------|
| SimpleCNN | 93K | 1.27 ms | ~85% |
| ResNet18 | 11M | 7.31 ms | ~92% |
| ResNet34 | 21M | 12.5 ms | ~94% |
| VGG16 | 138M | 25.3 ms | ~93% |

### Feature Comparison

| Feature | Extraction Time | Accuracy | Robustness |
|---------|----------------|----------|------------|
| Mel | Fast | Good | Medium |
| CQCC | Medium | Excellent | High |
| MFCC | Fast | Good | Medium |

---

## Additional Resources

- **Documentation**: See `docs/` folder
- **Examples**: See `examples/` folder
- **Tests**: Run `pytest` for unit tests
- **API Documentation**: See `src/deployment/README.md`
- **Profiling Guide**: See `src/profiling/README.md`

---

## Support

For issues or questions:
1. Check documentation
2. Run tests to verify installation
3. Check GitHub issues
4. Create new issue with details

---

## License

See LICENSE file for details.
