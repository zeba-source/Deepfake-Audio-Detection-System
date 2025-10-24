# Main Training Script - Complete Implementation

## ✅ Successfully Created

A comprehensive production-ready training script (`main_train.py`) that integrates all 22 steps of the deepfake audio detection system.

## 📝 Features

### Core Functionality
- ✅ **Data Loading**: Automatic dataset splitting (train/val/test)
- ✅ **Feature Extraction**: Mel spectrograms, CQCC, MFCC
- ✅ **Model Training**: ResNet18/34/50 architectures
- ✅ **Evaluation**: Comprehensive testing with metrics
- ✅ **Model Export**: TorchScript conversion (optional)
- ✅ **Performance Profiling**: Detailed benchmarking (optional)

### Training Features
- ✅ **Optimizers**: Adam, SGD, AdamW
- ✅ **Schedulers**: StepLR, CosineAnnealing, ReduceLROnPlateau
- ✅ **Data Augmentation**: Time stretch, pitch shift, noise injection
- ✅ **Checkpointing**: Save best model, resume training
- ✅ **Device Support**: Auto-detect CUDA, CPU fallback

### Output & Logging
- ✅ **Structured Outputs**: JSON configs and results
- ✅ **Training History**: Loss/accuracy tracking
- ✅ **Test Metrics**: Accuracy, precision, recall, F1-score
- ✅ **Progress Bars**: Real-time training progress
- ✅ **Experiment Management**: Organized output directories

## 🎯 Usage Examples

### Example 1: Basic Training

```bash
python main_train.py \
    --data_folder ./data/audio \
    --feature_type mel \
    --model_type resnet18
```

**Expected Output:**
```
================================================================================
                   DEEPFAKE AUDIO DETECTION - TRAINING
================================================================================
Experiment: resnet18_mel_20251021_225500
Model: resnet18
Features: mel
Epochs: 30
Batch size: 32
...
✅ ALL TRAINING COMPLETED!
Test accuracy: 92.34%
```

### Example 2: Advanced Training with CQCC

```bash
python main_train.py \
    --data_folder ./data/audio \
    --feature_type cqcc \
    --model_type resnet34 \
    --epochs 50 \
    --batch_size 32 \
    --augmentation \
    --export_model \
    --profile_model
```

### Example 3: Resume Training

```bash
python main_train.py \
    --data_folder ./data/audio \
    --feature_type mel \
    --model_type resnet18 \
    --resume models/checkpoint.pth
```

## 📊 Command Line Arguments

### Complete Reference

```
Required Arguments:
  --data_folder PATH              # Path to audio data folder

Feature Extraction:
  --feature_type {mel,cqcc,mfcc}  # Feature type (default: mel)
  --sample_rate 16000             # Sample rate in Hz
  --n_mels 128                    # Mel bands for mel spectrograms
  --n_mfcc 40                     # MFCC coefficients

Model Configuration:
  --model_type {resnet18,resnet34,resnet50,vgg16,simple_cnn}
  --num_classes 2                 # Output classes

Training Parameters:
  --epochs 30                     # Training epochs
  --batch_size 32                 # Batch size
  --learning_rate 0.001           # Initial learning rate
  --weight_decay 1e-4             # L2 regularization
  --optimizer {adam,sgd,adamw}    # Optimizer
  --scheduler {step,cosine,plateau,none}  # LR scheduler

Data Splits:
  --train_split 0.7               # Training proportion
  --val_split 0.15                # Validation proportion
  --test_split 0.15               # Test proportion

Augmentation:
  --augmentation                  # Enable augmentation
  --aug_prob 0.5                  # Augmentation probability

Device & Performance:
  --device {auto,cpu,cuda}        # Compute device
  --num_workers 0                 # Data loader workers

Output:
  --output_dir ./outputs          # Output directory
  --model_save_dir ./models       # Model directory
  --experiment_name NAME          # Experiment name

Additional Features:
  --export_model                  # Export to TorchScript
  --profile_model                 # Profile performance
  --resume PATH                   # Resume from checkpoint
  --seed 42                       # Random seed
```

## 📁 Output Structure

Running the script creates:

```
outputs/
└── {experiment_name}/
    ├── config.json              # Complete configuration
    ├── training_history.json    # Loss/accuracy per epoch
    ├── test_results.json        # Final test metrics
    └── profiling/               # Performance data (if --profile_model)
        ├── profiling.json
        └── report.txt

models/
├── {experiment_name}_best.pth      # Best model checkpoint
└── {experiment_name}_scripted.pt   # TorchScript model (if --export_model)
```

### config.json
```json
{
  "data_folder": "./data/audio",
  "feature_type": "mel",
  "model_type": "resnet18",
  "epochs": 30,
  "batch_size": 32,
  "learning_rate": 0.001,
  "timestamp": "2025-10-21T22:55:00"
}
```

### training_history.json
```json
{
  "train_loss": [0.623, 0.421, 0.312, ...],
  "train_acc": [0.734, 0.856, 0.892, ...],
  "val_loss": [0.512, 0.367, 0.289, ...],
  "val_acc": [0.812, 0.873, 0.923, ...]
}
```

### test_results.json
```json
{
  "accuracy": 0.9234,
  "precision": 0.9123,
  "recall": 0.9345,
  "f1": 0.9233,
  "confusion_matrix": [[245, 15], [12, 228]]
}
```

## 🔧 Integration with Project Components

The script integrates all 22 steps:

| Step | Component | Integration |
|------|-----------|-------------|
| 1-2 | Dataset Loading | `src.data.dataset.DeepfakeAudioDataset` |
| 3-5 | Feature Extraction | `feature_type` argument (mel/cqcc/mfcc) |
| 6-8 | Model Architecture | `src.models.resnet_model.ResNetDeepfakeDetector` |
| 9-11 | Training Loop | `src.training.train.train_model` |
| 12-14 | Testing/Evaluation | `src.testing.test.test_model` |
| 15-17 | Data Augmentation | `--augmentation` flag |
| 18-20 | Hyperparameter Tuning | Manual configuration via args |
| 21 | Model Export | `--export_model` flag |
| 22 | Performance Profiling | `--profile_model` flag |

## 🚀 Workflow Examples

### Workflow 1: Quick Experiment

```bash
# Fast training for testing
python main_train.py \
    --data_folder ./data \
    --model_type resnet18 \
    --epochs 10 \
    --batch_size 64
```

### Workflow 2: Production Training

```bash
# Full production pipeline
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet34 \
    --epochs 50 \
    --batch_size 32 \
    --augmentation \
    --learning_rate 0.0005 \
    --export_model \
    --profile_model \
    --experiment_name production_v1
```

### Workflow 3: Model Comparison

```bash
# Compare different architectures
for model in resnet18 resnet34 resnet50; do
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
# Compare different features
for feat in mel cqcc mfcc; do
    python main_train.py \
        --data_folder ./data \
        --feature_type $feat \
        --model_type resnet18 \
        --epochs 30 \
        --experiment_name features_$feat
done
```

## 📈 Training Progress Example

```
================================================================================
                   DEEPFAKE AUDIO DETECTION - TRAINING
================================================================================
Experiment: resnet18_mel_20251021_225500
Model: resnet18
Features: mel
Epochs: 30
Batch size: 32
Learning rate: 0.001
Optimizer: adam
Scheduler: step
Augmentation: No
================================================================================

================================================================================
[STEP 1/5] Creating Data Loaders
================================================================================

Loading data from: ./data/audio

Dataset sizes:
  Total: 1000
  Training: 700 (70%)
  Validation: 150 (15%)
  Testing: 150 (15%)
✓ Data loaders created successfully

================================================================================
[STEP 2/5] Creating Model
================================================================================

Creating model: resnet18
Total parameters: 11,173,962
Trainable parameters: 11,173,962
Model size: 42.65 MB (float32)
✓ Model created and moved to cuda
✓ Optimizer: adam
✓ Scheduler: step
✓ Loss function: CrossEntropyLoss

================================================================================
[STEP 3/5] Training Model
================================================================================

Epoch 1/30
----------
  Training: 100%|████████████████| 22/22 [00:08<00:00, 2.53batch/s]
  Train Loss: 0.5234, Train Acc: 73.45%
  Val Loss: 0.4123, Val Acc: 81.23%
  ✓ New best model saved!

Epoch 2/30
----------
  Training: 100%|████████████████| 22/22 [00:07<00:00, 2.87batch/s]
  Train Loss: 0.3421, Train Acc: 85.67%
  Val Loss: 0.3012, Val Acc: 87.34%
  ✓ New best model saved!

...

✓ Training history saved to: outputs/resnet18_mel_20251021_225500/training_history.json

================================================================================
[STEP 4/5] Evaluating on Test Set
================================================================================

Testing model...
Testing: 100%|████████████████| 5/5 [00:01<00:00, 3.45batch/s]

========================================
TEST RESULTS
========================================
Accuracy:    92.34%
Precision:  0.9123
Recall:     0.9345
F1-Score:   0.9233
========================================

✓ Test results saved to: outputs/resnet18_mel_20251021_225500/test_results.json

================================================================================
                          TRAINING COMPLETE!
================================================================================

Experiment: resnet18_mel_20251021_225500
Best model: models/resnet18_mel_20251021_225500_best.pth
Test accuracy: 92.34%
F1-Score: 0.9233

Output directory: outputs/resnet18_mel_20251021_225500

Generated files:
  ✓ resnet18_mel_20251021_225500_best.pth
  ✓ config.json
  ✓ training_history.json
  ✓ test_results.json

================================================================================
```

## 🔍 Validation

The script has been validated with:
- ✅ Help message displays correctly
- ✅ All command-line arguments work
- ✅ Integration with all project modules
- ✅ Error handling for missing data
- ✅ Checkpoint saving/loading
- ✅ Progress tracking
- ✅ Results saving

## 📚 Additional Resources

- **Usage Guide**: `docs/USAGE_GUIDE.md`
- **README**: `README.md`
- **Deployment**: `docs/DEPLOYMENT_GUIDE.md`
- **Step Summaries**: `docs/STEP_*_SUMMARY.md`

## 🎓 Next Steps

After training:

1. **View Results**: Check `outputs/{experiment_name}/`
2. **Test Model**: Use saved checkpoint for inference
3. **Deploy**: Export and deploy via API
4. **Profile**: Analyze performance metrics
5. **Iterate**: Adjust hyperparameters and retrain

## ✅ Status

**Production-ready training script** with comprehensive features, error handling, and integration of all 22 project steps!
