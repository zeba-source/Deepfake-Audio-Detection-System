# Complete Workflow: Training to Inference

This guide shows the complete workflow from preparing data to running inference.

## Overview

```
Data Preparation → Training → Evaluation → Export → Inference
```

---

## Step 1: Prepare Your Data

### Organize Audio Files

```
data/
├── real/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── fake/
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

### Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

### Convert Audio (if needed)

```bash
# Convert MP3 to WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Batch convert
for file in *.mp3; do
    ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

---

## Step 2: Train Your Model

### Basic Training

```bash
python main_train.py \
    --data_folder ./data \
    --feature_type mel \
    --model_type resnet18 \
    --epochs 30 \
    --batch_size 32
```

### Production Training (Recommended)

```bash
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet18 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --optimizer adam \
    --scheduler cosine \
    --augmentation \
    --device cuda \
    --export_model \
    --profile_model \
    --experiment_name production_v1
```

### Expected Training Output

```
======================================================================
                 DEEPFAKE AUDIO DETECTION - TRAINING
======================================================================
Experiment: production_v1
Device: cuda
Feature type: cqcc
Model: resnet18
======================================================================

[Step 1/5] Loading data...
✓ Training set: 1000 samples
✓ Validation set: 214 samples
✓ Test set: 214 samples

[Step 2/5] Creating model...
✓ Model created: ResNetDeepfakeDetector
✓ Parameters: 11,181,642
✓ Optimizer: Adam
✓ Scheduler: CosineAnnealingLR

[Step 3/5] Training model...
Epoch 1/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:45
Train Loss: 0.6234, Train Acc: 65.3%
Val Loss: 0.5123, Val Acc: 75.2%

...

Epoch 50/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:42
Train Loss: 0.0234, Train Acc: 99.1%
Val Loss: 0.0876, Val Acc: 97.2%

✓ Training completed!
✓ Best model saved: outputs/production_v1/best_model.pth

[Step 4/5] Testing model...
Test Accuracy: 96.73%
Test Precision: 97.12%
Test Recall: 96.34%
Test F1-Score: 96.73%

[Step 5/5] Exporting model...
✓ Model exported: outputs/production_v1/deployed_model.pt

[BONUS] Profiling model...
✓ Profiling complete!
✓ Report: outputs/production_v1/profiling/performance_report.json

======================================================================
TRAINING COMPLETED SUCCESSFULLY!
======================================================================
```

### Output Files

```
outputs/production_v1/
├── config.json                     # Training configuration
├── best_model.pth                  # Best model weights
├── deployed_model.pt               # Exported TorchScript model
├── training_history.json           # Loss/accuracy history
├── test_results.json               # Test metrics
└── profiling/
    ├── performance_report.json     # Performance metrics
    └── cpu_gpu_comparison.json     # CPU vs GPU benchmarks
```

---

## Step 3: Run Inference

### Single File Inference

```bash
python main_inference.py \
    --audio_path test_audio.wav \
    --model_checkpoint outputs/production_v1/best_model.pth
```

**Output:**

```
[2025-10-21 15:30:00] ============================================================
[2025-10-21 15:30:00] DEEPFAKE AUDIO DETECTION - INFERENCE
[2025-10-21 15:30:00] ============================================================
[2025-10-21 15:30:00] Initializing resnet18 model...
[2025-10-21 15:30:01] ✓ Model initialized
[2025-10-21 15:30:01] Device: cpu
[2025-10-21 15:30:01] Feature type: cqcc
[2025-10-21 15:30:01] 
[2025-10-21 15:30:01] Running single audio prediction...
[2025-10-21 15:30:01] ------------------------------------------------------------
[2025-10-21 15:30:02] File: test_audio.wav
[2025-10-21 15:30:02] Prediction: FAKE (Confidence: 97.3%)
[2025-10-21 15:30:02] ------------------------------------------------------------
[2025-10-21 15:30:02] ✓ Prediction completed successfully
[2025-10-21 15:30:02] 
[2025-10-21 15:30:02] ============================================================
[2025-10-21 15:30:02] INFERENCE COMPLETED SUCCESSFULLY!
[2025-10-21 15:30:02] ============================================================
```

### Batch Inference

```bash
python main_inference.py \
    --folder_path ./test_audio \
    --model_checkpoint outputs/production_v1/best_model.pth \
    --output_csv results.csv \
    --device cuda
```

**Output:**

```
[2025-10-21 15:35:00] ============================================================
[2025-10-21 15:35:00] DEEPFAKE AUDIO DETECTION - INFERENCE
[2025-10-21 15:35:00] ============================================================
[2025-10-21 15:35:00] Initializing resnet18 model...
[2025-10-21 15:35:01] ✓ Model initialized
[2025-10-21 15:35:01] Device: cuda
[2025-10-21 15:35:01] Feature type: cqcc
[2025-10-21 15:35:01] 
[2025-10-21 15:35:01] Running batch predictions...
[2025-10-21 15:35:01] ------------------------------------------------------------
Processing: 100%|████████████████████████| 150/150 [00:12<00:00, 12.5 files/s]
[2025-10-21 15:35:13] ------------------------------------------------------------
[2025-10-21 15:35:13] ✓ Batch predictions completed
[2025-10-21 15:35:13] ✓ Results saved to results.csv

Preview of results:
      filename prediction  confidence  processing_time_ms
   audio_001.wav      FAKE       0.973                45.2
   audio_002.wav      REAL       0.891                43.8
   audio_003.wav      FAKE       0.856                44.1
   ...
```

### Results CSV

**File: `results.csv`**

```csv
filename,prediction,confidence,processing_time_ms
audio_001.wav,FAKE,0.973,45.2
audio_002.wav,REAL,0.891,43.8
audio_003.wav,FAKE,0.856,44.1
audio_004.wav,REAL,0.924,42.9
audio_005.wav,FAKE,0.887,46.1
```

---

## Step 4: Deploy as API (Optional)

### Start API Server

```bash
python src/deployment/api.py \
    --model outputs/production_v1/deployed_model.pt \
    --port 5000
```

### Test API

```bash
# Using curl
curl -X POST -F "file=@test_audio.wav" http://localhost:5000/predict

# Using Python
python test_api_client.py --file test_audio.wav --url http://localhost:5000/predict
```

**API Response:**

```json
{
  "prediction": "fake",
  "confidence": 0.9734,
  "probabilities": {
    "real": 0.0266,
    "fake": 0.9734
  },
  "processing_time_ms": 45.2
}
```

---

## Complete Example Workflow

### Scenario: Train and Deploy Deepfake Detector

```bash
# 1. Prepare data
mkdir -p data/{real,fake}
cp /source/real_audio/* data/real/
cp /source/fake_audio/* data/fake/

# 2. Train model
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet18 \
    --epochs 50 \
    --batch_size 32 \
    --augmentation \
    --device cuda \
    --export_model \
    --profile_model \
    --experiment_name my_detector

# 3. Test on new audio
python main_inference.py \
    --audio_path suspicious_audio.wav \
    --model_checkpoint outputs/my_detector/best_model.pth

# 4. Batch process folder
python main_inference.py \
    --folder_path ./incoming_audio \
    --model_checkpoint outputs/my_detector/best_model.pth \
    --output_csv batch_results.csv \
    --device cuda

# 5. Deploy API
python src/deployment/api.py \
    --model outputs/my_detector/deployed_model.pt \
    --port 5000
```

---

## Production Workflow

### Daily Batch Processing Pipeline

```bash
#!/bin/bash
# production_pipeline.sh

DATE=$(date +%Y%m%d)
INPUT_DIR="/data/incoming/${DATE}"
OUTPUT_DIR="/data/results/${DATE}"
MODEL="/models/production/v1.0.0.pth"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run batch inference
python main_inference.py \
    --folder_path ${INPUT_DIR} \
    --model_checkpoint ${MODEL} \
    --output_csv ${OUTPUT_DIR}/predictions.csv \
    --device cuda \
    --feature_type cqcc

# Generate summary report
python scripts/generate_report.py \
    --csv ${OUTPUT_DIR}/predictions.csv \
    --output ${OUTPUT_DIR}/summary.pdf

# Send notification
python scripts/send_notification.py \
    --csv ${OUTPUT_DIR}/predictions.csv

# Archive processed files
mv ${INPUT_DIR}/* /data/archive/${DATE}/

echo "Pipeline completed: ${DATE}"
```

### Real-Time Monitoring

```python
# monitor.py
import time
from pathlib import Path
import subprocess

def monitor_folder(watch_dir, model, output_dir):
    """Monitor folder for new audio files."""
    processed = set()
    
    while True:
        # Find new files
        audio_files = list(Path(watch_dir).glob('*.wav'))
        new_files = [f for f in audio_files if f not in processed]
        
        if new_files:
            print(f"Found {len(new_files)} new files")
            
            for audio_file in new_files:
                # Run inference
                subprocess.run([
                    'python', 'main_inference.py',
                    '--audio_path', str(audio_file),
                    '--model_checkpoint', model
                ])
                
                processed.add(audio_file)
        
        time.sleep(10)  # Check every 10 seconds

# Run monitor
monitor_folder('/data/incoming', 'models/best_model.pth', '/data/results')
```

---

## Performance Optimization Tips

### 1. Use GPU for Large Batches

```bash
# CPU (slow)
python main_inference.py \
    --folder_path large_dataset/ \
    --model_checkpoint models/best_model.pth \
    --device cpu
# Speed: ~75 ms/file

# GPU (fast)
python main_inference.py \
    --folder_path large_dataset/ \
    --model_checkpoint models/best_model.pth \
    --device cuda
# Speed: ~8 ms/file (10x faster!)
```

### 2. Choose Right Feature Type

| Feature | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| CQCC | Medium | **High** | Best overall |
| MEL | **Fast** | Medium | Real-time |
| MFCC | Medium | Medium | General purpose |

### 3. Batch Size Optimization

```bash
# Small batches (default)
python main_train.py --batch_size 16  # Safer for memory

# Large batches (faster)
python main_train.py --batch_size 64 --device cuda  # Requires more VRAM
```

---

## Troubleshooting

### Issue: Model Not Found

```bash
# Error
✗ Error: Checkpoint not found: models/best_model.pth

# Solution: Check file exists
ls outputs/*/best_model.pth

# Use correct path
python main_inference.py \
    --audio_path sample.wav \
    --model_checkpoint outputs/production_v1/best_model.pth
```

### Issue: CUDA Out of Memory

```bash
# Error
RuntimeError: CUDA out of memory

# Solution 1: Use CPU
python main_inference.py --device cpu

# Solution 2: Reduce batch size
python main_train.py --batch_size 16
```

### Issue: Low Accuracy

```bash
# Check 1: Ensure enough training data
ls data/real/*.wav | wc -l  # Should be > 500
ls data/fake/*.wav | wc -l  # Should be > 500

# Check 2: Train longer
python main_train.py --epochs 50

# Check 3: Use augmentation
python main_train.py --augmentation

# Check 4: Try different features
python main_train.py --feature_type cqcc
```

---

## Summary of Commands

```bash
# Training
python main_train.py --data_folder ./data --feature_type cqcc --model_type resnet18 --epochs 50

# Single file inference
python main_inference.py --audio_path sample.wav --model_checkpoint outputs/exp/best_model.pth

# Batch inference
python main_inference.py --folder_path ./test --model_checkpoint outputs/exp/best_model.pth --output_csv results.csv

# API deployment
python src/deployment/api.py --model outputs/exp/deployed_model.pt
```

---

## Next Steps

1. **Experiment with different architectures**: Try `resnet34`, `resnet50`
2. **Fine-tune hyperparameters**: Use `src/hyperparameter_tuning/tune.py`
3. **Evaluate on real-world data**: Test with diverse audio samples
4. **Monitor performance**: Use profiling to optimize inference speed
5. **Deploy to production**: Use the Flask API or TorchScript export

---

## Documentation Links

- [Training Guide](docs/MAIN_TRAIN_GUIDE.md)
- [Inference Guide](docs/INFERENCE_GUIDE.md)
- [Inference Examples](INFERENCE_EXAMPLES.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Project Summary](docs/PROJECT_SUMMARY.md)
- [Quick Reference](QUICK_REFERENCE.md)

---

**Last Updated:** October 21, 2025
