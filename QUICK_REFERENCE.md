# Quick Reference Card - Deepfake Audio Detection

## 🚀 Quick Start Commands

### Extract Features
```bash
# Basic extraction
python main_extract_features.py --input_dir ./data --output_dir ./features --feature_type cqcc

# Multiple features with metadata
python main_extract_features.py --input_dir ./data --output_dir ./features --feature_type cqcc mel mfcc --save_metadata
```

### Train Model
```bash
# Basic
python main_train.py --data_folder ./data --feature_type mel --model_type resnet18

# Production
python main_train.py --data_folder ./data --feature_type cqcc --model_type resnet34 \
    --epochs 50 --batch_size 32 --augmentation --export_model --profile_model
```

### Run Inference
```bash
# Single file
python main_inference.py --audio_path sample.wav --model_checkpoint models/best_model.pth

# Batch folder
python main_inference.py --folder_path ./test_audio --model_checkpoint models/best_model.pth --output_csv results.csv
```

### Deploy API
```bash
python src/deployment/api.py --model models/deployed.pt
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

### Profile Model
```bash
python test_profiling.py
```

---

## 📋 Command Line Arguments Cheat Sheet

```bash
# Required
--data_folder PATH              # Audio data folder

# Features
--feature_type {mel,cqcc,mfcc}  # Feature extraction method
--sample_rate 16000             # Audio sample rate

# Model
--model_type {resnet18,resnet34,resnet50,vgg16,simple_cnn}
--num_classes 2                 # Output classes

# Training
--epochs 30                     # Training epochs
--batch_size 32                 # Batch size
--learning_rate 0.001           # Learning rate
--optimizer {adam,sgd,adamw}    # Optimizer
--scheduler {step,cosine,plateau,none}

# Augmentation
--augmentation                  # Enable augmentation
--aug_prob 0.5                  # Augmentation probability

# Device
--device {auto,cpu,cuda}        # Compute device
--num_workers 0                 # Data loader workers

# Output
--output_dir ./outputs          # Output directory
--model_save_dir ./models       # Model directory
--experiment_name NAME          # Experiment name

# Extra
--export_model                  # Export to TorchScript
--profile_model                 # Profile performance
--resume PATH                   # Resume from checkpoint
--seed 42                       # Random seed
```

---

## 📁 Data Structure

```
data/
├── real/          # Real audio files
│   ├── audio1.wav
│   └── audio2.wav
└── fake/          # Fake audio files
    ├── audio1.wav
    └── audio2.wav
```

---

## 🔑 Key File Locations

```
main_extract_features.py          # Feature extraction script
main_train.py                      # Main training script
main_inference.py                  # Main inference script
src/deployment/api.py              # Flask API server
src/profiling/profile_model.py     # Performance profiling
test_profiling.py                  # Profiling tests
test_deployment.py                 # Deployment tests
```

---

## 🎯 Common Workflows

### 1. Quick Test
```bash
python main_train.py --data_folder ./data --epochs 5
```

### 2. Full Training
```bash
python main_train.py --data_folder ./data --feature_type cqcc \
    --model_type resnet34 --epochs 50 --augmentation
```

### 3. Deploy & Test
```bash
# Terminal 1
python src/deployment/api.py --model models/best.pth

# Terminal 2
curl -X POST -F "file=@test.wav" http://localhost:5000/predict
```

---

## 📊 Model Comparison

| Model | Params | Speed | Accuracy |
|-------|--------|-------|----------|
| ResNet18 | 11M | Fast | ~92% |
| ResNet34 | 21M | Medium | ~94% |
| ResNet50 | 23M | Slow | ~95% |

---

## 🌟 Feature Comparison

| Feature | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| Mel | Fast | Good | General |
| CQCC | Medium | Best | Deepfake |
| MFCC | Fast | Good | Speech |

---

## 📝 Output Files

```
outputs/{experiment}/
├── config.json              # Configuration
├── training_history.json    # Training metrics
├── test_results.json        # Test results
└── profiling/               # Performance data

models/
├── {name}_best.pth         # Best model
└── {name}_scripted.pt      # TorchScript (if exported)
```

---

## 🔧 Troubleshooting

### Out of Memory
```bash
python main_train.py --batch_size 16 --device cpu
```

### Slow Training
```bash
python main_train.py --device cuda --batch_size 64
```

### Poor Accuracy
```bash
python main_train.py --augmentation --epochs 100 --feature_type cqcc
```

---

## 🌐 API Endpoints

```bash
GET  /              # API info
GET  /health        # Health check
GET  /info          # Model info
POST /predict       # Prediction (file upload)
```

---

## 📚 Documentation Quick Links

- **Usage Guide**: `docs/USAGE_GUIDE.md`
- **Training Guide**: `docs/MAIN_TRAIN_GUIDE.md`
- **Inference Guide**: `docs/INFERENCE_GUIDE.md`
- **Feature Extraction**: `docs/FEATURE_EXTRACTION_GUIDE.md`
- **Inference Examples**: `INFERENCE_EXAMPLES.md`
- **Deployment**: `docs/DEPLOYMENT_GUIDE.md`
- **Project Summary**: `docs/PROJECT_SUMMARY.md`
- **README**: `README.md`

---

## 🎓 Example Results

```json
{
  "prediction": "fake",
  "confidence": 0.9234,
  "probabilities": {
    "real": 0.0766,
    "fake": 0.9234
  }
}
```

---

## ⚡ Performance Tips

1. **Use CUDA**: `--device cuda` (10x faster)
2. **Batch Size**: Larger = better throughput
3. **Features**: CQCC for best accuracy
4. **Augmentation**: Enable for robustness
5. **Epochs**: 30-50 for convergence

---

## 🔍 Quick Checks

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# List models
ls models/

# View config
cat outputs/{experiment}/config.json

# Check results
cat outputs/{experiment}/test_results.json
```

---

## ✅ Installation

```bash
git clone <repo>
cd Deepfake
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

---

**For detailed information, see the full documentation in `docs/`**
