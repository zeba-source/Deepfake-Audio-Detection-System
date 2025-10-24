# 🚨 Why Your Model Says Human Audio is AI-Generated

## The Problem

The current model (`models/test_model.pth`) is showing **incorrect predictions** because:

1. ❌ **No Training Data** - Your `data/` folder is empty
2. ❌ **Untrained Model** - The test model wasn't trained on real data
3. ❌ **Random Predictions** - It's essentially guessing

## ✅ Solution: Train a Real Model

### Step 1: Prepare Training Data

You need **both human and AI-generated audio samples**:

```
data/
├── real/          # Put HUMAN voice recordings here
│   ├── human_001.wav
│   ├── human_002.wav
│   ├── human_003.wav
│   └── ... (at least 100+ files recommended)
└── fake/          # Put AI-generated audio here
    ├── ai_001.wav
    ├── ai_002.wav
    ├── ai_003.wav
    └── ... (at least 100+ files recommended)
```

### Step 2: Get Audio Samples

#### Option A: Record Your Own Data
```bash
# Record human voices (use any recording app)
# - Record yourself speaking
# - Record friends/family
# - Use different microphones
# - Vary recording conditions

# Generate AI voices using:
# - ElevenLabs (https://elevenlabs.io)
# - Google TTS
# - Microsoft Azure TTS
# - OpenAI TTS
# - Any text-to-speech service
```

#### Option B: Use Public Datasets

**For Audio Deepfake Detection (ASVspoof dataset):**
```bash
# Download ASVspoof 2019 dataset
# https://datashare.ed.ac.uk/handle/10283/3336

# Or other datasets:
# - VCTK (real voices)
# - FakeAVCeleb (deepfake dataset)
# - DFDC (DeepFake Detection Challenge)
```

#### Option C: Quick Test (Minimal Dataset)

Create a small test dataset:

```bash
# 1. Record 20-30 short clips of human voice
# 2. Generate 20-30 clips using TTS services
# 3. Place in data/real/ and data/fake/
```

### Step 3: Train Your Model

Once you have data in the folders:

```bash
# Basic training (30 epochs)
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet18 \
    --epochs 30 \
    --batch_size 16

# Better training (50 epochs with augmentation)
python main_train.py \
    --data_folder ./data \
    --feature_type cqcc \
    --model_type resnet18 \
    --epochs 50 \
    --batch_size 32 \
    --augmentation \
    --learning_rate 0.001 \
    --export_model
```

### Step 4: Use Your Trained Model

After training completes:

```bash
# The trained model will be saved to:
# outputs/{experiment_name}/best_model.pth

# Copy it to models folder:
copy outputs/your_experiment/best_model.pth models/trained_model.pth

# Or update app.py to use the new model directly
```

### Step 5: Restart Server with New Model

```bash
# Stop current server (Ctrl+C in terminal)

# Update app.py to load your trained model
# Then restart:
python app.py
```

---

## 🎯 Quick Start Guide (5 Minutes)

### Minimal Setup for Testing:

1. **Create folders:**
   ```bash
   mkdir data\real
   mkdir data\fake
   ```

2. **Add audio files:**
   - Put 10-20 human voice recordings in `data/real/`
   - Generate 10-20 AI voices and put in `data/fake/`

3. **Train model:**
   ```bash
   python main_train.py --data_folder ./data --epochs 10
   ```

4. **Use trained model:**
   ```bash
   # Model saved to: outputs/resnet18_cqcc_YYYYMMDD_HHMMSS/best_model.pth
   # Copy to: models/trained_model.pth
   ```

---

## 📊 Expected Results After Training

With proper training data:

- **Human Voice:** 85-95% confidence (correctly identified as REAL)
- **AI Voice:** 85-95% confidence (correctly identified as FAKE)
- **Training Time:** 5-30 minutes (depends on data size)
- **Minimum Data:** 50+ files per class (100 total)
- **Recommended Data:** 500+ files per class (1000 total)

---

## 🔍 Why AI Detection is Hard

The model learns to detect:

1. **Acoustic Artifacts** - AI voices have subtle imperfections
2. **Prosody Patterns** - Human speech has natural rhythm
3. **Spectral Features** - Frequency characteristics differ
4. **Background Noise** - Real recordings have environmental sounds
5. **Compression Artifacts** - Processing leaves traces

**Without training data, the model cannot learn these patterns!**

---

## 🛠️ Troubleshooting

### Issue: Still showing wrong predictions after training

**Possible causes:**
1. Not enough training data (need 100+ files minimum)
2. Low quality audio files
3. Similar quality between real and fake (hard to distinguish)
4. Training didn't converge (increase epochs)

**Solutions:**
```bash
# Train longer with more data
python main_train.py --data_folder ./data --epochs 100 --batch_size 32

# Use data augmentation
python main_train.py --data_folder ./data --augmentation --epochs 50

# Check training accuracy in outputs folder
cat outputs/{experiment}/test_results.json
```

### Issue: Model says everything is fake (or everything is real)

**Cause:** Imbalanced dataset or poor training

**Solution:**
```bash
# Ensure equal numbers of real and fake files
ls data/real/*.wav | wc -l    # Should be ~same
ls data/fake/*.wav | wc -l    # Should be ~same

# Retrain with balanced data
```

---

## 📥 Where to Get Training Data

### Free Datasets:

1. **ASVspoof 2019** (Recommended)
   - URL: https://datashare.ed.ac.uk/handle/10283/3336
   - Contains: Real + Fake audio for deepfake detection
   - Size: ~20GB

2. **VCTK Corpus** (Real voices)
   - URL: https://datashare.ed.ac.uk/handle/10283/3443
   - Contains: 110 speakers, clean recordings
   - Size: ~10GB

3. **LJSpeech** (Real voices)
   - URL: https://keithito.com/LJ-Speech-Dataset/
   - Contains: 13,100 audio clips
   - Size: ~2.6GB

### Generate Fake Audio:

1. **ElevenLabs** - https://elevenlabs.io (best quality)
2. **Google Cloud TTS** - https://cloud.google.com/text-to-speech
3. **Microsoft Azure** - https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/
4. **Coqui TTS** (free, local) - https://github.com/coqui-ai/TTS

---

## 🎓 Training Tips

### For Best Results:

1. ✅ **Balanced Dataset:** Equal real and fake samples
2. ✅ **Quality Audio:** Clear recordings, 16kHz+ sample rate
3. ✅ **Variety:** Different speakers, accents, languages
4. ✅ **Multiple Sources:** Use various TTS engines for fake audio
5. ✅ **Augmentation:** Enable `--augmentation` flag
6. ✅ **Enough Epochs:** At least 30-50 epochs
7. ✅ **Validation:** Check test accuracy (should be >85%)

### Training Command (Recommended):

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
    --device cpu \
    --export_model \
    --experiment_name my_deepfake_detector
```

---

## 🚀 Next Steps

1. **Get Training Data** (most important!)
2. **Organize into data/real/ and data/fake/**
3. **Train model with main_train.py**
4. **Test accuracy on validation set**
5. **Deploy trained model in web interface**

**Remember:** AI detection is only as good as your training data!

---

## 💡 Quick Demo (If You Don't Have Data Yet)

To just see how the interface works:

```bash
# The current model is not trained
# It will give random predictions
# This is EXPECTED until you train on real data
```

**Bottom Line:** You MUST train with real data to get accurate predictions!

---

Last Updated: October 21, 2025
