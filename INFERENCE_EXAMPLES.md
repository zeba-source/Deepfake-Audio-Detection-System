# Deepfake Audio Detection - Inference Examples

Quick reference for common inference scenarios.

## Single File Inference

### Basic Example

```bash
python main_inference.py \
    --audio_path sample.wav \
    --model_checkpoint models/best_model.pth
```

**Expected Output:**
```
[2025-10-21 14:30:00] ============================================================
[2025-10-21 14:30:00] DEEPFAKE AUDIO DETECTION - INFERENCE
[2025-10-21 14:30:00] ============================================================
[2025-10-21 14:30:00] Initializing resnet18 model...
[2025-10-21 14:30:01] ✓ Model initialized
[2025-10-21 14:30:01] Device: cpu
[2025-10-21 14:30:01] Feature type: cqcc
[2025-10-21 14:30:01] 
[2025-10-21 14:30:01] Running single audio prediction...
[2025-10-21 14:30:01] ------------------------------------------------------------
[2025-10-21 14:30:02] File: sample.wav
[2025-10-21 14:30:02] Prediction: FAKE (Confidence: 97.3%)
[2025-10-21 14:30:02] ------------------------------------------------------------
[2025-10-21 14:30:02] ✓ Prediction completed successfully
```

### With GPU Acceleration

```bash
python main_inference.py \
    --audio_path sample.wav \
    --model_checkpoint models/best_model.pth \
    --device cuda
```

### With Custom Features

```bash
python main_inference.py \
    --audio_path sample.wav \
    --model_checkpoint models/best_model.pth \
    --feature_type mel \
    --sample_rate 16000
```

---

## Batch Folder Inference

### Basic Batch Processing

```bash
python main_inference.py \
    --folder_path ./test_audio \
    --model_checkpoint models/best_model.pth \
    --output_csv results.csv
```

**Sample Output CSV:**

```csv
filename,prediction,confidence,processing_time_ms
audio_001.wav,FAKE,0.973,45.2
audio_002.wav,REAL,0.891,43.8
audio_003.wav,FAKE,0.856,44.1
audio_004.wav,REAL,0.924,42.9
```

### Batch with GPU

```bash
python main_inference.py \
    --folder_path ./large_dataset \
    --model_checkpoint models/best_model.pth \
    --output_csv results.csv \
    --device cuda
```

### Batch with Timestamped Output

```bash
# PowerShell
python main_inference.py `
    --folder_path ./test_audio `
    --model_checkpoint models/best_model.pth `
    --output_csv "results_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
```

---

## Advanced Examples

### Example 1: Multi-Stream Model

```bash
python main_inference.py \
    --audio_path sample.wav \
    --model_checkpoint models/multistream_best.pt \
    --model_type multistream \
    --feature_type mel
```

### Example 2: Production Pipeline

```bash
# Process incoming files
python main_inference.py \
    --folder_path data/incoming \
    --model_checkpoint models/production_v1.0.0.pt \
    --output_csv results/production_$(Get-Date -Format 'yyyyMMdd').csv \
    --device cuda \
    --feature_type cqcc
```

### Example 3: High Sample Rate Audio

```bash
python main_inference.py \
    --audio_path high_quality.flac \
    --model_checkpoint models/best_model.pth \
    --sample_rate 44100 \
    --feature_type mel
```

### Example 4: Batch Processing with Analysis

```bash
# Step 1: Run inference
python main_inference.py \
    --folder_path data/test_set \
    --model_checkpoint models/best_model.pth \
    --output_csv results.csv

# Step 2: Analyze results
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print(f'Total: {len(df)}')
print(f'Fake: {(df[\"prediction\"] == \"FAKE\").sum()}')
print(f'Real: {(df[\"prediction\"] == \"REAL\").sum()}')
print(f'Avg Confidence: {df[\"confidence\"].mean():.3f}')
"
```

---

## Use Case Scenarios

### Scenario 1: Verify Suspicious Audio

**Situation:** You received a suspicious voice message

```bash
# Run inference
python main_inference.py \
    --audio_path voicemail.wav \
    --model_checkpoint models/best_model.pth

# If FAKE detected, examine features
python src/features/visualize.py --audio_path voicemail.wav
```

### Scenario 2: Screen Customer Uploads

**Situation:** Batch screen user-uploaded audio files

```bash
# Process uploads folder daily
python main_inference.py \
    --folder_path uploads/$(Get-Date -Format 'yyyy-MM-dd') \
    --model_checkpoint models/best_model.pth \
    --output_csv results/screening_$(Get-Date -Format 'yyyyMMdd').csv \
    --device cuda
```

### Scenario 3: Forensic Analysis

**Situation:** Analyze evidence audio files

```bash
# Step 1: Run inference
python main_inference.py \
    --audio_path evidence_001.wav \
    --model_checkpoint models/best_model.pth \
    --feature_type cqcc \
    --device cpu

# Step 2: Generate detailed report
python scripts/forensic_report.py \
    --audio_path evidence_001.wav \
    --model_checkpoint models/best_model.pth
```

### Scenario 4: Dataset Validation

**Situation:** Validate labeled dataset for quality

```bash
# Process entire dataset
python main_inference.py \
    --folder_path data/labeled_dataset \
    --model_checkpoint models/best_model.pth \
    --output_csv validation_results.csv

# Compare with ground truth labels
python scripts/compare_labels.py \
    --predictions validation_results.csv \
    --ground_truth data/labels.csv
```

---

## PowerShell Automation Scripts

### Automated Daily Scanning

```powershell
# daily_scan.ps1
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$inputFolder = "E:\Audio\Incoming"
$outputCsv = "E:\Results\scan_$timestamp.csv"
$model = "E:\Models\best_model.pth"

Write-Host "Starting daily scan at $timestamp"

python main_inference.py `
    --folder_path $inputFolder `
    --model_checkpoint $model `
    --output_csv $outputCsv `
    --device cuda `
    --feature_type cqcc

Write-Host "Scan completed. Results: $outputCsv"

# Send email notification
# ... (email code)
```

### Batch Process Multiple Folders

```powershell
# batch_folders.ps1
$folders = @(
    "data/set_A",
    "data/set_B",
    "data/set_C"
)

foreach ($folder in $folders) {
    $name = Split-Path $folder -Leaf
    $output = "results/${name}_predictions.csv"
    
    Write-Host "Processing: $folder"
    
    python main_inference.py `
        --folder_path $folder `
        --model_checkpoint models/best_model.pth `
        --output_csv $output `
        --device cuda
    
    Write-Host "Completed: $output`n"
}

Write-Host "All folders processed!"
```

### Monitoring Script

```powershell
# monitor.ps1
while ($true) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $files = Get-ChildItem "E:\Monitoring\Incoming" -Filter *.wav
    
    if ($files.Count -gt 0) {
        Write-Host "[$timestamp] Found $($files.Count) new files"
        
        python main_inference.py `
            --folder_path "E:\Monitoring\Incoming" `
            --model_checkpoint "models/best_model.pth" `
            --output_csv "E:\Monitoring\Results\scan_$timestamp.csv" `
            --device cuda
        
        # Move processed files
        Move-Item "E:\Monitoring\Incoming\*.wav" "E:\Monitoring\Processed\"
        
        Write-Host "Processing complete. Sleeping..."
    }
    
    Start-Sleep -Seconds 300  # Check every 5 minutes
}
```

---

## Python Integration Examples

### Example 1: Simple Integration

```python
import subprocess
import json

def detect_deepfake(audio_path, model_path):
    """Run deepfake detection on audio file."""
    cmd = [
        'python', 'main_inference.py',
        '--audio_path', audio_path,
        '--model_checkpoint', model_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    for line in result.stdout.split('\n'):
        if 'Prediction:' in line:
            parts = line.split('Prediction:')[1].strip()
            prediction = parts.split('(')[0].strip()
            confidence = float(parts.split('Confidence:')[1].replace('%', '').replace(')', ''))
            
            return {
                'prediction': prediction,
                'confidence': confidence / 100.0
            }
    
    return None

# Usage
result = detect_deepfake('sample.wav', 'models/best_model.pth')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example 2: Batch Analysis

```python
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

def batch_analyze(folder, model, output_csv):
    """Run batch inference and analyze results."""
    # Run inference
    subprocess.run([
        'python', 'main_inference.py',
        '--folder_path', folder,
        '--model_checkpoint', model,
        '--output_csv', output_csv,
        '--device', 'cuda'
    ])
    
    # Load results
    df = pd.read_csv(output_csv)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:      {len(df):>6}")
    print(f"Fake detected:    {(df['prediction'] == 'FAKE').sum():>6}")
    print(f"Real detected:    {(df['prediction'] == 'REAL').sum():>6}")
    print(f"Avg confidence:   {df['confidence'].mean():>6.2%}")
    print(f"{'='*60}\n")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.savefig('confidence_distribution.png')
    print("Plot saved: confidence_distribution.png")
    
    return df

# Usage
df = batch_analyze('data/test', 'models/best_model.pth', 'results.csv')
```

### Example 3: Real-Time Monitoring

```python
import time
import subprocess
from pathlib import Path
from datetime import datetime

def monitor_folder(watch_folder, model, output_folder):
    """Monitor folder for new audio files and process them."""
    processed_files = set()
    
    print(f"Monitoring: {watch_folder}")
    print("Press Ctrl+C to stop\n")
    
    while True:
        # Find new files
        audio_files = list(Path(watch_folder).glob('*.wav'))
        new_files = [f for f in audio_files if f not in processed_files]
        
        if new_files:
            print(f"[{datetime.now()}] Found {len(new_files)} new files")
            
            for audio_file in new_files:
                print(f"Processing: {audio_file.name}")
                
                # Run inference
                subprocess.run([
                    'python', 'main_inference.py',
                    '--audio_path', str(audio_file),
                    '--model_checkpoint', model
                ])
                
                processed_files.add(audio_file)
        
        time.sleep(5)  # Check every 5 seconds

# Usage
try:
    monitor_folder('incoming/', 'models/best_model.pth', 'results/')
except KeyboardInterrupt:
    print("\nMonitoring stopped")
```

---

## Troubleshooting Common Issues

### Issue 1: File Not Found

```bash
# Error
✗ Error: Audio file not found: sample.wav

# Solution: Use absolute path
python main_inference.py `
    --audio_path "E:\Audio\sample.wav" `
    --model_checkpoint "E:\Models\best_model.pth"
```

### Issue 2: CUDA Not Available

```bash
# Error
RuntimeError: CUDA not available

# Solution: Use CPU
python main_inference.py `
    --audio_path sample.wav `
    --model_checkpoint models/best_model.pth `
    --device cpu
```

### Issue 3: Model Architecture Mismatch

```bash
# Error
RuntimeError: Error loading model checkpoint

# Solution: Specify correct model type
python main_inference.py `
    --audio_path sample.wav `
    --model_checkpoint models/multistream.pth `
    --model_type multistream  # Match architecture!
```

### Issue 4: Empty Batch Results

```bash
# Check folder contents
ls test_audio/

# Ensure correct audio formats
python main_inference.py `
    --folder_path test_audio `
    --model_checkpoint models/best_model.pth `
    --output_csv results.csv
```

---

## Performance Comparison

### Single File Processing

| Method | Command | Time |
|--------|---------|------|
| CPU | `--device cpu` | 75 ms |
| GPU | `--device cuda` | 8 ms |

### Batch Processing (100 files)

| Method | Command | Time |
|--------|---------|------|
| CPU | `--device cpu` | 7.5 sec |
| GPU | `--device cuda` | 0.8 sec |

**Recommendation:** Use GPU for batches > 10 files

---

## Quick Reference

```bash
# Single file
python main_inference.py --audio_path <file> --model_checkpoint <model>

# Batch folder
python main_inference.py --folder_path <folder> --model_checkpoint <model> --output_csv <csv>

# GPU inference
python main_inference.py --audio_path <file> --model_checkpoint <model> --device cuda

# Custom features
python main_inference.py --audio_path <file> --model_checkpoint <model> --feature_type mel

# Multi-stream model
python main_inference.py --audio_path <file> --model_checkpoint <model> --model_type multistream
```

---

**Last Updated:** October 21, 2025
