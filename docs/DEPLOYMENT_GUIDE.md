# Model Export and Deployment Guide

## Overview
Production-ready model deployment system with TorchScript export and Flask API for real-time deepfake audio detection.

## Features

### Model Export
- ✅ TorchScript conversion (torch.jit.script)
- ✅ Model validation and benchmarking
- ✅ Inference optimization
- ✅ Export metadata tracking
- ✅ File size comparison
- ✅ Performance metrics

### Flask API
- ✅ RESTful endpoints (/health, /info, /predict)
- ✅ Audio file upload handling
- ✅ Input validation and sanitization
- ✅ Comprehensive error handling
- ✅ Logging and monitoring
- ✅ Production-ready configuration

## Quick Start

### 1. Export Model to TorchScript

```python
from src.deployment.export_model import export_model

# Export trained model
export_info = export_model(
    model_path='models/best_model.pth',
    output_path='models/deployed_model.pt',
    model_type='resnet',
    input_shape=(1, 1, 32, 32),
    use_trace=False,  # Use torch.jit.script
    optimize=False,   # Set True for mobile deployment
    device='cpu'
)

print(f"Model exported: {export_info['scripted_model_path']}")
print(f"Speedup: {export_info['performance']['speedup']:.2f}x")
```

### 2. Start API Server

```bash
# Basic usage
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

### 3. Test API

```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/info

# Make prediction
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

### 4. Python Client

```python
import requests

# Test prediction
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
```

## Model Export Details

### Supported Export Methods

#### torch.jit.script (Recommended)
```python
export_info = export_model(
    model_path='model.pth',
    output_path='model.pt',
    use_trace=False  # Use scripting
)
```

**Pros:**
- More flexible, handles control flow
- Better for models with dynamic behavior
- No example input needed for complex models

**Cons:**
- May require model code modifications
- Slightly slower export process

#### torch.jit.trace
```python
export_info = export_model(
    model_path='model.pth',
    output_path='model.pt',
    use_trace=True  # Use tracing
)
```

**Pros:**
- Faster export
- Works well for simple models

**Cons:**
- Requires example input
- May miss dynamic behavior
- Can have compatibility issues

### Export Information

The `export_model()` function returns detailed information:

```json
{
    "original_model_path": "models/test_model.pth",
    "scripted_model_path": "models/test_model_scripted.pt",
    "model_type": "resnet",
    "input_shape": [1, 1, 32, 32],
    "export_method": "script",
    "optimized": false,
    "device": "cpu",
    "validation": {
        "max_output_difference": 0.0,
        "outputs_match": true
    },
    "file_sizes": {
        "original_mb": 42.69,
        "scripted_mb": 42.75,
        "size_change_percent": 0.14
    },
    "performance": {
        "original_ms": 9.51,
        "scripted_ms": 8.55,
        "speedup": 1.11
    }
}
```

## API Endpoints

### GET /
Root endpoint with API information.

**Response:**
```json
{
    "service": "Deepfake Audio Detection API",
    "version": "1.0.0",
    "endpoints": {
        "/health": "Health check (GET)",
        "/predict": "Audio prediction (POST)",
        "/info": "Model information (GET)"
    }
}
```

### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu"
}
```

### GET /info
Model and configuration information.

**Response:**
```json
{
    "model_path": "models/deployed_model.pt",
    "feature_type": "mel",
    "sample_rate": 16000,
    "device": "cpu",
    "max_file_size_mb": 10,
    "allowed_extensions": ["wav", "mp3", "flac", "ogg", "m4a"]
}
```

### POST /predict
Audio prediction endpoint.

**Request:**
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

**Success Response (200):**
```json
{
    "prediction": "fake",
    "confidence": 0.9234,
    "probabilities": {
        "real": 0.0766,
        "fake": 0.9234
    },
    "metadata": {
        "filename": "audio.wav",
        "duration_seconds": 2.5,
        "sample_rate": 16000,
        "feature_type": "mel",
        "feature_shape": [128, 100]
    }
}
```

**Error Response (400):**
```json
{
    "error": "Invalid file type",
    "message": "Allowed types: wav, mp3, flac, ogg, m4a",
    "received": "txt"
}
```

## Error Handling

### Client-side Errors (400)

1. **No file provided**
   ```json
   {
       "error": "No file provided",
       "message": "Please upload an audio file using the 'file' field"
   }
   ```

2. **Invalid file type**
   ```json
   {
       "error": "Invalid file type",
       "message": "Allowed types: wav, mp3, flac, ogg, m4a"
   }
   ```

3. **Audio too short**
   ```json
   {
       "error": "Audio too short",
       "message": "Audio must be at least 0.5 seconds (got 0.2s)"
   }
   ```

4. **Invalid audio file**
   ```json
   {
       "error": "Invalid audio file",
       "message": "Could not load audio file. Please ensure it is a valid audio file."
   }
   ```

### Server-side Errors (500)

1. **Feature extraction failed**
2. **Inference failed**
3. **Internal server error**

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run API
CMD ["python", "src/deployment/api.py", \
     "--model", "models/deployed_model.pt", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

Build and run:
```bash
docker build -t deepfake-detector .
docker run -p 5000:5000 deepfake-detector
```

### Using Gunicorn (Production WSGI)

```bash
# Install gunicorn
pip install gunicorn

# Create WSGI app
# app.py
from src.deployment.api import create_api

api = create_api(model_path='models/deployed_model.pt')
app = api.app

# Run with gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/deepfake-api
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Increase timeout for large files
        proxy_read_timeout 300s;
        client_max_body_size 10M;
    }
}
```

### Systemd Service

```ini
# /etc/systemd/system/deepfake-api.service
[Unit]
Description=Deepfake Audio Detection API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/Deepfake
Environment="PATH=/path/to/Deepfake/.venv/bin"
ExecStart=/path/to/Deepfake/.venv/bin/python src/deployment/api.py \
          --model models/deployed_model.pt \
          --host 127.0.0.1 \
          --port 5000

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable deepfake-api
sudo systemctl start deepfake-api
sudo systemctl status deepfake-api
```

## Testing

### 1. Run Deployment Test Suite

```bash
python test_deployment.py
```

This will:
- Create a test model
- Export to TorchScript
- Validate export
- Create test audio files
- Show API usage examples

### 2. Test API with Client

```bash
# Start server first
python src/deployment/api.py --model models/test_model_scripted.pt

# In another terminal, run client tests
python test_api_client.py --audio test_audio/test_audio_1.wav
```

### 3. Manual Testing

```bash
# Health check
curl http://localhost:5000/health

# Prediction
curl -X POST -F "file=@test_audio/test_audio_1.wav" \
     http://localhost:5000/predict | jq .
```

## Performance Optimization

### 1. Model Optimization

```python
# Enable optimization for inference
export_info = export_model(
    model_path='model.pth',
    output_path='model.pt',
    optimize=True  # Enable optimization
)
```

### 2. Batch Processing

```python
# Process multiple files
import requests

files_to_process = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = []

for audio_file in files_to_process:
    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/predict', files=files)
        results.append(response.json())
```

### 3. GPU Acceleration

```bash
# Start API with GPU
python src/deployment/api.py \
    --model models/deployed_model.pt \
    --device cuda
```

### 4. Caching

```python
from functools import lru_cache

# Cache feature extraction for repeated files
@lru_cache(maxsize=100)
def cached_extract_features(audio_hash):
    return extract_features(audio_path)
```

## Monitoring and Logging

### Application Logs

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection

```python
# Track API metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
```

## Security Best Practices

### 1. Input Validation
- ✅ File type checking
- ✅ File size limits
- ✅ Audio duration validation
- ✅ Filename sanitization

### 2. Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    ...
```

### 3. HTTPS/TLS
- Use HTTPS in production
- Configure SSL certificates
- Redirect HTTP to HTTPS

### 4. Authentication

```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    ...
```

## Troubleshooting

### Issue: Model loading fails
**Solution:** Check model path and format
```python
# Verify model exists
import os
assert os.path.exists('models/deployed_model.pt')

# Try loading manually
import torch
model = torch.jit.load('models/deployed_model.pt')
```

### Issue: Feature extraction errors
**Solution:** Check audio file format
```python
import librosa

# Test audio loading
audio, sr = librosa.load('audio.wav', sr=16000)
print(f"Duration: {len(audio)/sr:.2f}s, SR: {sr}Hz")
```

### Issue: Out of memory
**Solutions:**
1. Reduce batch size
2. Use CPU instead of GPU
3. Limit concurrent requests
4. Process shorter audio clips

### Issue: Slow inference
**Solutions:**
1. Use GPU (`--device cuda`)
2. Enable model optimization
3. Reduce audio duration
4. Use batch processing

## Example Integration

### JavaScript Frontend

```javascript
// Upload audio for prediction
async function predictAudio(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);
    
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    console.log(`Prediction: ${result.prediction}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
}
```

### Python Script

```python
import requests
from pathlib import Path

def detect_deepfake(audio_path):
    """Detect if audio is deepfake."""
    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post('http://localhost:5000/predict', files=files)
        response.raise_for_status()
        return response.json()

# Use function
result = detect_deepfake('suspicious_audio.wav')
if result['prediction'] == 'fake':
    print(f"⚠️  FAKE detected with {result['confidence']*100:.1f}% confidence")
else:
    print(f"✓ REAL audio ({result['confidence']*100:.1f}% confidence)")
```

## See Also

- `src/models/resnet_model.py` - Model architecture
- `src/features/extract.py` - Feature extraction
- `test_deployment.py` - Deployment testing
- `test_api_client.py` - API testing client
