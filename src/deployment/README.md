# Deployment Module

Production-ready model export and API deployment for deepfake audio detection.

## Overview

This module provides:
- **Model Export**: Convert PyTorch models to TorchScript for optimized inference
- **REST API**: Flask-based API for real-time audio predictions
- **Validation**: Automated testing and validation of exported models
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Files

### `export_model.py`
Model export utilities for converting trained PyTorch models to TorchScript format.

**Key Functions:**
- `export_model()`: Main export function with validation and benchmarking
- `load_scripted_model()`: Load exported TorchScript models
- `predict_with_scripted_model()`: Run inference with TorchScript models

**Usage:**
```python
from src.deployment.export_model import export_model

# Export model
export_info = export_model(
    model_path='models/best_model.pth',
    output_path='models/deployed_model.pt',
    model_type='resnet',
    input_shape=(1, 1, 32, 32)
)

# Verification
print(f"Speedup: {export_info['performance']['speedup']:.2f}x")
print(f"Size: {export_info['file_sizes']['scripted_mb']:.2f} MB")
print(f"Outputs match: {export_info['validation']['outputs_match']}")
```

**Export Process:**
1. ✅ Load original PyTorch model
2. ✅ Convert to TorchScript (script or trace)
3. ✅ Validate conversion (compare outputs)
4. ✅ Benchmark performance (speed comparison)
5. ✅ Save optimized model
6. ✅ Generate export metadata
7. ✅ Save validation report
8. ✅ Return comprehensive export info

**Export Methods:**
- **torch.jit.script**: More flexible, handles control flow (recommended)
- **torch.jit.trace**: Faster export, requires example input

### `api.py`
Flask REST API for serving deepfake audio detection models.

**Endpoints:**
- `GET /`: API information and available endpoints
- `GET /health`: Health check for monitoring
- `GET /info`: Model and configuration information
- `POST /predict`: Audio file prediction with confidence scores

**Features:**
- ✅ File upload handling with validation
- ✅ Supported formats: WAV, MP3, FLAC, OGG, M4A
- ✅ File size limit: 10 MB (configurable)
- ✅ Audio duration validation (min 0.5s)
- ✅ Secure filename handling
- ✅ Comprehensive error handling
- ✅ Request logging
- ✅ CORS support (optional)

**Usage:**
```bash
# Start API server
python src/deployment/api.py --model models/deployed_model.pt

# Custom configuration
python src/deployment/api.py \
    --model models/deployed_model.pt \
    --feature-type mel \
    --sample-rate 16000 \
    --device cpu \
    --host 0.0.0.0 \
    --port 5000
```

**API Request Example:**
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

**API Response Example:**
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

## Quick Start

### 1. Export a Model

```python
from src.deployment.export_model import export_model

# Export trained model to TorchScript
export_info = export_model(
    model_path='models/best_model.pth',
    output_path='models/deployed_model.pt',
    model_type='resnet',
    input_shape=(1, 1, 32, 32),
    use_trace=False,  # Use torch.jit.script
    optimize=False,   # Enable for mobile deployment
    device='cpu'
)

print(f"✓ Model exported: {export_info['scripted_model_path']}")
print(f"✓ Speedup: {export_info['performance']['speedup']:.2f}x")
print(f"✓ Size: {export_info['file_sizes']['scripted_mb']:.2f} MB")
```

### 2. Start API Server

```bash
# Basic usage
python src/deployment/api.py --model models/deployed_model.pt

# With GPU acceleration
python src/deployment/api.py --model models/deployed_model.pt --device cuda

# Custom host/port
python src/deployment/api.py --model models/deployed_model.pt --host 0.0.0.0 --port 8080
```

### 3. Test API

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Get Model Info:**
```bash
curl http://localhost:5000/info
```

**Make Prediction:**
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

**Python Client:**
```python
import requests

with open('audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
```

## Testing

### Run Deployment Test Suite

```bash
# Complete deployment testing
python test_deployment.py
```

**This will:**
1. ✅ Create a test model
2. ✅ Export to TorchScript
3. ✅ Validate export (output matching)
4. ✅ Benchmark performance (speed comparison)
5. ✅ Generate test audio files
6. ✅ Show API usage examples

**Expected Output:**
```
✅ Step 1: Creating test model...
   Model saved to: models/test_model.pth

✅ Step 2: Exporting model to TorchScript...
   Using torch.jit.script (more flexible)

✅ Step 3: Validating exported model...
   Max output difference: 0.00e+00
   Outputs match: True

✅ Export Summary:
   Original size: 42.69 MB
   Scripted size: 42.75 MB
   Speedup: 1.11x
   Outputs match: True

✅ All tests passed!
```

### Test API with Client

```bash
# Terminal 1: Start server
python src/deployment/api.py --model models/test_model_scripted.pt

# Terminal 2: Run API tests
python test_api_client.py --audio test_audio/test_audio_1.wav
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "src/deployment/api.py", "--model", "models/deployed_model.pt", "--host", "0.0.0.0"]
```

```bash
docker build -t deepfake-detector .
docker run -p 5000:5000 deepfake-detector
```

### Gunicorn (Production WSGI)

```bash
pip install gunicorn

# app.py
from src.deployment.api import create_api
api = create_api(model_path='models/deployed_model.pt')
app = api.app

# Run
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
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
ExecStart=/path/to/venv/bin/python src/deployment/api.py --model models/deployed_model.pt
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance Benchmarks

### Export Performance
- **Method**: torch.jit.script
- **Speedup**: 1.11x faster inference
- **Size**: ~0.14% larger (negligible)
- **Validation**: 100% output match

### API Performance
- **Average latency**: ~50-100ms per request
- **Throughput**: ~10-20 requests/second (single worker)
- **Memory usage**: ~500MB (model loaded)

### Optimization Tips
1. **GPU acceleration**: Use `--device cuda` for 10-50x speedup
2. **Batch processing**: Process multiple files together
3. **Model optimization**: Enable `optimize=True` for mobile/edge deployment
4. **Caching**: Cache feature extraction for repeated files
5. **Load balancing**: Use multiple workers with gunicorn

## Error Handling

### Common Errors

**1. Model Loading Failed**
```
Error: Could not load model from path
Solution: Verify model path and TorchScript format
```

**2. Invalid Audio File**
```
Error: Could not load audio file
Solution: Check file format (wav, mp3, flac, ogg, m4a)
```

**3. Audio Too Short**
```
Error: Audio must be at least 0.5 seconds
Solution: Provide longer audio samples
```

**4. File Too Large**
```
Error: File size exceeds 10 MB
Solution: Reduce file size or increase limit in api.py
```

### Debug Mode

```bash
# Enable debug logging
python src/deployment/api.py --model models/deployed_model.pt --debug
```

## Security

### Input Validation
- ✅ File type checking (whitelist)
- ✅ File size limits (10 MB default)
- ✅ Filename sanitization
- ✅ Audio duration validation
- ✅ MIME type validation

### Best Practices
1. Use HTTPS in production
2. Implement rate limiting
3. Add API key authentication
4. Enable CORS carefully
5. Monitor and log requests
6. Use firewall rules

### Rate Limiting Example

```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    ...
```

## Monitoring

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

```python
from prometheus_client import Counter, Histogram

predictions = Counter('predictions_total', 'Total predictions')
latency = Histogram('prediction_duration_seconds', 'Prediction duration')
```

## Dependencies

**Core:**
- torch >= 2.0.0
- flask >= 3.0.0
- librosa >= 0.10.0
- numpy >= 1.24.0

**Optional:**
- gunicorn (production WSGI)
- prometheus_client (metrics)
- flask-limiter (rate limiting)
- flask-cors (CORS support)

## API Reference

### DeepfakeDetectionAPI Class

```python
class DeepfakeDetectionAPI:
    """Flask API for deepfake audio detection."""
    
    def __init__(self, model_path, feature_type='mel', 
                 sample_rate=16000, device='cpu'):
        """Initialize API with model and configuration."""
        
    def predict(self):
        """POST endpoint for audio prediction."""
        
    def health(self):
        """GET endpoint for health check."""
        
    def info(self):
        """GET endpoint for model information."""
```

### Export Functions

```python
def export_model(model_path, output_path, model_type='resnet',
                 input_shape=(1, 1, 32, 32), use_trace=False,
                 optimize=False, device='cpu'):
    """Export PyTorch model to TorchScript."""
    
def load_scripted_model(model_path, device='cpu'):
    """Load TorchScript model."""
    
def predict_with_scripted_model(model, input_tensor):
    """Run inference with TorchScript model."""
```

## See Also

- [Deployment Guide](../../docs/DEPLOYMENT_GUIDE.md) - Complete deployment documentation
- [Model Architecture](../models/resnet_model.py) - ResNet model implementation
- [Feature Extraction](../features/extract.py) - Audio feature extraction
- [Test Suite](../../test_deployment.py) - Deployment testing
- [API Client](../../test_api_client.py) - API testing client

## License

See project LICENSE file.
