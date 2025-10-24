"""
Real-time Deepfake Audio Detection Web Interface

This server uses a trained model to actually detect if audio is human-generated or AI-generated.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
import torch
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.resnet_model import ResNetDeepfakeDetector
from src.features.extract import extract_cqcc_features, extract_mel_spectrogram

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Global model variables
model = None
device = None
feature_type = 'cqcc'
model_loaded = False


def load_model(model_path='models/trained_model.pth', device_name='cpu'):
    """Load the trained model."""
    global model, device, model_loaded
    
    try:
        device = torch.device(device_name)
        print(f"\n📥 Loading model from: {model_path}")
        print(f"🔧 Device: {device}")
        
        # Create model (ResNetDeepfakeDetector uses default 2 classes)
        model = ResNetDeepfakeDetector()
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            model_loaded = True
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model loaded successfully!")
            print(f"📊 Parameters: {total_params:,}")
            return True
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print(f"ℹ️  Server will run in DEMO mode")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"ℹ️  Server will run in DEMO mode")
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features_from_audio(filepath, feature_type='cqcc'):
    """Extract features from audio file."""
    try:
        if feature_type == 'cqcc':
            features = extract_cqcc_features(filepath, sr=16000)
        else:  # mel
            features = extract_mel_spectrogram(filepath, sr=16000, n_mels=128)
        
        # Convert to tensor and add batch dimension
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Add channel and batch dimensions if needed
        if features.ndim == 2:
            features = features.unsqueeze(0)  # Add channel dimension
        if features.ndim == 3:
            features = features.unsqueeze(0)  # Add batch dimension
        
        return features
        
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")


def predict_audio(filepath):
    """Predict if audio is real or fake."""
    global model, device, model_loaded, feature_type
    
    if not model_loaded or model is None:
        # Demo mode: random prediction
        import random
        is_fake = random.choice([True, False])
        confidence = random.uniform(0.7, 0.99)
        
        return {
            'prediction': 'FAKE' if is_fake else 'REAL',
            'confidence': confidence,
            'real_prob': 1 - confidence if is_fake else confidence,
            'fake_prob': confidence if is_fake else 1 - confidence,
            'demo_mode': True
        }
    
    # Real prediction with model
    try:
        start_time = time.time()
        
        # Extract features
        features = extract_features_from_audio(filepath, feature_type)
        features = features.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction (NO LABEL SWAP - test if model is actually correct)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Standard interpretation: Class 0 = FAKE, Class 1 = REAL
            fake_prob = probabilities[0, 0].item()
            real_prob = probabilities[0, 1].item()
            
            # Determine prediction
            prediction = 'FAKE' if predicted_class == 0 else 'REAL'
            confidence = fake_prob if predicted_class == 0 else real_prob
            
            # Debug logging
            print(f"\n{'='*60}")
            print(f"🔍 DEBUG - PREDICTION (NO LABEL SWAP):")
            print(f"   Raw logits: {outputs[0].cpu().numpy()}")
            print(f"   Predicted class: {predicted_class}")
            print(f"   Probabilities: FAKE={fake_prob:.4f}, REAL={real_prob:.4f}")
            print(f"   Final Result: {prediction} ({confidence:.2%})")
            print(f"{'='*60}\n")
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'real_prob': real_prob,
            'fake_prob': fake_prob,
            'processing_time_ms': processing_time,
            'demo_mode': False
        }
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index_real.html', model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and return prediction."""
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Run prediction
        prediction_result = predict_audio(filepath)
        
        # Prepare result
        result = {
            'success': True,
            'filename': filename,
            'prediction': prediction_result['prediction'],
            'confidence': round(prediction_result['confidence'], 4),
            'probabilities': {
                'real': round(prediction_result['real_prob'], 4),
                'fake': round(prediction_result['fake_prob'], 4)
            },
            'file_size_bytes': file_size,
            'processing_time_ms': round(prediction_result.get('processing_time_ms', 50), 2),
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model_loaded,
            'demo_mode': prediction_result.get('demo_mode', False)
        }
        
        if prediction_result.get('demo_mode'):
            result['message'] = 'DEMO MODE: Random prediction. Train a model for real results.'
        else:
            result['message'] = 'Real prediction from trained model'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'mode': 'DEMO' if not model_loaded else 'PRODUCTION',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/info')
def info():
    """API information endpoint."""
    return jsonify({
        'name': 'Deepfake Audio Detection API',
        'version': '1.0.0',
        'model_loaded': model_loaded,
        'mode': 'DEMO' if not model_loaded else 'PRODUCTION',
        'feature_type': feature_type,
        'device': str(device),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 16,
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'Upload audio for prediction (POST)',
            '/health': 'Health check',
            '/info': 'API information'
        }
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" " * 15 + "DEEPFAKE AUDIO DETECTION - REAL-TIME SERVER")
    print("="*70)
    
    # Try to load model - ✅ Use trained model!
    model_path = 'models/trained_model.pth'
    
    # Check for other model files if trained model doesn't exist
    if not os.path.exists(model_path):
        print(f"⚠️  Trained model not found at {model_path}, looking for alternatives...")
        # Try test model as fallback
        if os.path.exists('models/test_model.pth'):
            model_path = 'models/test_model.pth'
        else:
            # Look for any .pth files
            pth_files = list(Path('models').glob('*.pth'))
            if pth_files:
                model_path = str(pth_files[0])
            else:
                # Look in outputs directory
                output_models = list(Path('outputs').rglob('best_model.pth')) if os.path.exists('outputs') else []
                if output_models:
                    model_path = str(output_models[0])
    
    loaded = load_model(model_path, 'cpu')
    
    print("\n📌 Server starting on: http://localhost:5001")
    
    if loaded:
        print("\n✅ MODEL LOADED - Real predictions enabled!")
        print(f"📁 Model: {model_path}")
        print(f"🎯 Feature type: {feature_type}")
    else:
        print("\n⚠️  NO MODEL - Running in DEMO mode")
        print("\n💡 To enable real predictions:")
        print("   1. Train a model: python main_train.py --data_folder ./data --epochs 30")
        print("   2. Model will be saved to: outputs/{experiment}/best_model.pth")
        print("   3. Copy to: models/test_model.pth")
    
    print("\n" + "="*70)
    print("\n🌐 Open your browser and go to: http://localhost:5001")
    print("="*70 + "\n")
    
    # Force Flask to stay alive using a workaround
    import threading
    import time
    
    def run_app():
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)
    
    try:
        print("🟢 Server is starting and will keep running...")
        print("🛑 Press Ctrl+C in this terminal to stop\n")
        
        # Start Flask in a thread
        server_thread = threading.Thread(target=run_app)
        server_thread.daemon = False
        server_thread.start()
        
        # Keep main thread alive
        while server_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Goodbye!\n")
