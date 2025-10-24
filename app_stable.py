"""
Stable Flask server for deepfake detection - runs on port 5001
"""
from flask import Flask, render_template, request, jsonify
import os
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.resnet_model import ResNetDeepfakeDetector
from src.features.extract import extract_cqcc_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Global model variables
model = None
device = None
model_loaded = False


def load_model_once():
    """Load model at startup"""
    global model, device, model_loaded
    
    try:
        device = torch.device('cpu')
        model = ResNetDeepfakeDetector()
        
        model_path = 'models/trained_model.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            model_loaded = True
            print(f"✅ Model loaded: {model_path}")
            return True
        else:
            print(f"⚠️ Model not found: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_audio(filepath):
    """Predict if audio is real or fake"""
    global model, device, model_loaded
    
    if not model_loaded:
        return {'error': 'Model not loaded', 'prediction': 'UNKNOWN'}
    
    try:
        # Extract features
        features = extract_cqcc_features(filepath, sr=16000)
        
        if features is None:
            return {'error': 'Feature extraction failed'}
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()
        
        # Add dimensions
        if features_tensor.ndim == 2:
            features_tensor = features_tensor.unsqueeze(0)  # Add channel
        if features_tensor.ndim == 3:
            features_tensor = features_tensor.unsqueeze(0)  # Add batch
        
        features_tensor = features_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get raw prediction
            raw_predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # LABEL SWAP FIX: Flip prediction
            predicted_class = 1 - raw_predicted_class
            
            # SWAP PROBABILITIES
            real_prob = probabilities[0, 0].item()  # Model's 0 = actually REAL
            fake_prob = probabilities[0, 1].item()  # Model's 1 = actually FAKE
            
            prediction = 'FAKE' if predicted_class == 0 else 'REAL'
            confidence = fake_prob if predicted_class == 0 else real_prob
            
            print(f"\n🔍 Prediction: {prediction} ({confidence:.2%})")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'real_prob': real_prob,
            'fake_prob': fake_prob
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


@app.route('/')
def index():
    """Main page"""
    return render_template('index_real.html', model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"\n📁 Processing: {filename}")
        
        # Run prediction
        result = predict_audio(filepath)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 500
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'probabilities': {
                'real': round(result['real_prob'], 4),
                'fake': round(result['fake_prob'], 4)
            },
            'model_loaded': model_loaded
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "DEEPFAKE AUDIO DETECTION SERVER")
    print("="*70)
    
    # Load model at startup
    load_model_once()
    
    print("\n🌐 Server: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Use waitress production server (more stable than Flask dev server)
    try:
        from waitress import serve
        print("✅ Starting waitress server...")
        print("DEBUG: About to call serve()...")
        serve(app, host='0.0.0.0', port=5001, threads=4)
        print("DEBUG: serve() returned - THIS SHOULD NOT HAPPEN!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("DEBUG: In finally block")
        import time
        time.sleep(5)  # Keep alive for 5 seconds to see output
