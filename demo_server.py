"""
Demo Flask Web Interface for Deepfake Audio Detection

This is a demo server that shows the web interface without requiring a trained model.
Perfect for testing the UI and understanding the workflow.
"""

from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and return prediction (demo mode)."""
    
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
        
        # DEMO MODE: Return simulated prediction
        # In production, this would run the actual model
        import random
        is_fake = random.choice([True, False])
        confidence = random.uniform(0.7, 0.99)
        
        result = {
            'success': True,
            'filename': filename,
            'prediction': 'FAKE' if is_fake else 'REAL',
            'confidence': round(confidence, 4),
            'probabilities': {
                'real': round(1 - confidence if is_fake else confidence, 4),
                'fake': round(confidence if is_fake else 1 - confidence, 4)
            },
            'file_size_bytes': file_size,
            'processing_time_ms': round(random.uniform(40, 80), 2),
            'timestamp': datetime.now().isoformat(),
            'demo_mode': True,
            'message': 'This is a DEMO prediction. Train a model to get real results.'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'mode': 'DEMO',
        'timestamp': datetime.now().isoformat(),
        'message': 'Demo server is running. Upload audio to see the interface.'
    })


@app.route('/info')
def info():
    """API information endpoint."""
    return jsonify({
        'name': 'Deepfake Audio Detection API',
        'version': '1.0.0 (DEMO)',
        'mode': 'DEMO',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 16,
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'Upload audio for prediction (POST)',
            '/health': 'Health check',
            '/info': 'API information'
        },
        'note': 'This is a demo server. Train a model using main_train.py for real predictions.'
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" " * 15 + "DEEPFAKE AUDIO DETECTION - DEMO SERVER")
    print("="*70)
    print("\n📌 Server starting on: http://localhost:5000")
    print("\n🎵 Upload audio files to test the interface")
    print("⚠️  DEMO MODE: Predictions are randomly generated")
    print("\n💡 To get real predictions:")
    print("   1. Train a model: python main_train.py --data_folder ./data")
    print("   2. Export model: (done automatically with --export_model flag)")
    print("   3. Run real API: python src/deployment/api.py --model outputs/exp/deployed_model.pt")
    print("\n" + "="*70)
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
