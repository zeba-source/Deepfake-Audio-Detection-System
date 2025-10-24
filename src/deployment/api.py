"""Flask API for deepfake audio detection inference."""
import os
import tempfile
import logging
from typing import Dict, Any
from pathlib import Path

import torch
import numpy as np
import librosa
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from src.features.extract import extract_mel_spectrogram, extract_cqcc_features
from src.deployment.export_model import load_scripted_model, predict_with_scripted_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepfakeDetectionAPI:
    """Flask API for deepfake audio detection."""
    
    def __init__(
        self,
        model_path: str,
        feature_type: str = 'mel',
        sample_rate: int = 16000,
        device: str = 'cpu',
        max_file_size_mb: int = 10,
        allowed_extensions: set = None
    ):
        """Initialize the API.
        
        Args:
            model_path: Path to TorchScript model (.pt file)
            feature_type: Type of features ('mel' or 'cqcc')
            sample_rate: Audio sample rate
            device: Device for inference ('cpu' or 'cuda')
            max_file_size_mb: Maximum upload file size in MB
            allowed_extensions: Set of allowed file extensions
        """
        self.model_path = model_path
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.device = device
        self.max_file_size_mb = max_file_size_mb
        
        if allowed_extensions is None:
            self.allowed_extensions = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
        else:
            self.allowed_extensions = allowed_extensions
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = load_scripted_model(model_path, device=device)
        logger.info(f"Model loaded successfully on {device}")
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = max_file_size_mb * 1024 * 1024
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes."""
        self.app.route('/')(self.index)
        self.app.route('/health', methods=['GET'])(self.health)
        self.app.route('/predict', methods=['POST'])(self.predict)
        self.app.route('/info', methods=['GET'])(self.info)
    
    def index(self):
        """Root endpoint."""
        return jsonify({
            'service': 'Deepfake Audio Detection API',
            'version': '1.0.0',
            'endpoints': {
                '/health': 'Health check (GET)',
                '/predict': 'Audio prediction (POST)',
                '/info': 'Model information (GET)'
            }
        })
    
    def health(self):
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'device': str(self.device)
        })
    
    def info(self):
        """Model information endpoint."""
        return jsonify({
            'model_path': self.model_path,
            'feature_type': self.feature_type,
            'sample_rate': self.sample_rate,
            'device': str(self.device),
            'max_file_size_mb': self.max_file_size_mb,
            'allowed_extensions': list(self.allowed_extensions)
        })
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed.
        
        Args:
            filename: Name of uploaded file
            
        Returns:
            True if extension is allowed
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature array
        """
        if self.feature_type == 'mel':
            features = extract_mel_spectrogram(
                audio_path,
                sr=self.sample_rate,
                n_mels=128
            )
        elif self.feature_type == 'cqcc':
            features = extract_cqcc_features(
                audio_path,
                sr=self.sample_rate
            )
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return features
    
    def _preprocess_features(self, features: np.ndarray) -> torch.Tensor:
        """Preprocess features for model input.
        
        Args:
            features: Feature array (height, width)
            
        Returns:
            Preprocessed tensor (1, 1, height, width)
        """
        # Add batch and channel dimensions
        features = features[np.newaxis, np.newaxis, :, :]
        
        # Convert to tensor
        tensor = torch.from_numpy(features).float()
        
        return tensor
    
    def predict(self):
        """Prediction endpoint.
        
        Accepts audio file upload and returns prediction.
        """
        try:
            # Check if file is present
            if 'file' not in request.files:
                logger.warning("No file provided in request")
                return jsonify({
                    'error': 'No file provided',
                    'message': 'Please upload an audio file using the "file" field'
                }), 400
            
            file = request.files['file']
            
            # Check if filename is empty
            if file.filename == '':
                logger.warning("Empty filename")
                return jsonify({
                    'error': 'No file selected',
                    'message': 'Please select a file to upload'
                }), 400
            
            # Check file extension
            if not self._allowed_file(file.filename):
                logger.warning(f"Invalid file extension: {file.filename}")
                return jsonify({
                    'error': 'Invalid file type',
                    'message': f'Allowed types: {", ".join(self.allowed_extensions)}',
                    'received': file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'none'
                }), 400
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, filename)
                file.save(file_path)
                
                logger.info(f"Processing file: {filename}")
                
                # Validate audio file
                try:
                    audio, sr = librosa.load(file_path, sr=None, duration=10.0)
                    duration = len(audio) / sr
                    
                    if duration < 0.5:
                        return jsonify({
                            'error': 'Audio too short',
                            'message': f'Audio must be at least 0.5 seconds (got {duration:.2f}s)'
                        }), 400
                    
                    logger.info(f"Audio duration: {duration:.2f}s, sample rate: {sr}Hz")
                    
                except Exception as e:
                    logger.error(f"Error loading audio: {e}")
                    return jsonify({
                        'error': 'Invalid audio file',
                        'message': 'Could not load audio file. Please ensure it is a valid audio file.'
                    }), 400
                
                # Extract features
                try:
                    features = self._extract_features(file_path)
                    logger.info(f"Features extracted: shape={features.shape}")
                except Exception as e:
                    logger.error(f"Error extracting features: {e}")
                    return jsonify({
                        'error': 'Feature extraction failed',
                        'message': str(e)
                    }), 500
                
                # Preprocess features
                input_tensor = self._preprocess_features(features)
                
                # Run inference
                try:
                    predicted_class, confidence, probabilities = predict_with_scripted_model(
                        self.model,
                        input_tensor,
                        device=self.device
                    )
                    
                    result = {
                        'prediction': 'fake' if predicted_class == 1 else 'real',
                        'confidence': float(confidence),
                        'probabilities': {
                            'real': float(probabilities[0]),
                            'fake': float(probabilities[1])
                        },
                        'metadata': {
                            'filename': filename,
                            'duration_seconds': float(duration),
                            'sample_rate': int(sr),
                            'feature_type': self.feature_type,
                            'feature_shape': list(features.shape)
                        }
                    }
                    
                    logger.info(f"Prediction: {result['prediction']} ({confidence*100:.2f}%)")
                    
                    return jsonify(result), 200
                    
                except Exception as e:
                    logger.error(f"Error during inference: {e}")
                    return jsonify({
                        'error': 'Inference failed',
                        'message': str(e)
                    }), 500
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """Run the Flask application.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_api(
    model_path: str,
    feature_type: str = 'mel',
    sample_rate: int = 16000,
    device: str = 'cpu'
) -> DeepfakeDetectionAPI:
    """Create and configure the API.
    
    Args:
        model_path: Path to TorchScript model
        feature_type: Feature extraction type
        sample_rate: Audio sample rate
        device: Inference device
        
    Returns:
        Configured API instance
    """
    return DeepfakeDetectionAPI(
        model_path=model_path,
        feature_type=feature_type,
        sample_rate=sample_rate,
        device=device
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Audio Detection API')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TorchScript model (.pt file)')
    parser.add_argument('--feature-type', type=str, default='mel',
                        choices=['mel', 'cqcc'],
                        help='Feature extraction type')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Inference device')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port number')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and run API
    api = create_api(
        model_path=args.model,
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        device=args.device
    )
    
    api.run(host=args.host, port=args.port, debug=args.debug)
