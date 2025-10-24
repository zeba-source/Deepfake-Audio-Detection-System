"""
Quick test script to verify model predictions
Tests the model on generated dataset files
"""

import torch
import numpy as np
import librosa
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.models.resnet_model import ResNetDeepfakeDetector

def extract_cqcc_features(audio_path, sr=16000):
    """Extract CQCC features from audio file"""
    try:
        audio, _ = librosa.load(audio_path, sr=sr, duration=5.0)
        cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=512, n_bins=84))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        target_frames = 256
        if cqt_db.shape[1] < target_frames:
            pad_width = target_frames - cqt_db.shape[1]
            cqt_db = np.pad(cqt_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            cqt_db = cqt_db[:, :target_frames]
        
        return cqt_db.T
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_model():
    print("\n" + "="*70)
    print("  🧪 TESTING MODEL PREDICTIONS")
    print("="*70)
    
    # Load model
    print("\n📥 Loading model...")
    model = ResNetDeepfakeDetector()
    checkpoint = torch.load('models/trained_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Test files
    test_files = [
        ('data/real/human_001.wav', 'REAL/HUMAN', 1),
        ('data/real/human_002.wav', 'REAL/HUMAN', 1),
        ('data/real/human_003.wav', 'REAL/HUMAN', 1),
        ('data/fake/ai_001.wav', 'FAKE/AI', 0),
        ('data/fake/ai_002.wav', 'FAKE/AI', 0),
        ('data/fake/ai_003.wav', 'FAKE/AI', 0),
    ]
    
    print("\n" + "="*70)
    print("  📊 PREDICTIONS")
    print("="*70)
    
    correct = 0
    total = 0
    
    for filepath, label, expected_class in test_files:
        if not Path(filepath).exists():
            print(f"\n❌ File not found: {filepath}")
            continue
        
        # Extract features
        features = extract_cqcc_features(filepath)
        if features is None:
            continue
        
        # Prepare input
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Results
        fake_prob = probabilities[0].item()  # Class 0 = FAKE
        real_prob = probabilities[1].item()  # Class 1 = REAL
        
        is_correct = (predicted_class == expected_class)
        correct += is_correct
        total += 1
        
        status = "✅" if is_correct else "❌"
        
        print(f"\n{status} File: {Path(filepath).name}")
        print(f"   Expected: {label}")
        print(f"   Predicted: {'FAKE/AI' if predicted_class == 0 else 'REAL/HUMAN'}")
        print(f"   Confidence: {max(fake_prob, real_prob)*100:.1f}%")
        print(f"   Probabilities: FAKE={fake_prob*100:.1f}% | REAL={real_prob*100:.1f}%")
    
    print("\n" + "="*70)
    print(f"  ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")
    print("="*70)
    print()

if __name__ == '__main__':
    test_model()
