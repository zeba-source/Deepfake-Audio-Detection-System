"""
Test single file prediction to debug the issue
"""

import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.resnet_model import ResNetDeepfakeDetector
from src.features.extract import extract_cqcc_features

def test_prediction(audio_file, model_path='models/trained_model.pth'):
    """Test prediction on a single file"""
    
    print(f"\n{'='*70}")
    print(f"  🧪 TESTING PREDICTION")
    print(f"{'='*70}")
    
    # Load model
    device = torch.device('cpu')
    model = ResNetDeepfakeDetector()
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"\n📁 Testing file: {audio_file}")
    
    # Extract features
    features = extract_cqcc_features(audio_file, sr=16000)
    
    # Convert to tensor
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    
    # Add dimensions
    if features.ndim == 2:
        features = features.unsqueeze(0)  # Add channel
    if features.ndim == 3:
        features = features.unsqueeze(0)  # Add batch
    
    print(f"📊 Feature shape: {features.shape}")
    
    # Get prediction
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        
        fake_prob = probabilities[0, 0].item()
        real_prob = probabilities[0, 1].item()
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        print(f"\n🎯 RAW MODEL OUTPUT:")
        print(f"   Logits: {outputs[0].cpu().numpy()}")
        print(f"   Softmax: {probabilities[0].cpu().numpy()}")
        
        print(f"\n📊 PROBABILITIES:")
        print(f"   Class 0 (FAKE): {fake_prob:.4f} ({fake_prob*100:.2f}%)")
        print(f"   Class 1 (REAL): {real_prob:.4f} ({real_prob*100:.2f}%)")
        
        print(f"\n✅ PREDICTION:")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Label: {'FAKE' if predicted_class == 0 else 'REAL'}")
        
        # Determine expected label from file path
        file_str = str(audio_file).lower()
        if 'fake' in file_str or 'ai_' in file_str or 'ai_generated' in file_str:
            expected = 'FAKE'
            expected_class = 0
        elif 'real' in file_str or 'human_' in file_str or 'human_voice' in file_str:
            expected = 'REAL'
            expected_class = 1
        else:
            expected = 'UNKNOWN'
            expected_class = None
        
        print(f"\n🎯 EXPECTED:")
        print(f"   Based on filename: {expected}")
        
        if expected_class is not None:
            if predicted_class == expected_class:
                print(f"\n✅ CORRECT PREDICTION!")
            else:
                print(f"\n❌ INCORRECT PREDICTION!")
                print(f"   Expected class {expected_class} ({expected})")
                print(f"   Got class {predicted_class} ({'FAKE' if predicted_class == 0 else 'REAL'})")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    # Test all 4 sample files
    test_files = [
        'test_samples_download/human_voice_sample_1.wav',
        'test_samples_download/human_voice_sample_2.wav',
        'test_samples_download/ai_generated_sample_1.wav',
        'test_samples_download/ai_generated_sample_2.wav',
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            test_prediction(test_file)
        else:
            print(f"❌ File not found: {test_file}")
