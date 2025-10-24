"""
Test the model on actual data from the training folders
"""

import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.resnet_model import ResNetDeepfakeDetector
from src.features.extract import extract_cqcc_features

def test_on_real_data():
    """Test model on actual files from data folders"""
    
    print("\n" + "="*70)
    print("  🧪 TESTING MODEL ON ACTUAL DATASET FILES")
    print("="*70)
    
    # Load model
    device = torch.device('cpu')
    model = ResNetDeepfakeDetector()
    
    checkpoint = torch.load('models/trained_model.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("\n✅ Model loaded successfully")
    
    # Test files from data/real (should predict REAL - class 1)
    real_files = list(Path('data/real').glob('*.wav'))[:5]
    
    # Test files from data/fake (should predict FAKE - class 0)
    fake_files = list(Path('data/fake').glob('*.wav'))[:5]
    
    print(f"\n📊 Testing on data/real/ files (Expected: REAL/class 1):")
    print("="*70)
    
    real_correct = 0
    for audio_file in real_files:
        features = extract_cqcc_features(str(audio_file), sr=16000)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            is_correct = (predicted_class == 1)
            real_correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {audio_file.name}: Predicted={predicted_class} ({'REAL' if predicted_class == 1 else 'FAKE'}), Prob={probabilities[0, predicted_class].item():.4f}")
    
    print(f"\n✅ Real files accuracy: {real_correct}/{len(real_files)} = {real_correct/len(real_files)*100:.1f}%")
    
    print(f"\n📊 Testing on data/fake/ files (Expected: FAKE/class 0):")
    print("="*70)
    
    fake_correct = 0
    for audio_file in fake_files:
        features = extract_cqcc_features(str(audio_file), sr=16000)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            is_correct = (predicted_class == 0)
            fake_correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {audio_file.name}: Predicted={predicted_class} ({'FAKE' if predicted_class == 0 else 'REAL'}), Prob={probabilities[0, predicted_class].item():.4f}")
    
    print(f"\n✅ Fake files accuracy: {fake_correct}/{len(fake_files)} = {fake_correct/len(fake_files)*100:.1f}%")
    
    total_correct = real_correct + fake_correct
    total_files = len(real_files) + len(fake_files)
    
    print(f"\n{'='*70}")
    print(f"  📊 OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Total Accuracy: {total_correct}/{total_files} = {total_correct/total_files*100:.1f}%")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    test_on_real_data()
