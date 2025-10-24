"""
Test the label swap fix by simulating what the server does
"""

import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.resnet_model import ResNetDeepfakeDetector
from src.features.extract import extract_cqcc_features

def test_with_label_swap(audio_file, model_path='models/trained_model.pth'):
    """Test prediction with label swap fix"""
    
    print(f"\n{'='*70}")
    print(f"  🧪 TESTING WITH LABEL SWAP FIX")
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
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim == 3:
        features = features.unsqueeze(0)
    
    # Get prediction WITH LABEL SWAP
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Raw prediction from model
        raw_predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # ✅✅✅ APPLY LABEL SWAP FIX
        predicted_class = 1 - raw_predicted_class  # Flip: 0→1, 1→0
        
        # ✅✅✅ SWAP PROBABILITIES
        real_prob = probabilities[0, 0].item()  # Model's 0 = actually REAL
        fake_prob = probabilities[0, 1].item()  # Model's 1 = actually FAKE
        
        prediction = 'FAKE' if predicted_class == 0 else 'REAL'
        confidence = fake_prob if predicted_class == 0 else real_prob
        
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Raw logits: {outputs[0].cpu().numpy()}")
        print(f"   Raw softmax: {probabilities[0].cpu().numpy()}")
        print(f"   Raw prediction: {raw_predicted_class}")
        print(f"   ✅ Flipped prediction: {predicted_class}")
        print(f"   ✅ Probabilities: REAL={real_prob:.4f}, FAKE={fake_prob:.4f}")
        
        print(f"\n✅ FINAL RESULT:")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.2%}")
        
        # Determine expected label
        file_str = str(audio_file).lower()
        if 'fake' in file_str or 'ai_' in file_str or 'ai_generated' in file_str:
            expected = 'FAKE'
        elif 'real' in file_str or 'human_' in file_str or 'human_voice' in file_str:
            expected = 'REAL'
        else:
            expected = 'UNKNOWN'
        
        print(f"\n🎯 EXPECTED: {expected}")
        
        if expected != 'UNKNOWN':
            if prediction == expected:
                print(f"✅ ✅ ✅ CORRECT! Prediction matches expected label!")
            else:
                print(f"❌ ❌ ❌ INCORRECT! Expected {expected}, got {prediction}")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    # Test all 4 sample files
    test_files = [
        'test_samples_download/human_voice_sample_1.wav',
        'test_samples_download/human_voice_sample_2.wav',
        'test_samples_download/ai_generated_sample_1.wav',
        'test_samples_download/ai_generated_sample_2.wav',
    ]
    
    print("\n" + "="*70)
    print("  📊 TESTING LABEL SWAP FIX ON ALL SAMPLES")
    print("="*70)
    
    for test_file in test_files:
        if Path(test_file).exists():
            test_with_label_swap(test_file)
        else:
            print(f"❌ File not found: {test_file}")
    
    print("\n" + "="*70)
    print("  🎯 TEST COMPLETE!")
    print("="*70 + "\n")
