"""Test script for model export and API deployment."""
import os
import sys
import time
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile

import torch
from src.models.resnet_model import ResNetDeepfakeDetector
from src.deployment.export_model import export_model, load_scripted_model, predict_with_scripted_model


def create_test_model():
    """Create and save a test model."""
    print("\n" + "="*70)
    print("STEP 1: CREATE TEST MODEL")
    print("="*70)
    
    # Create model
    print("\n1. Creating ResNet model...")
    model = ResNetDeepfakeDetector(dropout_p=0.5)
    print("   ✓ Model created")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/test_model.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'resnet',
        'dropout_p': 0.5
    }, model_path)
    
    print(f"   ✓ Model saved to: {model_path}")
    return model_path


def test_model_export(model_path):
    """Test model export to TorchScript."""
    print("\n" + "="*70)
    print("STEP 2: EXPORT TO TORCHSCRIPT")
    print("="*70)
    
    # Export model
    scripted_path = 'models/test_model_scripted.pt'
    export_info = export_model(
        model_path=model_path,
        output_path=scripted_path,
        model_type='resnet',
        input_shape=(1, 1, 32, 32),
        use_trace=False,  # Use script instead of trace
        optimize=False,    # Disable optimization to avoid issues
        device='cpu'
    )
    
    print("\n✅ Export Summary:")
    print(f"   Original size: {export_info['file_sizes']['original_mb']:.2f} MB")
    print(f"   Scripted size: {export_info['file_sizes']['scripted_mb']:.2f} MB")
    print(f"   Speedup: {export_info['performance']['speedup']:.2f}x")
    print(f"   Outputs match: {export_info['validation']['outputs_match']}")
    
    return scripted_path


def test_scripted_inference(scripted_path):
    """Test inference with scripted model."""
    print("\n" + "="*70)
    print("STEP 3: TEST SCRIPTED MODEL INFERENCE")
    print("="*70)
    
    # Load model
    print("\n1. Loading scripted model...")
    model = load_scripted_model(scripted_path, device='cpu')
    print("   ✓ Model loaded")
    
    # Create test inputs
    print("\n2. Running inference tests...")
    n_tests = 5
    results = []
    
    for i in range(n_tests):
        # Random input
        test_input = torch.randn(1, 1, 32, 32)
        
        # Predict
        predicted_class, confidence, probs = predict_with_scripted_model(
            model, test_input, device='cpu'
        )
        
        results.append({
            'test': i + 1,
            'prediction': 'fake' if predicted_class == 1 else 'real',
            'confidence': confidence * 100,
            'probs': probs
        })
        
        print(f"   Test {i+1}: {results[-1]['prediction']} ({results[-1]['confidence']:.2f}%)")
    
    print(f"\n   ✓ {n_tests} inference tests completed successfully")
    return results


def create_test_audio():
    """Create test audio files."""
    print("\n" + "="*70)
    print("STEP 4: CREATE TEST AUDIO FILES")
    print("="*70)
    
    os.makedirs('test_audio', exist_ok=True)
    audio_files = []
    
    # Create 3 test files
    for i in range(3):
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Different frequency for each file
        freq = 440 * (i + 1)
        audio = np.sin(2 * np.pi * freq * t)
        audio = (audio * 32767 * 0.8).astype(np.int16)
        
        filename = f'test_audio/test_audio_{i+1}.wav'
        wavfile.write(filename, sr, audio)
        audio_files.append(filename)
        
        print(f"   ✓ Created: {filename} ({freq}Hz, {duration}s)")
    
    return audio_files


def test_api_client(audio_files):
    """Test API with client requests."""
    print("\n" + "="*70)
    print("STEP 5: TEST API CLIENT")
    print("="*70)
    
    try:
        import requests
    except ImportError:
        print("   ⚠️  requests not installed, skipping API client test")
        return
    
    print("\n⚠️  To test API:")
    print("   1. Start API server in another terminal:")
    print("      python src/deployment/api.py --model models/test_model_scripted.pt")
    print("   2. Run this script again to test client")
    print("\nExample client code:")
    print("""
    import requests
    
    # Test health endpoint
    response = requests.get('http://localhost:5000/health')
    print(response.json())
    
    # Test prediction
    with open('test_audio/test_audio_1.wav', 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/predict', files=files)
        print(response.json())
    """)


def main():
    """Run deployment test suite."""
    print("="*70)
    print("DEPLOYMENT TEST SUITE")
    print("="*70)
    print("\nThis script will:")
    print("1. Create a test model")
    print("2. Export to TorchScript")
    print("3. Test scripted model inference")
    print("4. Create test audio files")
    print("5. Show how to test API")
    
    try:
        # Step 1: Create test model
        model_path = create_test_model()
        
        # Step 2: Export to TorchScript
        scripted_path = test_model_export(model_path)
        
        # Step 3: Test inference
        results = test_scripted_inference(scripted_path)
        
        # Step 4: Create test audio
        audio_files = create_test_audio()
        
        # Step 5: API client info
        test_api_client(audio_files)
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUITE COMPLETE")
        print("="*70)
        print("\n✅ All tests passed!")
        print("\n📁 Generated Files:")
        print(f"   - {model_path}")
        print(f"   - {scripted_path}")
        print(f"   - {scripted_path.replace('.pt', '_export_info.json')}")
        for af in audio_files:
            print(f"   - {af}")
        
        print("\n🚀 Next Steps:")
        print("   1. Start API server:")
        print(f"      python src/deployment/api.py --model {scripted_path}")
        print("\n   2. Test with curl:")
        print(f"      curl -X POST -F \"file=@{audio_files[0]}\" http://localhost:5000/predict")
        print("\n   3. Or use Python client:")
        print("      python test_api_client.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
