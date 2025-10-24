"""
Quick Demo Dataset Generator
Creates synthetic audio files for testing the deepfake detection system
NO NEED TO COLLECT REAL DATA - This generates everything automatically!
"""

import numpy as np
import os
from scipy.io import wavfile
import json

def generate_human_like_audio(duration=3, sample_rate=16000):
    """Generate audio that simulates human voice characteristics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base frequency (human voice range: 85-255 Hz)
    base_freq = np.random.uniform(120, 200)
    
    # Create harmonic structure (mimics human vocal cords)
    audio = np.zeros_like(t)
    for harmonic in range(1, 8):
        freq = base_freq * harmonic
        amplitude = 1.0 / harmonic  # Natural harmonic decay
        # Add natural vibrato
        vibrato = 5 * np.sin(2 * np.pi * 5 * t)
        audio += amplitude * np.sin(2 * np.pi * (freq + vibrato) * t)
    
    # Add natural speech-like amplitude modulation
    envelope = np.random.uniform(0.5, 1.0, 100)
    envelope = np.interp(t, np.linspace(0, duration, 100), envelope)
    audio *= envelope
    
    # Add slight background noise (realistic recordings have this)
    noise = np.random.normal(0, 0.02, len(audio))
    audio += noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return (audio * 32767).astype(np.int16)

def generate_ai_like_audio(duration=3, sample_rate=16000):
    """Generate audio that simulates AI-generated voice characteristics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # AI voices often have more precise, uniform characteristics
    base_freq = np.random.uniform(130, 190)
    
    # Create TOO perfect harmonic structure (unrealistic for humans)
    audio = np.zeros_like(t)
    for harmonic in range(1, 12):  # More harmonics (overly synthetic)
        freq = base_freq * harmonic
        amplitude = 1.0 / (harmonic * 1.2)
        # AI voices have less natural vibrato variation
        vibrato = 2 * np.sin(2 * np.pi * 6.5 * t)  # More mechanical
        audio += amplitude * np.sin(2 * np.pi * (freq + vibrato) * t)
    
    # AI voices often have more uniform amplitude
    envelope = np.ones(100) * 0.9  # Too consistent
    envelope = np.interp(t, np.linspace(0, duration, 100), envelope)
    audio *= envelope
    
    # Less background noise (too clean)
    noise = np.random.normal(0, 0.005, len(audio))
    audio += noise
    
    # Add slight digital artifacts (compression-like)
    quantization_noise = np.random.choice([-0.01, 0, 0.01], len(audio))
    audio += quantization_noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return (audio * 32767).astype(np.int16)

def create_demo_dataset(num_files_per_class=50):
    """Create a complete demo dataset with real and fake audio"""
    
    print("=" * 70)
    print("  🎵 CREATING DEMO DATASET - NO DATA COLLECTION NEEDED!")
    print("=" * 70)
    print()
    
    # Create directories
    os.makedirs('data/real', exist_ok=True)
    os.makedirs('data/fake', exist_ok=True)
    
    sample_rate = 16000
    
    # Generate REAL (human-like) audio files
    print(f"📝 Generating {num_files_per_class} HUMAN voice samples...")
    for i in range(num_files_per_class):
        duration = np.random.uniform(2, 4)  # Vary duration
        audio = generate_human_like_audio(duration, sample_rate)
        filename = f'data/real/human_{i+1:03d}.wav'
        wavfile.write(filename, sample_rate, audio)
        
        if (i + 1) % 10 == 0:
            print(f"   ✅ Created {i+1}/{num_files_per_class} human samples")
    
    print(f"✅ Created {num_files_per_class} HUMAN voice samples")
    print()
    
    # Generate FAKE (AI-like) audio files
    print(f"📝 Generating {num_files_per_class} AI voice samples...")
    for i in range(num_files_per_class):
        duration = np.random.uniform(2, 4)
        audio = generate_ai_like_audio(duration, sample_rate)
        filename = f'data/fake/ai_{i+1:03d}.wav'
        wavfile.write(filename, sample_rate, audio)
        
        if (i + 1) % 10 == 0:
            print(f"   ✅ Created {i+1}/{num_files_per_class} AI samples")
    
    print(f"✅ Created {num_files_per_class} AI voice samples")
    print()
    
    # Save metadata
    metadata = {
        'created': '2025-10-21',
        'total_files': num_files_per_class * 2,
        'real_files': num_files_per_class,
        'fake_files': num_files_per_class,
        'sample_rate': sample_rate,
        'description': 'Synthetic demo dataset for deepfake detection testing',
        'characteristics': {
            'real': 'Natural harmonics, vibrato, background noise, amplitude variation',
            'fake': 'Perfect harmonics, mechanical vibrato, clean signal, uniform amplitude'
        }
    }
    
    with open('data/dataset_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 70)
    print("  ✅ DATASET CREATED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print(f"📁 Location: data/")
    print(f"   - Real audio: data/real/ ({num_files_per_class} files)")
    print(f"   - Fake audio: data/fake/ ({num_files_per_class} files)")
    print(f"   - Metadata: data/dataset_info.json")
    print()
    print("🚀 NEXT STEPS:")
    print()
    print("1️⃣  Train the model:")
    print("   python main_train.py --data_folder ./data --epochs 20 --batch_size 16")
    print()
    print("2️⃣  After training completes, copy the model:")
    print("   copy outputs\\[experiment_name]\\best_model.pth models\\trained_model.pth")
    print()
    print("3️⃣  Update app.py to use the trained model (line ~55):")
    print("   model_path = 'models/trained_model.pth'")
    print()
    print("4️⃣  Restart the server:")
    print("   python app.py")
    print()
    print("=" * 70)
    print()

if __name__ == '__main__':
    # Create dataset with 50 files per class (takes ~10 seconds)
    create_demo_dataset(num_files_per_class=50)
    print("✨ Done! You can now train your model with real data.")
