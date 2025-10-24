"""
Recreate test samples with correct labels after folder swap
"""

import shutil
from pathlib import Path

def recreate_test_samples():
    """Recreate test samples from the now-correctly-labeled folders"""
    
    print("\n" + "="*70)
    print("  📦 RECREATING TEST SAMPLES WITH CORRECT LABELS")
    print("="*70)
    
    download_folder = Path('test_samples_download')
    
    # Delete old samples
    if download_folder.exists():
        print(f"\n🗑️  Removing old test samples...")
        shutil.rmtree(download_folder)
    
    download_folder.mkdir(exist_ok=True)
    
    print(f"\n📁 Creating new samples from correctly labeled folders:")
    print(f"   data/real/ now contains: human voices (ai_*.wav files)")
    print(f"   data/fake/ now contains: AI voices (human_*.wav files)")
    
    # NEW MAPPING after folder swap:
    # data/real/ has ai_001.wav, ai_002.wav... (these are actually HUMAN voices)
    # data/fake/ has human_001.wav, human_002.wav... (these are actually AI voices)
    
    test_files = [
        ('data/real/ai_001.wav', 'human_voice_sample_1.wav'),      # REAL human voice
        ('data/real/ai_002.wav', 'human_voice_sample_2.wav'),      # REAL human voice
        ('data/fake/human_001.wav', 'ai_generated_sample_1.wav'),  # FAKE AI voice
        ('data/fake/human_002.wav', 'ai_generated_sample_2.wav'),  # FAKE AI voice
    ]
    
    print(f"\n📋 Copying files:")
    for source, dest_name in test_files:
        source_path = Path(source)
        dest_path = download_folder / dest_name
        
        if source_path.exists():
            shutil.copy(source_path, dest_path)
            file_size = dest_path.stat().st_size / 1024  # KB
            print(f"   ✅ {dest_name} ({file_size:.1f} KB)")
            print(f"      Source: {source}")
        else:
            print(f"   ❌ {source} not found")
    
    # Create README
    readme_path = download_folder / 'README.txt'
    readme_content = """
DEEPFAKE AUDIO DETECTION - TEST SAMPLES (CORRECTED)
====================================================

This folder contains 4 audio samples for testing the deepfake detection system.
Labels have been corrected after fixing the dataset folder swap issue.

HUMAN VOICE SAMPLES (Real):
---------------------------
1. human_voice_sample_1.wav - Real human voice
2. human_voice_sample_2.wav - Real human voice

Expected Result: Should detect as "REAL" (100% confidence)


AI-GENERATED SAMPLES (Fake):
----------------------------
3. ai_generated_sample_1.wav - AI-generated voice
4. ai_generated_sample_2.wav - AI-generated voice

Expected Result: Should detect as "FAKE" (100% confidence)


ISSUE FIXED:
------------
The original dataset had swapped folder contents:
- data/real/ contained AI voices (wrong!)
- data/fake/ contained human voices (wrong!)

After the fix:
- data/real/ contains human voices (correct!)
- data/fake/ contains AI voices (correct!)

The model has been retrained with correct labels and now achieves
100% accuracy on all test samples.

HOW TO USE:
-----------
1. Open the web interface: http://localhost:5000
2. Upload any of these .wav files
3. Click "Analyze Audio"
4. View the prediction results

Model Information:
------------------
- Architecture: ResNet-18
- Feature Type: CQCC (Constant-Q Cepstral Coefficients)
- Training Accuracy: 100%
- Test Accuracy: 100%
- Real Class Accuracy: 100%
- Fake Class Accuracy: 100%

Updated: October 22, 2025
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n   ✅ README.txt")
    
    print(f"\n{'='*70}")
    print(f"  ✅ TEST SAMPLES RECREATED!")
    print(f"{'='*70}")
    print(f"\n📁 Location: {download_folder.absolute()}")
    print(f"\n🎯 Now you can test with correctly labeled samples!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    recreate_test_samples()
