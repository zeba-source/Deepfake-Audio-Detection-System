"""Demo script for testing augmentation validation."""
import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from src.utils.augmentation_validator import (
    validate_augmentation,
    batch_validate_augmentation
)


def create_test_audio_files(output_dir: str, num_files: int = 3):
    """Create synthetic test audio files.
    
    Args:
        output_dir: Directory to save test files
        num_files: Number of files to create
        
    Returns:
        List of (file_path, label) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []
    
    sr = 16000
    duration = 2.0
    
    for i in range(num_files):
        # Alternate between real and fake labels
        label = i % 2
        label_name = 'fake' if label == 1 else 'real'
        
        # Create audio with different frequencies
        t = np.linspace(0, duration, int(sr * duration))
        
        if label == 0:  # Real - cleaner sine wave
            audio = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        else:  # Fake - more complex waveform
            audio = (np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 5 * t) +
                    0.2 * np.sin(2 * np.pi * 1320 * t))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = (audio * 32767).astype(np.int16)
        
        # Save file
        filename = os.path.join(output_dir, f'{label_name}_{i}.wav')
        wavfile.write(filename, sr, audio)
        files.append((filename, label))
        print(f"Created: {filename} (label={label})")
    
    return files


def main():
    """Run augmentation validation demo."""
    print("="*70)
    print("AUGMENTATION VALIDATION DEMO")
    print("="*70)
    
    # Create temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix='augmentation_test_')
    print(f"\nCreating test audio files in: {test_dir}")
    
    # Create synthetic test files
    test_files = create_test_audio_files(test_dir, num_files=3)
    
    # Test 1: Single file validation
    print("\n" + "="*70)
    print("TEST 1: Single Audio Validation")
    print("="*70)
    audio_path, label = test_files[0]
    
    result = validate_augmentation(
        audio_path=audio_path,
        label=label,
        output_dir="results/augmentation_validation",
        save_plots=True
    )
    
    print("\n📊 Validation Result:")
    print(f"  - Audio path: {result['audio_path']}")
    print(f"  - Original label: {result['original_label']}")
    print(f"  - Labels preserved: {result['labels_preserved']}")
    print(f"  - Number of augmentations: {result['num_augmentations']}")
    print(f"  - Audio duration: {result['audio_stats']['duration_seconds']:.2f}s")
    print(f"  - Audio mean: {result['audio_stats']['mean']:.4f}")
    print(f"  - Audio std: {result['audio_stats']['std']:.4f}")
    
    print("\n📁 Augmentation details:")
    for aug_name, aug_data in result['augmentations'].items():
        print(f"  ✓ {aug_name}: {aug_data['description']}, label={aug_data['label']}")
    
    # Test 2: Batch validation
    print("\n" + "="*70)
    print("TEST 2: Batch Audio Validation")
    print("="*70)
    
    audio_paths = [f[0] for f in test_files]
    labels = [f[1] for f in test_files]
    
    batch_result = batch_validate_augmentation(
        audio_paths=audio_paths,
        labels=labels,
        output_dir="results/augmentation_validation",
        max_samples=3
    )
    
    print("\n✅ All tests completed successfully!")
    print(f"\nResults saved to: results/augmentation_validation/")
    print(f"Test files created in: {test_dir}")
    print("\nGenerated plots:")
    print("  - augmentation_validation_*.png (comprehensive comparison)")
    print("  - augmentation_detailed_*.png (detailed side-by-side view)")
    
    # Cleanup test directory (optional)
    import shutil
    try:
        shutil.rmtree(test_dir)
        print(f"\n🗑️  Cleaned up test directory: {test_dir}")
    except Exception as e:
        print(f"\n⚠️  Could not clean up test directory: {e}")


if __name__ == '__main__':
    main()
