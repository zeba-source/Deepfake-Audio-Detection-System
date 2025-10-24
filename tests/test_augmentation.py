"""Unit tests for augmentation validation utilities."""
import os
import tempfile
import shutil

import pytest
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing

from src.utils.augmentation_validator import (
    add_gaussian_noise,
    pitch_shift_audio,
    time_stretch_audio,
    apply_spec_augment,
    validate_augmentation,
    batch_validate_augmentation
)


@pytest.fixture
def sample_audio():
    """Create a sample audio signal."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_audio_file():
    """Create a temporary audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(f.name, sr, audio)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


# Test 1: Gaussian noise augmentation
def test_add_gaussian_noise(sample_audio):
    """Test that Gaussian noise augmentation preserves audio length and changes values."""
    audio, _ = sample_audio
    noise_factor = 0.005
    
    augmented = add_gaussian_noise(audio, noise_factor)
    
    # Check shape preserved
    assert augmented.shape == audio.shape, "Shape should be preserved"
    
    # Check that values changed
    assert not np.array_equal(augmented, audio), "Audio should be modified"
    
    # Check dtype
    assert augmented.dtype == np.float32, "Should return float32"
    
    # Check that noise is relatively small
    diff = np.abs(augmented - audio)
    assert np.mean(diff) < 0.1, "Noise should be relatively small"


# Test 2: Pitch shift augmentation
def test_pitch_shift_audio(sample_audio):
    """Test that pitch shift preserves audio length."""
    audio, sr = sample_audio
    n_steps = 2.0
    
    augmented = pitch_shift_audio(audio, sr, n_steps)
    
    # Check shape preserved (approximately, may have small differences)
    assert abs(augmented.shape[0] - audio.shape[0]) < 100, "Shape should be approximately preserved"
    
    # Check that values changed
    assert not np.array_equal(augmented, audio), "Audio should be modified"


# Test 3: Time stretch augmentation
def test_time_stretch_audio(sample_audio):
    """Test that time stretch changes audio length appropriately."""
    audio, _ = sample_audio
    rate = 1.1
    
    augmented = time_stretch_audio(audio, rate)
    
    # Check that length changed (faster = shorter)
    assert augmented.shape[0] < audio.shape[0], "Time-stretched audio should be shorter for rate > 1"
    
    # Check approximate length
    expected_length = int(audio.shape[0] / rate)
    assert abs(augmented.shape[0] - expected_length) < 500, "Length should be approximately correct"


# Test 4: SpecAugment
def test_apply_spec_augment():
    """Test that SpecAugment masks spectrogram correctly."""
    # Create dummy spectrogram
    spectrogram = np.random.randn(128, 100).astype(np.float32)
    
    augmented = apply_spec_augment(
        spectrogram,
        time_mask_param=10,
        freq_mask_param=10,
        num_time_masks=1,
        num_freq_masks=1
    )
    
    # Check shape preserved
    assert augmented.shape == spectrogram.shape, "Shape should be preserved"
    
    # Check that masking occurred (some zeros should be present)
    assert np.sum(augmented == 0) > 0, "Should have masked regions (zeros)"
    
    # Check that not everything is masked
    assert np.sum(augmented != 0) > 0, "Should have unmasked regions"


# Test 5: validate_augmentation function
def test_validate_augmentation(sample_audio_file, temp_output_dir):
    """Test the main validation function."""
    result = validate_augmentation(
        audio_path=sample_audio_file,
        label=0,
        output_dir=temp_output_dir,
        sr=16000,
        save_plots=True
    )
    
    # Check result structure
    assert 'audio_path' in result
    assert 'original_label' in result
    assert 'labels_preserved' in result
    assert 'num_augmentations' in result
    assert 'augmentations' in result
    assert 'original_audio' in result
    
    # Check label preservation
    assert result['labels_preserved'] == True, "Labels should be preserved"
    assert result['original_label'] == 0
    
    # Check augmentations
    assert result['num_augmentations'] == 4, "Should have 4 augmentations"
    assert 'gaussian_noise' in result['augmentations']
    assert 'pitch_shift' in result['augmentations']
    assert 'time_stretch' in result['augmentations']
    assert 'spec_augment' in result['augmentations']
    
    # Check all labels match
    for aug_name, aug_data in result['augmentations'].items():
        assert aug_data['label'] == 0, f"{aug_name} should have label 0"
    
    # Check audio stats
    assert 'audio_stats' in result
    assert 'mean' in result['audio_stats']
    assert 'std' in result['audio_stats']
    assert 'duration_seconds' in result['audio_stats']
    
    # Check plots created
    assert os.path.exists(temp_output_dir), "Output directory should exist"
    plot_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.png')]
    assert len(plot_files) >= 2, "Should create at least 2 plots"


# Test 6: batch_validate_augmentation function
def test_batch_validate_augmentation(temp_output_dir):
    """Test batch validation function."""
    # Create multiple test audio files
    audio_paths = []
    labels = []
    
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sr = 16000
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            audio = (audio * 32767).astype(np.int16)
            wavfile.write(f.name, sr, audio)
            audio_paths.append(f.name)
            labels.append(i % 2)
    
    try:
        # Run batch validation
        result = batch_validate_augmentation(
            audio_paths=audio_paths,
            labels=labels,
            output_dir=temp_output_dir,
            max_samples=3
        )
        
        # Check result structure
        assert 'num_samples' in result
        assert 'results' in result
        assert 'all_labels_preserved' in result
        
        # Check number of samples
        assert result['num_samples'] == 3, "Should process 3 samples"
        assert len(result['results']) == 3
        
        # Check that all labels preserved
        assert result['all_labels_preserved'] == True
        
        # Check each result
        for i, res in enumerate(result['results']):
            assert res['original_label'] == labels[i]
            assert res['labels_preserved'] == True
    
    finally:
        # Cleanup temp files
        for path in audio_paths:
            if os.path.exists(path):
                os.unlink(path)


# Test 7: Label verification with fake samples
def test_validate_augmentation_fake_label(sample_audio_file, temp_output_dir):
    """Test validation with fake label (label=1)."""
    result = validate_augmentation(
        audio_path=sample_audio_file,
        label=1,  # Fake label
        output_dir=temp_output_dir,
        save_plots=False
    )
    
    # Check label is correct
    assert result['original_label'] == 1
    
    # Check all augmentations have label=1
    for aug_data in result['augmentations'].values():
        assert aug_data['label'] == 1, "All augmentations should have label=1"
    
    assert result['labels_preserved'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
