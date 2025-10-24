"""Unit tests for deepfake audio detection features and models."""
import os
import tempfile
import shutil

import pytest
import numpy as np
import torch
import scipy.io.wavfile as wavfile

from src.features.extract import extract_cqcc_features, extract_mel_spectrogram
from src.features.feature_extractor import FeatureExtractor
from src.models.dataset import DeepfakeAudioDataset
from src.models.resnet_model import ResNetDeepfakeDetector


# Fixtures for test data
@pytest.fixture
def sample_audio_file():
    """Create a temporary synthetic audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sr = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sr * duration))
        # Generate a simple sine wave
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(f.name, sr, audio)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_audio_short():
    """Create a short synthetic audio file (0.5 seconds)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 880 * t)  # 880 Hz tone
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(f.name, sr, audio)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_audio_long():
    """Create a longer synthetic audio file (3 seconds)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 220 * t)  # 220 Hz tone
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(f.name, sr, audio)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_dataset_folder():
    """Create a temporary folder with real and fake audio files."""
    tmpdir = tempfile.mkdtemp()
    real_dir = os.path.join(tmpdir, 'real')
    fake_dir = os.path.join(tmpdir, 'fake')
    os.makedirs(real_dir)
    os.makedirs(fake_dir)
    
    sr = 16000
    duration = 0.5
    
    # Create 3 real files
    for i in range(3):
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * (440 + i * 100) * t)
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(os.path.join(real_dir, f'real_{i}.wav'), sr, audio)
    
    # Create 2 fake files
    for i in range(2):
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * (880 + i * 100) * t)
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(os.path.join(fake_dir, f'fake_{i}.wav'), sr, audio)
    
    yield tmpdir
    shutil.rmtree(tmpdir)


# Test 1: CQCC extraction returns correct shape
def test_cqcc_extraction_shape(sample_audio_file):
    """Test that CQCC extraction returns expected shape."""
    features = extract_cqcc_features(sample_audio_file, sr=16000)
    
    assert features is not None, "CQCC features should not be None"
    assert features.ndim == 2, "CQCC features should be 2D (frames x coefficients)"
    assert features.shape[1] == 20, "CQCC should have 20 coefficients"
    assert features.shape[0] > 0, "CQCC should have at least one frame"


# Test 2: Mel-spectrogram extraction handles different audio lengths
def test_mel_spectrogram_different_lengths(sample_audio_short, sample_audio_long):
    """Test mel-spectrogram extraction with different audio lengths."""
    # Short audio
    features_short = extract_mel_spectrogram(sample_audio_short, sr=16000, n_mels=128)
    assert features_short is not None
    assert features_short.ndim == 2
    assert features_short.shape[0] == 128, "Should have 128 mel bands"
    
    # Long audio
    features_long = extract_mel_spectrogram(sample_audio_long, sr=16000, n_mels=128)
    assert features_long is not None
    assert features_long.ndim == 2
    assert features_long.shape[0] == 128, "Should have 128 mel bands"
    
    # Long audio should have more frames
    assert features_long.shape[1] > features_short.shape[1], "Longer audio should have more frames"
    
    # All values should be in [0, 1] (normalized)
    assert np.all(features_short >= 0) and np.all(features_short <= 1), "Values should be normalized to [0,1]"
    assert np.all(features_long >= 0) and np.all(features_long <= 1), "Values should be normalized to [0,1]"


# Test 3: Feature extractor processes folders correctly
def test_feature_extractor_folder_processing(sample_dataset_folder):
    """Test that FeatureExtractor processes folder structure correctly."""
    extractor = FeatureExtractor(feature_type='cqcc', sample_rate=16000)
    
    output_dir = tempfile.mkdtemp()
    try:
        features, labels, df = extractor.process_dataset(
            data_folder=sample_dataset_folder,
            output_folder=output_dir,
            labels_csv=os.path.join(output_dir, 'labels.csv')
        )
        
        # Check shapes
        assert features.shape[0] == 5, "Should have 5 samples (3 real + 2 fake)"
        assert labels.shape[0] == 5, "Should have 5 labels"
        
        # Check labels
        assert np.sum(labels == 0) == 3, "Should have 3 real samples (label 0)"
        assert np.sum(labels == 1) == 2, "Should have 2 fake samples (label 1)"
        
        # Check DataFrame
        assert len(df) == 5, "DataFrame should have 5 rows"
        assert 'file_path' in df.columns
        assert 'label' in df.columns
        
        # Check files saved
        assert os.path.exists(os.path.join(output_dir, 'features.npy'))
        assert os.path.exists(os.path.join(output_dir, 'labels.npy'))
        assert os.path.exists(os.path.join(output_dir, 'labels.csv'))
    finally:
        shutil.rmtree(output_dir)


# Test 4: Dataset class returns correct tensor shapes
def test_dataset_tensor_shapes():
    """Test that DeepfakeAudioDataset returns correct tensor shapes."""
    # Create dummy features and labels
    num_samples = 10
    feature_dim = 20
    features = np.random.randn(num_samples, feature_dim).astype(np.float32)
    labels = np.array([0, 1] * (num_samples // 2))
    
    # Create dataset
    dataset = DeepfakeAudioDataset(features, labels, train=False)
    
    # Check length
    assert len(dataset) == num_samples, f"Dataset should have {num_samples} samples"
    
    # Check single item
    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor), "Should return torch.Tensor"
    assert isinstance(label, int), "Label should be int"
    assert tensor.ndim == 3, "Tensor should be 3D (channels, height, width)"
    assert tensor.shape[0] == 1, "First dimension should be 1 (channels)"
    assert label in [0, 1], "Label should be 0 or 1"


# Test 5: Model forward pass works with expected input
def test_resnet_model_forward_pass():
    """Test ResNet model forward pass with expected input shapes."""
    model = ResNetDeepfakeDetector()
    model.eval()
    
    # Create dummy input (batch_size=4, channels=1, height=32, width=32)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 32, 32)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (batch_size, 2), f"Output should be (batch_size, 2), got {output.shape}"
    
    # Check that output can be converted to probabilities
    probs = torch.softmax(output, dim=1)
    assert probs.shape == (batch_size, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "Probabilities should sum to 1"


# Test 6 (Bonus): Test embedding extraction
def test_model_embedding_extraction():
    """Test that get_embedding method works correctly."""
    model = ResNetDeepfakeDetector()
    model.eval()
    
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 32, 32)
    
    with torch.no_grad():
        embedding = model.get_embedding(dummy_input)
    
    # Embedding should be 2D (batch_size, embedding_dim)
    assert embedding.ndim == 2, "Embedding should be 2D"
    assert embedding.shape[0] == batch_size, f"First dim should be batch_size={batch_size}"
    assert embedding.shape[1] == 512, "ResNet18 embedding should be 512-dim"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
