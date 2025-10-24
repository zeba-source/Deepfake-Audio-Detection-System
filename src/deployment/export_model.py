"""Model export and TorchScript conversion for production deployment."""
import os
from typing import Dict, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.models.resnet_model import ResNetDeepfakeDetector
from src.models.multistream_model import MultiStreamDetector


class ModelWrapper(nn.Module):
    """Wrapper for models to ensure JIT compatibility."""
    
    def __init__(self, model: nn.Module):
        """Initialize wrapper.
        
        Args:
            model: PyTorch model to wrap
        """
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits tensor (batch_size, 2)
        """
        return self.model(x)


def export_model(
    model_path: str,
    output_path: str,
    model_type: str = 'resnet',
    input_shape: Tuple[int, int, int, int] = (1, 1, 32, 32),
    use_trace: bool = True,
    optimize: bool = True,
    device: str = 'cpu'
) -> Dict[str, any]:
    """Export PyTorch model to TorchScript for production deployment.
    
    This function:
    1. Loads trained PyTorch model
    2. Converts to TorchScript using torch.jit.trace or torch.jit.script
    3. Optimizes for inference
    4. Saves optimized model for deployment
    5. Validates exported model
    
    Args:
        model_path: Path to saved PyTorch model (.pth file)
        output_path: Path to save TorchScript model (.pt file)
        model_type: Type of model ('resnet' or 'multistream')
        input_shape: Expected input shape (batch, channels, height, width)
        use_trace: If True, use torch.jit.trace; else use torch.jit.script
        optimize: Whether to optimize for mobile/production
        device: Device to use for export ('cpu' or 'cuda')
        
    Returns:
        Dictionary with export information and validation results
    """
    print(f"\n{'='*70}")
    print("MODEL EXPORT TO TORCHSCRIPT")
    print(f"{'='*70}")
    print(f"Input model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Model type: {model_type}")
    print(f"Input shape: {input_shape}")
    print(f"Method: {'torch.jit.trace' if use_trace else 'torch.jit.script'}")
    print(f"Optimize: {optimize}")
    print(f"Device: {device}")
    
    # Set device
    device = torch.device(device)
    
    # Load model
    print(f"\n1. Loading PyTorch model...")
    if model_type == 'resnet':
        model = ResNetDeepfakeDetector()
    elif model_type == 'multistream':
        model = MultiStreamDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"   ✓ Model loaded successfully")
    
    # Create example input
    print(f"\n2. Creating example input...")
    example_input = torch.randn(input_shape).to(device)
    print(f"   ✓ Example input shape: {example_input.shape}")
    
    # Test original model
    print(f"\n3. Testing original model...")
    with torch.no_grad():
        original_output = model(example_input)
    print(f"   ✓ Original output shape: {original_output.shape}")
    print(f"   ✓ Original output: {original_output[0].cpu().numpy()}")
    
    # Convert to TorchScript
    print(f"\n4. Converting to TorchScript...")
    if use_trace:
        # Use tracing (records operations)
        with torch.no_grad():
            scripted_model = torch.jit.trace(model, example_input)
        print(f"   ✓ Model traced successfully")
    else:
        # Use scripting (analyzes Python code)
        scripted_model = torch.jit.script(model)
        print(f"   ✓ Model scripted successfully")
    
    # Optimize for inference
    if optimize:
        print(f"\n5. Optimizing for inference...")
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        print(f"   ✓ Model optimized")
    
    # Validate exported model
    print(f"\n6. Validating exported model...")
    with torch.no_grad():
        scripted_output = scripted_model(example_input)
    
    # Check outputs match
    output_diff = torch.abs(original_output - scripted_output).max().item()
    print(f"   ✓ Scripted output shape: {scripted_output.shape}")
    print(f"   ✓ Scripted output: {scripted_output[0].cpu().numpy()}")
    print(f"   ✓ Max difference: {output_diff:.2e}")
    
    if output_diff > 1e-5:
        print(f"   ⚠️  Warning: Outputs differ by {output_diff:.2e}")
    else:
        print(f"   ✓ Outputs match (diff < 1e-5)")
    
    # Save TorchScript model
    print(f"\n7. Saving TorchScript model...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    scripted_model.save(output_path)
    
    # Get file sizes
    original_size = os.path.getsize(model_path)
    scripted_size = os.path.getsize(output_path)
    
    print(f"   ✓ Model saved to: {output_path}")
    print(f"   ✓ Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"   ✓ Scripted size: {scripted_size / 1024 / 1024:.2f} MB")
    print(f"   ✓ Size change: {(scripted_size - original_size) / original_size * 100:+.1f}%")
    
    # Benchmark inference speed
    print(f"\n8. Benchmarking inference speed...")
    n_iterations = 100
    
    # Original model
    with torch.no_grad():
        import time
        start = time.time()
        for _ in range(n_iterations):
            _ = model(example_input)
        original_time = (time.time() - start) / n_iterations * 1000
    
    # Scripted model
    with torch.no_grad():
        start = time.time()
        for _ in range(n_iterations):
            _ = scripted_model(example_input)
        scripted_time = (time.time() - start) / n_iterations * 1000
    
    print(f"   ✓ Original model: {original_time:.2f} ms/inference")
    print(f"   ✓ Scripted model: {scripted_time:.2f} ms/inference")
    print(f"   ✓ Speedup: {original_time / scripted_time:.2f}x")
    
    # Create export info
    export_info = {
        'original_model_path': model_path,
        'scripted_model_path': output_path,
        'model_type': model_type,
        'input_shape': input_shape,
        'export_method': 'trace' if use_trace else 'script',
        'optimized': optimize,
        'device': str(device),
        'validation': {
            'max_output_difference': float(output_diff),
            'outputs_match': output_diff < 1e-5
        },
        'file_sizes': {
            'original_mb': original_size / 1024 / 1024,
            'scripted_mb': scripted_size / 1024 / 1024,
            'size_change_percent': (scripted_size - original_size) / original_size * 100
        },
        'performance': {
            'original_ms': original_time,
            'scripted_ms': scripted_time,
            'speedup': original_time / scripted_time
        }
    }
    
    # Save export info
    import json
    info_path = output_path.replace('.pt', '_export_info.json')
    with open(info_path, 'w') as f:
        json.dump(export_info, f, indent=4)
    print(f"\n   ✓ Export info saved to: {info_path}")
    
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"✅ TorchScript model ready for deployment!")
    print(f"✅ Use scripted_model = torch.jit.load('{output_path}')")
    
    return export_info


def load_scripted_model(model_path: str, device: str = 'cpu') -> torch.jit.ScriptModule:
    """Load a TorchScript model for inference.
    
    Args:
        model_path: Path to TorchScript model (.pt file)
        device: Device to load model on
        
    Returns:
        Loaded TorchScript model
    """
    device = torch.device(device)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def predict_with_scripted_model(
    model: torch.jit.ScriptModule,
    input_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[int, float, np.ndarray]:
    """Run inference with TorchScript model.
    
    Args:
        model: Loaded TorchScript model
        input_tensor: Input tensor (batch, channels, height, width)
        device: Device to use for inference
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
    """
    device = torch.device(device)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return (
        predicted.item(),
        confidence.item(),
        probabilities.cpu().numpy()[0]
    )


if __name__ == '__main__':
    # Example usage
    print("Model Export Example")
    print("="*70)
    
    # Create a dummy model for demonstration
    print("\n1. Creating dummy model for demonstration...")
    model = ResNetDeepfakeDetector(dropout_p=0.5)
    
    # Save it
    dummy_model_path = 'models/dummy_model.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), dummy_model_path)
    print(f"   ✓ Dummy model saved to: {dummy_model_path}")
    
    # Export to TorchScript
    print("\n2. Exporting to TorchScript...")
    export_info = export_model(
        model_path=dummy_model_path,
        output_path='models/dummy_model_scripted.pt',
        model_type='resnet',
        input_shape=(1, 1, 32, 32),
        use_trace=True,
        optimize=True,
        device='cpu'
    )
    
    # Test loading and inference
    print("\n3. Testing deployed model...")
    scripted_model = load_scripted_model('models/dummy_model_scripted.pt', device='cpu')
    
    # Create test input
    test_input = torch.randn(1, 1, 32, 32)
    predicted_class, confidence, probs = predict_with_scripted_model(
        scripted_model, test_input, device='cpu'
    )
    
    print(f"   ✓ Prediction: {'Fake' if predicted_class == 1 else 'Real'}")
    print(f"   ✓ Confidence: {confidence * 100:.2f}%")
    print(f"   ✓ Probabilities: Real={probs[0]*100:.2f}%, Fake={probs[1]*100:.2f}%")
    
    print("\n✅ Demo complete!")
