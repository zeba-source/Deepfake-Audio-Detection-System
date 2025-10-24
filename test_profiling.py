"""
Test Model Performance Profiling

Tests the profiling functionality with a sample model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.profiling.profile_model import (
    profile_model,
    measure_inference_time,
    measure_batch_inference,
    measure_memory_usage,
    count_parameters,
    calculate_flops,
    generate_performance_report
)


class SimpleTestModel(nn.Module):
    """Simple CNN model for testing."""
    
    def __init__(self, num_classes=2):
        super(SimpleTestModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetTestModel(nn.Module):
    """ResNet-like model for testing."""
    
    def __init__(self, num_classes=2):
        super(ResNetTestModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block may have stride > 1
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None
        
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


def test_simple_model():
    """Test profiling with simple CNN model."""
    print("\n" + "="*70)
    print("TEST 1: Simple CNN Model")
    print("="*70)
    
    model = SimpleTestModel(num_classes=2)
    input_shape = (1, 1, 32, 32)  # Batch=1, Channels=1, H=32, W=32
    
    # Profile model
    results = profile_model(
        model=model,
        input_shape=input_shape,
        model_name='SimpleCNN',
        devices=['cpu'],
        batch_sizes=[1, 4, 8, 16],
        num_iterations=50,
        output_dir='profiling_results'
    )
    
    # Generate report
    report = generate_performance_report(
        results,
        output_path='profiling_results/SimpleCNN_report.txt'
    )
    
    print("\n" + report)
    
    return results


def test_resnet_model():
    """Test profiling with ResNet-like model."""
    print("\n" + "="*70)
    print("TEST 2: ResNet-like Model")
    print("="*70)
    
    model = ResNetTestModel(num_classes=2)
    input_shape = (1, 1, 64, 64)  # Larger input for ResNet
    
    # Determine available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, testing CPU only")
    
    # Profile model
    results = profile_model(
        model=model,
        input_shape=input_shape,
        model_name='ResNet',
        devices=devices,
        batch_sizes=[1, 2, 4, 8, 16, 32],
        num_iterations=100,
        output_dir='profiling_results'
    )
    
    # Generate report
    report = generate_performance_report(
        results,
        output_path='profiling_results/ResNet_report.txt'
    )
    
    print("\n" + report)
    
    return results


def test_individual_functions():
    """Test individual profiling functions."""
    print("\n" + "="*70)
    print("TEST 3: Individual Profiling Functions")
    print("="*70)
    
    model = SimpleTestModel(num_classes=2)
    input_shape = (1, 1, 32, 32)
    
    print("\n1. Testing measure_inference_time()...")
    timing = measure_inference_time(model, input_shape, device='cpu', num_iterations=50)
    print(f"   Mean inference time: {timing['mean_ms']:.2f} ms")
    print(f"   Throughput: {timing['throughput_fps']:.2f} FPS")
    
    print("\n2. Testing count_parameters()...")
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    
    print("\n3. Testing measure_memory_usage()...")
    memory = measure_memory_usage(model, input_shape, device='cpu')
    print(f"   Device: {memory['device']}")
    if 'memory_increase_mb' in memory:
        print(f"   Memory increase: {memory['memory_increase_mb']:.2f} MB")
    
    print("\n4. Testing calculate_flops()...")
    flops = calculate_flops(model, input_shape, device='cpu')
    if flops['available']:
        print(f"   FLOPs: {flops['flops_formatted']}")
        print(f"   Params: {flops['params_formatted']}")
    else:
        print(f"   ⚠️  FLOPs calculation not available")
        print(f"   Reason: {flops.get('error', 'Unknown')}")
    
    print("\n5. Testing measure_batch_inference()...")
    batch_results = measure_batch_inference(
        model, 
        input_shape[1:],  # Remove batch dimension
        batch_sizes=[1, 4, 8],
        device='cpu',
        num_iterations=20
    )
    print(f"   Tested batch sizes: {list(batch_results.keys())}")
    for bs, stats in batch_results.items():
        print(f"   Batch {bs}: {stats['mean_ms']:.2f} ms, "
              f"{stats['samples_per_second']:.2f} samples/s")


def test_with_trained_model():
    """Test profiling with a trained model if available."""
    print("\n" + "="*70)
    print("TEST 4: Trained Model Profiling (if available)")
    print("="*70)
    
    model_path = Path('models/best_model.pth')
    
    if not model_path.exists():
        print("⚠️  No trained model found at models/best_model.pth")
        print("   Skipping trained model test")
        return None
    
    try:
        # Try to load model
        from src.models.resnet_model import ResNetModel
        
        print(f"✓ Loading model from {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = ResNetModel(num_classes=2)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Profile
        input_shape = (1, 1, 128, 100)  # Typical mel spectrogram shape
        
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        results = profile_model(
            model=model,
            input_shape=input_shape,
            model_name='TrainedResNet',
            devices=devices,
            batch_sizes=[1, 4, 8, 16],
            num_iterations=100,
            output_dir='profiling_results'
        )
        
        # Generate report
        report = generate_performance_report(
            results,
            output_path='profiling_results/TrainedResNet_report.txt'
        )
        
        print("\n" + report)
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        return None


def compare_models():
    """Compare performance of different models."""
    print("\n" + "="*70)
    print("TEST 5: Model Comparison")
    print("="*70)
    
    models = {
        'SimpleCNN': (SimpleTestModel(num_classes=2), (1, 1, 32, 32)),
        'ResNet': (ResNetTestModel(num_classes=2), (1, 1, 64, 64))
    }
    
    comparison = {}
    
    for name, (model, input_shape) in models.items():
        print(f"\nProfiling {name}...")
        
        # Quick profile (fewer iterations)
        results = profile_model(
            model=model,
            input_shape=input_shape,
            model_name=name,
            devices=['cpu'],
            batch_sizes=[1, 8],
            num_iterations=30,
            output_dir=None  # Don't save
        )
        
        comparison[name] = {
            'params': results['parameters']['total'],
            'inference_time_ms': results['profiles']['cpu']['single_inference']['mean_ms'],
            'throughput_fps': results['profiles']['cpu']['single_inference']['throughput_fps']
        }
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} | {'Parameters':>12} | {'Time (ms)':>10} | {'FPS':>8}")
    print("-" * 70)
    
    for name, stats in comparison.items():
        print(f"{name:<15} | {stats['params']:>12,} | "
              f"{stats['inference_time_ms']:>10.2f} | {stats['throughput_fps']:>8.2f}")
    
    print("="*70)


def main():
    """Run all profiling tests."""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE PROFILING TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    try:
        # Test 1: Simple model
        print("\nRunning Test 1: Simple CNN Model...")
        test_simple_model()
        
        # Test 2: ResNet model
        print("\nRunning Test 2: ResNet-like Model...")
        test_resnet_model()
        
        # Test 3: Individual functions
        print("\nRunning Test 3: Individual Functions...")
        test_individual_functions()
        
        # Test 4: Trained model (if available)
        print("\nRunning Test 4: Trained Model...")
        test_with_trained_model()
        
        # Test 5: Model comparison
        print("\nRunning Test 5: Model Comparison...")
        compare_models()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  - profiling_results/SimpleCNN_profiling.json")
        print("  - profiling_results/SimpleCNN_report.txt")
        print("  - profiling_results/ResNet_profiling.json")
        print("  - profiling_results/ResNet_report.txt")
        if Path('models/best_model.pth').exists():
            print("  - profiling_results/TrainedResNet_profiling.json")
            print("  - profiling_results/TrainedResNet_report.txt")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
