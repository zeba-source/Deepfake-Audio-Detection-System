# Model Performance Profiling

Comprehensive performance profiling tools for PyTorch models.

## Overview

This module provides detailed performance analysis including:
- **Inference Time**: Single and batch inference speed measurements
- **Memory Usage**: CPU and GPU memory tracking
- **Parameter Counting**: Total, trainable, and layer-wise parameters
- **FLOPs Calculation**: Computational complexity analysis
- **Device Comparison**: CPU vs GPU performance benchmarks
- **Performance Reports**: Automated report generation

## Features

### ✅ Inference Time Profiling
- Single prediction timing with statistical analysis
- Batch inference speed testing with multiple batch sizes
- Warmup iterations for accurate measurements
- Throughput calculation (FPS)
- Percentile statistics (P95, P99)

### ✅ Memory Profiling
- GPU memory tracking (allocated, peak, total)
- CPU memory monitoring
- Per-inference memory usage

### ✅ Model Analysis
- Total parameter count
- Trainable vs non-trainable parameters
- Layer-wise parameter breakdown
- Model size calculation

### ✅ Computational Complexity
- FLOPs calculation using thop
- Parameter-efficient model comparison
- Computational cost analysis

### ✅ Multi-Device Support
- CPU profiling
- GPU profiling (if CUDA available)
- Automatic GPU speedup calculation
- Device-specific optimizations

## Quick Start

### Basic Profiling

```python
import torch
import torch.nn as nn
from src.profiling import profile_model

# Your model
model = YourModel()
input_shape = (1, 1, 32, 32)  # (batch, channels, height, width)

# Profile model
results = profile_model(
    model=model,
    input_shape=input_shape,
    model_name='YourModel',
    devices=['cpu', 'cuda'],  # Test both CPU and GPU
    batch_sizes=[1, 4, 8, 16, 32],
    num_iterations=100,
    output_dir='profiling_results'
)
```

### Generate Report

```python
from src.profiling import generate_performance_report

# Generate formatted report
report = generate_performance_report(
    profiling_results=results,
    output_path='profiling_results/report.txt'
)

print(report)
```

### Individual Metrics

```python
from src.profiling import (
    measure_inference_time,
    measure_batch_inference,
    measure_memory_usage,
    count_parameters,
    calculate_flops
)

# Inference time
timing = measure_inference_time(model, input_shape, device='cpu', num_iterations=100)
print(f"Mean: {timing['mean_ms']:.2f} ms")
print(f"Throughput: {timing['throughput_fps']:.2f} FPS")

# Memory usage
memory = measure_memory_usage(model, input_shape, device='cuda')
print(f"Peak memory: {memory['peak_memory_mb']:.2f} MB")

# Parameter count
params = count_parameters(model)
print(f"Total parameters: {params['total']:,}")

# FLOPs
flops = calculate_flops(model, input_shape)
print(f"FLOPs: {flops['flops_formatted']}")
```

## Functions

### profile_model()

Comprehensive model profiling with all metrics.

```python
def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    model_name: str = 'model',
    devices: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    num_iterations: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, any]:
    """
    Args:
        model: PyTorch model to profile
        input_shape: Input tensor shape (batch, channels, height, width)
        model_name: Name for reporting
        devices: Devices to test ['cpu', 'cuda'] (default: auto-detect)
        batch_sizes: Batch sizes to test (default: [1, 4, 8, 16, 32])
        num_iterations: Number of timing iterations
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comprehensive profiling results
    """
```

**Example:**
```python
results = profile_model(
    model=resnet_model,
    input_shape=(1, 1, 64, 64),
    model_name='ResNet18',
    devices=['cpu', 'cuda'],
    batch_sizes=[1, 8, 16, 32],
    num_iterations=100,
    output_dir='results'
)
```

### measure_inference_time()

Measure single inference time with statistics.

```python
def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Returns:
        {
            'mean_ms': float,          # Mean time in milliseconds
            'std_ms': float,           # Standard deviation
            'min_ms': float,           # Minimum time
            'max_ms': float,           # Maximum time
            'median_ms': float,        # Median time
            'p95_ms': float,           # 95th percentile
            'p99_ms': float,           # 99th percentile
            'throughput_fps': float    # Frames per second
        }
    """
```

### measure_batch_inference()

Test batch inference with multiple batch sizes.

```python
def measure_batch_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int],
    device: str = 'cpu',
    num_iterations: int = 50
) -> Dict[int, Dict[str, float]]:
    """
    Returns:
        {
            batch_size: {
                'mean_ms': float,
                'samples_per_second': float,
                'ms_per_sample': float
            }
        }
    """
```

### measure_memory_usage()

Measure memory consumption during inference.

```python
def measure_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Returns (GPU):
        {
            'device': 'cuda',
            'model_memory_mb': float,
            'peak_memory_mb': float,
            'total_allocated_mb': float
        }
        
    Returns (CPU):
        {
            'device': 'cpu',
            'memory_increase_mb': float
        }
    """
```

### count_parameters()

Count model parameters.

```python
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Returns:
        {
            'total': int,
            'trainable': int,
            'non_trainable': int,
            'breakdown': Dict[str, int]  # Layer-wise counts
        }
    """
```

### calculate_flops()

Calculate FLOPs using thop library.

```python
def calculate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, Union[int, float, str]]:
    """
    Returns:
        {
            'flops': int,
            'flops_formatted': str,      # e.g., "9.741M"
            'params': int,
            'params_formatted': str,
            'available': bool
        }
    """
```

## Profiling Results Structure

```json
{
  "model_name": "ResNet",
  "input_shape": [1, 1, 64, 64],
  "timestamp": "2025-10-21T22:53:38.343808",
  "devices_tested": ["cpu", "cuda"],
  "batch_sizes_tested": [1, 4, 8, 16, 32],
  
  "parameters": {
    "total": 2777474,
    "trainable": 2777474,
    "non_trainable": 0,
    "breakdown": {
      "conv1": 3136,
      "layer1": 147968,
      ...
    }
  },
  
  "flops": {
    "flops": 108844000,
    "flops_formatted": "108.844M",
    "params_formatted": "2.777M",
    "available": true
  },
  
  "profiles": {
    "cpu": {
      "single_inference": {
        "mean_ms": 7.31,
        "std_ms": 1.40,
        "throughput_fps": 136.86,
        ...
      },
      "batch_inference": {
        "1": {"mean_ms": 7.50, "samples_per_second": 133.36},
        "8": {"mean_ms": 27.70, "samples_per_second": 288.85},
        ...
      },
      "memory": {
        "device": "cpu",
        "memory_increase_mb": 0.95
      }
    },
    "cuda": {
      ...
    }
  },
  
  "gpu_speedup": {
    "cpu_time_ms": 7.31,
    "gpu_time_ms": 1.23,
    "speedup": 5.94
  }
}
```

## Example Output

### Console Output

```
============================================================
Model Performance Profiling: ResNet
============================================================

1. Counting parameters...
   Total parameters: 2,777,474
   Trainable parameters: 2,777,474
   Model size: 10.60 MB (float32)

2. Profiling on CPU...

   a) Measuring single inference time...
      Mean: 7.31 ms
      Std: 1.40 ms
      Throughput: 136.86 FPS

   b) Measuring batch inference speed...
      Batch size | Time (ms) | Samples/sec | ms/sample
      -------------------------------------------------------
               1 |      7.50 |      133.36 |      7.50
               8 |     27.70 |      288.85 |      3.46
              16 |     51.26 |      312.13 |      3.20

   c) Measuring memory usage...
      Memory increase: 0.95 MB

   d) Calculating FLOPs...
      FLOPs: 108.844M
      Params: 2.777M

3. GPU Speedup: 5.94x faster than CPU
```

### Performance Report

```
======================================================================
MODEL PERFORMANCE REPORT: ResNet
======================================================================
Timestamp: 2025-10-21T22:53:38.343808
Input Shape: [1, 1, 64, 64]

PARAMETERS
----------------------------------------------------------------------
Total Parameters:            2,777,474
Trainable Parameters:        2,777,474
Model Size (float32):            10.60 MB

COMPUTATIONAL COMPLEXITY
----------------------------------------------------------------------
FLOPs:                        108.844M

CPU PERFORMANCE
----------------------------------------------------------------------
Single Inference Time:
  Mean:                           7.31 ms
  Throughput:                   136.86 FPS
  
Batch Inference:
   Batch |  Time (ms) |  Samples/s |  ms/sample
  ------ | ---------- | ---------- | ----------
       1 |       7.50 |     133.36 |       7.50
       8 |      27.70 |     288.85 |       3.46

GPU PERFORMANCE
----------------------------------------------------------------------
Single Inference Time:
  Mean:                           1.23 ms
  Throughput:                   813.01 FPS

GPU vs CPU COMPARISON
----------------------------------------------------------------------
GPU Speedup:                      5.94x
======================================================================
```

## Use Cases

### 1. Model Selection

Compare different architectures:

```python
models = {
    'SimpleCNN': simple_cnn,
    'ResNet18': resnet18,
    'ResNet34': resnet34
}

for name, model in models.items():
    results = profile_model(model, input_shape, model_name=name)
    print(f"{name}: {results['profiles']['cpu']['single_inference']['mean_ms']:.2f} ms")
```

### 2. Optimization Validation

Before/after optimization:

```python
# Before optimization
results_before = profile_model(model_original, input_shape, model_name='Before')

# After optimization (quantization, pruning, etc.)
results_after = profile_model(model_optimized, input_shape, model_name='After')

# Compare
speedup = results_before['profiles']['cpu']['single_inference']['mean_ms'] / \
          results_after['profiles']['cpu']['single_inference']['mean_ms']
print(f"Speedup: {speedup:.2f}x")
```

### 3. Batch Size Tuning

Find optimal batch size:

```python
results = profile_model(
    model, 
    input_shape,
    batch_sizes=[1, 2, 4, 8, 16, 32, 64]
)

# Find best throughput
batch_results = results['profiles']['cpu']['batch_inference']
best_bs = max(batch_results.items(), 
              key=lambda x: x[1]['samples_per_second'])[0]
print(f"Optimal batch size: {best_bs}")
```

### 4. Hardware Selection

Determine if GPU is worth it:

```python
results = profile_model(model, input_shape, devices=['cpu', 'cuda'])

if 'gpu_speedup' in results:
    speedup = results['gpu_speedup']['speedup']
    if speedup > 5:
        print(f"GPU recommended: {speedup:.1f}x faster")
    else:
        print(f"CPU sufficient: only {speedup:.1f}x speedup")
```

## Performance Tips

### 1. Accurate Timing
- Use sufficient warmup iterations (10+)
- Run multiple iterations (100+)
- Avoid other processes during profiling

### 2. Memory Profiling
- Clear GPU cache before profiling
- Profile with realistic batch sizes
- Monitor peak memory, not just average

### 3. FLOPs Calculation
- Install thop: `pip install thop`
- Some models may not support FLOPs calculation
- Use as relative comparison, not absolute measure

### 4. Batch Size Selection
- Larger batches improve throughput
- Consider memory constraints
- Balance latency vs throughput

## Dependencies

**Required:**
- torch
- numpy
- psutil (CPU memory tracking)

**Optional:**
- thop (FLOPs calculation)
- CUDA (GPU profiling)

Install:
```bash
pip install psutil thop
```

## Testing

Run the test suite:

```bash
python test_profiling.py
```

This will:
- Profile SimpleCNN model
- Profile ResNet model
- Test individual functions
- Compare model performance
- Generate reports

## See Also

- [Model Export](../deployment/README.md) - Model deployment
- [Training](../training/README.md) - Model training
- [Testing](../testing/README.md) - Model evaluation
