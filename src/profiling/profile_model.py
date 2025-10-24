"""
Model Performance Profiling

Comprehensive profiling tools for measuring model performance including:
- Inference time (single and batch)
- Memory usage
- Parameter counting
- FLOPs calculation
- CPU vs GPU comparison
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import psutil
import gc
from datetime import datetime


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure inference time for single prediction.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        device: Device to run on ('cpu' or 'cuda')
        num_iterations: Number of iterations for timing
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    print(f"  Running {warmup_iterations} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Actual timing
    print(f"  Running {num_iterations} timed iterations...")
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'num_iterations': num_iterations,
        'throughput_fps': 1000.0 / float(np.mean(times))
    }


def measure_batch_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int],
    device: str = 'cpu',
    num_iterations: int = 50
) -> Dict[int, Dict[str, float]]:
    """
    Test batch inference speed with different batch sizes.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of single input (channels, height, width)
        batch_sizes: List of batch sizes to test
        device: Device to run on ('cpu' or 'cuda')
        num_iterations: Number of iterations per batch size
        
    Returns:
        Dictionary mapping batch size to timing statistics
    """
    model.eval()
    model = model.to(device)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"  Testing batch size: {batch_size}")
        
        # Create batched input
        batched_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(batched_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
        
        # Timing
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(dummy_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
        
        times = np.array(times)
        mean_time = float(np.mean(times))
        
        results[batch_size] = {
            'mean_ms': mean_time,
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'samples_per_second': batch_size * 1000.0 / mean_time,
            'ms_per_sample': mean_time / batch_size
        }
    
    return results


def measure_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure memory usage during inference.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of input tensor
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with memory statistics
    """
    model.eval()
    model = model.to(device)
    
    # Clear memory
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    results = {}
    
    if device == 'cuda' and torch.cuda.is_available():
        # GPU memory tracking
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        results['device'] = 'cuda'
        results['model_memory_mb'] = (memory_before) / (1024 ** 2)
        results['inference_memory_mb'] = (memory_after - memory_before) / (1024 ** 2)
        results['peak_memory_mb'] = peak_memory / (1024 ** 2)
        results['total_allocated_mb'] = memory_after / (1024 ** 2)
        
    else:
        # CPU memory tracking
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        dummy_input = torch.randn(input_shape).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_after = process.memory_info().rss
        
        results['device'] = 'cpu'
        results['memory_before_mb'] = memory_before / (1024 ** 2)
        results['memory_after_mb'] = memory_after / (1024 ** 2)
        results['memory_increase_mb'] = (memory_after - memory_before) / (1024 ** 2)
    
    return results


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Parameter breakdown by layer type
    param_breakdown = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else name
        if layer_type not in param_breakdown:
            param_breakdown[layer_type] = 0
        param_breakdown[layer_type] += param.numel()
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'breakdown': param_breakdown
    }


def calculate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, Union[int, float, str]]:
    """
    Calculate FLOPs (Floating Point Operations) for the model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor
        device: Device to run on
        
    Returns:
        Dictionary with FLOPs information
    """
    try:
        from thop import profile, clever_format
        
        model.eval()
        model = model.to(device)
        dummy_input = torch.randn(input_shape).to(device)
        
        # Calculate FLOPs and params
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        
        return {
            'flops': int(flops),
            'flops_formatted': flops_formatted,
            'params': int(params),
            'params_formatted': params_formatted,
            'method': 'thop',
            'available': True
        }
        
    except ImportError:
        print("  Warning: 'thop' package not installed. FLOPs calculation skipped.")
        print("  Install with: pip install thop")
        return {
            'flops': 0,
            'flops_formatted': 'N/A',
            'params': 0,
            'params_formatted': 'N/A',
            'method': 'none',
            'available': False,
            'error': 'thop package not installed'
        }
    except Exception as e:
        print(f"  Warning: FLOPs calculation failed: {e}")
        return {
            'flops': 0,
            'flops_formatted': 'N/A',
            'params': 0,
            'params_formatted': 'N/A',
            'method': 'none',
            'available': False,
            'error': str(e)
        }


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
    Comprehensive model profiling.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        model_name: Name of the model for reporting
        devices: List of devices to test on (default: ['cpu', 'cuda'] if CUDA available)
        batch_sizes: List of batch sizes to test (default: [1, 4, 8, 16, 32])
        num_iterations: Number of iterations for timing
        output_dir: Directory to save profiling results
        
    Returns:
        Dictionary with comprehensive profiling results
    """
    print(f"\n{'='*60}")
    print(f"Model Performance Profiling: {model_name}")
    print(f"{'='*60}\n")
    
    # Default devices
    if devices is None:
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
    
    # Default batch sizes
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]
    
    # Extract single input shape (remove batch dimension if present)
    if len(input_shape) == 4 and input_shape[0] == 1:
        single_input_shape = input_shape
        batch_input_shape = input_shape[1:]
    else:
        single_input_shape = (1,) + input_shape
        batch_input_shape = input_shape
    
    results = {
        'model_name': model_name,
        'input_shape': list(input_shape),
        'timestamp': datetime.now().isoformat(),
        'devices_tested': devices,
        'batch_sizes_tested': batch_sizes,
        'profiles': {}
    }
    
    # Count parameters (device-independent)
    print("1. Counting parameters...")
    param_info = count_parameters(model)
    results['parameters'] = param_info
    print(f"   Total parameters: {param_info['total']:,}")
    print(f"   Trainable parameters: {param_info['trainable']:,}")
    print(f"   Model size: {param_info['total'] * 4 / (1024**2):.2f} MB (float32)")
    
    # Profile on each device
    for device in devices:
        print(f"\n2. Profiling on {device.upper()}...")
        
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"   ⚠️  CUDA not available, skipping GPU profiling")
            continue
        
        device_results = {}
        
        # Single inference timing
        print("\n   a) Measuring single inference time...")
        try:
            inference_time = measure_inference_time(
                model, single_input_shape, device, num_iterations
            )
            device_results['single_inference'] = inference_time
            print(f"      Mean: {inference_time['mean_ms']:.2f} ms")
            print(f"      Std: {inference_time['std_ms']:.2f} ms")
            print(f"      Throughput: {inference_time['throughput_fps']:.2f} FPS")
        except Exception as e:
            print(f"      ❌ Error: {e}")
            device_results['single_inference'] = {'error': str(e)}
        
        # Batch inference timing
        print("\n   b) Measuring batch inference speed...")
        try:
            batch_results = measure_batch_inference(
                model, batch_input_shape, batch_sizes, device
            )
            device_results['batch_inference'] = batch_results
            print("\n      Batch size | Time (ms) | Samples/sec | ms/sample")
            print("      " + "-" * 55)
            for bs, stats in batch_results.items():
                print(f"      {bs:10d} | {stats['mean_ms']:9.2f} | "
                      f"{stats['samples_per_second']:11.2f} | {stats['ms_per_sample']:9.2f}")
        except Exception as e:
            print(f"      ❌ Error: {e}")
            device_results['batch_inference'] = {'error': str(e)}
        
        # Memory usage
        print("\n   c) Measuring memory usage...")
        try:
            memory_info = measure_memory_usage(model, single_input_shape, device)
            device_results['memory'] = memory_info
            if device == 'cuda':
                print(f"      Model memory: {memory_info['model_memory_mb']:.2f} MB")
                print(f"      Peak memory: {memory_info['peak_memory_mb']:.2f} MB")
                print(f"      Total allocated: {memory_info['total_allocated_mb']:.2f} MB")
            else:
                print(f"      Memory increase: {memory_info.get('memory_increase_mb', 0):.2f} MB")
        except Exception as e:
            print(f"      ❌ Error: {e}")
            device_results['memory'] = {'error': str(e)}
        
        # FLOPs calculation (only once, device-independent)
        if device == devices[0]:
            print("\n   d) Calculating FLOPs...")
            try:
                flops_info = calculate_flops(model, single_input_shape, device)
                results['flops'] = flops_info
                if flops_info['available']:
                    print(f"      FLOPs: {flops_info['flops_formatted']}")
                    print(f"      Params: {flops_info['params_formatted']}")
                else:
                    print(f"      ⚠️  FLOPs calculation not available")
            except Exception as e:
                print(f"      ❌ Error: {e}")
                results['flops'] = {'error': str(e), 'available': False}
        
        results['profiles'][device] = device_results
    
    # Compare CPU vs GPU if both available
    if 'cpu' in results['profiles'] and 'cuda' in results['profiles']:
        cpu_time = results['profiles']['cpu'].get('single_inference', {}).get('mean_ms', 0)
        gpu_time = results['profiles']['cuda'].get('single_inference', {}).get('mean_ms', 0)
        
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            results['gpu_speedup'] = {
                'cpu_time_ms': cpu_time,
                'gpu_time_ms': gpu_time,
                'speedup': speedup
            }
            print(f"\n3. GPU Speedup: {speedup:.2f}x faster than CPU")
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_path / f"{model_name}_profiling.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n4. Results saved to: {json_path}")
    
    print(f"\n{'='*60}")
    print("Profiling Complete!")
    print(f"{'='*60}\n")
    
    return results


def generate_performance_report(
    profiling_results: Dict[str, any],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a formatted performance report from profiling results.
    
    Args:
        profiling_results: Results from profile_model()
        output_path: Optional path to save the report
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"MODEL PERFORMANCE REPORT: {profiling_results['model_name']}")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {profiling_results['timestamp']}")
    report_lines.append(f"Input Shape: {profiling_results['input_shape']}")
    report_lines.append("")
    
    # Parameters
    params = profiling_results.get('parameters', {})
    report_lines.append("PARAMETERS")
    report_lines.append("-" * 70)
    report_lines.append(f"Total Parameters:      {params.get('total', 0):>15,}")
    report_lines.append(f"Trainable Parameters:  {params.get('trainable', 0):>15,}")
    report_lines.append(f"Non-trainable Params:  {params.get('non_trainable', 0):>15,}")
    report_lines.append(f"Model Size (float32):  {params.get('total', 0) * 4 / (1024**2):>15.2f} MB")
    report_lines.append("")
    
    # FLOPs
    if 'flops' in profiling_results:
        flops = profiling_results['flops']
        report_lines.append("COMPUTATIONAL COMPLEXITY")
        report_lines.append("-" * 70)
        if flops.get('available', False):
            report_lines.append(f"FLOPs:                 {flops.get('flops_formatted', 'N/A'):>15}")
            report_lines.append(f"Method:                {flops.get('method', 'N/A'):>15}")
        else:
            report_lines.append(f"FLOPs:                 {'N/A':>15} (thop not installed)")
        report_lines.append("")
    
    # Device profiles
    for device, profile in profiling_results.get('profiles', {}).items():
        report_lines.append(f"{device.upper()} PERFORMANCE")
        report_lines.append("-" * 70)
        
        # Single inference
        if 'single_inference' in profile and 'error' not in profile['single_inference']:
            single = profile['single_inference']
            report_lines.append(f"Single Inference Time:")
            report_lines.append(f"  Mean:                {single.get('mean_ms', 0):>15.2f} ms")
            report_lines.append(f"  Std:                 {single.get('std_ms', 0):>15.2f} ms")
            report_lines.append(f"  Min:                 {single.get('min_ms', 0):>15.2f} ms")
            report_lines.append(f"  Max:                 {single.get('max_ms', 0):>15.2f} ms")
            report_lines.append(f"  Median:              {single.get('median_ms', 0):>15.2f} ms")
            report_lines.append(f"  P95:                 {single.get('p95_ms', 0):>15.2f} ms")
            report_lines.append(f"  P99:                 {single.get('p99_ms', 0):>15.2f} ms")
            report_lines.append(f"  Throughput:          {single.get('throughput_fps', 0):>15.2f} FPS")
        
        # Memory
        if 'memory' in profile and 'error' not in profile['memory']:
            memory = profile['memory']
            report_lines.append(f"Memory Usage:")
            if device == 'cuda':
                report_lines.append(f"  Model Memory:        {memory.get('model_memory_mb', 0):>15.2f} MB")
                report_lines.append(f"  Peak Memory:         {memory.get('peak_memory_mb', 0):>15.2f} MB")
                report_lines.append(f"  Total Allocated:     {memory.get('total_allocated_mb', 0):>15.2f} MB")
            else:
                report_lines.append(f"  Memory Increase:     {memory.get('memory_increase_mb', 0):>15.2f} MB")
        
        # Batch inference
        if 'batch_inference' in profile and 'error' not in profile['batch_inference']:
            report_lines.append(f"Batch Inference:")
            report_lines.append(f"  {'Batch':>6} | {'Time (ms)':>10} | {'Samples/s':>10} | {'ms/sample':>10}")
            report_lines.append(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
            for bs, stats in profile['batch_inference'].items():
                report_lines.append(f"  {bs:>6} | {stats['mean_ms']:>10.2f} | "
                                  f"{stats['samples_per_second']:>10.2f} | "
                                  f"{stats['ms_per_sample']:>10.2f}")
        
        report_lines.append("")
    
    # GPU speedup
    if 'gpu_speedup' in profiling_results:
        speedup = profiling_results['gpu_speedup']
        report_lines.append("GPU vs CPU COMPARISON")
        report_lines.append("-" * 70)
        report_lines.append(f"CPU Time:              {speedup.get('cpu_time_ms', 0):>15.2f} ms")
        report_lines.append(f"GPU Time:              {speedup.get('gpu_time_ms', 0):>15.2f} ms")
        report_lines.append(f"GPU Speedup:           {speedup.get('speedup', 0):>15.2f}x")
        report_lines.append("")
    
    report_lines.append("=" * 70)
    
    report = "\n".join(report_lines)
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report
