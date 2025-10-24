"""
Model Performance Profiling Module

This module provides comprehensive performance profiling for deepfake detection models.
"""

from .profile_model import (
    profile_model,
    measure_inference_time,
    measure_batch_inference,
    measure_memory_usage,
    count_parameters,
    calculate_flops,
    generate_performance_report
)

__all__ = [
    'profile_model',
    'measure_inference_time',
    'measure_batch_inference',
    'measure_memory_usage',
    'count_parameters',
    'calculate_flops',
    'generate_performance_report'
]
