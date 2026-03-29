#!/usr/bin/env python
# coding: utf-8
"""
PyTorch CPU Detection and Mock Support

This module provides utilities to detect PyTorch CPU environment
and provides mock support for testing without GPU.
"""

import sys
from unittest.mock import MagicMock, patch


def is_torch_cpu_environment():
    """
    Detect if PyTorch is running in CPU-only mode.
    
    Returns:
        tuple: (is_cpu, device_info)
            - is_cpu: bool, True if running on CPU only
            - device_info: dict containing device information
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        is_cpu = not cuda_available or device_count == 0
        
        device_info = {
            "cuda_available": cuda_available,
            "device_count": device_count,
            "is_cpu": is_cpu,
            "torch_version": torch.__version__,
        }
        
        if cuda_available:
            device_info["cuda_version"] = torch.version.cuda
            device_info["device_name"] = torch.cuda.get_device_name(0)
        
        return is_cpu, device_info
    
    except ImportError:
        return True, {"error": "PyTorch not installed"}


def get_mock_torch_device():
    """
    Get a mocked PyTorch device for CPU-only environments.
    
    Returns:
        MagicMock: A mocked device object
    """
    mock_device = MagicMock()
    mock_device.type = "cpu"
    mock_device.__repr__ = lambda self: "MockedCPUDevice"
    return mock_device


class PyTorchMockContext:
    """
    Context manager for mocking PyTorch in CPU environments.
    
    Usage:
        with PyTorchMockContext():
            # Your test code here
            import torch
            model = torch.nn.Linear(10, 10)  # Will work with mocks
    """
    
    def __init__(self, mock_device=True, mock_cuda=True):
        self.mock_device = mock_device
        self.mock_cuda = mock_cuda
        self.patches = []
    
    def __enter__(self):
        if self.mock_cuda:
            cuda_patch = patch('torch.cuda.is_available', return_value=False)
            self.patches.append(cuda_patch)
            cuda_patch.start()
            
            device_count_patch = patch('torch.cuda.device_count', return_value=0)
            self.patches.append(device_count_patch)
            device_count_patch.start()
        
        if self.mock_device:
            device_patch = patch('torch.device', return_value=MagicMock(type='cpu'))
            self.patches.append(device_patch)
            device_patch.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch in reversed(self.patches):
            patch.stop()
        return False


def require_cpu_environment(func):
    """
    Decorator that skips test if CUDA is available.
    
    Use this decorator for tests that should only run in CPU environment.
    """
    import unittest
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_cpu, _ = is_torch_cpu_environment()
        if not is_cpu:
            raise unittest.SkipTest("Test requires CPU environment (CUDA is available)")
        return func(*args, **kwargs)
    
    return wrapper


def require_cuda_environment(func):
    """
    Decorator that skips test if CUDA is not available.
    
    Use this decorator for tests that should only run in CUDA environment.
    """
    import unittest
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_cpu, _ = is_torch_cpu_environment()
        if is_cpu:
            raise unittest.SkipTest("Test requires CUDA environment (CPU only detected)")
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    is_cpu, device_info = is_torch_cpu_environment()
    
    print("=" * 60)
    print("PyTorch Environment Detection")
    print("=" * 60)
    
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print("-" * 60)
    if is_cpu:
        print("Status: Running on CPU-only environment")
        print("Mock support: Enabled")
    else:
        print("Status: CUDA is available")
        print("Mock support: Not needed")
    
    print("=" * 60)
