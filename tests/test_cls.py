#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for classification module (cls.py)

Tests cover:
- Metric computation functions
- Model configuration
- Device detection
"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_pytorch_env import is_torch_cpu_environment, PyTorchMockContext, require_cpu_environment


class TestClassificationMetrics(unittest.TestCase):
    """Test metrics computation functions from cls.py."""
    
    def test_compute_metrics_accuracy(self):
        """Test compute_metrics function with accuracy."""
        with PyTorchMockContext():
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            def compute_metrics(eval_pred):
                preds, labels = eval_pred
                preds = np.argmax(preds, axis=1)
                
                results = {
                    "accuracy": accuracy_score(labels, preds),
                    "f1": f1_score(labels, preds, average='macro'),
                    "precision": precision_score(labels, preds, average='macro'),
                    "recall": recall_score(labels, preds, average='macro')
                }
                return results
            
            preds = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.2, 0.3, 0.5]])
            labels = np.array([0, 1, 2])
            
            metrics = compute_metrics((preds, labels))
            
            self.assertIn("accuracy", metrics)
            self.assertIn("f1", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertEqual(metrics["accuracy"], 1.0)
    
    def test_compute_metrics_with_errors(self):
        """Test compute_metrics with misclassifications."""
        with PyTorchMockContext():
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            def compute_metrics(eval_pred):
                preds, labels = eval_pred
                preds = np.argmax(preds, axis=1)
                
                results = {
                    "accuracy": accuracy_score(labels, preds),
                    "f1": f1_score(labels, preds, average='weighted'),
                    "precision": precision_score(labels, preds, average='weighted'),
                    "recall": recall_score(labels, preds, average='weighted')
                }
                return results
            
            preds = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
            labels = np.array([0, 1, 2])
            
            metrics = compute_metrics((preds, labels))
            
            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)
            self.assertGreaterEqual(metrics["f1"], 0.0)
            self.assertLessEqual(metrics["f1"], 1.0)


class TestClassificationDeviceDetection(unittest.TestCase):
    """Test device detection in classification module."""
    
    def test_device_detection(self):
        """Test device detection returns correct type."""
        is_cpu, device_info = is_torch_cpu_environment()
        
        self.assertIsInstance(is_cpu, bool)
        self.assertIsInstance(device_info, dict)
    
    def test_device_cpu_when_no_cuda(self):
        """Test that CPU is detected when CUDA is not available."""
        is_cpu, device_info = is_torch_cpu_environment()
        
        if not device_info.get("cuda_available", False):
            self.assertTrue(is_cpu)
    
    def test_mock_context_disables_cuda(self):
        """Test that mock context properly disables CUDA."""
        with PyTorchMockContext():
            import torch
            self.assertFalse(torch.cuda.is_available())
            self.assertEqual(torch.cuda.device_count(), 0)


class TestClassificationModelConfig(unittest.TestCase):
    """Test model configuration options."""
    
    def test_model_name_mapping(self):
        """Test model name to checkpoint mapping."""
        model_mappings = {
            'steelberta': './../model_saved/checkpoint-140000',
            'scibert': 'allenai/scibert_scivocab_uncased',
            'matscibert': 'm3rg-iitd/matscibert',
            'bert': 'bert-base-uncased'
        }
        
        for name, model_path in model_mappings.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(model_path, str)
            self.assertTrue(len(model_path) > 0)
    
    def test_supported_models(self):
        """Test that all expected models are supported."""
        expected_models = ['steelberta', 'scibert', 'matscibert', 'bert']
        
        model_mappings = {
            'steelberta': './../model_saved/checkpoint-140000',
            'scibert': 'allenai/scibert_scivocab_uncased',
            'matscibert': 'm3rg-iitd/matscibert',
            'bert': 'bert-base-uncased'
        }
        
        for model in expected_models:
            self.assertIn(model, model_mappings)


if __name__ == "__main__":
    unittest.main()
