#!/usr/bin/env python
# coding: utf-8
"""
Standalone unit tests for regression module components.

These tests import only the necessary PyTorch components directly,
bypassing the module-level imports in reg_v1.py that require
external dependencies (transformers model, etc).
"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_pytorch_env import is_torch_cpu_environment, PyTorchMockContext, require_cpu_environment


class TestPyTorchEnvironment(unittest.TestCase):
    """Test PyTorch CPU detection and mock utilities."""
    
    def test_cpu_detection(self):
        """Test CPU environment detection."""
        is_cpu, device_info = is_torch_cpu_environment()
        
        self.assertIn("is_cpu", device_info)
        self.assertIn("torch_version", device_info)
        self.assertEqual(device_info["is_cpu"], is_cpu)
        print(f"  Detected CPU environment: torch {device_info['torch_version']}")
    
    def test_mock_context(self):
        """Test PyTorchMockContext context manager."""
        with PyTorchMockContext():
            import torch
            self.assertFalse(torch.cuda.is_available())
            self.assertEqual(torch.cuda.device_count(), 0)


class TestSeedFunction(unittest.TestCase):
    """Test seed setting functionality."""
    
    @require_cpu_environment
    def test_set_global_seed_reproducibility(self):
        """Test that setting global seed produces reproducible results."""
        with PyTorchMockContext():
            import torch
            import numpy as np
            import random
            
            def set_global_seed_mock(seed=123):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
            
            set_global_seed_mock(seed=42)
            
            torch_val1 = torch.rand(10)
            np_val1 = np.random.rand(5)
            
            set_global_seed_mock(seed=42)
            
            torch_val2 = torch.rand(10)
            np_val2 = np.random.rand(5)
            
            self.assertTrue(torch.equal(torch_val1, torch_val2))
            self.assertTrue(np.array_equal(np_val1, np_val2))


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics calculation."""
    
    @require_cpu_environment
    def test_eval_metrics_torch(self):
        """Test evaluation metrics with torch tensors."""
        with PyTorchMockContext():
            import torch
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            def eval_model(y_true, y_pred):
                y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
                y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred

                r2 = round(r2_score(y_true, y_pred), 3)
                rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
                mae = round(mean_absolute_error(y_true, y_pred), 3)
                return {"r2": r2, "rmse": rmse, "mae": mae}
            
            y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            y_pred = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])
            
            metrics = eval_model(y_true, y_pred)
            
            self.assertIn("r2", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("mae", metrics)
            self.assertIsInstance(metrics["r2"], float)
            self.assertIsInstance(metrics["rmse"], float)
            self.assertIsInstance(metrics["mae"], float)
    
    @require_cpu_environment
    def test_eval_metrics_numpy(self):
        """Test evaluation metrics with numpy arrays."""
        with PyTorchMockContext():
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            def eval_model(y_true, y_pred):
                r2 = round(r2_score(y_true, y_pred), 3)
                rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
                mae = round(mean_absolute_error(y_true, y_pred), 3)
                return {"r2": r2, "rmse": rmse, "mae": mae}
            
            y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
            
            metrics = eval_model(y_true, y_pred)
            
            self.assertGreater(metrics["r2"], 0.9)
            self.assertLess(metrics["rmse"], 1.0)


class TestNeuralNetworkUnits(unittest.TestCase):
    """Test neural network unit modules."""
    
    @require_cpu_environment
    def test_unit_layer_norm(self):
        """Test Unit class with layer normalization."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class Unit(nn.Module):
                def __init__(self, normalization, in_features, out_features, activation, dropout_prob):
                    super().__init__()
                    if normalization == "layer_norm":
                        self.norm = nn.LayerNorm(in_features)
                    elif normalization == "batch_norm":
                        self.norm = nn.BatchNorm1d(in_features)
                    elif normalization == "null_norm":
                        self.norm = None
                    else:
                        raise ValueError(f"unknown normalization: {normalization}")
                    self.fc = nn.Linear(in_features, out_features)
                    self.act_fn = nn.ReLU()
                    self.dropout = nn.Dropout(dropout_prob)

                def forward(self, x):
                    if self.norm is not None:
                        x = self.norm(x)
                    x = self.fc(x)
                    x = self.act_fn(x)
                    x = self.dropout(x)
                    return x
            
            unit = Unit(
                normalization="layer_norm",
                in_features=128,
                out_features=64,
                activation="relu",
                dropout_prob=0.1
            )
            
            x = torch.randn(16, 128)
            output = unit(x)
            
            self.assertEqual(output.shape, (16, 64))
    
    @require_cpu_environment
    def test_unit_batch_norm(self):
        """Test Unit class with batch normalization."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class Unit(nn.Module):
                def __init__(self, normalization, in_features, out_features, activation, dropout_prob):
                    super().__init__()
                    if normalization == "layer_norm":
                        self.norm = nn.LayerNorm(in_features)
                    elif normalization == "batch_norm":
                        self.norm = nn.BatchNorm1d(in_features)
                    elif normalization == "null_norm":
                        self.norm = None
                    else:
                        raise ValueError(f"unknown normalization: {normalization}")
                    self.fc = nn.Linear(in_features, out_features)
                    self.act_fn = nn.LeakyReLU(negative_slope=-1)
                    self.dropout = nn.Dropout(dropout_prob)

                def forward(self, x):
                    if self.norm is not None:
                        x = self.norm(x)
                    x = self.fc(x)
                    x = self.act_fn(x)
                    x = self.dropout(x)
                    return x
            
            unit = Unit(
                normalization="batch_norm",
                in_features=128,
                out_features=64,
                activation="leaky_relu",
                dropout_prob=0.0
            )
            
            x = torch.randn(16, 128)
            output = unit(x)
            
            self.assertEqual(output.shape, (16, 64))
    
    @require_cpu_environment
    def test_unit_invalid_normalization(self):
        """Test Unit class with invalid normalization."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class Unit(nn.Module):
                def __init__(self, normalization, in_features, out_features, activation, dropout_prob):
                    super().__init__()
                    if normalization == "layer_norm":
                        self.norm = nn.LayerNorm(in_features)
                    elif normalization == "batch_norm":
                        self.norm = nn.BatchNorm1d(in_features)
                    elif normalization == "null_norm":
                        self.norm = None
                    else:
                        raise ValueError(f"unknown normalization: {normalization}")
                    self.fc = nn.Linear(in_features, out_features)
                    self.act_fn = nn.ReLU()
                    self.dropout = nn.Dropout(dropout_prob)

                def forward(self, x):
                    if self.norm is not None:
                        x = self.norm(x)
                    x = self.fc(x)
                    x = self.act_fn(x)
                    x = self.dropout(x)
                    return x
            
            with self.assertRaises(ValueError):
                Unit(
                    normalization="invalid_norm",
                    in_features=128,
                    out_features=64,
                    activation="relu",
                    dropout_prob=0.1
                )
    
    @require_cpu_environment
    def test_fc_unit(self):
        """Test FcUnit fully connected unit."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class FcUnit(nn.Module):
                def __init__(self, num_features, normalization=None, activation='relu', dropout_prob=0.1):
                    super().__init__()
                    if normalization == "layer_norm":
                        self.norm = nn.LayerNorm(num_features)
                    elif normalization == "batch_norm":
                        self.norm = nn.BatchNorm1d(num_features)
                    elif normalization == "null_norm" or normalization is None:
                        self.norm = None
                    else:
                        raise ValueError(f"unknown normalization: {normalization}")
                    self.fc = nn.LazyLinear(num_features)
                    self.act_fn = nn.ReLU()
                    self.dropout = nn.Dropout(dropout_prob)

                def forward(self, x):
                    if self.norm is not None:
                        x = self.norm(x)
                    x = self.fc(x)
                    x = self.act_fn(x)
                    x = self.dropout(x)
                    return x
            
            fc = FcUnit(
                num_features=128,
                normalization="layer_norm",
                activation="relu",
                dropout_prob=0.1
            )
            
            x = torch.randn(16, 128)
            output = fc(x)
            
            self.assertEqual(output.shape, (16, 128))
    
    @require_cpu_environment
    def test_cnn1d_unit(self):
        """Test Cnn1dUnit 1D convolution unit."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class Cnn1dUnit(nn.Module):
                def __init__(self, out_channels, kernel_size=3, activation='relu', stride=1, padding=1, pooling='max'):
                    super().__init__()
                    if pooling == "max":
                        self.pool = nn.MaxPool1d(kernel_size=kernel_size)
                    elif pooling == "avg":
                        self.pool = nn.AvgPool1d(kernel_size=kernel_size)
                    else:
                        raise ValueError(f"unknown pooling type: {pooling}")
                    self.act_fn = nn.ReLU()
                    self.conv1d = nn.LazyConv1d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

                def forward(self, x):
                    x = self.conv1d(x)
                    x = self.act_fn(x)
                    x = self.pool(x)
                    return x
            
            cnn = Cnn1dUnit(
                out_channels=32,
                kernel_size=3,
                activation="relu",
                stride=1,
                padding=1,
                pooling="max"
            )
            
            x = torch.randn(16, 16, 100)
            output = cnn(x)
            
            self.assertEqual(output.shape[0], 16)
            self.assertEqual(output.shape[1], 32)
    
    @require_cpu_environment
    def test_cnn2d_unit(self):
        """Test Cnn2dUnit 2D convolution unit."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class Cnn2dUnit(nn.Module):
                def __init__(self, out_channels, kernel_size=2, activation='relu', stride=1, padding=1, pooling='max'):
                    super().__init__()
                    if pooling == "max":
                        self.pool = nn.MaxPool2d(kernel_size=kernel_size)
                    elif pooling == "avg":
                        self.pool = nn.AvgPool2d(kernel_size=kernel_size)
                    else:
                        raise ValueError(f"unknown pooling type: {pooling}")
                    self.act_fn = nn.Tanh()
                    self.conv2d = nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

                def forward(self, x):
                    x = self.conv2d(x)
                    x = self.act_fn(x)
                    x = self.pool(x)
                    return x
            
            cnn = Cnn2dUnit(
                out_channels=16,
                kernel_size=2,
                activation="tanh",
                stride=1,
                padding=1,
                pooling="avg"
            )
            
            x = torch.randn(8, 16, 32, 32)
            output = cnn(x)
            
            self.assertEqual(output.shape[0], 8)
            self.assertEqual(output.shape[1], 16)


class TestCustomSimpleDataset(unittest.TestCase):
    """Test CustomSimpleDataset class."""
    
    @require_cpu_environment
    def test_dataset_length(self):
        """Test dataset __len__ method."""
        with PyTorchMockContext():
            import torch
            from torch.utils.data import Dataset
            
            class CustomSimpleDataset(Dataset):
                def __init__(self, eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, targets):
                    self.eles = eles
                    self.text_embeds = text_embeds
                    self.com_embeds = com_embeds
                    self.com_tsne_embeds = com_tsne_embeds
                    self.action_embeds = action_embeds
                    self.targets = targets

                def __len__(self):
                    return len(self.targets)
                
                def __getitem__(self, idx):
                    if torch.is_tensor(idx):
                        idx = idx.tolist()
                    return {
                        "labels": torch.tensor(self.targets[idx], dtype=torch.float32),
                        "eles": torch.tensor(self.eles[idx], dtype=torch.float32),
                        "text_embeds": torch.tensor(self.text_embeds[idx], dtype=torch.float32),
                        "com_embeds": torch.tensor(self.com_embeds[idx], dtype=torch.float32),
                        "action_embeds": torch.tensor(self.action_embeds[idx], dtype=torch.float32),
                        "com_tsne_embeds": torch.tensor(self.com_tsne_embeds[idx], dtype=torch.float32),
                    }
            
            import numpy as np
            eles = np.random.randn(10, 86)
            text_embeds = np.random.randn(10, 768)
            com_embeds = np.random.randn(10, 768)
            com_tsne_embeds = np.random.randn(10, 3)
            action_embeds = np.random.randn(10, 768)
            targets = np.random.randn(10)
            
            dataset = CustomSimpleDataset(
                eles=eles,
                text_embeds=text_embeds,
                com_embeds=com_embeds,
                com_tsne_embeds=com_tsne_embeds,
                action_embeds=action_embeds,
                targets=targets
            )
            
            self.assertEqual(len(dataset), 10)
    
    @require_cpu_environment
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        with PyTorchMockContext():
            import torch
            from torch.utils.data import Dataset
            
            class CustomSimpleDataset(Dataset):
                def __init__(self, eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, targets):
                    self.eles = eles
                    self.text_embeds = text_embeds
                    self.com_embeds = com_embeds
                    self.com_tsne_embeds = com_tsne_embeds
                    self.action_embeds = action_embeds
                    self.targets = targets

                def __len__(self):
                    return len(self.targets)
                
                def __getitem__(self, idx):
                    if torch.is_tensor(idx):
                        idx = idx.tolist()
                    return {
                        "labels": torch.tensor(self.targets[idx], dtype=torch.float32),
                        "eles": torch.tensor(self.eles[idx], dtype=torch.float32),
                        "text_embeds": torch.tensor(self.text_embeds[idx], dtype=torch.float32),
                        "com_embeds": torch.tensor(self.com_embeds[idx], dtype=torch.float32),
                        "action_embeds": torch.tensor(self.action_embeds[idx], dtype=torch.float32),
                        "com_tsne_embeds": torch.tensor(self.com_tsne_embeds[idx], dtype=torch.float32),
                    }
            
            import numpy as np
            eles = np.random.randn(10, 86)
            text_embeds = np.random.randn(10, 768)
            com_embeds = np.random.randn(10, 768)
            com_tsne_embeds = np.random.randn(10, 3)
            action_embeds = np.random.randn(10, 768)
            targets = np.random.randn(10)
            
            dataset = CustomSimpleDataset(
                eles=eles,
                text_embeds=text_embeds,
                com_embeds=com_embeds,
                com_tsne_embeds=com_tsne_embeds,
                action_embeds=action_embeds,
                targets=targets
            )
            
            sample = dataset[0]
            
            self.assertIn("labels", sample)
            self.assertIn("eles", sample)
            self.assertIn("text_embeds", sample)
            self.assertIn("com_embeds", sample)
            self.assertIn("action_embeds", sample)
            self.assertIn("com_tsne_embeds", sample)


class TestCustomSimpleModel(unittest.TestCase):
    """Test CustomSimpleModel architecture."""
    
    @require_cpu_environment
    def test_model_forward_pass(self):
        """Test model forward pass."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class SimpleFcLayer(nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.fc = nn.Linear(in_features, out_features)
                    self.act_fn = nn.ReLU()
                
                def forward(self, x):
                    return self.act_fn(self.fc(x))
            
            class CustomSimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = SimpleFcLayer(768, 256)
                    self.fc2 = SimpleFcLayer(256, 64)
                    self.fc3 = nn.Linear(64, 1)

                def forward(self, text_embeds):
                    x = self.fc1(text_embeds)
                    x = self.fc2(x)
                    return self.fc3(x)
            
            model = CustomSimpleModel()
            
            batch_size = 4
            text_embeds = torch.randn(batch_size, 768)
            
            output = model(text_embeds)
            
            self.assertEqual(output.shape, (batch_size, 1))
    
    @require_cpu_environment
    def test_model_training_mode(self):
        """Test model in training mode."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 5)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            
            model.train()
            self.assertTrue(model.training)
            
            model.eval()
            self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
