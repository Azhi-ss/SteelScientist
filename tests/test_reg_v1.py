#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for regression module (reg_v1.py)

Tests cover:
- Utility functions (range_data, eval_model, set_global_seed)
- Model architecture (Unit, FcUnit, Cnn1dUnit, Cnn2dUnit, CustomSimpleModel)
- Dataset class (CustomSimpleDataset)
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
    
    def test_mock_context(self):
        """Test PyTorchMockContext context manager."""
        with PyTorchMockContext():
            import torch
            self.assertFalse(torch.cuda.is_available())
            self.assertEqual(torch.cuda.device_count(), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions from reg_v1.py."""
    
    @require_cpu_environment
    def test_set_global_seed(self):
        """Test global seed setting function."""
        with PyTorchMockContext():
            import torch
            import numpy as np
            import random
            
            with patch.dict('sys.modules', {'transformers': MagicMock(), 'datasets': MagicMock()}):
                from regression.reg_v1 import set_global_seed
            
            set_global_seed(seed=42)
            
            torch_val1 = torch.rand(10)
            set_global_seed(seed=42)
            torch_val2 = torch.rand(10)
            
            self.assertTrue(torch.equal(torch_val1, torch_val2))
    
    @require_cpu_environment
    def test_eval_model(self):
        """Test evaluation metrics function."""
        with PyTorchMockContext():
            import torch
            with patch.dict('sys.modules', {'transformers': MagicMock(), 'datasets': MagicMock(), 'tqdm': MagicMock()}):
                from regression.reg_v1 import eval_model
            
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
    def test_eval_model_with_numpy(self):
        """Test eval_model with numpy arrays."""
        with PyTorchMockContext():
            import numpy as np
            with patch.dict('sys.modules', {'transformers': MagicMock(), 'datasets': MagicMock(), 'tqdm': MagicMock()}):
                from regression.reg_v1 import eval_model
            
            y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
            
            metrics = eval_model(y_true, y_pred)
            
            self.assertGreater(metrics["r2"], 0.9)


class TestNeuralNetworkUnits(unittest.TestCase):
    """Test neural network unit modules."""
    
    @require_cpu_environment
    def test_unit_layer_norm(self):
        """Test Unit class with layer normalization."""
        with PyTorchMockContext():
            import torch
            from torch import nn
            from regression.reg_v1 import Unit
            
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
            from regression.reg_v1 import Unit
            
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
            from regression.reg_v1 import Unit
            
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
            from regression.reg_v1 import FcUnit
            
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
            from regression.reg_v1 import Cnn1dUnit
            
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
            from regression.reg_v1 import Cnn2dUnit
            
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


class TestCustomSimpleModel(unittest.TestCase):
    """Test CustomSimpleModel architecture."""
    
    @require_cpu_environment
    def test_model_forward_pass(self):
        """Test model forward pass."""
        with PyTorchMockContext():
            import torch
            from regression.reg_v1 import CustomSimpleModel
            
            model = CustomSimpleModel(
                simple_layer_list=[128, 64],
                concat_layer_list=[32, 16, 8, 4, 1],
                seq_embed_con1d_list=[8, 16],
                seq_embed_fc_list=[32, 16],
                seq_embed_con2d_list=[8, 16],
                seq_embed_2d_fc_list=[32, 16],
                simple_layer_drop_prob=0.0,
                concat_layer_drop_prob=0.0,
            )
            
            batch_size = 4
            eles = torch.randn(batch_size, 86)
            text_embeds = torch.randn(batch_size, 768)
            com_embeds = torch.randn(batch_size, 768)
            com_tsne_embeds = torch.randn(batch_size, 3)
            action_embeds = torch.randn(batch_size, 768)
            labels = torch.randn(batch_size, 1)
            
            output = model(eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, labels)
            
            self.assertEqual(output.shape, (batch_size, 1))
    
    @require_cpu_environment
    def test_model_training_mode(self):
        """Test model in training mode."""
        with PyTorchMockContext():
            import torch
            from regression.reg_v1 import CustomSimpleModel
            
            model = CustomSimpleModel(
                simple_layer_list=[64],
                concat_layer_list=[16, 8, 4, 1],
                seq_embed_con1d_list=[4, 8],
                seq_embed_fc_list=[16],
                seq_embed_con2d_list=[4, 8],
                seq_embed_2d_fc_list=[16],
                simple_layer_drop_prob=0.1,
                concat_layer_drop_prob=0.1,
            )
            
            model.train()
            self.assertTrue(model.training)
            
            model.eval()
            self.assertFalse(model.training)


class TestCustomSimpleDataset(unittest.TestCase):
    """Test CustomSimpleDataset class."""
    
    @require_cpu_environment
    def test_dataset_length(self):
        """Test dataset __len__ method."""
        with PyTorchMockContext():
            import torch
            import pandas as pd
            from regression.reg_v1 import CustomSimpleDataset
            
            eles = pd.DataFrame(torch.randn(10, 86))
            text_embeds = pd.DataFrame(torch.randn(10, 768))
            com_embeds = pd.DataFrame(torch.randn(10, 768))
            com_tsne_embeds = pd.DataFrame(torch.randn(10, 3))
            action_embeds = pd.DataFrame(torch.randn(10, 768))
            targets = pd.Series(torch.randn(10))
            
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
            import pandas as pd
            from regression.reg_v1 import CustomSimpleDataset
            
            eles = pd.DataFrame(torch.randn(10, 86))
            text_embeds = pd.DataFrame(torch.randn(10, 768))
            com_embeds = pd.DataFrame(torch.randn(10, 768))
            com_tsne_embeds = pd.DataFrame(torch.randn(10, 3))
            action_embeds = pd.DataFrame(torch.randn(10, 768))
            targets = pd.Series(torch.randn(10))
            
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


if __name__ == "__main__":
    unittest.main()
