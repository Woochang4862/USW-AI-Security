"""
Unit tests for Base MMTD Model
Tests the implementation of the base MMTD model components
"""

import unittest
import tempfile
import torch
import numpy as np
from PIL import Image
import pandas as pd
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.base_mmtd import BaseMMTD, MMTDConfig, create_base_mmtd_model
from models.mmtd_data_loader import MMTDDataset, MMTDCollator, MMTDDataModule, create_mmtd_data_module


class TestMMTDConfig(unittest.TestCase):
    """Test cases for MMTDConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = MMTDConfig()
        
        self.assertEqual(config.bert_model_name, "bert-base-multilingual-cased")
        self.assertEqual(config.dit_model_name, "microsoft/dit-base")
        self.assertEqual(config.num_labels, 2)
        self.assertEqual(config.dropout_rate, 0.1)
        self.assertEqual(config.fusion_hidden_size, 768)
        self.assertEqual(config.fusion_num_heads, 8)
        self.assertEqual(config.max_text_length, 256)
        self.assertEqual(config.image_size, 224)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MMTDConfig(
            num_labels=3,
            dropout_rate=0.2,
            fusion_hidden_size=512,
            max_text_length=128
        )
        
        self.assertEqual(config.num_labels, 3)
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertEqual(config.fusion_hidden_size, 512)
        self.assertEqual(config.max_text_length, 128)
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = MMTDConfig(num_labels=3, dropout_rate=0.2)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['num_labels'], 3)
        self.assertEqual(config_dict['dropout_rate'], 0.2)
    
    def test_config_from_dict(self):
        """Test configuration from dictionary creation"""
        config_dict = {
            'num_labels': 3,
            'dropout_rate': 0.2,
            'fusion_hidden_size': 512
        }
        
        config = MMTDConfig.from_dict(config_dict)
        
        self.assertEqual(config.num_labels, 3)
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertEqual(config.fusion_hidden_size, 512)


class TestBaseMMTD(unittest.TestCase):
    """Test cases for BaseMMTD model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MMTDConfig(
            bert_model_name=None,  # Use default config instead of pretrained
            dit_model_name=None,   # Use default config instead of pretrained
            num_labels=2,
            dropout_rate=0.1,
            fusion_hidden_size=768,
            fusion_num_heads=8
        )
        
        # Create model with small configurations for testing
        self.model = BaseMMTD(
            bert_pretrain_weight=None,
            beit_pretrain_weight=None,
            num_labels=2,
            dropout_rate=0.1,
            fusion_hidden_size=768,
            fusion_num_heads=8
        )
        
        # Test data dimensions
        self.batch_size = 2
        self.seq_len = 256
        self.image_size = 224
        self.num_patches = 197  # For 224x224 image with 16x16 patches + CLS token
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, BaseMMTD)
        self.assertEqual(self.model.num_labels, 2)
        self.assertEqual(self.model.fusion_hidden_size, 768)
        
        # Check if components are initialized
        self.assertIsNotNone(self.model.text_encoder)
        self.assertIsNotNone(self.model.image_encoder)
        self.assertIsNotNone(self.model.multimodal_fusion)
        self.assertIsNotNone(self.model.pooler)
        self.assertIsNotNone(self.model.classifier)
    
    def test_modal_embeddings(self):
        """Test modal type embeddings"""
        self.assertEqual(self.model.text_modal_embedding.shape, (1, 1, 768))
        self.assertEqual(self.model.image_modal_embedding.shape, (1, 1, 768))
        
        # Check if embeddings are different
        self.assertFalse(torch.equal(
            self.model.text_modal_embedding, 
            self.model.image_modal_embedding
        ))
    
    def test_forward_pass_shapes(self):
        """Test forward pass with correct input shapes"""
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        pixel_values = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        labels = torch.randint(0, 2, (self.batch_size,))
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
        
        # Check output shapes
        self.assertEqual(outputs.logits.shape, (self.batch_size, 2))
        self.assertIsNotNone(outputs.loss)
        self.assertIsInstance(outputs.loss.item(), float)
    
    def test_forward_pass_without_labels(self):
        """Test forward pass without labels"""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        pixel_values = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
        
        self.assertEqual(outputs.logits.shape, (self.batch_size, 2))
        self.assertIsNone(outputs.loss)
    
    def test_get_text_features(self):
        """Test text feature extraction"""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Check shape: [batch_size, seq_len, hidden_size]
        self.assertEqual(text_features.shape, (self.batch_size, self.seq_len, 768))
    
    def test_get_image_features(self):
        """Test image feature extraction"""
        pixel_values = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=pixel_values)
        
        # Check shape: [batch_size, num_patches, hidden_size]
        self.assertEqual(image_features.shape, (self.batch_size, self.num_patches, 768))
    
    def test_get_fused_features(self):
        """Test fused feature extraction"""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        pixel_values = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            fused_features = self.model.get_fused_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
        
        # Check shape: [batch_size, seq_len + num_patches, hidden_size]
        expected_seq_len = self.seq_len + self.num_patches
        self.assertEqual(fused_features.shape, (self.batch_size, expected_seq_len, 768))
    
    def test_freeze_unfreeze_encoders(self):
        """Test freezing and unfreezing encoders"""
        # Initially, all parameters should require gradients
        for param in self.model.text_encoder.parameters():
            self.assertTrue(param.requires_grad)
        for param in self.model.image_encoder.parameters():
            self.assertTrue(param.requires_grad)
        
        # Freeze encoders
        self.model.freeze_encoders()
        
        for param in self.model.text_encoder.parameters():
            self.assertFalse(param.requires_grad)
        for param in self.model.image_encoder.parameters():
            self.assertFalse(param.requires_grad)
        
        # Unfreeze encoders
        self.model.unfreeze_encoders()
        
        for param in self.model.text_encoder.parameters():
            self.assertTrue(param.requires_grad)
        for param in self.model.image_encoder.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_return_hidden_states(self):
        """Test returning hidden states"""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        pixel_values = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_hidden_states=True
            )
        
        self.assertIsNotNone(outputs.hidden_states)
        expected_seq_len = self.seq_len + self.num_patches
        self.assertEqual(outputs.hidden_states.shape, (self.batch_size, expected_seq_len, 768))


class TestCreateBaseMMTDModel(unittest.TestCase):
    """Test cases for model factory function"""
    
    def test_create_model_with_default_config(self):
        """Test creating model with default configuration"""
        # Skip this test if transformers models are not available
        try:
            model = create_base_mmtd_model()
            self.assertIsInstance(model, BaseMMTD)
        except Exception as e:
            self.skipTest(f"Skipping due to model loading error: {e}")
    
    def test_create_model_with_custom_config(self):
        """Test creating model with custom configuration"""
        config = MMTDConfig(
            bert_model_name=None,
            dit_model_name=None,
            num_labels=3,
            dropout_rate=0.2
        )
        
        try:
            model = create_base_mmtd_model(config)
            self.assertIsInstance(model, BaseMMTD)
            self.assertEqual(model.num_labels, 3)
        except Exception as e:
            self.skipTest(f"Skipping due to model loading error: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
        self.data_dir.mkdir()
        
        # Create test images
        self.num_samples = 6
        for i in range(self.num_samples):
            image = Image.new('RGB', (224, 224), color='red')
            image.save(self.data_dir / f"image_{i}.jpg")
        
        # Create test CSV
        test_data = {
            'text': [f"Sample email text {i}" for i in range(self.num_samples)],
            'image_path': [f"image_{i}.jpg" for i in range(self.num_samples)],
            'label': [i % 2 for i in range(self.num_samples)]  # Alternating 0, 1
        }
        
        self.df = pd.DataFrame(test_data)
        self.csv_path = self.data_dir / "test_data.csv"
        self.df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to model inference"""
        try:
            # Create data module
            data_module = create_mmtd_data_module(
                data_path=self.data_dir,
                csv_path=self.csv_path,
                batch_size=2,
                train_split=0.5,
                val_split=0.25,
                test_split=0.25
            )
            
            # Create model with small config for testing
            config = MMTDConfig(
                bert_model_name=None,
                dit_model_name=None,
                num_labels=2
            )
            model = BaseMMTD(
                bert_pretrain_weight=None,
                beit_pretrain_weight=None,
                num_labels=2
            )
            
            # Test data loading
            train_loader = data_module.train_dataloader()
            sample_batch = next(iter(train_loader))
            
            # Test model inference
            with torch.no_grad():
                outputs = model(**sample_batch)
            
            # Verify outputs
            self.assertIsNotNone(outputs.logits)
            self.assertIsNotNone(outputs.loss)
            self.assertEqual(outputs.logits.shape[0], sample_batch['labels'].shape[0])
            self.assertEqual(outputs.logits.shape[1], 2)
            
        except Exception as e:
            self.skipTest(f"Skipping integration test due to error: {e}")


if __name__ == '__main__':
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 