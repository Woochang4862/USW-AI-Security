"""
Unit tests for data preprocessing pipeline
Tests all components of the EDP dataset preprocessing
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Download required NLTK data for tests
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from data_preprocessing import (
    TextPreprocessor, 
    ImagePreprocessor, 
    DataSplitter, 
    EDPDatasetPreprocessor
)
from data_loader import EDPDataset, EDPCollator, EDPDataModule


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.text_processor = TextPreprocessor()
        self.sample_texts = [
            "This is a normal email message.",
            "URGENT!!! Click here: http://spam.com for FREE MONEY!!!",
            "Hello, this email contains HTML <b>tags</b> and email@example.com",
            "Texto en español con caracteres especiales: ñáéíóú",
            ""  # Empty text
        ]
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        # Test normal text
        cleaned = self.text_processor.clean_text(self.sample_texts[0])
        self.assertEqual(cleaned, "this is a normal email message.")
        
        # Test HTML removal
        cleaned = self.text_processor.clean_text(self.sample_texts[2])
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("</b>", cleaned)
        self.assertNotIn("email@example.com", cleaned)
        
        # Test empty text
        cleaned = self.text_processor.clean_text(self.sample_texts[4])
        self.assertEqual(cleaned, "")
        
        # Test non-string input
        cleaned = self.text_processor.clean_text(None)
        self.assertEqual(cleaned, "")
    
    def test_tokenize_and_filter(self):
        """Test tokenization and filtering"""
        text = "This is a test message with stopwords."
        tokens = self.text_processor.tokenize_and_filter(text)
        
        # Check that tokens are returned
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Check stopword removal
        tokens_with_stopwords = self.text_processor.tokenize_and_filter(text, remove_stopwords=False)
        tokens_without_stopwords = self.text_processor.tokenize_and_filter(text, remove_stopwords=True)
        self.assertGreaterEqual(len(tokens_with_stopwords), len(tokens_without_stopwords))
    
    def test_bert_tokenize(self):
        """Test BERT tokenization"""
        text = "This is a test message."
        result = self.text_processor.bert_tokenize(text)
        
        # Check required keys
        required_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], torch.Tensor)
        
        # Check tensor shapes
        max_length = self.text_processor.max_length
        self.assertEqual(result['input_ids'].shape, (max_length,))
        self.assertEqual(result['attention_mask'].shape, (max_length,))
        self.assertEqual(result['token_type_ids'].shape, (max_length,))
    
    def test_preprocess_text(self):
        """Test complete text preprocessing pipeline"""
        text = "This is a test email message!"
        result = self.text_processor.preprocess_text(text)
        
        # Check required keys
        required_keys = ['cleaned_text', 'tokens', 'stemmed_tokens', 'bert_tokens']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check types
        self.assertIsInstance(result['cleaned_text'], str)
        self.assertIsInstance(result['tokens'], list)
        self.assertIsInstance(result['stemmed_tokens'], list)
        self.assertIsInstance(result['bert_tokens'], dict)


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_processor = ImagePreprocessor()
        
        # Create a temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test image
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_image(self):
        """Test image loading"""
        # Test valid image
        image = self.image_processor.load_image(self.test_image_path)
        self.assertIsNotNone(image)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, 'RGB')
        
        # Test invalid path
        image = self.image_processor.load_image("nonexistent.jpg")
        self.assertIsNone(image)
    
    def test_preprocess_image_basic(self):
        """Test basic image preprocessing"""
        image = Image.new('RGB', (100, 100), color='blue')
        tensor = self.image_processor.preprocess_image_basic(image)
        
        self.assertIsInstance(tensor, torch.Tensor)
        # Check shape: (channels, height, width)
        expected_shape = (3, *self.image_processor.target_size)
        self.assertEqual(tensor.shape, expected_shape)
    
    def test_beit_preprocess(self):
        """Test BEiT preprocessing"""
        image = Image.new('RGB', (100, 100), color='green')
        result = self.image_processor.beit_preprocess(image)
        
        self.assertIn('pixel_values', result)
        self.assertIsInstance(result['pixel_values'], torch.Tensor)
        # BEiT expects (channels, height, width)
        self.assertEqual(len(result['pixel_values'].shape), 3)
    
    def test_preprocess_image(self):
        """Test complete image preprocessing pipeline"""
        result = self.image_processor.preprocess_image(self.test_image_path)
        
        self.assertIsNotNone(result)
        
        # Check required keys
        required_keys = ['basic_tensor', 'beit_pixel_values']
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], torch.Tensor)
        
        # Test with augmentation
        result_aug = self.image_processor.preprocess_image(self.test_image_path, augment=True)
        self.assertIn('augmented_tensor', result_aug)


class TestDataSplitter(unittest.TestCase):
    """Test cases for DataSplitter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_splitter = DataSplitter(n_splits=3, random_state=42)
        
        # Create test dataframe
        self.test_df = pd.DataFrame({
            'text': [f"Sample text {i}" for i in range(100)],
            'image_path': [f"image_{i}.jpg" for i in range(100)],
            'label': [i % 2 for i in range(100)]  # Binary labels
        })
    
    def test_create_splits(self):
        """Test cross-validation split creation"""
        splits = self.data_splitter.create_splits(self.test_df)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that all indices are covered
        all_indices = set()
        for train_idx, test_idx in splits:
            all_indices.update(train_idx)
            all_indices.update(test_idx)
        
        self.assertEqual(len(all_indices), len(self.test_df))
    
    def test_get_fold_data(self):
        """Test getting data for specific fold"""
        train_df, test_df = self.data_splitter.get_fold_data(self.test_df, 0)
        
        # Check that we get dataframes
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        
        # Check that train and test don't overlap
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)
        
        # Check total size
        self.assertEqual(len(train_df) + len(test_df), len(self.test_df))


class TestEDPDataset(unittest.TestCase):
    """Test cases for EDPDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock preprocessed data
        self.preprocessed_data = {
            'text_data': [
                {
                    'cleaned_text': f'sample text {i}',
                    'tokens': [f'token{j}' for j in range(5)],
                    'stemmed_tokens': [f'stem{j}' for j in range(5)],
                    'bert_tokens': {
                        'input_ids': torch.randint(0, 1000, (512,)),
                        'attention_mask': torch.ones(512),
                        'token_type_ids': torch.zeros(512)
                    }
                } for i in range(10)
            ],
            'image_data': [
                {
                    'basic_tensor': torch.randn(3, 224, 224),
                    'beit_pixel_values': torch.randn(3, 224, 224)
                } for i in range(10)
            ],
            'labels': [i % 2 for i in range(10)],
            'indices': list(range(10))
        }
        
        self.dataset = EDPDataset(self.preprocessed_data)
    
    def test_dataset_length(self):
        """Test dataset length"""
        self.assertEqual(len(self.dataset), 10)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        item = self.dataset[0]
        
        # Check required keys
        required_keys = [
            'input_ids', 'attention_mask', 'token_type_ids',
            'pixel_values', 'labels', 'text_tokens', 'cleaned_text', 'original_index'
        ]
        for key in required_keys:
            self.assertIn(key, item)
        
        # Check tensor types
        tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'pixel_values', 'labels']
        for key in tensor_keys:
            self.assertIsInstance(item[key], torch.Tensor)


class TestEDPCollator(unittest.TestCase):
    """Test cases for EDPCollator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collator = EDPCollator()
        
        # Create mock batch
        self.batch = [
            {
                'input_ids': torch.randint(0, 1000, (512,)),
                'attention_mask': torch.ones(512),
                'token_type_ids': torch.zeros(512),
                'pixel_values': torch.randn(3, 224, 224),
                'labels': torch.tensor(0),
                'text_tokens': ['token1', 'token2'],
                'cleaned_text': 'sample text',
                'original_index': i
            } for i in range(4)
        ]
    
    def test_collate_batch(self):
        """Test batch collation"""
        batched = self.collator(self.batch)
        
        # Check tensor shapes
        self.assertEqual(batched['input_ids'].shape, (4, 512))
        self.assertEqual(batched['attention_mask'].shape, (4, 512))
        self.assertEqual(batched['token_type_ids'].shape, (4, 512))
        self.assertEqual(batched['pixel_values'].shape, (4, 3, 224, 224))
        self.assertEqual(batched['labels'].shape, (4,))
        
        # Check metadata
        self.assertEqual(len(batched['text_tokens']), 4)
        self.assertEqual(len(batched['cleaned_text']), 4)
        self.assertEqual(len(batched['original_indices']), 4)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete preprocessing pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
        self.data_dir.mkdir()
        
        # Create pics directory
        pics_dir = self.data_dir / "pics"
        pics_dir.mkdir()
        
        # Create test images (more samples for cross-validation)
        num_samples = 20  # Increased from 5 to 20
        for i in range(num_samples):
            image = Image.new('RGB', (100, 100), color='red')
            image.save(pics_dir / f"image_{i}.jpg")
        
        # Create test CSV with more balanced data
        test_data = {
            'text': [f"Sample email text {i}" for i in range(num_samples)],
            'image_path': [f"image_{i}.jpg" for i in range(num_samples)],
            'label': ['spam' if i % 2 == 0 else 'ham' for i in range(num_samples)]
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(self.data_dir / "EDP.csv", index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline"""
        # Initialize preprocessor
        preprocessor = EDPDatasetPreprocessor(str(self.data_dir))
        
        # Load dataset
        df = preprocessor.load_dataset()
        self.assertEqual(len(df), 20)
        
        # Validate data
        df = preprocessor.validate_data(df)
        self.assertEqual(len(df), 20)  # All images should exist
        
        # Create cross-validation splits
        cv_splits = preprocessor.create_cross_validation_splits(df)
        # The actual number of splits will be min(n_splits, min_samples_per_class)
        # With 20 samples and 2 classes (10 each), we get min(5, 10) = 5 splits
        self.assertEqual(len(cv_splits), 5)  # 5-fold CV (adjusted from 20)
        
        # Test preprocessing on first fold
        train_df, test_df = cv_splits[0]
        
        # Preprocess data
        train_data = preprocessor.preprocess_dataset(train_df, augment=False)
        
        # Check preprocessed data structure
        required_keys = ['text_data', 'image_data', 'labels', 'indices']
        for key in required_keys:
            self.assertIn(key, train_data)
        
        # Check data consistency
        self.assertEqual(len(train_data['text_data']), len(train_data['image_data']))
        self.assertEqual(len(train_data['text_data']), len(train_data['labels']))


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTextPreprocessor,
        TestImagePreprocessor,
        TestDataSplitter,
        TestEDPDataset,
        TestEDPCollator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1) 