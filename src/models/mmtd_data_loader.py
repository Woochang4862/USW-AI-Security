"""
MMTD Data Loader Implementation
Based on the original MMTD Email_dataset.py with improvements for interpretability research
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from transformers import (
    BertTokenizerFast, 
    BeitFeatureExtractor,
    AutoTokenizer,
    AutoFeatureExtractor
)
from torchvision.transforms import (
    Resize, RandomResizedCrop, Normalize, Compose, 
    CenterCrop, ToTensor, RandomHorizontalFlip
)

logger = logging.getLogger(__name__)


class MMTDDataset(Dataset):
    """
    Dataset class for MMTD multimodal email data
    Based on the original EDPDataset with enhancements
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        data_df: pd.DataFrame,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ):
        """
        Initialize MMTD Dataset
        
        Args:
            data_path: Path to the directory containing images
            data_df: DataFrame with columns ['text', 'image_path', 'label']
            transform: Optional transform for images
            target_transform: Optional transform for labels
        """
        super(MMTDDataset, self).__init__()
        self.data_path = Path(data_path)
        self.data = data_df.copy()
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Initialized MMTDDataset with {len(self.data)} samples")
    
    def _validate_data(self):
        """Validate that the dataframe has required columns"""
        required_columns = ['texts', 'pics', 'labels']  # Match EDP dataset columns
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"✅ Data validation passed. Columns: {self.data.columns.tolist()}")
        logger.info(f"📊 Dataset size: {len(self.data)} samples")
    
    def __getitem__(self, idx: int) -> Tuple[str, Image.Image, int]:
        """Get a single sample from the dataset"""
        row = self.data.iloc[idx]
        
        # Get text (handle NaN values)
        text = row['texts'] if pd.notna(row['texts']) else ""
        
        # Get image
        image_path = self.data_path / row['pics']
        if not image_path.exists():
            # Create a dummy image if file doesn't exist
            image = Image.new('RGB', (224, 224), color='white')
            logger.warning(f"Image not found: {image_path}, using dummy image")
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Error loading image {image_path}: {e}, using dummy image")
                image = Image.new('RGB', (224, 224), color='white')
        
        # Get label
        label = int(row['labels'])
        
        return text, image, label
    
    def __len__(self) -> int:
        """Return the number of samples"""
        return len(self.data)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes"""
        return self.data['labels'].value_counts().to_dict()
    
    def get_sample_by_class(self, class_label: int, n_samples: int = 5) -> List[int]:
        """Get sample indices for a specific class"""
        class_indices = self.data[self.data['labels'] == class_label].index.tolist()
        return class_indices[:n_samples]


class MMTDCollator:
    """
    Data collator for MMTD model
    Based on the original EDPCollator with improvements
    """
    
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        max_length: int = 256,
        image_size: int = 224
    ):
        """
        Initialize MMTD Collator
        
        Args:
            tokenizer: Text tokenizer (default: BERT multilingual)
            feature_extractor: Image feature extractor (default: DiT)
            max_length: Maximum text sequence length
            image_size: Target image size
        """
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        else:
            self.tokenizer = tokenizer
        
        # Initialize feature extractor
        if feature_extractor is None:
            self.feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/dit-base')
        else:
            self.feature_extractor = feature_extractor
        
        self.max_length = max_length
        self.image_size = image_size
        
        logger.info(f"Initialized MMTDCollator with max_length={max_length}, image_size={image_size}")
    
    def text_process(self, text_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process text data
        
        Args:
            text_list: List of text strings
            
        Returns:
            Dictionary with tokenized text tensors
        """
        text_tensor = self.tokenizer(
            text_list,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        return text_tensor
    
    def image_process(self, image_list: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Process image data
        
        Args:
            image_list: List of PIL Images
            
        Returns:
            Dictionary with processed image tensors
        """
        pixel_values = self.feature_extractor(image_list, return_tensors='pt')
        return pixel_values
    
    def __call__(self, batch: List[Tuple[str, Image.Image, int]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples
        
        Args:
            batch: List of (text, image, label) tuples
            
        Returns:
            Dictionary with batched tensors
        """
        text_list, image_list, label_list = zip(*batch)
        
        # Process text
        text_tensors = self.text_process(list(text_list))
        
        # Process images
        image_tensors = self.image_process(image_list)
        
        # Process labels
        labels = torch.LongTensor(label_list)
        
        # Combine all inputs
        inputs = {}
        inputs.update(text_tensors)
        inputs.update(image_tensors)
        inputs['labels'] = labels
        
        return inputs


class MMTDTextOnlyCollator(MMTDCollator):
    """Collator for text-only processing"""
    
    def __call__(self, batch: List[Tuple[str, Image.Image, int]]) -> Dict[str, torch.Tensor]:
        """Process only text data"""
        text_list, _, label_list = zip(*batch)
        
        text_tensors = self.text_process(list(text_list))
        labels = torch.LongTensor(label_list)
        
        inputs = {}
        inputs.update(text_tensors)
        inputs['labels'] = labels
        
        return inputs


class MMTDImageOnlyCollator(MMTDCollator):
    """Collator for image-only processing"""
    
    def __call__(self, batch: List[Tuple[str, Image.Image, int]]) -> Dict[str, torch.Tensor]:
        """Process only image data"""
        _, image_list, label_list = zip(*batch)
        
        image_tensors = self.image_process(image_list)
        labels = torch.LongTensor(label_list)
        
        inputs = {}
        inputs.update(image_tensors)
        inputs['labels'] = labels
        
        return inputs


class MMTDDataModule:
    """
    Data module for managing MMTD datasets and dataloaders
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        collator_type: str = "multimodal",  # "multimodal", "text_only", "image_only"
        **collator_kwargs
    ):
        """
        Initialize MMTD Data Module
        
        Args:
            data_path: Path to image directory
            train_df: Training dataframe
            val_df: Validation dataframe  
            test_df: Test dataframe
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            collator_type: Type of collator to use
            **collator_kwargs: Additional arguments for collator
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create datasets
        self.train_dataset = MMTDDataset(data_path, train_df)
        self.val_dataset = MMTDDataset(data_path, val_df)
        self.test_dataset = MMTDDataset(data_path, test_df)
        
        # Create collator
        if collator_type == "multimodal":
            self.collator = MMTDCollator(**collator_kwargs)
        elif collator_type == "text_only":
            self.collator = MMTDTextOnlyCollator(**collator_kwargs)
        elif collator_type == "image_only":
            self.collator = MMTDImageOnlyCollator(**collator_kwargs)
        else:
            raise ValueError(f"Unknown collator_type: {collator_type}")
        
        logger.info(f"Initialized MMTDDataModule with {len(self.train_dataset)} train, "
                   f"{len(self.val_dataset)} val, {len(self.test_dataset)} test samples")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=False
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        train_labels = self.train_dataset.data['labels'].values
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        
        # Calculate inverse frequency weights
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)
    
    def get_sample_batch(self, dataset_type: str = "train", batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        if dataset_type == "train":
            dataset = self.train_dataset
        elif dataset_type == "val":
            dataset = self.val_dataset
        elif dataset_type == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        # Get random samples
        indices = np.random.choice(len(dataset), size=min(batch_size, len(dataset)), replace=False)
        samples = [dataset[i] for i in indices]
        
        # Collate samples
        return self.collator(samples)


def create_mmtd_data_module(
    data_path: Union[str, Path],
    csv_path: Union[str, Path],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    max_samples: Optional[int] = None,
    **kwargs
) -> MMTDDataModule:
    """
    Factory function to create MMTDDataModule from CSV file
    
    Args:
        data_path: Path to image directory
        csv_path: Path to CSV file with data
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        random_state: Random seed
        max_samples: Maximum samples to use (for testing)
        **kwargs: Additional arguments for MMTDDataModule
        
    Returns:
        MMTDDataModule instance
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Apply max_samples if specified
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=random_state).reset_index(drop=True)
        logger.info(f"Limited dataset to {len(df)} samples")
    
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Train, val, and test splits must sum to 1.0")
    
    # Split data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    logger.info(f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Remove max_samples from kwargs since it's not needed for MMTDDataModule
    kwargs.pop('max_samples', None)
    
    return MMTDDataModule(
        data_path=data_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        **kwargs
    )


if __name__ == "__main__":
    # Test data module creation
    import tempfile
    
    # Create dummy data for testing
    dummy_data = {
        'text': ['Sample email text 1', 'Sample email text 2', 'Sample email text 3'],
        'image_path': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
        'label': [0, 1, 0]
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy CSV
        df = pd.DataFrame(dummy_data)
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Create dummy images
        for img_name in dummy_data['image_path']:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(os.path.join(temp_dir, img_name))
        
        try:
            # Test data module creation
            data_module = create_mmtd_data_module(
                data_path=temp_dir,
                csv_path=csv_path,
                batch_size=2
            )
            
            # Test dataloaders
            train_loader = data_module.train_dataloader()
            sample_batch = next(iter(train_loader))
            
            print(f"Sample batch keys: {sample_batch.keys()}")
            print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
            print(f"Pixel values shape: {sample_batch['pixel_values'].shape}")
            print(f"Labels shape: {sample_batch['labels'].shape}")
            print("✅ MMTDDataModule test passed!")
            
        except Exception as e:
            print(f"❌ Data module test failed: {e}") 