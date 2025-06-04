"""
PyTorch DataLoader and Dataset classes for Interpretable Multimodal Spam Detection
Handles efficient loading and batching of preprocessed EDP dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EDPDataset(Dataset):
    """
    PyTorch Dataset class for EDP (Email Data with Pictures) dataset
    """
    
    def __init__(self, 
                 preprocessed_data: Dict[str, List],
                 transform_text: Optional[callable] = None,
                 transform_image: Optional[callable] = None):
        """
        Initialize EDP Dataset
        
        Args:
            preprocessed_data: Dictionary containing preprocessed text and image data
            transform_text: Optional text transformation function
            transform_image: Optional image transformation function
        """
        self.text_data = preprocessed_data['text_data']
        self.image_data = preprocessed_data['image_data']
        self.labels = preprocessed_data['labels']
        self.indices = preprocessed_data['indices']
        
        self.transform_text = transform_text
        self.transform_image = transform_image
        
        # Validate data consistency
        assert len(self.text_data) == len(self.image_data) == len(self.labels), \
            "Text, image, and label data must have the same length"
    
    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing text features, image features, and label
        """
        # Get text data
        text_item = self.text_data[idx]
        
        # Get image data
        image_item = self.image_data[idx]
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply transforms if provided
        if self.transform_text:
            text_item = self.transform_text(text_item)
        
        if self.transform_image:
            image_item = self.transform_image(image_item)
        
        return {
            'input_ids': text_item['bert_tokens']['input_ids'],
            'attention_mask': text_item['bert_tokens']['attention_mask'],
            'token_type_ids': text_item['bert_tokens']['token_type_ids'],
            'pixel_values': image_item['beit_pixel_values'],
            'labels': label,
            'text_tokens': text_item['tokens'],
            'cleaned_text': text_item['cleaned_text'],
            'original_index': self.indices[idx]
        }


class EDPCollator:
    """
    Custom collate function for batching EDP dataset samples
    """
    
    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator
        
        Args:
            pad_token_id: Token ID used for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary with padded tensors
        """
        # Extract individual components
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Stack tensors
        batched_data = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'token_type_ids': torch.stack(token_type_ids),
            'pixel_values': torch.stack(pixel_values),
            'labels': torch.stack(labels)
        }
        
        # Add metadata (not stacked)
        batched_data['text_tokens'] = [item['text_tokens'] for item in batch]
        batched_data['cleaned_text'] = [item['cleaned_text'] for item in batch]
        batched_data['original_indices'] = [item['original_index'] for item in batch]
        
        return batched_data


class EDPDataModule:
    """
    Data module for managing EDP dataset loading and cross-validation
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 shuffle_train: bool = True):
        """
        Initialize data module
        
        Args:
            data_dir: Directory containing preprocessed data
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            shuffle_train: Whether to shuffle training data
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        
        self.collator = EDPCollator()
        
        # Storage for datasets
        self.train_datasets = []
        self.test_datasets = []
        self.train_loaders = []
        self.test_loaders = []
    
    def load_fold_data(self, fold_idx: int) -> Tuple[Dataset, Dataset]:
        """
        Load preprocessed data for a specific fold
        
        Args:
            fold_idx: Fold index (0-based)
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_file = self.data_dir / f"preprocessed_train_fold{fold_idx + 1}.pt"
        test_file = self.data_dir / f"preprocessed_test_fold{fold_idx + 1}.pt"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test data file not found: {test_file}")
        
        # Load preprocessed data
        train_data = torch.load(train_file)
        test_data = torch.load(test_file)
        
        # Create datasets
        train_dataset = EDPDataset(train_data)
        test_dataset = EDPDataset(test_data)
        
        logger.info(f"Loaded fold {fold_idx + 1}: Train={len(train_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def create_data_loaders(self, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for a specific fold
        
        Args:
            fold_idx: Fold index (0-based)
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        train_dataset, test_dataset = self.load_fold_data(fold_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=False
        )
        
        return train_loader, test_loader
    
    def setup_all_folds(self, num_folds: int = 5):
        """
        Set up datasets and loaders for all folds
        
        Args:
            num_folds: Number of cross-validation folds
        """
        self.train_datasets = []
        self.test_datasets = []
        self.train_loaders = []
        self.test_loaders = []
        
        for fold_idx in range(num_folds):
            try:
                train_dataset, test_dataset = self.load_fold_data(fold_idx)
                train_loader, test_loader = self.create_data_loaders(fold_idx)
                
                self.train_datasets.append(train_dataset)
                self.test_datasets.append(test_dataset)
                self.train_loaders.append(train_loader)
                self.test_loaders.append(test_loader)
                
            except FileNotFoundError as e:
                logger.warning(f"Could not load fold {fold_idx + 1}: {e}")
                continue
        
        logger.info(f"Successfully set up {len(self.train_loaders)} folds")
    
    def get_fold_loaders(self, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
        """
        Get DataLoaders for a specific fold
        
        Args:
            fold_idx: Fold index (0-based)
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        if fold_idx >= len(self.train_loaders):
            raise IndexError(f"Fold {fold_idx} not available. Only {len(self.train_loaders)} folds loaded.")
        
        return self.train_loaders[fold_idx], self.test_loaders[fold_idx]
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded datasets
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.train_datasets:
            return {"error": "No datasets loaded"}
        
        stats = {
            "num_folds": len(self.train_datasets),
            "fold_stats": []
        }
        
        for i, (train_ds, test_ds) in enumerate(zip(self.train_datasets, self.test_datasets)):
            fold_stat = {
                "fold": i + 1,
                "train_size": len(train_ds),
                "test_size": len(test_ds),
                "total_size": len(train_ds) + len(test_ds)
            }
            
            # Calculate label distribution for training set
            train_labels = [train_ds.labels[j] for j in range(len(train_ds))]
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            fold_stat["train_label_distribution"] = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            # Calculate label distribution for test set
            test_labels = [test_ds.labels[j] for j in range(len(test_ds))]
            unique_labels, counts = np.unique(test_labels, return_counts=True)
            fold_stat["test_label_distribution"] = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            stats["fold_stats"].append(fold_stat)
        
        return stats


class DataAugmentation:
    """
    Additional data augmentation techniques for text and images
    """
    
    def __init__(self):
        """Initialize data augmentation"""
        pass
    
    @staticmethod
    def text_augmentation(text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply text augmentation techniques
        
        Args:
            text_data: Dictionary containing text features
            
        Returns:
            Augmented text data
        """
        # For now, return original data
        # Can be extended with techniques like:
        # - Synonym replacement
        # - Random insertion
        # - Random swap
        # - Random deletion
        return text_data
    
    @staticmethod
    def image_augmentation(image_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply image augmentation techniques
        
        Args:
            image_data: Dictionary containing image tensors
            
        Returns:
            Augmented image data
        """
        # For now, return original data
        # Additional augmentations can be applied here
        return image_data


def create_data_loaders_for_fold(data_dir: str, 
                                fold_idx: int,
                                batch_size: int = 32,
                                num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create data loaders for a specific fold
    
    Args:
        data_dir: Directory containing preprocessed data
        fold_idx: Fold index (0-based)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    data_module = EDPDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return data_module.create_data_loaders(fold_idx)


def main():
    """
    Example usage of the data loading pipeline
    """
    # Initialize data module
    data_dir = "preprocessed_data"  # Adjust path as needed
    data_module = EDPDataModule(
        data_dir=data_dir,
        batch_size=16,
        num_workers=2
    )
    
    try:
        # Set up all folds
        data_module.setup_all_folds(num_folds=5)
        
        # Get statistics
        stats = data_module.get_dataset_stats()
        print("Dataset Statistics:")
        print(f"Number of folds: {stats['num_folds']}")
        
        for fold_stat in stats['fold_stats']:
            print(f"\nFold {fold_stat['fold']}:")
            print(f"  Train size: {fold_stat['train_size']}")
            print(f"  Test size: {fold_stat['test_size']}")
            print(f"  Train label distribution: {fold_stat['train_label_distribution']}")
            print(f"  Test label distribution: {fold_stat['test_label_distribution']}")
        
        # Test loading a batch from first fold
        if data_module.train_loaders:
            train_loader, test_loader = data_module.get_fold_loaders(0)
            
            print(f"\nTesting batch loading from fold 1:")
            for batch_idx, batch in enumerate(train_loader):
                print(f"Batch {batch_idx + 1}:")
                print(f"  Input IDs shape: {batch['input_ids'].shape}")
                print(f"  Attention mask shape: {batch['attention_mask'].shape}")
                print(f"  Pixel values shape: {batch['pixel_values'].shape}")
                print(f"  Labels shape: {batch['labels'].shape}")
                print(f"  Batch size: {len(batch['cleaned_text'])}")
                
                # Only test first batch
                break
        
        print("\nData loading test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data loading test: {e}")
        raise


if __name__ == "__main__":
    main() 