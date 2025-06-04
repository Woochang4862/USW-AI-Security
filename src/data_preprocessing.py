"""
Data Preprocessing Pipeline for Interpretable Multimodal Spam Detection
Implements comprehensive preprocessing for EDP (Email Data with Pictures) dataset
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string

# Image processing
from PIL import Image
import cv2
from torchvision import transforms
import torch

# ML utilities
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BeitImageProcessor

# Download required NLTK data
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Handles text preprocessing for multilingual email data
    """
    
    def __init__(self, 
                 languages: List[str] = ['english', 'spanish', 'french', 'german'],
                 max_length: int = 512,
                 bert_model: str = 'bert-base-multilingual-cased'):
        """
        Initialize text preprocessor
        
        Args:
            languages: List of languages to handle
            max_length: Maximum sequence length for BERT tokenization
            bert_model: BERT model name for tokenization
        """
        self.languages = languages
        self.max_length = max_length
        self.bert_model = bert_model
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # Initialize stopwords for multiple languages
        self.stopwords = set()
        for lang in languages:
            try:
                self.stopwords.update(stopwords.words(lang))
            except OSError:
                logger.warning(f"Stopwords for {lang} not available")
        
        # Initialize stemmers
        self.stemmers = {lang: SnowballStemmer(lang) for lang in languages}
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text and optionally remove stopwords
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove empty tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def stem_tokens(self, tokens: List[str], language: str = 'english') -> List[str]:
        """
        Apply stemming to tokens
        
        Args:
            tokens: List of tokens
            language: Language for stemming
            
        Returns:
            Stemmed tokens
        """
        if language in self.stemmers:
            stemmer = self.stemmers[language]
            return [stemmer.stem(token) for token in tokens]
        return tokens
    
    def bert_tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using BERT tokenizer
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze()
        }
    
    def preprocess_text(self, text: str, language: str = 'english') -> Dict[str, Union[str, List[str], torch.Tensor]]:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Raw text input
            language: Text language
            
        Returns:
            Dictionary with processed text in various formats
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Stem tokens
        stemmed_tokens = self.stem_tokens(tokens, language)
        
        # BERT tokenization
        bert_tokens = self.bert_tokenize(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'stemmed_tokens': stemmed_tokens,
            'bert_tokens': bert_tokens
        }


class ImagePreprocessor:
    """
    Handles image preprocessing for email screenshots
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 beit_model: str = 'microsoft/beit-base-patch16-224'):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target image size (height, width)
            beit_model: BEiT model name for preprocessing
        """
        self.target_size = target_size
        self.beit_model = beit_model
        
        # Initialize BEiT image processor
        self.beit_processor = BeitImageProcessor.from_pretrained(beit_model)
        
        # Standard image transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.Resize((int(target_size[0] * 1.1), int(target_size[1] * 1.1))),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if loading fails
        """
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def preprocess_image_basic(self, image: Image.Image) -> torch.Tensor:
        """
        Basic image preprocessing without augmentation
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        return self.basic_transform(image)
    
    def preprocess_image_augmented(self, image: Image.Image) -> torch.Tensor:
        """
        Image preprocessing with augmentation
        
        Args:
            image: PIL Image
            
        Returns:
            Augmented and preprocessed image tensor
        """
        return self.augment_transform(image)
    
    def beit_preprocess(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Preprocess image using BEiT processor
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with pixel_values tensor
        """
        inputs = self.beit_processor(image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze()
        }
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> Dict[str, torch.Tensor]:
        """
        Complete image preprocessing pipeline
        
        Args:
            image_path: Path to image file
            augment: Whether to apply augmentation
            
        Returns:
            Dictionary with processed image tensors
        """
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Basic preprocessing
        basic_tensor = self.preprocess_image_basic(image)
        
        # BEiT preprocessing
        beit_inputs = self.beit_preprocess(image)
        
        result = {
            'basic_tensor': basic_tensor,
            'beit_pixel_values': beit_inputs['pixel_values']
        }
        
        # Add augmented version if requested
        if augment:
            result['augmented_tensor'] = self.preprocess_image_augmented(image)
        
        return result


class DataSplitter:
    """
    Handles cross-validation splits for the dataset
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize data splitter
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
    
    def create_splits(self, df: pd.DataFrame, target_column: str = 'label') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified k-fold splits
        
        Args:
            df: DataFrame with data
            target_column: Name of target column
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        X = df.index.values
        y = df[target_column].values
        
        # Check if we have enough samples for the requested number of splits
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples_per_class = min(counts)
        
        # Adjust n_splits if necessary
        effective_n_splits = min(self.n_splits, min_samples_per_class)
        
        if effective_n_splits < self.n_splits:
            logger.warning(f"Reducing n_splits from {self.n_splits} to {effective_n_splits} "
                          f"due to insufficient samples per class (min: {min_samples_per_class})")
        
        # Create StratifiedKFold with adjusted splits
        skf = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=self.random_state)
        
        splits = []
        for train_idx, test_idx in skf.split(X, y):
            splits.append((train_idx, test_idx))
        
        return splits
    
    def get_fold_data(self, df: pd.DataFrame, fold_idx: int, target_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and test data for a specific fold
        
        Args:
            df: DataFrame with data
            fold_idx: Fold index (0-based)
            target_column: Name of target column
            
        Returns:
            Tuple of (train_df, test_df)
        """
        splits = self.create_splits(df, target_column)
        train_idx, test_idx = splits[fold_idx]
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        return train_df, test_df


class EDPDatasetPreprocessor:
    """
    Main preprocessing class for EDP dataset
    """
    
    def __init__(self, 
                 data_dir: str,
                 csv_file: str = 'EDP.csv',
                 image_dir: str = 'pics',
                 text_column: str = 'text',
                 image_column: str = 'image_path',
                 label_column: str = 'label'):
        """
        Initialize EDP dataset preprocessor
        
        Args:
            data_dir: Directory containing the dataset
            csv_file: Name of CSV file with metadata
            image_dir: Directory containing images
            text_column: Name of text column in CSV
            image_column: Name of image path column in CSV
            label_column: Name of label column in CSV
        """
        self.data_dir = Path(data_dir)
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.text_column = text_column
        self.image_column = image_column
        self.label_column = label_column
        
        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor()
        self.data_splitter = DataSplitter()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the EDP dataset
        
        Returns:
            DataFrame with dataset
        """
        csv_path = self.data_dir / self.csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Validate required columns
        required_columns = [self.text_column, self.image_column, self.label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Encode labels
        df[self.label_column] = self.label_encoder.fit_transform(df[self.label_column])
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        
        # Remove rows with missing text
        df = df.dropna(subset=[self.text_column])
        
        # Remove rows with missing image paths
        df = df.dropna(subset=[self.image_column])
        
        # Validate image files exist
        image_dir_path = self.data_dir / self.image_dir
        valid_images = []
        
        for idx, row in df.iterrows():
            image_path = image_dir_path / row[self.image_column]
            if image_path.exists():
                valid_images.append(idx)
            else:
                logger.warning(f"Image not found: {image_path}")
        
        df = df.loc[valid_images]
        
        final_count = len(df)
        logger.info(f"Dataset validation: {initial_count} -> {final_count} samples")
        
        return df.reset_index(drop=True)
    
    def preprocess_dataset(self, df: pd.DataFrame, augment: bool = False) -> Dict[str, List]:
        """
        Preprocess the entire dataset
        
        Args:
            df: Input DataFrame
            augment: Whether to apply data augmentation
            
        Returns:
            Dictionary with preprocessed data
        """
        preprocessed_data = {
            'text_data': [],
            'image_data': [],
            'labels': [],
            'indices': []
        }
        
        image_dir_path = self.data_dir / self.image_dir
        
        for idx, row in df.iterrows():
            # Preprocess text
            text_result = self.text_preprocessor.preprocess_text(row[self.text_column])
            
            # Preprocess image
            image_path = image_dir_path / row[self.image_column]
            image_result = self.image_preprocessor.preprocess_image(str(image_path), augment=augment)
            
            if image_result is not None:
                preprocessed_data['text_data'].append(text_result)
                preprocessed_data['image_data'].append(image_result)
                preprocessed_data['labels'].append(row[self.label_column])
                preprocessed_data['indices'].append(idx)
        
        logger.info(f"Preprocessed {len(preprocessed_data['labels'])} samples")
        
        return preprocessed_data
    
    def create_cross_validation_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create cross-validation splits
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of (train_df, test_df) tuples for each fold
        """
        return [self.data_splitter.get_fold_data(df, i, self.label_column) 
                for i in range(self.data_splitter.n_splits)]
    
    def save_preprocessed_data(self, preprocessed_data: Dict, output_path: str):
        """
        Save preprocessed data to disk
        
        Args:
            preprocessed_data: Preprocessed data dictionary
            output_path: Path to save the data
        """
        torch.save(preprocessed_data, output_path)
        logger.info(f"Saved preprocessed data to {output_path}")
    
    def load_preprocessed_data(self, input_path: str) -> Dict:
        """
        Load preprocessed data from disk
        
        Args:
            input_path: Path to load the data from
            
        Returns:
            Preprocessed data dictionary
        """
        preprocessed_data = torch.load(input_path)
        logger.info(f"Loaded preprocessed data from {input_path}")
        return preprocessed_data


def main():
    """
    Example usage of the preprocessing pipeline
    """
    # Initialize preprocessor
    data_dir = "MMTD/DATA/email_data"  # Adjust path as needed
    preprocessor = EDPDatasetPreprocessor(data_dir)
    
    try:
        # Load dataset
        df = preprocessor.load_dataset()
        
        # Validate data
        df = preprocessor.validate_data(df)
        
        # Create cross-validation splits
        cv_splits = preprocessor.create_cross_validation_splits(df)
        
        # Preprocess first fold as example
        train_df, test_df = cv_splits[0]
        
        print(f"Fold 1 - Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Preprocess training data
        train_data = preprocessor.preprocess_dataset(train_df, augment=True)
        test_data = preprocessor.preprocess_dataset(test_df, augment=False)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(train_data, "preprocessed_train_fold1.pt")
        preprocessor.save_preprocessed_data(test_data, "preprocessed_test_fold1.pt")
        
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()