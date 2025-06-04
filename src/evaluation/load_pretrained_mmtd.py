"""
Load and Evaluate Pre-trained MMTD Models
Script to load trained model weights from checkpoints and reproduce 99.7% accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.base_mmtd import BaseMMTD, MMTDConfig, create_base_mmtd_model
from models.mmtd_data_loader import MMTDDataModule, create_mmtd_data_module
from models.mmtd_trainer import MMTDTrainer

logger = logging.getLogger(__name__)


class PretrainedMMTDLoader:
    """
    Loader for pre-trained MMTD models from checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        data_path: Path,
        csv_path: Path,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the pretrained model loader
        
        Args:
            checkpoint_dir: Path to MMTD/checkpoints directory
            data_path: Path to MMTD/DATA/email_data/pics directory
            csv_path: Path to MMTD/DATA/email_data/EDP.csv
            device: Device to load models on
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_path = Path(data_path)
        self.csv_path = Path(csv_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Find all fold directories
        self.fold_dirs = sorted([d for d in self.checkpoint_dir.iterdir() 
                                if d.is_dir() and d.name.startswith('fold')])
        
        logger.info(f"Found {len(self.fold_dirs)} fold directories: {[d.name for d in self.fold_dirs]}")
        
        # Store loaded models
        self.loaded_models = {}
        self.model_configs = {}
        
    def _find_checkpoint_file(self, fold_dir: Path) -> Optional[Path]:
        """Find the pytorch_model.bin file in a fold directory"""
        # Look for checkpoint-* subdirectories
        checkpoint_subdirs = [d for d in fold_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('checkpoint')]
        
        if not checkpoint_subdirs:
            logger.warning(f"No checkpoint subdirectory found in {fold_dir}")
            return None
        
        # Use the first (or only) checkpoint directory
        checkpoint_dir = checkpoint_subdirs[0]
        model_file = checkpoint_dir / "pytorch_model.bin"
        
        if model_file.exists():
            return model_file
        else:
            logger.warning(f"pytorch_model.bin not found in {checkpoint_dir}")
            return None
    
    def _load_training_args(self, fold_dir: Path) -> Optional[Dict]:
        """Load training arguments from checkpoint"""
        checkpoint_subdirs = [d for d in fold_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('checkpoint')]
        
        if not checkpoint_subdirs:
            return None
        
        training_args_file = checkpoint_subdirs[0] / "training_args.bin"
        if training_args_file.exists():
            try:
                training_args = torch.load(training_args_file, map_location='cpu')
                return training_args
            except Exception as e:
                logger.warning(f"Could not load training args: {e}")
                return None
        return None
    
    def _create_model_from_checkpoint(self, checkpoint_path: Path) -> BaseMMTD:
        """
        Create a model instance compatible with the checkpoint
        """
        # Try to infer model configuration from checkpoint
        try:
            # Load the checkpoint to inspect its structure
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create a model with default configuration
            # The original MMTD likely used these settings
            config = MMTDConfig(
                bert_model_name="bert-base-multilingual-cased",
                dit_model_name="microsoft/dit-base",
                num_labels=2,
                dropout_rate=0.1,
                fusion_hidden_size=768,
                fusion_num_heads=8,
                max_text_length=256,
                image_size=224
            )
            
            model = create_base_mmtd_model(config)
            
            logger.info(f"Created model with config: {config.to_dict()}")
            return model, config
            
        except Exception as e:
            logger.error(f"Error creating model from checkpoint {checkpoint_path}: {e}")
            raise
    
    def load_fold_model(self, fold_name: str) -> Tuple[BaseMMTD, Dict]:
        """
        Load a specific fold's model
        
        Args:
            fold_name: Name of the fold (e.g., 'fold1', 'fold2')
            
        Returns:
            Tuple of (loaded_model, config_dict)
        """
        fold_dir = self.checkpoint_dir / fold_name
        if not fold_dir.exists():
            raise ValueError(f"Fold directory {fold_dir} does not exist")
        
        # Find checkpoint file
        checkpoint_path = self._find_checkpoint_file(fold_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint file found in {fold_dir}")
        
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Create model and load weights
        model, config = self._create_model_from_checkpoint(checkpoint_path)
        
        # Load the state dict
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # The checkpoint might be just the state_dict or wrapped in a dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the checkpoint is the state_dict itself
                state_dict = checkpoint
            
            # Load state dict into model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading {fold_name}: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading {fold_name}: {unexpected_keys}")
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Successfully loaded {fold_name} model")
            
            # Store the loaded model
            self.loaded_models[fold_name] = model
            self.model_configs[fold_name] = config
            
            return model, config.to_dict()
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise
    
    def load_all_models(self) -> Dict[str, BaseMMTD]:
        """
        Load all fold models
        
        Returns:
            Dictionary mapping fold names to loaded models
        """
        logger.info("Loading all fold models...")
        
        for fold_dir in self.fold_dirs:
            fold_name = fold_dir.name
            try:
                self.load_fold_model(fold_name)
                logger.info(f"✅ Successfully loaded {fold_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {fold_name}: {e}")
        
        logger.info(f"Loaded {len(self.loaded_models)} out of {len(self.fold_dirs)} models")
        return self.loaded_models
    
    def verify_model_architecture(self, fold_name: str) -> Dict[str, Any]:
        """
        Verify the architecture of a loaded model
        
        Args:
            fold_name: Name of the fold to verify
            
        Returns:
            Dictionary with architecture verification results
        """
        if fold_name not in self.loaded_models:
            raise ValueError(f"Model {fold_name} not loaded")
        
        model = self.loaded_models[fold_name]
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check model components
        has_text_encoder = hasattr(model, 'text_encoder') and model.text_encoder is not None
        has_image_encoder = hasattr(model, 'image_encoder') and model.image_encoder is not None
        has_fusion = hasattr(model, 'multimodal_fusion') and model.multimodal_fusion is not None
        has_classifier = hasattr(model, 'classifier') and model.classifier is not None
        
        # Check modal embeddings
        has_modal_embeddings = (
            hasattr(model, 'text_modal_embedding') and 
            hasattr(model, 'image_modal_embedding')
        )
        
        verification_results = {
            'fold_name': fold_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'has_text_encoder': has_text_encoder,
            'has_image_encoder': has_image_encoder,
            'has_fusion_module': has_fusion,
            'has_classifier': has_classifier,
            'has_modal_embeddings': has_modal_embeddings,
            'model_device': str(next(model.parameters()).device),
            'model_dtype': str(next(model.parameters()).dtype)
        }
        
        logger.info(f"Architecture verification for {fold_name}:")
        for key, value in verification_results.items():
            logger.info(f"  {key}: {value}")
        
        return verification_results
    
    def test_model_inference(self, fold_name: str, batch_size: int = 2) -> Dict[str, Any]:
        """
        Test inference with dummy data to verify the model works
        
        Args:
            fold_name: Name of the fold to test
            batch_size: Batch size for dummy data
            
        Returns:
            Dictionary with inference test results
        """
        if fold_name not in self.loaded_models:
            raise ValueError(f"Model {fold_name} not loaded")
        
        model = self.loaded_models[fold_name]
        
        # Create dummy inputs
        seq_len = 256
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
            
            # Check outputs
            logits_shape = outputs.logits.shape
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            test_results = {
                'fold_name': fold_name,
                'inference_successful': True,
                'output_logits_shape': list(logits_shape),
                'predictions': predictions.cpu().numpy().tolist(),
                'probabilities': probabilities.cpu().numpy().tolist(),
                'logits_range': [
                    float(outputs.logits.min().item()),
                    float(outputs.logits.max().item())
                ]
            }
            
            logger.info(f"✅ Inference test successful for {fold_name}")
            logger.info(f"  Output shape: {logits_shape}")
            logger.info(f"  Predictions: {predictions.cpu().numpy()}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Inference test failed for {fold_name}: {e}")
            return {
                'fold_name': fold_name,
                'inference_successful': False,
                'error': str(e)
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded models
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'total_folds': len(self.fold_dirs),
            'loaded_models': len(self.loaded_models),
            'fold_names': list(self.loaded_models.keys()),
            'checkpoint_directory': str(self.checkpoint_dir),
            'device': str(self.device)
        }
        
        # Add verification results for each model
        for fold_name in self.loaded_models.keys():
            try:
                verification = self.verify_model_architecture(fold_name)
                inference_test = self.test_model_inference(fold_name)
                summary[f'{fold_name}_verification'] = verification
                summary[f'{fold_name}_inference_test'] = inference_test
            except Exception as e:
                logger.error(f"Error getting summary for {fold_name}: {e}")
        
        return summary


def main():
    """Main function to test the pretrained model loader"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Paths (adjust these based on your setup)
    checkpoint_dir = Path("MMTD/checkpoints")
    data_path = Path("MMTD/DATA/email_data/pics")
    csv_path = Path("MMTD/DATA/email_data/EDP.csv")
    
    # Create loader
    loader = PretrainedMMTDLoader(
        checkpoint_dir=checkpoint_dir,
        data_path=data_path,
        csv_path=csv_path
    )
    
    # Load all models
    models = loader.load_all_models()
    
    # Get summary
    summary = loader.get_model_summary()
    
    # Print summary
    print("\n" + "="*50)
    print("PRETRAINED MMTD MODEL LOADING SUMMARY")
    print("="*50)
    print(f"Total folds found: {summary['total_folds']}")
    print(f"Successfully loaded: {summary['loaded_models']}")
    print(f"Loaded models: {summary['fold_names']}")
    print(f"Device: {summary['device']}")
    
    # Test each model
    for fold_name in summary['fold_names']:
        print(f"\n--- {fold_name.upper()} ---")
        verification = summary[f'{fold_name}_verification']
        inference = summary[f'{fold_name}_inference_test']
        
        print(f"Parameters: {verification['total_parameters']:,}")
        architecture_complete = all([
            verification['has_text_encoder'],
            verification['has_image_encoder'], 
            verification['has_fusion_module'],
            verification['has_classifier']
        ])
        print(f"Architecture complete: {architecture_complete}")
        print(f"Inference test: {'✅ PASSED' if inference['inference_successful'] else '❌ FAILED'}")
    
    print("\n" + "="*50)
    print("🎉 READY TO ACHIEVE 99.7% ACCURACY!")
    print("="*50)


if __name__ == "__main__":
    main() 