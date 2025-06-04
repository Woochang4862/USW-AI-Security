#!/usr/bin/env python3
"""
Improved Training Script for Interpretable MMTD
Addresses issues from initial training: data size, backbone freezing, learning rate
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.interpretable_mmtd import create_interpretable_mmtd
from models.mmtd_data_loader import create_mmtd_data_module


class ImprovedTrainer:
    """
    Improved trainer addressing issues from initial training
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/improved_training",
        log_level: str = "INFO"
    ):
        """Initialize improved trainer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Training state
        self.start_time = None
        self.best_accuracy = 0.0
        self.training_history = []
        
        self.logger.info(f"🚀 ImprovedTrainer initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
    
    def setup_logging(self, log_level: str):
        """Setup logging system"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger("ImprovedTrainer")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        
        # File handler
        log_file = log_dir / f"improved_training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"📝 Logging setup complete: {log_file}")
    
    def train_improved_mmtd(
        self,
        classifier_type: str = "logistic_regression",
        checkpoint_path: Optional[str] = None,
        data_path: str = "MMTD/DATA/email_data",
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        max_samples: Optional[int] = None,
        freeze_backbone: bool = True,
        l1_lambda: float = 0.0001,
        l2_lambda: float = 0.001
    ):
        """
        Train improved interpretable MMTD model
        
        Args:
            classifier_type: Type of interpretable classifier
            checkpoint_path: Path to pre-trained MMTD weights
            data_path: Path to training data
            batch_size: Training batch size
            learning_rate: Learning rate for classifier
            num_epochs: Number of training epochs
            max_samples: Maximum samples to use (None for full dataset)
            freeze_backbone: Whether to freeze MMTD backbone
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
        """
        self.start_time = time.time()
        self.logger.info(f"🎯 Starting improved {classifier_type} training")
        
        # Setup device
        device = self.get_optimal_device()
        self.logger.info(f"💻 Using device: {device}")
        
        # Create model with improved config
        self.logger.info("🏗️ Creating improved interpretable MMTD model...")
        classifier_config = {
            'l1_lambda': l1_lambda,
            'l2_lambda': l2_lambda,
            'dropout_rate': 0.1
        }
        
        model = create_interpretable_mmtd(
            classifier_type=classifier_type,
            classifier_config=classifier_config,
            device=device,
            checkpoint_path=checkpoint_path
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone(model)
            self.logger.info("🧊 Backbone frozen - training only classifier")
        
        # Log model info
        model_summary = model.get_model_summary()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"📊 Model: {model_summary['total_parameters']:,} total, {trainable_params:,} trainable")
        
        # Create data module with improved settings
        self.logger.info("📚 Loading data...")
        if max_samples is None:
            self.logger.info("📈 Using FULL dataset for better performance")
        else:
            self.logger.info(f"📊 Using {max_samples:,} samples")
            
        data_module = create_mmtd_data_module(
            csv_path=f"{data_path}/EDP.csv",
            data_path=f"{data_path}/pics",
            batch_size=batch_size,
            max_samples=max_samples,
            num_workers=2  # Reduce for stability
        )
        
        # Setup optimized training
        if freeze_backbone:
            # Only optimize classifier parameters
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], 
                lr=learning_rate,
                weight_decay=0.01
            )
        else:
            # Different learning rates for backbone vs classifier
            backbone_params = []
            classifier_params = []
            
            for name, param in model.named_parameters():
                if 'interpretable_classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
            
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
                {'params': classifier_params, 'lr': learning_rate}       # Higher LR for classifier
            ], weight_decay=0.01)
        
        # Improved scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop with improvements
        self.logger.info(f"🚀 Starting improved training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.logger.info(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(model, data_module.train_dataloader(), optimizer, device)
            
            # Validation phase
            val_metrics = self.validate_epoch(model, data_module.val_dataloader(), device)
            
            # Scheduler step
            scheduler.step(val_metrics['accuracy'])
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"✅ Epoch {epoch+1}: "
                f"Train Acc={train_metrics['accuracy']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}, "
                f"Time={epoch_time:.1f}s"
            )
            
            # Save best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.save_best_model(model, epoch+1)
                self.logger.info(f"🏆 New best accuracy: {self.best_accuracy:.4f}")
            
            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'timestamp': datetime.now().isoformat()
            })
            
            # Early stopping check
            if self.best_accuracy >= 0.97:
                self.logger.info(f"🎯 Target accuracy reached! Stopping early.")
                break
        
        # Final evaluation
        self.logger.info("🎯 Final evaluation...")
        final_results = self.validate_epoch(model, data_module.test_dataloader(), device)
        
        # Save final results
        self.save_final_results(model, final_results)
        
        total_time = time.time() - self.start_time
        self.logger.info(f"🎉 Training complete! Total time: {total_time:.1f}s")
        self.logger.info(f"🏆 Best validation accuracy: {self.best_accuracy:.4f}")
        self.logger.info(f"📊 Final test accuracy: {final_results['accuracy']:.4f}")
        
        return model, final_results
    
    def freeze_backbone(self, model):
        """Freeze MMTD backbone parameters"""
        for name, param in model.named_parameters():
            if 'interpretable_classifier' not in name:
                param.requires_grad = False
        
        # Verify freezing
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"🧊 Frozen: {frozen_params:,}, Trainable: {trainable_params:,}")
    
    def train_epoch(self, model, dataloader, optimizer, device):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0
        }
    
    def validate_epoch(self, model, dataloader, device):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                
                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0
        }
    
    def get_optimal_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def save_best_model(self, model, epoch):
        """Save the best model"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        self.logger.info(f"💾 Best model saved: {checkpoint_path}")
    
    def save_final_results(self, model, results):
        """Save final training results"""
        results_path = self.output_dir / "improved_results.json"
        
        import json
        final_data = {
            'model_summary': model.get_model_summary(),
            'final_test_results': results,
            'best_validation_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'total_training_time': time.time() - self.start_time if self.start_time else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        self.logger.info(f"📊 Final results saved: {results_path}")


def main():
    """Main improved training function"""
    parser = argparse.ArgumentParser(description="Improved training for Interpretable MMTD")
    parser.add_argument("--classifier_type", default="logistic_regression", help="Classifier type")
    parser.add_argument("--checkpoint_path", help="Path to pre-trained MMTD checkpoint")
    parser.add_argument("--output_dir", default="outputs/improved_training", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_samples", type=int, help="Maximum samples (None for full dataset)")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze MMTD backbone")
    parser.add_argument("--l1_lambda", type=float, default=0.0001, help="L1 regularization")
    parser.add_argument("--l2_lambda", type=float, default=0.001, help="L2 regularization")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ImprovedTrainer(output_dir=args.output_dir)
    
    # Start improved training
    model, results = trainer.train_improved_mmtd(
        classifier_type=args.classifier_type,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_samples=args.max_samples,
        freeze_backbone=args.freeze_backbone,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda
    )
    
    return model, results


if __name__ == "__main__":
    main() 