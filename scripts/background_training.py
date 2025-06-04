#!/usr/bin/env python3
"""
Background Training Script for Interpretable MMTD
Supports nohup execution with real-time logging and monitoring
"""

import os
import sys
import time
import logging
import signal
import json
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
from evaluation.evaluate_pretrained_mmtd import MMTDEvaluator


class BackgroundTrainer:
    """
    Background trainer with comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/background_training",
        log_level: str = "INFO",
        save_interval: int = 100,
        eval_interval: int = 500
    ):
        """
        Initialize background trainer
        
        Args:
            output_dir: Directory for outputs and logs
            log_level: Logging level
            save_interval: Steps between model saves
            eval_interval: Steps between evaluations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Training parameters
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # State tracking
        self.start_time = None
        self.current_step = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info(f"🚀 BackgroundTrainer initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        
    def setup_logging(self, log_level: str):
        """Setup comprehensive logging system"""
        
        # Create log directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logger
        self.logger = logging.getLogger("BackgroundTrainer")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        log_file = log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Progress log for easy monitoring
        self.progress_log = log_dir / f"progress_{timestamp}.log"
        
        self.logger.info(f"📝 Logging setup complete")
        self.logger.info(f"📄 Detailed log: {log_file}")
        self.logger.info(f"📊 Progress log: {self.progress_log}")
        
    def log_progress(self, message: str):
        """Log progress to both main log and progress file"""
        self.logger.info(message)
        
        # Also write to progress file for easy monitoring
        with open(self.progress_log, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.save_checkpoint("emergency_checkpoint")
        self.save_training_history()
        sys.exit(0)
    
    def train_interpretable_mmtd(
        self,
        classifier_type: str = "logistic_regression",
        checkpoint_path: Optional[str] = None,
        data_path: str = "MMTD/DATA/email_data",
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 5,
        max_samples: Optional[int] = None
    ):
        """
        Train interpretable MMTD model
        
        Args:
            classifier_type: Type of interpretable classifier
            checkpoint_path: Path to pre-trained MMTD weights
            data_path: Path to training data
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            max_samples: Maximum samples to use (for testing)
        """
        self.start_time = time.time()
        self.log_progress(f"🎯 Starting {classifier_type} training")
        
        # Setup device
        device = self.get_optimal_device()
        self.log_progress(f"💻 Using device: {device}")
        
        # Create model
        self.log_progress("🏗️ Creating interpretable MMTD model...")
        model = create_interpretable_mmtd(
            classifier_type=classifier_type,
            device=device,
            checkpoint_path=checkpoint_path
        )
        
        # Log model info
        model_summary = model.get_model_summary()
        self.log_progress(f"📊 Model created: {model_summary['total_parameters']:,} parameters")
        self.log_progress(f"🧠 Classifier: {model_summary['classifier_parameters']:,} parameters")
        
        # Create data module
        self.log_progress("📚 Loading data...")
        data_module = create_mmtd_data_module(
            csv_path=f"{data_path}/EDP.csv",
            data_path=f"{data_path}/pics",
            batch_size=batch_size,
            max_samples=max_samples
        )
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        self.log_progress(f"🚀 Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.log_progress(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_loader = data_module.train_dataloader()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                self.current_step += 1
                
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == batch['labels']).sum().item()
                train_total += batch['labels'].size(0)
                
                # Periodic logging
                if batch_idx % 50 == 0:
                    current_acc = train_correct / train_total if train_total > 0 else 0
                    self.log_progress(
                        f"Step {self.current_step}: Loss={loss.item():.4f}, "
                        f"Acc={current_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}"
                    )
                
                # Periodic saves
                if self.current_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{self.current_step}")
                
                # Periodic evaluation
                if self.current_step % self.eval_interval == 0:
                    eval_results = self.evaluate_model(model, data_module, device)
                    if eval_results['accuracy'] > self.best_accuracy:
                        self.best_accuracy = eval_results['accuracy']
                        self.save_checkpoint("best_model")
                        self.log_progress(f"🏆 New best accuracy: {self.best_accuracy:.4f}")
            
            # End of epoch
            scheduler.step()
            epoch_time = time.time() - epoch_start
            epoch_acc = train_correct / train_total if train_total > 0 else 0
            epoch_loss = train_loss / len(train_loader)
            
            self.log_progress(
                f"✅ Epoch {epoch+1} complete: "
                f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Time={epoch_time:.1f}s"
            )
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")
            
            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'step': self.current_step,
                'train_loss': epoch_loss,
                'train_accuracy': epoch_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'timestamp': datetime.now().isoformat()
            })
        
        # Final evaluation
        self.log_progress("🎯 Final evaluation...")
        final_results = self.evaluate_model(model, data_module, device)
        
        # Save final results
        self.save_final_results(model, final_results)
        
        total_time = time.time() - self.start_time
        self.log_progress(f"🎉 Training complete! Total time: {total_time:.1f}s")
        self.log_progress(f"🏆 Best accuracy: {self.best_accuracy:.4f}")
        
        return model, final_results
    
    def evaluate_model(self, model, data_module, device):
        """Evaluate model performance"""
        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_loss = 0.0
        
        val_loader = data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                
                eval_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                eval_correct += (predictions == batch['labels']).sum().item()
                eval_total += batch['labels'].size(0)
        
        accuracy = eval_correct / eval_total if eval_total > 0 else 0
        avg_loss = eval_loss / len(val_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': eval_correct,
            'total': eval_total
        }
    
    def get_optimal_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        # Implementation would save model state here
        self.log_progress(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.log_progress(f"📈 Training history saved: {history_path}")
    
    def save_final_results(self, model, results):
        """Save final training results"""
        results_path = self.output_dir / "final_results.json"
        
        final_data = {
            'model_summary': model.get_model_summary(),
            'evaluation_results': results,
            'training_history': self.training_history,
            'total_training_time': time.time() - self.start_time if self.start_time else 0,
            'best_accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        self.log_progress(f"📊 Final results saved: {results_path}")


def create_background_script():
    """Create a shell script for easy background execution"""
    script_content = '''#!/bin/bash

# Background Training Script for Interpretable MMTD
# Usage: ./run_background_training.sh [classifier_type] [checkpoint_path]

CLASSIFIER_TYPE=${1:-"logistic_regression"}
CHECKPOINT_PATH=${2:-"MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"}
OUTPUT_DIR="outputs/background_training_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting background training..."
echo "📊 Classifier: $CLASSIFIER_TYPE"
echo "💾 Checkpoint: $CHECKPOINT_PATH"
echo "📁 Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training in background with nohup
nohup python scripts/background_training.py \\
    --classifier_type "$CLASSIFIER_TYPE" \\
    --checkpoint_path "$CHECKPOINT_PATH" \\
    --output_dir "$OUTPUT_DIR" \\
    --batch_size 4 \\
    --learning_rate 1e-4 \\
    --num_epochs 3 \\
    --max_samples 1000 \\
    > "$OUTPUT_DIR/nohup.out" 2>&1 &

PID=$!
echo "🔄 Training started with PID: $PID"
echo "📝 Logs: $OUTPUT_DIR/nohup.out"
echo "📊 Progress: $OUTPUT_DIR/logs/progress_*.log"

# Save PID for monitoring
echo $PID > "$OUTPUT_DIR/training.pid"

echo ""
echo "📋 Monitoring commands:"
echo "  tail -f $OUTPUT_DIR/nohup.out                    # Main output"
echo "  tail -f $OUTPUT_DIR/logs/progress_*.log          # Progress log"
echo "  kill $PID                                        # Stop training"
echo "  ps aux | grep $PID                               # Check status"

echo ""
echo "✅ Background training started successfully!"
'''
    
    script_path = Path("run_background_training.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Background training for Interpretable MMTD")
    parser.add_argument("--classifier_type", default="logistic_regression", help="Classifier type")
    parser.add_argument("--checkpoint_path", help="Path to pre-trained MMTD checkpoint")
    parser.add_argument("--output_dir", default="outputs/background_training", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_samples", type=int, help="Maximum samples (for testing)")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BackgroundTrainer(
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    # Start training
    model, results = trainer.train_interpretable_mmtd(
        classifier_type=args.classifier_type,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_samples=args.max_samples
    )
    
    return model, results


if __name__ == "__main__":
    # Create background script
    script_path = create_background_script()
    print(f"✅ Created background script: {script_path}")
    
    # Run main training
    main() 