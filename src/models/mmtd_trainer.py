"""
MMTD Training Utilities
Based on the original MMTD utils.py with enhancements for interpretability research
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import wandb
from datetime import datetime

from .base_mmtd import BaseMMTD, MMTDConfig
from .mmtd_data_loader import MMTDDataModule

logger = logging.getLogger(__name__)


class MMTDTrainer:
    """
    Trainer class for MMTD model
    Based on the original MMTD training approach with enhancements
    """
    
    def __init__(
        self,
        model: BaseMMTD,
        data_module: MMTDDataModule,
        config: Optional[MMTDConfig] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        wandb_project: str = "interpretable-mmtd",
        save_dir: Union[str, Path] = "./outputs"
    ):
        """
        Initialize MMTD Trainer
        
        Args:
            model: MMTD model to train
            data_module: Data module with train/val/test dataloaders
            config: Model configuration
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            device: Device to use for training
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            save_dir: Directory to save outputs
        """
        self.model = model
        self.data_module = data_module
        self.config = config or MMTDConfig()
        
        # Setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=5e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Setup logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project)
            wandb.config.update(self.config.to_dict())
        
        # Setup save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        logger.info(f"Initialized MMTDTrainer with device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate AUC if binary classification
        if len(np.unique(all_labels)) == 2:
            probabilities = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
            auc_score = roc_auc_score(all_labels, probabilities)
        else:
            auc_score = 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc_score
        }
        
        return metrics
    
    def train(
        self,
        num_epochs: int = 3,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            early_stopping_patience: Early stopping patience (None to disable)
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_dataloader)
            val_loss = val_metrics['val_loss']
            val_accuracy = val_metrics['val_accuracy']
            
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **val_metrics
            }
            
            if self.use_wandb:
                wandb.log(log_dict)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if save_best and val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.save_model("best_model.pt")
                patience_counter = 0
                logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_model("final_model.pt")
        
        # Return training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        return history
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        save_results: bool = True,
        save_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            dataloader: DataLoader to evaluate on (default: test dataloader)
            save_results: Whether to save evaluation results
            save_name: Name for saved results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if dataloader is None:
            dataloader = self.data_module.test_dataloader()
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        # Classification report
        class_names = ['Ham', 'Spam']  # Assuming binary classification
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # ROC curve and AUC for binary classification
        if len(np.unique(all_labels)) == 2:
            probabilities = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
            fpr, tpr, _ = roc_curve(all_labels, probabilities)
            auc_score = auc(fpr, tpr)
        else:
            fpr, tpr, auc_score = None, None, 0.0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc': auc_score,
            'fpr': fpr.tolist() if fpr is not None else None,
            'tpr': tpr.tolist() if tpr is not None else None
        }
        
        # Save results
        if save_results:
            self._save_evaluation_results(results, save_name, class_names)
        
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def _save_evaluation_results(
        self, 
        results: Dict[str, Any], 
        save_name: str,
        class_names: List[str]
    ):
        """Save evaluation results with visualizations"""
        results_dir = self.save_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics as JSON/YAML
        metrics_file = results_dir / f"{save_name}_metrics.yaml"
        with open(metrics_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Save confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {save_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(results_dir / f"{save_name}_confusion_matrix.png", dpi=300)
        plt.close()
        
        # Save ROC curve if available
        if results['fpr'] is not None and results['tpr'] is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(results['fpr'], results['tpr'], 
                    label=f'ROC Curve (AUC = {results["auc"]:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {save_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(results_dir / f"{save_name}_roc_curve.png", dpi=300)
            plt.close()
        
        logger.info(f"Evaluation results saved to {results_dir}")
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'epoch': self.current_epoch,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        logger.info(f"Model loaded from {load_path}")
    
    def plot_training_history(self, save_plot: bool = True):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.save_dir / "training_history.png", dpi=300)
            logger.info(f"Training history plot saved to {self.save_dir}")
        
        plt.show()


class CrossValidationTrainer:
    """
    Cross-validation trainer for MMTD model
    Based on the original MMTD k-fold approach
    """
    
    def __init__(
        self,
        model_factory: callable,
        data_path: Union[str, Path],
        csv_path: Union[str, Path],
        k_folds: int = 5,
        config: Optional[MMTDConfig] = None,
        use_wandb: bool = False,
        wandb_project: str = "interpretable-mmtd-cv",
        save_dir: Union[str, Path] = "./outputs/cv"
    ):
        """
        Initialize Cross-Validation Trainer
        
        Args:
            model_factory: Function that creates a new model instance
            data_path: Path to image directory
            csv_path: Path to CSV file
            k_folds: Number of folds for cross-validation
            config: Model configuration
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            save_dir: Directory to save outputs
        """
        self.model_factory = model_factory
        self.data_path = Path(data_path)
        self.csv_path = Path(csv_path)
        self.k_folds = k_folds
        self.config = config or MMTDConfig()
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and split data
        self._prepare_cv_splits()
        
        # Results storage
        self.fold_results = []
        
        logger.info(f"Initialized CrossValidationTrainer with {k_folds} folds")
    
    def _prepare_cv_splits(self):
        """Prepare cross-validation splits"""
        from .mmtd_data_loader import create_mmtd_data_module
        
        # Load data
        df = pd.read_csv(self.csv_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create k-fold splits
        fold_size = len(df) // self.k_folds
        self.cv_splits = []
        
        for i in range(self.k_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.k_folds - 1 else len(df)
            
            test_df = df.iloc[start_idx:end_idx].copy()
            train_df = pd.concat([
                df.iloc[:start_idx],
                df.iloc[end_idx:]
            ]).copy()
            
            # Further split train into train/val (80/20)
            val_size = len(train_df) // 5
            val_df = train_df.iloc[:val_size].copy()
            train_df = train_df.iloc[val_size:].copy()
            
            self.cv_splits.append((train_df, val_df, test_df))
        
        logger.info(f"Created {self.k_folds} CV splits")
    
    def run_cv(
        self,
        num_epochs: int = 3,
        **trainer_kwargs
    ) -> Dict[str, Any]:
        """
        Run cross-validation
        
        Args:
            num_epochs: Number of epochs per fold
            **trainer_kwargs: Additional arguments for trainer
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting {self.k_folds}-fold cross-validation")
        
        for fold in range(self.k_folds):
            logger.info(f"Training fold {fold + 1}/{self.k_folds}")
            
            # Initialize W&B for this fold
            if self.use_wandb:
                wandb.init(
                    project=self.wandb_project,
                    name=f"fold_{fold + 1}",
                    reinit=True
                )
            
            # Get data for this fold
            train_df, val_df, test_df = self.cv_splits[fold]
            
            # Create data module
            from .mmtd_data_loader import MMTDDataModule
            data_module = MMTDDataModule(
                data_path=self.data_path,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                **trainer_kwargs.get('data_module_kwargs', {})
            )
            
            # Create model
            model = self.model_factory()
            
            # Create trainer
            fold_save_dir = self.save_dir / f"fold_{fold + 1}"
            trainer = MMTDTrainer(
                model=model,
                data_module=data_module,
                config=self.config,
                use_wandb=self.use_wandb,
                save_dir=fold_save_dir,
                **{k: v for k, v in trainer_kwargs.items() 
                   if k != 'data_module_kwargs'}
            )
            
            # Train
            history = trainer.train(num_epochs=num_epochs)
            
            # Evaluate
            results = trainer.evaluate(save_name=f"fold_{fold + 1}")
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'history': history,
                'evaluation': results
            }
            self.fold_results.append(fold_result)
            
            if self.use_wandb:
                wandb.finish()
        
        # Aggregate results
        cv_results = self._aggregate_cv_results()
        
        # Save CV results
        self._save_cv_results(cv_results)
        
        return cv_results
    
    def _aggregate_cv_results(self) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        aggregated = {}
        for metric in metrics:
            values = [fold['evaluation'][metric] for fold in self.fold_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_values'] = values
        
        # Best fold
        accuracies = [fold['evaluation']['accuracy'] for fold in self.fold_results]
        best_fold_idx = np.argmax(accuracies)
        aggregated['best_fold'] = best_fold_idx + 1
        aggregated['best_accuracy'] = accuracies[best_fold_idx]
        
        logger.info(f"CV Results - Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        
        return aggregated
    
    def _save_cv_results(self, cv_results: Dict[str, Any]):
        """Save cross-validation results"""
        # Save detailed results
        results_file = self.save_dir / "cv_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump({
                'cv_summary': cv_results,
                'fold_details': self.fold_results
            }, f, default_flow_style=False)
        
        # Create summary plot
        self._plot_cv_results(cv_results)
        
        logger.info(f"CV results saved to {self.save_dir}")
    
    def _plot_cv_results(self, cv_results: Dict[str, Any]):
        """Plot cross-validation results"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        means = [cv_results[f'{metric}_mean'] for metric in metrics]
        stds = [cv_results[f'{metric}_std'] for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color='skyblue', edgecolor='navy', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(f'{self.k_folds}-Fold Cross-Validation Results')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "cv_results.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    # Test trainer creation
    from .base_mmtd import create_base_mmtd_model
    from .mmtd_data_loader import create_mmtd_data_module
    
    print("Testing MMTD Trainer...")
    
    # This would be used with real data
    # config = MMTDConfig()
    # model = create_base_mmtd_model(config)
    # data_module = create_mmtd_data_module(...)
    # trainer = MMTDTrainer(model, data_module, config)
    
    print("✅ MMTD Trainer implementation completed!") 