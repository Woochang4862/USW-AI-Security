"""
Evaluate Pre-trained MMTD Models on EDP Dataset
Script to reproduce 99.7% accuracy using the trained checkpoints
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.original_mmtd_model import OriginalMMTD, OriginalMMTDLoader
from models.mmtd_data_loader import MMTDDataset, MMTDCollator, create_mmtd_data_module

logger = logging.getLogger(__name__)


class MMTDEvaluator:
    """
    Comprehensive evaluator for pre-trained MMTD models
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        data_path: Path,
        csv_path: Path,
        device: Optional[torch.device] = None,
        batch_size: int = 16
    ):
        """
        Initialize the MMTD evaluator
        
        Args:
            checkpoint_dir: Path to MMTD/checkpoints directory
            data_path: Path to MMTD/DATA/email_data/pics directory
            csv_path: Path to MMTD/DATA/email_data/EDP.csv
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_path = Path(data_path)
        self.csv_path = Path(csv_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Initialize model loader
        self.model_loader = OriginalMMTDLoader(
            checkpoint_dir=str(checkpoint_dir),
            device=self.device
        )
        
        # Load data
        self.data_df = self._load_and_prepare_data()
        
        # Initialize collator
        self.collator = MMTDCollator(max_length=256, image_size=224)
        
        logger.info(f"Initialized MMTDEvaluator with {len(self.data_df)} samples")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {batch_size}")
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the EDP dataset"""
        logger.info(f"Loading data from {self.csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Check columns
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # Prepare data based on CSV structure
        if 'texts' in df.columns and 'pics' in df.columns and 'labels' in df.columns:
            # Standard format
            df_clean = df[['texts', 'pics', 'labels']].copy()
            df_clean.columns = ['text', 'image_path', 'label']
        else:
            raise ValueError(f"Unexpected CSV format. Columns: {df.columns.tolist()}")
        
        # Clean data
        df_clean = df_clean.dropna()
        
        # Verify image paths exist
        valid_rows = []
        for idx, row in df_clean.iterrows():
            image_path = self.data_path / row['image_path']
            if image_path.exists():
                valid_rows.append(row)
            elif idx < 10:  # Log first few missing files
                logger.warning(f"Image not found: {image_path}")
        
        df_final = pd.DataFrame(valid_rows)
        logger.info(f"Final dataset: {len(df_final)} samples with valid images")
        
        # Check label distribution
        label_counts = df_final['label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        return df_final
    
    def create_kfold_splits(self, n_splits: int = 5, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create K-fold splits matching the original training"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, test_idx in kfold.split(self.data_df):
            train_df = self.data_df.iloc[train_idx].copy()
            test_df = self.data_df.iloc[test_idx].copy()
            splits.append((train_df, test_df))
        
        logger.info(f"Created {n_splits} K-fold splits")
        for i, (train_df, test_df) in enumerate(splits):
            logger.info(f"Fold {i+1}: Train={len(train_df)}, Test={len(test_df)}")
        
        return splits
    
    def evaluate_fold(self, fold_name: str, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a specific fold model
        
        Args:
            fold_name: Name of the fold (e.g., 'fold1')
            test_df: Test dataframe for this fold
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {fold_name} on {len(test_df)} test samples")
        
        # Load model
        model = self.model_loader.load_fold_model(fold_name)
        model.eval()
        
        # Create dataset and dataloader
        test_dataset = MMTDDataset(
            data_path=self.data_path,
            data_df=test_df
        )
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=2,
            pin_memory=True
        )
        
        # Evaluation
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Evaluating {fold_name}"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Collect results
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    num_batches += 1
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'fold_name': fold_name,
            'num_test_samples': len(test_df),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        logger.info(f"{fold_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def evaluate_all_folds(self) -> Dict[str, Any]:
        """
        Evaluate all fold models using K-fold cross-validation
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("Starting comprehensive evaluation of all folds")
        
        # Create K-fold splits
        splits = self.create_kfold_splits(n_splits=5)
        
        # Load all models
        models = self.model_loader.load_all_models()
        
        # Evaluate each fold
        fold_results = {}
        for i, (fold_name, model) in enumerate(models.items()):
            train_df, test_df = splits[i]
            results = self.evaluate_fold(fold_name, test_df)
            fold_results[fold_name] = results
        
        # Calculate overall statistics
        accuracies = [results['accuracy'] for results in fold_results.values()]
        precisions = [results['precision'] for results in fold_results.values()]
        recalls = [results['recall'] for results in fold_results.values()]
        f1_scores = [results['f1_score'] for results in fold_results.values()]
        
        overall_results = {
            'fold_results': fold_results,
            'overall_stats': {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'mean_recall': np.mean(recalls),
                'std_recall': np.std(recalls),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'accuracy_range': [np.min(accuracies), np.max(accuracies)],
                'individual_accuracies': accuracies
            },
            'target_accuracy': 0.997,  # 99.7%
            'achieved_target': np.mean(accuracies) >= 0.997
        }
        
        # Log overall results
        logger.info("\n" + "="*60)
        logger.info("OVERALL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Mean Accuracy: {overall_results['overall_stats']['mean_accuracy']:.4f} ± {overall_results['overall_stats']['std_accuracy']:.4f}")
        logger.info(f"Accuracy Range: {overall_results['overall_stats']['accuracy_range'][0]:.4f} - {overall_results['overall_stats']['accuracy_range'][1]:.4f}")
        logger.info(f"Target (99.7%): {'✅ ACHIEVED' if overall_results['achieved_target'] else '❌ NOT ACHIEVED'}")
        logger.info(f"Individual Fold Accuracies:")
        for fold_name, accuracy in zip(fold_results.keys(), accuracies):
            logger.info(f"  {fold_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return overall_results
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save evaluation results to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: Path):
        """Create visualization plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Accuracy comparison plot
        fold_names = list(results['fold_results'].keys())
        accuracies = [results['fold_results'][fold]['accuracy'] for fold in fold_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(fold_names, accuracies, color='skyblue', alpha=0.7)
        plt.axhline(y=0.997, color='red', linestyle='--', label='Target (99.7%)')
        plt.axhline(y=results['overall_stats']['mean_accuracy'], color='green', linestyle='-', label=f"Mean ({results['overall_stats']['mean_accuracy']:.3f})")
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.ylabel('Accuracy')
        plt.title('MMTD Model Accuracy by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_fold.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (fold_name, fold_result) in enumerate(results['fold_results'].items()):
            cm = np.array(fold_result['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{fold_name} (Acc: {fold_result["accuracy"]:.3f})')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main evaluation function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Paths
    checkpoint_dir = Path("MMTD/checkpoints")
    data_path = Path("MMTD/DATA/email_data/pics")
    csv_path = Path("MMTD/DATA/email_data/EDP.csv")
    output_dir = Path("outputs/evaluation")
    
    # Create evaluator
    evaluator = MMTDEvaluator(
        checkpoint_dir=checkpoint_dir,
        data_path=data_path,
        csv_path=csv_path,
        batch_size=16
    )
    
    # Run evaluation
    results = evaluator.evaluate_all_folds()
    
    # Save results
    evaluator.save_results(results, output_dir / "evaluation_results.json")
    
    # Create visualizations
    evaluator.create_visualizations(results, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("🎯 MMTD MODEL EVALUATION COMPLETE")
    print("="*80)
    print(f"📊 Mean Accuracy: {results['overall_stats']['mean_accuracy']:.4f} ({results['overall_stats']['mean_accuracy']*100:.2f}%)")
    print(f"🎯 Target Accuracy: 99.7%")
    print(f"✅ Target Achieved: {'YES' if results['achieved_target'] else 'NO'}")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main() 