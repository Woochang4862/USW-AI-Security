#!/usr/bin/env python3
"""
Interpretability Analysis for Logistic Regression MMTD
Comprehensive analysis of feature importance and model interpretability
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.interpretable_mmtd import create_interpretable_mmtd
from models.mmtd_data_loader import create_mmtd_data_module


class MMTDInterpretabilityAnalyzer:
    """
    Comprehensive interpretability analyzer for Logistic Regression MMTD
    """
    
    def __init__(
        self,
        model_checkpoint_path: str,
        mmtd_checkpoint_path: str,
        output_dir: str = "outputs/interpretability_analysis"
    ):
        """
        Initialize the interpretability analyzer
        
        Args:
            model_checkpoint_path: Path to trained interpretable model
            mmtd_checkpoint_path: Path to original MMTD checkpoint
            output_dir: Directory for analysis outputs
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.mmtd_checkpoint_path = mmtd_checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
        print(f"🔍 Interpretability Analyzer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💻 Device: {self.device}")
    
    def _load_model(self):
        """Load the trained interpretable model"""
        print("🏗️ Loading interpretable MMTD model...")
        
        # Create model
        model = create_interpretable_mmtd(
            classifier_type="logistic_regression",
            device=self.device,
            checkpoint_path=self.mmtd_checkpoint_path
        )
        
        # Load trained weights
        checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded with accuracy: {checkpoint.get('accuracy', 'N/A')}")
        return model
    
    def analyze_feature_importance(self) -> Dict:
        """
        Analyze feature importance from Logistic Regression weights
        """
        print("\n🔍 Analyzing Feature Importance...")
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        importance_np = importance.cpu().numpy()
        
        # Get weights and bias
        classifier = self.model.interpretable_classifier
        weights = classifier.linear.weight.data.cpu().numpy()  # Shape: (2, 768)
        bias = classifier.linear.bias.data.cpu().numpy()       # Shape: (2,)
        
        # Calculate statistics
        analysis = {
            'feature_importance': {
                'values': importance_np.tolist(),
                'mean': float(np.mean(importance_np)),
                'std': float(np.std(importance_np)),
                'max': float(np.max(importance_np)),
                'min': float(np.min(importance_np)),
                'top_10_indices': np.argsort(importance_np)[-10:].tolist(),
                'top_10_values': importance_np[np.argsort(importance_np)[-10:]].tolist()
            },
            'weights': {
                'spam_weights': weights[1].tolist(),  # Class 1 (Spam)
                'ham_weights': weights[0].tolist(),   # Class 0 (Ham)
                'bias': bias.tolist(),
                'weight_statistics': {
                    'spam_mean': float(np.mean(weights[1])),
                    'spam_std': float(np.std(weights[1])),
                    'ham_mean': float(np.mean(weights[0])),
                    'ham_std': float(np.std(weights[0])),
                    'weight_difference_mean': float(np.mean(weights[1] - weights[0]))
                }
            },
            'regularization_effects': {
                'l1_norm': float(classifier.get_feature_importance().sum()),
                'l2_norm': float(torch.norm(classifier.linear.weight, p=2)),
                'sparsity': float((torch.abs(classifier.linear.weight) < 1e-6).float().mean())
            }
        }
        
        print(f"✅ Feature importance analysis complete")
        print(f"📊 Top feature importance: {analysis['feature_importance']['max']:.4f}")
        print(f"📊 Average importance: {analysis['feature_importance']['mean']:.4f}")
        print(f"📊 Sparsity level: {analysis['regularization_effects']['sparsity']:.4f}")
        
        return analysis
    
    def visualize_feature_importance(self, analysis: Dict, top_k: int = 50):
        """
        Create visualizations for feature importance
        """
        print(f"\n📊 Creating feature importance visualizations (top {top_k})...")
        
        importance = np.array(analysis['feature_importance']['values'])
        spam_weights = np.array(analysis['weights']['spam_weights'])
        ham_weights = np.array(analysis['weights']['ham_weights'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Logistic Regression MMTD - Feature Importance Analysis', fontsize=16)
        
        # 1. Top K Feature Importance
        top_indices = np.argsort(importance)[-top_k:]
        top_importance = importance[top_indices]
        
        axes[0, 0].barh(range(top_k), top_importance)
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_ylabel('Feature Index')
        axes[0, 0].set_title(f'Top {top_k} Most Important Features')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Weight Distribution
        axes[0, 1].hist(spam_weights, bins=50, alpha=0.7, label='Spam Weights', color='red')
        axes[0, 1].hist(ham_weights, bins=50, alpha=0.7, label='Ham Weights', color='blue')
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Weight Distribution by Class')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Spam vs Ham Weight Comparison
        weight_diff = spam_weights - ham_weights
        axes[1, 0].scatter(range(len(weight_diff)), weight_diff, alpha=0.6, s=1)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Weight Difference (Spam - Ham)')
        axes[1, 0].set_title('Feature Preference: Spam vs Ham')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance Distribution
        axes[1, 1].hist(importance, bins=50, alpha=0.7, color='green')
        axes[1, 1].axvline(x=np.mean(importance), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(importance):.4f}')
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Overall Feature Importance Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "feature_importance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to: {plot_path}")
        
        return plot_path
    
    def analyze_decision_boundary(self, analysis: Dict) -> Dict:
        """
        Analyze the decision boundary characteristics
        """
        print("\n🎯 Analyzing Decision Boundary...")
        
        spam_weights = np.array(analysis['weights']['spam_weights'])
        ham_weights = np.array(analysis['weights']['ham_weights'])
        bias = np.array(analysis['weights']['bias'])
        
        # Decision boundary analysis
        decision_analysis = {
            'bias_effect': {
                'spam_bias': float(bias[1]),
                'ham_bias': float(bias[0]),
                'bias_difference': float(bias[1] - bias[0])
            },
            'separability': {
                'weight_magnitude_ratio': float(np.linalg.norm(spam_weights) / np.linalg.norm(ham_weights)),
                'cosine_similarity': float(np.dot(spam_weights, ham_weights) / 
                                         (np.linalg.norm(spam_weights) * np.linalg.norm(ham_weights))),
                'euclidean_distance': float(np.linalg.norm(spam_weights - ham_weights))
            },
            'feature_discrimination': {
                'highly_discriminative_features': int(np.sum(np.abs(spam_weights - ham_weights) > 0.1)),
                'moderately_discriminative_features': int(np.sum(
                    (np.abs(spam_weights - ham_weights) > 0.01) & 
                    (np.abs(spam_weights - ham_weights) <= 0.1)
                )),
                'low_discriminative_features': int(np.sum(np.abs(spam_weights - ham_weights) <= 0.01))
            }
        }
        
        print(f"✅ Decision boundary analysis complete")
        print(f"📊 Highly discriminative features: {decision_analysis['feature_discrimination']['highly_discriminative_features']}")
        print(f"📊 Cosine similarity: {decision_analysis['separability']['cosine_similarity']:.4f}")
        
        return decision_analysis
    
    def interpret_predictions(self, data_module, num_samples: int = 10) -> Dict:
        """
        Interpret individual predictions with feature contributions
        """
        print(f"\n🔍 Interpreting {num_samples} sample predictions...")
        
        # Get sample data
        test_loader = data_module.test_dataloader()
        sample_batch = next(iter(test_loader))
        
        # Move to device
        sample_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in sample_batch.items()}
        
        # Get actual batch size and limit to num_samples
        actual_batch_size = sample_batch['input_ids'].size(0)
        effective_samples = min(num_samples, actual_batch_size)
        
        # Limit to effective_samples
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key][:effective_samples]
        
        # Get predictions and features
        with torch.no_grad():
            # Extract features
            features = self.model.extract_features(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                pixel_values=sample_batch['pixel_values']
            )
            
            # Get predictions
            outputs = self.model(**sample_batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Get weights
            classifier = self.model.interpretable_classifier
            weights = classifier.linear.weight.data  # Shape: (2, 768)
            
            # Calculate feature contributions for each sample
            interpretations = []
            for i in range(effective_samples):
                sample_features = features[i]  # Shape: (768,)
                true_label = sample_batch['labels'][i].item()
                pred_label = predictions[i].item()
                pred_prob = probabilities[i].cpu().numpy()
                
                # Calculate contributions (feature * weight)
                spam_contributions = (sample_features * weights[1]).cpu().numpy()
                ham_contributions = (sample_features * weights[0]).cpu().numpy()
                
                # Top contributing features
                top_spam_indices = np.argsort(np.abs(spam_contributions))[-10:]
                top_ham_indices = np.argsort(np.abs(ham_contributions))[-10:]
                
                interpretation = {
                    'sample_id': i,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'prediction_probability': pred_prob.tolist(),
                    'correct_prediction': true_label == pred_label,
                    'feature_contributions': {
                        'spam_total': float(spam_contributions.sum()),
                        'ham_total': float(ham_contributions.sum()),
                        'top_spam_features': {
                            'indices': top_spam_indices.tolist(),
                            'contributions': spam_contributions[top_spam_indices].tolist()
                        },
                        'top_ham_features': {
                            'indices': top_ham_indices.tolist(),
                            'contributions': ham_contributions[top_ham_indices].tolist()
                        }
                    }
                }
                interpretations.append(interpretation)
        
        print(f"✅ Sample interpretations complete")
        print(f"📊 Correct predictions: {sum(1 for i in interpretations if i['correct_prediction'])}/{effective_samples}")
        
        return {'sample_interpretations': interpretations}
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate a comprehensive interpretability report
        """
        print("\n📋 Generating Comprehensive Interpretability Report...")
        
        # Load test data
        data_module = create_mmtd_data_module(
            csv_path="MMTD/DATA/email_data/EDP.csv",
            data_path="MMTD/DATA/email_data/pics",
            batch_size=16,
            max_samples=1000,
            num_workers=2
        )
        
        # Perform all analyses
        feature_analysis = self.analyze_feature_importance()
        decision_analysis = self.analyze_decision_boundary(feature_analysis)
        prediction_analysis = self.interpret_predictions(data_module, num_samples=20)
        
        # Create visualizations
        plot_path = self.visualize_feature_importance(feature_analysis)
        
        # Compile comprehensive report
        report = {
            'model_info': self.model.get_model_summary(),
            'feature_importance_analysis': feature_analysis,
            'decision_boundary_analysis': decision_analysis,
            'prediction_interpretations': prediction_analysis,
            'visualizations': {
                'feature_importance_plot': str(plot_path)
            },
            'interpretability_insights': self._generate_insights(
                feature_analysis, decision_analysis, prediction_analysis
            )
        }
        
        # Save report
        report_path = self.output_dir / "comprehensive_interpretability_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Comprehensive report saved to: {report_path}")
        return report
    
    def _generate_insights(self, feature_analysis: Dict, decision_analysis: Dict, 
                          prediction_analysis: Dict) -> Dict:
        """Generate human-readable insights from the analysis"""
        
        insights = {
            'key_findings': [],
            'feature_insights': [],
            'model_behavior': [],
            'interpretability_summary': ""
        }
        
        # Feature insights
        top_importance = feature_analysis['feature_importance']['max']
        avg_importance = feature_analysis['feature_importance']['mean']
        sparsity = feature_analysis['regularization_effects']['sparsity']
        
        if sparsity > 0.1:
            insights['feature_insights'].append(
                f"Model shows {sparsity:.1%} sparsity, indicating effective feature selection"
            )
        
        if top_importance > avg_importance * 5:
            insights['feature_insights'].append(
                "Some features are significantly more important than others, suggesting clear discriminative patterns"
            )
        
        # Decision boundary insights
        cosine_sim = decision_analysis['separability']['cosine_similarity']
        if cosine_sim < 0.5:
            insights['model_behavior'].append(
                "Spam and Ham weight vectors are well-separated, indicating good class discrimination"
            )
        
        # Prediction insights
        correct_predictions = sum(1 for i in prediction_analysis['sample_interpretations'] 
                                if i['correct_prediction'])
        total_predictions = len(prediction_analysis['sample_interpretations'])
        
        insights['key_findings'].append(
            f"Achieved {correct_predictions}/{total_predictions} correct predictions in sample analysis"
        )
        
        # Summary
        insights['interpretability_summary'] = (
            "The Logistic Regression classifier provides full interpretability through "
            "feature weights and contributions, enabling understanding of spam detection decisions."
        )
        
        return insights


def main():
    """Main analysis function"""
    # Paths
    model_checkpoint = "outputs/improved_training_20250530_184957/checkpoints/best_model.pt"
    mmtd_checkpoint = "MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"
    
    # Create analyzer
    analyzer = MMTDInterpretabilityAnalyzer(
        model_checkpoint_path=model_checkpoint,
        mmtd_checkpoint_path=mmtd_checkpoint
    )
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    print("\n🎉 Interpretability Analysis Complete!")
    print(f"📊 Report saved to: {analyzer.output_dir}")
    
    return report


if __name__ == "__main__":
    main() 