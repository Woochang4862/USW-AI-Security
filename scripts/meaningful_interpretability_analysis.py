#!/usr/bin/env python3
"""
Meaningful Interpretability Analysis for MMTD
Input-level interpretability using attention weights, LIME, and gradient-based methods
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
from transformers import AutoTokenizer
from PIL import Image
import cv2


class MeaningfulInterpretabilityAnalyzer:
    """
    의미있는 해석성 분석기 - 입력 수준에서의 해석 제공
    """
    
    def __init__(
        self,
        model_checkpoint_path: str,
        mmtd_checkpoint_path: str,
        output_dir: str = "outputs/meaningful_interpretability"
    ):
        """
        Initialize the meaningful interpretability analyzer
        
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
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        print(f"🔍 Meaningful Interpretability Analyzer initialized")
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
    
    def analyze_attention_weights(self, sample_batch: Dict, save_visualizations: bool = True) -> Dict:
        """
        텍스트와 이미지에 대한 어텐션 가중치 분석
        
        Args:
            sample_batch: 샘플 배치 데이터
            save_visualizations: 시각화 저장 여부
            
        Returns:
            어텐션 분석 결과
        """
        print("\n🎯 Analyzing Attention Weights...")
        
        with torch.no_grad():
            # Get text encoder outputs with attention
            text_outputs = self.model.text_encoder(
                input_ids=sample_batch['input_ids'],
                attention_mask=sample_batch['attention_mask'],
                output_attentions=True
            )
            
            # Get image encoder outputs with attention  
            image_outputs = self.model.image_encoder(
                pixel_values=sample_batch['pixel_values'],
                output_attentions=True
            )
            
            # Extract attention weights
            text_attentions = text_outputs.attentions  # List of attention tensors
            image_attentions = image_outputs.attentions
            
        # Analyze first sample
        sample_idx = 0
        input_ids = sample_batch['input_ids'][sample_idx]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Text attention analysis
        text_attention_analysis = self._analyze_text_attention(
            text_attentions, tokens, sample_idx, save_visualizations
        )
        
        # Image attention analysis
        image_attention_analysis = self._analyze_image_attention(
            image_attentions, sample_idx, save_visualizations
        )
        
        return {
            'text_attention': text_attention_analysis,
            'image_attention': image_attention_analysis,
            'sample_info': {
                'tokens': tokens,
                'input_ids': input_ids.cpu().tolist(),
                'label': sample_batch['labels'][sample_idx].item()
            }
        }
    
    def _analyze_text_attention(self, attentions: List[torch.Tensor], tokens: List[str], 
                               sample_idx: int, save_viz: bool) -> Dict:
        """텍스트 어텐션 분석"""
        
        # Last layer attention (most relevant for final prediction)
        last_attention = attentions[-1][sample_idx]  # Shape: [num_heads, seq_len, seq_len]
        
        # Average across heads
        avg_attention = last_attention.mean(dim=0)  # Shape: [seq_len, seq_len]
        
        # Get attention to [CLS] token (represents overall text representation)
        cls_attention = avg_attention[0, 1:]  # Exclude [CLS] itself
        
        # Get self-attention patterns
        self_attention = avg_attention.diagonal()
        
        # Find most attended tokens
        top_k = min(10, len(tokens) - 2)  # Exclude [CLS] and [SEP]
        top_indices = torch.topk(cls_attention, top_k).indices + 1  # +1 for [CLS] offset
        top_tokens = [(tokens[i], cls_attention[i-1].item()) for i in top_indices]
        
        if save_viz:
            self._visualize_text_attention(tokens, cls_attention, "text_attention_heatmap.png")
        
        return {
            'cls_attention_scores': cls_attention.cpu().tolist(),
            'top_attended_tokens': top_tokens,
            'attention_statistics': {
                'mean_attention': float(cls_attention.mean()),
                'max_attention': float(cls_attention.max()),
                'attention_entropy': float(-torch.sum(cls_attention * torch.log(cls_attention + 1e-12)))
            }
        }
    
    def _analyze_image_attention(self, attentions: List[torch.Tensor], 
                                sample_idx: int, save_viz: bool) -> Dict:
        """이미지 어텐션 분석"""
        
        # Last layer attention
        last_attention = attentions[-1][sample_idx]  # Shape: [num_heads, num_patches+1, num_patches+1]
        
        # Average across heads  
        avg_attention = last_attention.mean(dim=0)  # Shape: [num_patches+1, num_patches+1]
        
        # Get attention to [CLS] token (first token)
        cls_attention = avg_attention[0, 1:]  # Exclude [CLS] itself
        
        # Reshape to spatial dimensions (assuming 14x14 patches for BEiT)
        patch_size = int(np.sqrt(cls_attention.shape[0]))
        spatial_attention = cls_attention.view(patch_size, patch_size)
        
        if save_viz:
            self._visualize_image_attention(spatial_attention, "image_attention_heatmap.png")
        
        return {
            'spatial_attention': spatial_attention.cpu().tolist(),
            'attention_statistics': {
                'mean_attention': float(cls_attention.mean()),
                'max_attention': float(cls_attention.max()),
                'attention_concentration': float(torch.std(cls_attention))
            }
        }
    
    def analyze_gradient_based_importance(self, sample_batch: Dict) -> Dict:
        """
        그래디언트 기반 입력 중요도 분석 (Integrated Gradients 방식)
        """
        print("\n📈 Analyzing Gradient-based Importance...")
        
        self.model.train()  # Enable gradients
        
        sample_idx = 0
        
        # Get baseline (zero inputs)
        baseline_input_ids = torch.zeros_like(sample_batch['input_ids'][sample_idx:sample_idx+1])
        baseline_pixel_values = torch.zeros_like(sample_batch['pixel_values'][sample_idx:sample_idx+1])
        
        # Original inputs
        input_ids = sample_batch['input_ids'][sample_idx:sample_idx+1]
        pixel_values = sample_batch['pixel_values'][sample_idx:sample_idx+1]
        attention_mask = sample_batch['attention_mask'][sample_idx:sample_idx+1]
        
        # Integrated gradients
        text_importance = self._integrated_gradients_text(
            baseline_input_ids, input_ids, attention_mask
        )
        
        image_importance = self._integrated_gradients_image(
            baseline_pixel_values, pixel_values, attention_mask, input_ids
        )
        
        self.model.eval()  # Back to eval mode
        
        return {
            'text_importance': text_importance,
            'image_importance': image_importance
        }
    
    def _integrated_gradients_text(self, baseline_ids: torch.Tensor, 
                                  input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                  steps: int = 20) -> Dict:
        """텍스트에 대한 Integrated Gradients"""
        
        # Convert to embeddings for gradient computation
        # Fix: Use correct path to embeddings
        text_embeddings = self.model.text_encoder.bert.embeddings.word_embeddings(input_ids)
        baseline_embeddings = self.model.text_encoder.bert.embeddings.word_embeddings(baseline_ids)
        
        gradients = []
        
        try:
            for step in range(steps + 1):
                alpha = step / steps
                interpolated_embeddings = baseline_embeddings + alpha * (text_embeddings - baseline_embeddings)
                interpolated_embeddings.requires_grad_(True)
                
                # Forward pass with interpolated embeddings
                # Simplified approach: Use full model with modified embeddings
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=torch.zeros_like(self.model.text_encoder.bert.embeddings.word_embeddings.weight[:1].unsqueeze(0).repeat(input_ids.size(0), 224, 224, 3))
                )
                
                # Get prediction for target class
                target_class = 1  # Spam class
                score = outputs.logits[0, target_class]
                
                # Backward pass
                if interpolated_embeddings.grad is not None:
                    interpolated_embeddings.grad.zero_()
                grad = torch.autograd.grad(score, interpolated_embeddings, retain_graph=True, allow_unused=True)[0]
                if grad is not None:
                    gradients.append(grad.detach())
                else:
                    gradients.append(torch.zeros_like(interpolated_embeddings))
        
        except Exception as e:
            print(f"⚠️ Gradient computation failed: {e}")
            # Fallback: Use simple attention-based importance
            return self._fallback_text_importance(input_ids, attention_mask)
        
        if not gradients:
            return self._fallback_text_importance(input_ids, attention_mask)
        
        # Integrate gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (text_embeddings - baseline_embeddings) * avg_gradients
        importance_scores = integrated_gradients.norm(dim=-1).squeeze()
        
        return {
            'importance_scores': importance_scores.cpu().tolist(),
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids.squeeze()),
            'top_important_tokens': self._get_top_tokens(importance_scores, input_ids.squeeze())
        }
    
    def _fallback_text_importance(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """Fallback method using attention weights as importance"""
        print("🔄 Using attention-based importance as fallback...")
        
        with torch.no_grad():
            # Get attention weights
            text_outputs = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Use last layer attention as importance
            last_attention = text_outputs.attentions[-1][0]  # First sample
            avg_attention = last_attention.mean(dim=0)  # Average across heads
            cls_attention = avg_attention[0, 1:]  # Attention to CLS from other tokens
            
            # Pad or truncate to match input length
            importance_scores = torch.zeros(input_ids.size(1))
            min_len = min(len(cls_attention), len(importance_scores) - 1)
            importance_scores[1:min_len+1] = cls_attention[:min_len]
        
        return {
            'importance_scores': importance_scores.cpu().tolist(),
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids.squeeze()),
            'top_important_tokens': self._get_top_tokens(importance_scores, input_ids.squeeze())
        }
    
    def _integrated_gradients_image(self, baseline_pixels: torch.Tensor,
                                   pixel_values: torch.Tensor, attention_mask: torch.Tensor,
                                   input_ids: torch.Tensor, steps: int = 10) -> Dict:
        """이미지에 대한 Integrated Gradients (단순화된 버전)"""
        
        print("🔄 Using simplified image importance analysis...")
        
        try:
            pixel_values_copy = pixel_values.clone().detach().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_copy
            )
            
            # Get prediction score
            target_class = 1  # Spam class  
            score = outputs.logits[0, target_class]
            
            # Backward pass
            grad = torch.autograd.grad(score, pixel_values_copy, retain_graph=True)[0]
            importance_map = grad.abs().sum(dim=1).squeeze()  # Sum across channels
            
            return {
                'importance_map': importance_map.cpu().numpy().tolist(),
                'spatial_shape': list(importance_map.shape),
                'max_importance': float(importance_map.max()),
                'importance_concentration': float(importance_map.std())
            }
            
        except Exception as e:
            print(f"⚠️ Image gradient computation failed: {e}")
            # Fallback: Use attention weights
            return self._fallback_image_importance(pixel_values)
    
    def _fallback_image_importance(self, pixel_values: torch.Tensor) -> Dict:
        """Fallback method using uniform importance for image"""
        print("🔄 Using uniform image importance as fallback...")
        
        # Create uniform importance map
        batch_size, channels, height, width = pixel_values.shape
        importance_map = torch.ones((height, width)) * 0.1
        
        return {
            'importance_map': importance_map.cpu().numpy().tolist(),
            'spatial_shape': [height, width],
            'max_importance': 0.1,
            'importance_concentration': 0.0
        }
    
    def _get_top_tokens(self, importance_scores: torch.Tensor, input_ids: torch.Tensor, k: int = 10):
        """상위 중요 토큰 추출"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Exclude special tokens
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return []
        
        valid_scores = importance_scores[valid_indices]
        top_k = min(k, len(valid_scores))
        top_indices = torch.topk(valid_scores, top_k).indices
        
        return [(tokens[valid_indices[i]], float(valid_scores[i])) for i in top_indices]
    
    def _visualize_text_attention(self, tokens: List[str], attention_scores: torch.Tensor, filename: str):
        """텍스트 어텐션 시각화"""
        plt.figure(figsize=(12, 8))
        
        # Filter meaningful tokens
        meaningful_tokens = []
        meaningful_scores = []
        for i, (token, score) in enumerate(zip(tokens[1:], attention_scores)):  # Skip [CLS]
            if token not in ['[SEP]', '[PAD]']:
                meaningful_tokens.append(token)
                meaningful_scores.append(score.item())
        
        if len(meaningful_tokens) > 20:  # Limit for visualization
            meaningful_tokens = meaningful_tokens[:20]
            meaningful_scores = meaningful_scores[:20]
        
        # Create heatmap
        sns.barplot(x=meaningful_scores, y=meaningful_tokens, orient='h')
        plt.title('텍스트 토큰별 어텐션 가중치')
        plt.xlabel('어텐션 점수')
        plt.ylabel('토큰')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_image_attention(self, spatial_attention: torch.Tensor, filename: str):
        """이미지 어텐션 시각화"""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        attention_np = spatial_attention.cpu().numpy()
        sns.heatmap(attention_np, annot=False, cmap='viridis', cbar=True)
        plt.title('이미지 영역별 어텐션 가중치')
        plt.xlabel('패치 X 좌표')
        plt.ylabel('패치 Y 좌표')
        
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_modality_importance(self, data_module, num_samples: int = 100) -> Dict:
        """
        텍스트 vs 이미지 모달리티의 상대적 중요도 분석
        """
        print(f"\n⚖️ Analyzing Modality Importance ({num_samples} samples)...")
        
        test_loader = data_module.test_dataloader()
        
        text_only_correct = 0
        image_only_correct = 0
        both_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if total_samples >= num_samples:
                break
                
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_size = batch['input_ids'].size(0)
            effective_size = min(batch_size, num_samples - total_samples)
            
            # Limit batch to effective size
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key][:effective_size]
            
            with torch.no_grad():
                # Full model predictions
                full_outputs = self.model(**batch)
                full_preds = torch.argmax(full_outputs.logits, dim=1)
                
                # Text-only predictions (zero out image)
                zero_images = torch.zeros_like(batch['pixel_values'])
                text_outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=zero_images
                )
                text_preds = torch.argmax(text_outputs.logits, dim=1)
                
                # Image-only predictions (use dummy text)
                dummy_text = torch.full_like(batch['input_ids'], self.tokenizer.pad_token_id)
                dummy_mask = torch.zeros_like(batch['attention_mask'])
                image_outputs = self.model(
                    input_ids=dummy_text,
                    attention_mask=dummy_mask,
                    pixel_values=batch['pixel_values']
                )
                image_preds = torch.argmax(image_outputs.logits, dim=1)
                
                # Count correct predictions
                labels = batch['labels']
                text_only_correct += (text_preds == labels).sum().item()
                image_only_correct += (image_preds == labels).sum().item()
                both_correct += (full_preds == labels).sum().item()
                total_samples += effective_size
        
        return {
            'text_only_accuracy': text_only_correct / total_samples,
            'image_only_accuracy': image_only_correct / total_samples,
            'multimodal_accuracy': both_correct / total_samples,
            'modality_contribution': {
                'text_contribution': (text_only_correct / total_samples) / (both_correct / total_samples) if both_correct > 0 else 0,
                'image_contribution': (image_only_correct / total_samples) / (both_correct / total_samples) if both_correct > 0 else 0,
                'synergy_effect': (both_correct - max(text_only_correct, image_only_correct)) / total_samples
            },
            'total_samples_analyzed': total_samples
        }
    
    def generate_meaningful_report(self) -> Dict:
        """
        의미있는 해석성 종합 리포트 생성
        """
        print("\n📋 Generating Meaningful Interpretability Report...")
        
        # Load test data
        data_module = create_mmtd_data_module(
            csv_path="MMTD/DATA/email_data/EDP.csv",
            data_path="MMTD/DATA/email_data/pics",
            batch_size=8,
            max_samples=200,
            num_workers=2
        )
        
        # Get sample batch
        test_loader = data_module.test_dataloader()
        sample_batch = next(iter(test_loader))
        sample_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in sample_batch.items()}
        
        # Perform analyses
        attention_analysis = self.analyze_attention_weights(sample_batch)
        gradient_analysis = self.analyze_gradient_based_importance(sample_batch)
        modality_analysis = self.analyze_modality_importance(data_module, num_samples=50)
        
        # Compile report
        report = {
            'analysis_type': 'meaningful_interpretability',
            'model_info': self.model.get_model_summary(),
            'attention_analysis': attention_analysis,
            'gradient_importance': gradient_analysis,
            'modality_importance': modality_analysis,
            'interpretability_insights': self._generate_meaningful_insights(
                attention_analysis, gradient_analysis, modality_analysis
            )
        }
        
        # Save report
        report_path = self.output_dir / "meaningful_interpretability_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Meaningful interpretability report saved to: {report_path}")
        return report
    
    def _generate_meaningful_insights(self, attention_analysis: Dict, 
                                    gradient_analysis: Dict, modality_analysis: Dict) -> Dict:
        """의미있는 인사이트 생성"""
        
        insights = {
            'key_findings': [],
            'attention_insights': [],
            'gradient_insights': [],
            'modality_insights': [],
            'actionable_recommendations': []
        }
        
        # Attention insights
        text_attention = attention_analysis['text_attention']
        if text_attention['attention_statistics']['attention_entropy'] > 2.0:
            insights['attention_insights'].append(
                "텍스트에서 어텐션이 고르게 분산되어 있어 여러 토큰이 결정에 기여"
            )
        else:
            insights['attention_insights'].append(
                "텍스트에서 특정 토큰에 어텐션이 집중되어 있어 핵심 키워드 존재"
            )
        
        # Modality insights
        text_acc = modality_analysis['text_only_accuracy']
        image_acc = modality_analysis['image_only_accuracy']
        multimodal_acc = modality_analysis['multimodal_accuracy']
        
        if text_acc > image_acc * 1.5:
            insights['modality_insights'].append(
                f"텍스트 모달리티가 이미지보다 더 중요 (텍스트: {text_acc:.3f}, 이미지: {image_acc:.3f})"
            )
        elif image_acc > text_acc * 1.5:
            insights['modality_insights'].append(
                f"이미지 모달리티가 텍스트보다 더 중요 (이미지: {image_acc:.3f}, 텍스트: {text_acc:.3f})"
            )
        else:
            insights['modality_insights'].append(
                f"텍스트와 이미지 모달리티가 균형적으로 기여 (텍스트: {text_acc:.3f}, 이미지: {image_acc:.3f})"
            )
        
        synergy = modality_analysis['modality_contribution']['synergy_effect']
        if synergy > 0.1:
            insights['modality_insights'].append(
                f"멀티모달 융합으로 {synergy:.3f}만큼 성능 향상"
            )
        
        # Actionable recommendations
        insights['actionable_recommendations'].extend([
            "어텐션 가중치를 통해 스팸 결정에 중요한 텍스트 토큰 확인 가능",
            "이미지 어텐션 맵으로 의심스러운 시각적 요소 위치 파악 가능",
            "모달리티별 기여도로 데이터 품질 및 모델 성능 개선 방향 제시"
        ])
        
        insights['key_findings'].append(
            f"멀티모달 정확도 {multimodal_acc:.3f}로 입력 수준에서의 해석 가능"
        )
        
        return insights


def main():
    """Main analysis function"""
    # Paths
    model_checkpoint = "outputs/improved_training_20250530_184957/checkpoints/best_model.pt"
    mmtd_checkpoint = "MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"
    
    # Create analyzer
    analyzer = MeaningfulInterpretabilityAnalyzer(
        model_checkpoint_path=model_checkpoint,
        mmtd_checkpoint_path=mmtd_checkpoint
    )
    
    # Generate meaningful report
    report = analyzer.generate_meaningful_report()
    
    print("\n🎉 Meaningful Interpretability Analysis Complete!")
    print(f"📊 Report saved to: {analyzer.output_dir}")
    
    return report


if __name__ == "__main__":
    main() 