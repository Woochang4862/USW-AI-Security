"""
Attention-based Interpretability Analysis for MMTD Models
다중모달 스팸 탐지 모델의 Attention 기반 해석가능성 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import cv2
from PIL import Image
import logging
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """MMTD 모델의 Attention 기반 해석가능성 분석기"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: 훈련된 MMTD 모델
            tokenizer: BERT 토크나이저
            device: 계산 디바이스
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Attention hook 등록
        self.attention_weights = {}
        self._register_hooks()
        
        logger.info("🔍 Attention Analyzer 초기화 완료")
    
    def _register_hooks(self):
        """Attention weight 추출을 위한 hook 등록"""
        
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(output, 'attentions') and output.attentions is not None:
                    self.attention_weights[name] = output.attentions
                elif isinstance(output, tuple) and len(output) > 1:
                    # attention이 tuple의 일부로 반환되는 경우
                    for i, item in enumerate(output):
                        if hasattr(item, 'shape') and len(item.shape) == 4:  # attention 형태
                            self.attention_weights[f"{name}_{i}"] = item
            return hook
        
        # 각 인코더에 hook 등록
        if hasattr(self.model, 'text_encoder'):
            self.model.text_encoder.register_forward_hook(hook_fn('text_encoder'))
        if hasattr(self.model, 'image_encoder'):
            self.model.image_encoder.register_forward_hook(hook_fn('image_encoder'))
        if hasattr(self.model, 'fusion_layer'):
            self.model.fusion_layer.register_forward_hook(hook_fn('fusion_layer'))
    
    def extract_attention_weights(self, input_ids: torch.Tensor, 
                                pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """모든 레벨의 attention weights 추출"""
        
        self.attention_weights.clear()
        
        with torch.no_grad():
            # Forward pass with attention extraction
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_attentions=True
            )
        
        # Attention weights 정리
        processed_attentions = {}
        
        for name, weights in self.attention_weights.items():
            if isinstance(weights, (list, tuple)):
                # 여러 레이어의 attention
                processed_attentions[name] = [w.cpu() for w in weights]
            else:
                processed_attentions[name] = weights.cpu()
        
        return processed_attentions, outputs
    
    def analyze_cross_modal_attention(self, fusion_attentions: torch.Tensor,
                                    text_length: int, image_length: int) -> Dict[str, torch.Tensor]:
        """Cross-modal attention 분석"""
        
        if isinstance(fusion_attentions, (list, tuple)):
            # 마지막 레이어 사용
            attention_matrix = fusion_attentions[-1]
        else:
            attention_matrix = fusion_attentions
        
        # 헤드 평균
        if len(attention_matrix.shape) == 4:  # [batch, heads, seq, seq]
            attention_matrix = attention_matrix.mean(dim=1)
        
        batch_size, seq_len, _ = attention_matrix.shape
        
        # 시퀀스 구조: [CLS, 텍스트토큰들, SEP, 이미지패치들]
        text_end = text_length
        image_start = text_length
        image_end = image_start + image_length
        
        # 영역별 attention 추출
        regions = {
            'text_to_text': attention_matrix[:, 1:text_end, 1:text_end],
            'text_to_image': attention_matrix[:, 1:text_end, image_start:image_end],
            'image_to_text': attention_matrix[:, image_start:image_end, 1:text_end],
            'image_to_image': attention_matrix[:, image_start:image_end, image_start:image_end],
            'cls_to_text': attention_matrix[:, 0:1, 1:text_end],
            'cls_to_image': attention_matrix[:, 0:1, image_start:image_end],
            'full_matrix': attention_matrix
        }
        
        return regions
    
    def get_token_importance(self, cross_modal_attention: Dict[str, torch.Tensor],
                           tokens: List[str]) -> List[Dict[str, Any]]:
        """텍스트 토큰별 중요도 계산"""
        
        text_to_image = cross_modal_attention['text_to_image']
        cls_to_text = cross_modal_attention['cls_to_text']
        
        # 여러 중요도 메트릭 계산
        token_importance = []
        
        for i, token in enumerate(tokens[1:]):  # CLS 제외
            if i >= text_to_image.shape[1]:
                break
                
            # 1. 이미지에 대한 attention 합계
            image_attention = text_to_image[0, i, :].sum().item()
            
            # 2. CLS 토큰으로부터의 attention
            cls_attention = cls_to_text[0, 0, i].item() if i < cls_to_text.shape[2] else 0
            
            # 3. 자기 자신에 대한 attention (text_to_text)
            if 'text_to_text' in cross_modal_attention:
                text_attention = cross_modal_attention['text_to_text'][0, i, i].item() if i < cross_modal_attention['text_to_text'].shape[1] else 0
            else:
                text_attention = 0
            
            # 4. 종합 중요도 (가중평균)
            combined_importance = (image_attention * 0.4 + cls_attention * 0.4 + text_attention * 0.2)
            
            token_importance.append({
                'token': token,
                'index': i,
                'image_attention': image_attention,
                'cls_attention': cls_attention,
                'text_attention': text_attention,
                'combined_importance': combined_importance,
                'is_special': token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
            })
        
        # 특수 토큰 제외하고 정렬
        filtered_importance = [t for t in token_importance if not t['is_special']]
        return sorted(filtered_importance, key=lambda x: x['combined_importance'], reverse=True)
    
    def get_patch_importance(self, cross_modal_attention: Dict[str, torch.Tensor],
                           patch_coordinates: Optional[List[Tuple[int, int]]] = None) -> List[Dict[str, Any]]:
        """이미지 패치별 중요도 계산"""
        
        image_to_text = cross_modal_attention['image_to_text']
        cls_to_image = cross_modal_attention['cls_to_image']
        
        num_patches = image_to_text.shape[1]
        
        # 패치 좌표가 없으면 생성 (14x14 grid 가정)
        if patch_coordinates is None:
            patch_size = int(np.sqrt(num_patches))
            patch_coordinates = [(i // patch_size, i % patch_size) for i in range(num_patches)]
        
        patch_importance = []
        
        for i in range(min(num_patches, len(patch_coordinates))):
            # 1. 텍스트로부터 받는 attention 합계
            text_attention = image_to_text[0, i, :].sum().item()
            
            # 2. CLS 토큰으로부터의 attention
            cls_attention = cls_to_image[0, 0, i].item() if i < cls_to_image.shape[2] else 0
            
            # 3. 다른 패치로부터의 attention
            if 'image_to_image' in cross_modal_attention:
                image_attention = cross_modal_attention['image_to_image'][0, i, :].sum().item() - cross_modal_attention['image_to_image'][0, i, i].item()
            else:
                image_attention = 0
            
            # 4. 종합 중요도
            combined_importance = (text_attention * 0.5 + cls_attention * 0.3 + image_attention * 0.2)
            
            patch_importance.append({
                'patch_index': i,
                'coordinates': patch_coordinates[i] if i < len(patch_coordinates) else (0, 0),
                'text_attention': text_attention,
                'cls_attention': cls_attention,
                'image_attention': image_attention,
                'combined_importance': combined_importance
            })
        
        return sorted(patch_importance, key=lambda x: x['combined_importance'], reverse=True)
    
    def attention_rollout(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """Attention Rollout: 레이어별 attention 누적 계산"""
        
        if not attentions:
            return None
        
        # 첫 번째 attention으로 초기화
        result = attentions[0].mean(dim=1)  # 헤드 평균
        
        # 각 레이어의 attention을 순차적으로 곱함
        for attention in attentions[1:]:
            avg_attention = attention.mean(dim=1)
            # Residual connection 고려
            avg_attention = avg_attention + torch.eye(avg_attention.size(-1)).to(avg_attention.device)
            avg_attention = avg_attention / avg_attention.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(avg_attention, result)
        
        return result
    
    def gradient_weighted_attention(self, input_ids: torch.Tensor,
                                  pixel_values: torch.Tensor,
                                  target_class: int = 1) -> Dict[str, torch.Tensor]:
        """Gradient-weighted Attention 계산"""
        
        # 그래디언트 계산을 위해 requires_grad 설정
        input_ids = input_ids.clone().detach().requires_grad_(True)
        pixel_values = pixel_values.clone().detach().requires_grad_(True)
        
        self.model.train()  # gradient 계산을 위해 train 모드
        
        try:
            # Forward pass
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, output_attentions=True)
            
            # Target class의 logit에 대한 loss 계산
            target_logit = outputs.logits[0, target_class]
            
            # Backward pass
            target_logit.backward()
            
            # Attention과 gradient의 곱
            weighted_attentions = {}
            
            for name, attentions in self.attention_weights.items():
                if isinstance(attentions, (list, tuple)):
                    weighted_attentions[name] = []
                    for attention in attentions:
                        if attention.requires_grad:
                            grad = torch.autograd.grad(target_logit, attention, retain_graph=True)[0]
                            weighted = attention * grad
                            weighted_attentions[name].append(weighted.mean(dim=1))  # 헤드 평균
                else:
                    if attentions.requires_grad:
                        grad = torch.autograd.grad(target_logit, attentions, retain_graph=True)[0]
                        weighted = attentions * grad
                        weighted_attentions[name] = weighted.mean(dim=1)
            
            return weighted_attentions
            
        finally:
            self.model.eval()  # 다시 eval 모드로
    
    def explain_prediction(self, text: str, image: torch.Tensor,
                         return_attention_maps: bool = True) -> Dict[str, Any]:
        """예측에 대한 종합적인 해석 제공"""
        
        # 텍스트 토큰화
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 이미지 준비
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        # 예측 수행
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, pixel_values=image)
            prediction = torch.sigmoid(outputs.logits).cpu().numpy()[0, 0]
            predicted_class = int(prediction > 0.5)
        
        # Attention 분석
        attentions, full_outputs = self.extract_attention_weights(input_ids, image)
        
        # 시퀀스 길이 계산
        text_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
        image_length = 196  # 14x14 patches for BEiT
        
        # Cross-modal attention 분석
        if 'fusion_layer' in attentions:
            cross_modal = self.analyze_cross_modal_attention(
                attentions['fusion_layer'], text_length, image_length
            )
        else:
            cross_modal = {}
        
        # 토큰 중요도
        token_importance = self.get_token_importance(cross_modal, tokens) if cross_modal else []
        
        # 패치 중요도
        patch_importance = self.get_patch_importance(cross_modal) if cross_modal else []
        
        # 결과 정리
        explanation = {
            'prediction': {
                'score': float(prediction),
                'class': predicted_class,
                'confidence': abs(prediction - 0.5) * 2,
                'label': 'SPAM' if predicted_class == 1 else 'HAM'
            },
            'text_analysis': {
                'original_text': text,
                'tokens': tokens,
                'important_tokens': token_importance[:10],
                'text_length': text_length
            },
            'image_analysis': {
                'important_patches': patch_importance[:20],
                'image_shape': image.shape,
                'num_patches': image_length
            },
            'cross_modal_analysis': {
                'text_to_image_strength': float(cross_modal.get('text_to_image', torch.tensor(0)).mean()) if cross_modal else 0,
                'image_to_text_strength': float(cross_modal.get('image_to_text', torch.tensor(0)).mean()) if cross_modal else 0,
                'modality_balance': self._calculate_modality_balance(cross_modal) if cross_modal else 0.5
            }
        }
        
        if return_attention_maps:
            explanation['attention_maps'] = {
                'cross_modal_attention': cross_modal,
                'all_attentions': attentions
            }
        
        return explanation
    
    def _calculate_modality_balance(self, cross_modal: Dict[str, torch.Tensor]) -> float:
        """모달리티간 균형 계산 (0: 텍스트 중심, 1: 이미지 중심)"""
        
        if not cross_modal:
            return 0.5
        
        text_contribution = 0
        image_contribution = 0
        
        if 'cls_to_text' in cross_modal:
            text_contribution += cross_modal['cls_to_text'].sum().item()
        if 'cls_to_image' in cross_modal:
            image_contribution += cross_modal['cls_to_image'].sum().item()
        
        total = text_contribution + image_contribution
        if total > 0:
            return image_contribution / total
        return 0.5
    
    def save_explanation(self, explanation: Dict[str, Any], 
                        output_path: str, include_attention_maps: bool = False):
        """해석 결과를 파일로 저장"""
        
        # Attention maps는 용량이 크므로 선택적으로 저장
        if not include_attention_maps and 'attention_maps' in explanation:
            del explanation['attention_maps']
        
        # Tensor를 리스트로 변환
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            return obj
        
        explanation_serializable = convert_tensors(explanation)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(explanation_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 해석 결과 저장: {output_path}") 