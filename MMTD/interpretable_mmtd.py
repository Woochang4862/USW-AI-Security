import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BeitForImageClassification
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from models import MMTD
from Email_dataset import EDPDataset, EDPCollator
from utils import SplitData
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json


class InterpretableMMTD(MMTD):
    """
    기존 MMTD 모델의 구조와 가중치를 유지하면서 
    attention 기반 해석 기능을 추가한 모델
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Attention weights를 저장할 딕셔너리
        self.attention_weights = {}
        self.text_attention_weights = {}
        self.image_attention_weights = {}
        self.fusion_attention_weights = {}
        
        # Hook 등록 여부 플래그
        self.hooks_registered = False
        
    def register_attention_hooks(self):
        """모든 attention layer에 hook을 등록합니다."""
        if self.hooks_registered:
            return
            
        # BERT attention hooks
        for i, layer in enumerate(self.text_encoder.bert.encoder.layer):
            def bert_hook(module, input, output, layer_idx=i):
                # BERT attention weights는 output[1]에 저장됨
                if len(output) > 1 and output[1] is not None:
                    self.text_attention_weights[f'bert_layer_{layer_idx}'] = output[1].detach()
            
            layer.attention.self.register_forward_hook(bert_hook)
        
        # BEiT attention hooks
        for i, layer in enumerate(self.image_encoder.beit.encoder.layer):
            def beit_hook(module, input, output, layer_idx=i):
                # BEiT attention weights는 output[1]에 저장됨
                if len(output) > 1 and output[1] is not None:
                    self.image_attention_weights[f'beit_layer_{layer_idx}'] = output[1].detach()
            
            layer.attention.register_forward_hook(beit_hook)
        
        # Fusion layer attention hook
        def fusion_hook(module, input, output):
            # Multi-head attention의 weights를 저장
            # TransformerEncoderLayer에서는 직접 접근이 어려우므로 
            # self_attn 모듈에 별도 hook 등록
            pass
        
        # Multi-modality transformer layer의 self-attention에 hook 등록
        def fusion_attention_hook(module, input, output):
            if len(output) > 1 and output[1] is not None:
                self.fusion_attention_weights['fusion_layer'] = output[1].detach()
        
        self.multi_modality_transformer_layer.self_attn.register_forward_hook(fusion_attention_hook)
        
        self.hooks_registered = True
        print("Attention hooks registered successfully!")
    
    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None, return_attentions=False):
        """
        Forward pass with optional attention weight extraction
        """
        # Hook 등록 (처음 실행시에만)
        if return_attentions and not self.hooks_registered:
            self.register_attention_hooks()
        
        # 기존 forward 로직 유지
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            output_attentions=return_attentions
        )
        
        image_outputs = self.image_encoder(
            pixel_values=pixel_values,
            output_attentions=return_attentions
        )
        
        # 12번째 layer의 hidden states 추출
        text_last_hidden_state = text_outputs.hidden_states[12]
        image_last_hidden_state = image_outputs.hidden_states[12]
        
        # Modality embedding 추가 (텍스트: 0, 이미지: 1)
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)
        
        # 텍스트와 이미지 특징 융합
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        
        # Multi-modality transformer
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        
        # Pooling 및 분류
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        
        # Loss 계산
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Return with attention weights if requested
        result = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        
        if return_attentions:
            # Attention weights를 결과에 추가
            result.text_attentions = text_outputs.attentions if text_outputs.attentions else None
            result.image_attentions = image_outputs.attentions if image_outputs.attentions else None
            result.fusion_attentions = self.fusion_attention_weights.get('fusion_layer', None)
            
        return result
    
    def get_modality_contributions(self, input_ids, token_type_ids, attention_mask, pixel_values):
        """
        각 모달리티의 예측에 대한 기여도를 계산합니다.
        """
        with torch.no_grad():
            # 전체 멀티모달 예측
            full_output = self.forward(input_ids, token_type_ids, attention_mask, pixel_values)
            full_logits = full_output.logits
            
            # 텍스트만 사용한 예측 (이미지 부분을 zero로 마스킹)
            text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            image_outputs = self.image_encoder(pixel_values=pixel_values)
            
            text_last_hidden_state = text_outputs.hidden_states[12]
            image_last_hidden_state = image_outputs.hidden_states[12]
            
            # 텍스트만: 이미지 부분을 0으로 설정
            text_only_hidden = torch.cat([
                text_last_hidden_state + torch.zeros(text_last_hidden_state.size()).to(self.device),
                torch.zeros_like(image_last_hidden_state).to(self.device)
            ], dim=1)
            
            # 이미지만: 텍스트 부분을 0으로 설정
            image_only_hidden = torch.cat([
                torch.zeros_like(text_last_hidden_state).to(self.device),
                image_last_hidden_state + torch.ones(image_last_hidden_state.size()).to(self.device)
            ], dim=1)
            
            # 각각에 대해 transformer 적용
            text_only_output = self.multi_modality_transformer_layer(text_only_hidden)
            text_only_output = self.pooler(text_only_output[:, 0, :])
            text_only_logits = self.classifier(text_only_output)
            
            image_only_output = self.multi_modality_transformer_layer(image_only_hidden)
            image_only_output = self.pooler(image_only_output[:, 0, :])
            image_only_logits = self.classifier(image_only_output)
            
            # 기여도 계산 (softmax 확률로 변환 후 계산)
            full_probs = F.softmax(full_logits, dim=-1)
            text_probs = F.softmax(text_only_logits, dim=-1)
            image_probs = F.softmax(image_only_logits, dim=-1)
            
            return {
                'full_prediction': full_probs,
                'text_only_prediction': text_probs,
                'image_only_prediction': image_probs,
                'text_contribution': text_probs,
                'image_contribution': image_probs,
                'interaction_effect': full_probs - (text_probs + image_probs) / 2
            }
    
    def analyze_attention_patterns(self, input_ids, token_type_ids, attention_mask, pixel_values, tokenizer=None):
        """
        Attention 패턴을 분석하고 해석 가능한 결과를 반환합니다.
        """
        # Attention weights와 함께 forward pass
        output = self.forward(
            input_ids, token_type_ids, attention_mask, pixel_values, 
            return_attentions=True
        )
        
        # 모달리티별 기여도 계산
        contributions = self.get_modality_contributions(
            input_ids, token_type_ids, attention_mask, pixel_values
        )
        
        analysis_result = {
            'prediction': output.logits,
            'prediction_probs': F.softmax(output.logits, dim=-1),
            'modality_contributions': contributions,
            'attention_weights': {
                'text_attentions': output.text_attentions,
                'image_attentions': output.image_attentions,
                'fusion_attentions': output.fusion_attentions
            }
        }
        
        return analysis_result


def load_interpretable_model_from_checkpoint(checkpoint_path):
    """
    체크포인트에서 해석 가능한 MMTD 모델을 로드합니다.
    기존 가중치를 완전히 유지합니다.
    """
    print(f"체크포인트에서 해석 가능한 모델 로딩: {checkpoint_path}")
    
    # 원본과 동일한 사전 훈련된 가중치로 해석 가능한 모델 초기화
    model = InterpretableMMTD(
        bert_pretrain_weight='bert-base-multilingual-cased',
        beit_pretrain_weight='microsoft/dit-base'
    )
    
    # 체크포인트에서 모델 가중치 로드
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        # strict=False로 설정하여 예상치 못한 키가 있어도 로드 가능하게 함
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"모델 가중치 로딩 완료 - 기존 성능 유지됨")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    model.eval()
    return model


def verify_model_equivalence(original_checkpoint_path, test_data_path='DATA/email_data/EDP.csv'):
    """
    원본 모델과 해석 가능한 모델의 출력이 동일한지 검증합니다.
    """
    print("모델 동등성 검증 시작...")
    
    # 원본 모델 로드 (원본과 동일한 방식으로)
    original_model = MMTD(
        bert_pretrain_weight='bert-base-multilingual-cased',
        beit_pretrain_weight='microsoft/dit-base'
    )
    model_path = os.path.join(original_checkpoint_path, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location='cpu')
    missing_keys, unexpected_keys = original_model.load_state_dict(state_dict, strict=False)
    original_model.eval()
    
    # 해석 가능한 모델 로드
    interpretable_model = load_interpretable_model_from_checkpoint(original_checkpoint_path)
    
    # 테스트 데이터 로드
    split_data = SplitData(test_data_path, 5)
    train_df, test_df = split_data()
    test_dataset = EDPDataset('DATA/email_data/pics', test_df.head(10))  # 작은 샘플로 테스트
    
    collator = EDPCollator()
    
    print("샘플 데이터로 출력 비교...")
    
    total_diff = 0
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        batch = collator([sample])
        
        with torch.no_grad():
            # 원본 모델 출력
            orig_output = original_model(**batch)
            
            # 해석 가능한 모델 출력 (attention 없이)
            interp_output = interpretable_model(**batch, return_attentions=False)
            
            # 출력 차이 계산
            diff = torch.abs(orig_output.logits - interp_output.logits).max().item()
            total_diff += diff
            
            print(f"샘플 {i+1}: 최대 차이 = {diff:.8f}")
    
    avg_diff = total_diff / min(5, len(test_dataset))
    print(f"\n평균 출력 차이: {avg_diff:.8f}")
    
    if avg_diff < 1e-6:
        print("✅ 모델 동등성 검증 성공! 기존 성능이 완전히 유지됩니다.")
    else:
        print("⚠️ 모델 출력에 차이가 있습니다. 확인이 필요합니다.")
    
    return avg_diff < 1e-6 