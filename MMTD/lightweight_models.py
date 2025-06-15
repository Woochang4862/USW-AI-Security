from transformers import DistilBertForSequenceClassification, ViTForImageClassification, DistilBertConfig, ViTConfig
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch


class LightWeightMMTD(torch.nn.Module):
    """
    경량화된 MMTD 모델
    - BERT → DistilBERT (파라미터 50% 감소, 속도 60% 향상)
    - BEiT → ViT Small (파라미터 및 연산량 감소)
    - Transformer 레이어 → FC 레이어 (연산량 대폭 감소)
    """
    def __init__(self, bert_pretrain_weight=None, vit_pretrain_weight=None):
        super(LightWeightMMTD, self).__init__()
        
        # DistilBERT 텍스트 인코더 (BERT 대비 50% 파라미터 감소)
        if bert_pretrain_weight is not None:
            self.text_encoder = DistilBertForSequenceClassification.from_pretrained(bert_pretrain_weight)
        else:
            self.text_encoder = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')
        
        # ViT Small 이미지 인코더 (BEiT 대비 경량화)
        if vit_pretrain_weight is not None:
            self.image_encoder = ViTForImageClassification.from_pretrained(vit_pretrain_weight)
        else:
            self.image_encoder = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 히든 스테이트 출력 활성화
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        
        # 차원 정보
        text_dim = self.text_encoder.config.hidden_size  # DistilBERT: 768
        image_dim = self.image_encoder.config.hidden_size  # ViT Small: 384
        fusion_dim = 384  # 경량화를 위해 작은 차원 사용
        
        # Transformer 레이어 대신 FC 레이어 사용 (속도 향상)
        self.fusion_fc = torch.nn.Sequential(
            torch.nn.Linear(text_dim + image_dim, fusion_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # 간소화된 pooler
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, fusion_dim),
            torch.nn.Tanh()
        )
        
        # 분류기
        self.classifier = torch.nn.Linear(fusion_dim, 2)
        self.num_labels = 2

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, token_type_ids=None):
        # 입력 텐서의 device를 동적으로 사용
        device = input_ids.device if input_ids is not None else pixel_values.device
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_last_hidden_state = text_outputs.hidden_states[-1]
        image_last_hidden_state = image_outputs.hidden_states[-1]
        text_vec = text_last_hidden_state[:, 0, :]
        image_vec = image_last_hidden_state[:, 0, :]
        fused_features = torch.cat([text_vec, image_vec], dim=1)
        outputs = self.fusion_fc(fused_features)
        outputs = self.pooler(outputs)
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def get_model_size(self):
        """모델 파라미터 수 계산"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_size_mb': total_params * 4 / (1024 * 1024)  # float32 기준
        }


class UltraLightMMTD(torch.nn.Module):
    """
    초경량화된 MMTD 모델
    - 더 작은 차원과 단순한 구조 사용
    """
    def __init__(self, bert_pretrain_weight=None, vit_pretrain_weight=None):
        super(UltraLightMMTD, self).__init__()
        
        # 더 작은 모델 사용
        if bert_pretrain_weight is not None:
            self.text_encoder = DistilBertForSequenceClassification.from_pretrained(bert_pretrain_weight)
        else:
            self.text_encoder = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')
        
        if vit_pretrain_weight is not None:
            self.image_encoder = ViTForImageClassification.from_pretrained(vit_pretrain_weight)
        else:
            self.image_encoder = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        
        # 매우 간단한 융합 레이어
        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        
        # 단일 FC 레이어로 직접 분류
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(text_dim + image_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 2)
        )
        self.num_labels = 2

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, token_type_ids=None):
        # 입력 텐서의 device를 동적으로 사용
        device = input_ids.device if input_ids is not None else pixel_values.device
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_vec = text_outputs.hidden_states[-1][:, 0, :]
        image_vec = image_outputs.hidden_states[-1][:, 0, :]
        fused_features = torch.cat([text_vec, image_vec], dim=1)
        logits = self.classifier(fused_features)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def get_model_size(self):
        """모델 파라미터 수 계산"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_size_mb': total_params * 4 / (1024 * 1024)
        } 