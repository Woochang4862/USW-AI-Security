# my_custom_models.py

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, ConvNextForImageClassification, BertConfig, ConvNextConfig
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss

class ConvNextMMTD(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), convnext_cfg=None, bert_pretrain_weight=None, convnext_pretrain_weight=None):
        super(ConvNextMMTD, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        
        if convnext_pretrain_weight is not None:
            self.image_encoder = ConvNextForImageClassification.from_pretrained(
                convnext_pretrain_weight,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        else:
            if convnext_cfg is None:
                convnext_cfg = ConvNextConfig.from_pretrained("facebook/convnext-base-224")
            self.image_encoder = ConvNextForImageClassification(convnext_cfg)

        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True

        # ConvNeXt의 hidden_size(1024)를 BERT의 hidden_size(768)에 맞춰주는 선형 레이어 추가
        self.image_projection = nn.Linear(1024, 768) # ConvNeXt-base의 기본 hidden_size는 1024입니다.

        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh()
        )
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        
        text_last_hidden_state = text_outputs.hidden_states[12] # [batch_size, sequence_length, 768]
        
        # ConvNeXt의 출력에서 이미지 특징을 추출하고 차원을 맞춥니다.
        image_feature = image_outputs.hidden_states[-1] 
            
        # 이전 차원 오류 해결을 위해 Global Average Pooling 후 unsqueeze(1) 했던 로직
        if image_feature.dim() == 4:
            image_feature = image_feature.mean(dim=(-2, -1)) # [batch_size, channels] (여기서 channels는 1024)

        # 1024 차원의 이미지 피처를 768 차원으로 변환
        image_projected_feature = self.image_projection(image_feature) # [batch_size, 768]
            
        # 텍스트 hidden state와 차원을 맞추기 위해 unsqueeze(1)로 중간 차원을 추가합니다.
        image_last_hidden_state = image_projected_feature.unsqueeze(1) # [batch_size, 1, 768]


        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)

        # 이제 두 텐서의 hidden_size(dim=2)가 768로 일치하므로 연결 가능
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        
        outputs = self.pooler(outputs[:, 0, :]) 
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_size_mb = total_params * 4 / (1024 * 1024) # Float32 기준

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_size_mb': total_size_mb
        }