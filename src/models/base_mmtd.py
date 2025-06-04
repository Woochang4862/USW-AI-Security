"""
Base MMTD Model Implementation
Based on the original MMTD code with Document Image Transformer (DiT) architecture
"""

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification, 
    BeitForImageClassification, 
    BertConfig, 
    BeitConfig
)
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseMMTD(nn.Module):
    """
    Base MMTD Model implementing the original architecture from the paper
    
    Architecture:
    1. Multilingual Text Encoder (MTE): BERT-base multilingual cased
    2. Document Image Encoder (DIE): Document Image Transformer (DiT)
    3. Multimodal Fusion Module: Modal type embedding + concatenation + transformer
    4. MLP Classifier (to be replaced with interpretable classifiers)
    """
    
    def __init__(
        self,
        bert_config: Optional[BertConfig] = None,
        beit_config: Optional[BeitConfig] = None,
        bert_pretrain_weight: Optional[str] = "bert-base-multilingual-cased",
        beit_pretrain_weight: Optional[str] = "microsoft/dit-base",
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        fusion_hidden_size: int = 768,
        fusion_num_heads: int = 8
    ):
        super(BaseMMTD, self).__init__()
        
        # Initialize configurations
        self.num_labels = num_labels
        self.fusion_hidden_size = fusion_hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Multilingual Text Encoder (MTE) - BERT
        if bert_pretrain_weight:
            self.text_encoder = BertForSequenceClassification.from_pretrained(
                bert_pretrain_weight, 
                num_labels=num_labels
            )
        else:
            bert_config = bert_config or BertConfig()
            self.text_encoder = BertForSequenceClassification(bert_config)
        
        # Enable hidden states output for fusion
        self.text_encoder.config.output_hidden_states = True
        
        # 2. Document Image Encoder (DIE) - DiT
        if beit_pretrain_weight:
            self.image_encoder = BeitForImageClassification.from_pretrained(
                beit_pretrain_weight,
                num_labels=num_labels
            )
        else:
            beit_config = beit_config or BeitConfig()
            self.image_encoder = BeitForImageClassification(beit_config)
        
        # Enable hidden states output for fusion
        self.image_encoder.config.output_hidden_states = True
        
        # 3. Multimodal Fusion Module
        self.multimodal_fusion = nn.TransformerEncoderLayer(
            d_model=fusion_hidden_size,
            nhead=fusion_num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Modal type embeddings (to distinguish text vs image features)
        self.text_modal_embedding = nn.Parameter(torch.zeros(1, 1, fusion_hidden_size))
        self.image_modal_embedding = nn.Parameter(torch.ones(1, 1, fusion_hidden_size))
        
        # 4. Pooler and Classifier (MLP head to be replaced)
        self.pooler = nn.Sequential(
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fusion_hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize modal type embeddings"""
        nn.init.normal_(self.text_modal_embedding, std=0.02)
        nn.init.normal_(self.image_modal_embedding, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> SequenceClassifierOutput:
        """
        Forward pass of the MMTD model
        
        Args:
            input_ids: Text input token ids [batch_size, seq_len]
            token_type_ids: Token type ids for BERT
            attention_mask: Attention mask for text
            pixel_values: Image pixel values [batch_size, channels, height, width]
            labels: Ground truth labels [batch_size]
            return_hidden_states: Whether to return intermediate hidden states
            
        Returns:
            SequenceClassifierOutput with loss, logits, and optional hidden states
        """
        
        # 1. Text encoding with BERT MTE
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        # Get last hidden state from BERT (layer 12)
        text_hidden_state = text_outputs.hidden_states[12]  # [batch_size, seq_len, 768]
        
        # 2. Image encoding with DiT DIE
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        
        # Get last hidden state from DiT (layer 12)
        image_hidden_state = image_outputs.hidden_states[12]  # [batch_size, num_patches, 768]
        
        # 3. Add modal type embeddings
        batch_size = text_hidden_state.size(0)
        
        # Add text modal embedding (zeros)
        text_modal_emb = self.text_modal_embedding.expand(
            batch_size, text_hidden_state.size(1), -1
        )
        text_hidden_state = text_hidden_state + text_modal_emb
        
        # Add image modal embedding (ones)
        image_modal_emb = self.image_modal_embedding.expand(
            batch_size, image_hidden_state.size(1), -1
        )
        image_hidden_state = image_hidden_state + image_modal_emb
        
        # 4. Concatenate text and image features
        # Text: [batch_size, 256, 768] -> [40, 256, 768] (논문 기준)
        # Image: [batch_size, 197, 768] -> [40, 197, 768] (논문 기준)
        fused_hidden_state = torch.cat([text_hidden_state, image_hidden_state], dim=1)
        # Result: [batch_size, 453, 768] -> [40, 453, 768] (논문 기준)
        
        # 5. Multimodal fusion with transformer
        fusion_output = self.multimodal_fusion(fused_hidden_state)
        
        # 6. Pooling (use CLS token equivalent - first token)
        pooled_output = self.pooler(fusion_output[:, 0, :])
        
        # 7. Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 8. Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Prepare output
        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=fusion_output if return_hidden_states else None,
            attentions=None,
        )
        
        return output
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract text features only"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        return text_outputs.hidden_states[12]
    
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features only"""
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        return image_outputs.hidden_states[12]
    
    def get_fused_features(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None
    ) -> torch.Tensor:
        """Get fused multimodal features before classification"""
        # Get individual modality features
        text_hidden_state = self.get_text_features(input_ids, token_type_ids, attention_mask)
        image_hidden_state = self.get_image_features(pixel_values)
        
        # Add modal type embeddings
        batch_size = text_hidden_state.size(0)
        
        text_modal_emb = self.text_modal_embedding.expand(
            batch_size, text_hidden_state.size(1), -1
        )
        text_hidden_state = text_hidden_state + text_modal_emb
        
        image_modal_emb = self.image_modal_embedding.expand(
            batch_size, image_hidden_state.size(1), -1
        )
        image_hidden_state = image_hidden_state + image_modal_emb
        
        # Concatenate and fuse
        fused_hidden_state = torch.cat([text_hidden_state, image_hidden_state], dim=1)
        fusion_output = self.multimodal_fusion(fused_hidden_state)
        
        return fusion_output
    
    def freeze_encoders(self):
        """Freeze text and image encoders for fine-tuning only the fusion layer"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze text and image encoders"""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.image_encoder.parameters():
            param.requires_grad = True


class MMTDConfig:
    """Configuration class for MMTD model"""
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-multilingual-cased",
        dit_model_name: str = "microsoft/dit-base",
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        fusion_hidden_size: int = 768,
        fusion_num_heads: int = 8,
        max_text_length: int = 256,
        image_size: int = 224,
        **kwargs
    ):
        self.bert_model_name = bert_model_name
        self.dit_model_name = dit_model_name
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_heads = fusion_num_heads
        self.max_text_length = max_text_length
        self.image_size = image_size
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MMTDConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


def create_base_mmtd_model(config: Optional[MMTDConfig] = None) -> BaseMMTD:
    """
    Factory function to create a BaseMMTD model
    
    Args:
        config: MMTDConfig object with model parameters
        
    Returns:
        BaseMMTD model instance
    """
    if config is None:
        config = MMTDConfig()
    
    model = BaseMMTD(
        bert_pretrain_weight=config.bert_model_name,
        beit_pretrain_weight=config.dit_model_name,
        num_labels=config.num_labels,
        dropout_rate=config.dropout_rate,
        fusion_hidden_size=config.fusion_hidden_size,
        fusion_num_heads=config.fusion_num_heads
    )
    
    logger.info(f"Created BaseMMTD model with config: {config.to_dict()}")
    return model


if __name__ == "__main__":
    # Test model creation
    config = MMTDConfig()
    model = create_base_mmtd_model(config)
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 256
    num_patches = 197  # For 224x224 image with 16x16 patches
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    
    try:
        output = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            pixel_values=dummy_pixel_values,
            labels=dummy_labels
        )
        
        print(f"Model output shape: {output.logits.shape}")
        print(f"Loss: {output.loss}")
        print("✅ BaseMMTD model test passed!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}") 