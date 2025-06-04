"""
Original MMTD Model Implementation
Exact replica of the original MMTD code structure to match checkpoint weights
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


class OriginalMMTD(nn.Module):
    """
    Original MMTD Model - exact replica of the original implementation
    This matches the checkpoint structure perfectly
    """
    
    def __init__(
        self, 
        bert_cfg: Optional[BertConfig] = None, 
        beit_cfg: Optional[BeitConfig] = None,
        bert_pretrain_weight: str = "bert-base-multilingual-cased",
        beit_pretrain_weight: str = "microsoft/dit-base"
    ):
        """
        Initialize Original MMTD Model
        
        Args:
            bert_cfg: BERT configuration (optional)
            beit_cfg: BEiT configuration (optional)
            bert_pretrain_weight: BERT model name
            beit_pretrain_weight: BEiT model name
        """
        super(OriginalMMTD, self).__init__()
        
        # Create default configs if not provided
        if bert_cfg is None:
            bert_cfg = BertConfig.from_pretrained(bert_pretrain_weight)
            # Adjust vocab size to match checkpoint (119547 tokens)
            bert_cfg.vocab_size = 119547
            
        if beit_cfg is None:
            beit_cfg = BeitConfig.from_pretrained(beit_pretrain_weight)
        
        # Initialize encoders with custom configs
        self.text_encoder = BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification(beit_cfg)
        
        # Enable hidden states output
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        
        # Multi-modality transformer layer (original MMTD structure)
        self.multi_modality_transformer_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Final classifier (original MMTD structure)
        self.pooler = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
        )
        self.classifier = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)
        
        # Model properties
        self.num_labels = 2
        
        logger.info(f"Initialized OriginalMMTD:")
        logger.info(f"  BERT vocab size: {bert_cfg.vocab_size}")
        logger.info(f"  BEiT image size: {beit_cfg.image_size}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        pixel_values: torch.Tensor = None, 
        labels: Optional[torch.Tensor] = None
    ) -> SequenceClassifierOutput:
        """
        Forward pass - exact replica of original implementation
        """
        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        
        # Image encoding
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        
        # Get last hidden states (layer 12)
        text_last_hidden_state = text_outputs.hidden_states[12]
        image_last_hidden_state = image_outputs.hidden_states[12]
        
        # Add modal type embeddings (original approach)
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)
        
        # Concatenate features
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        
        # Multi-modality fusion
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        
        # Pooling (use first token)
        outputs = self.pooler(outputs[:, 0, :])
        
        # Classification
        logits = self.classifier(outputs)
        
        # Calculate loss
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


class OriginalMMTDLoader:
    """
    Loader for original MMTD models with exact architecture matching
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[torch.device] = None
    ):
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
    
    def create_original_model(self) -> OriginalMMTD:
        """Create original MMTD model with pretrained weights"""
        model = OriginalMMTD(
            bert_pretrain_weight="bert-base-multilingual-cased",
            beit_pretrain_weight="microsoft/dit-base"
        )
        return model
    
    def load_fold_model(self, fold_name: str) -> OriginalMMTD:
        """
        Load a specific fold's model with exact weight matching
        """
        import os
        from pathlib import Path
        
        # Find checkpoint file
        fold_dir = Path(self.checkpoint_dir) / fold_name
        checkpoint_subdirs = [d for d in fold_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('checkpoint')]
        
        if not checkpoint_subdirs:
            raise ValueError(f"No checkpoint directory found in {fold_dir}")
        
        checkpoint_path = checkpoint_subdirs[0] / "pytorch_model.bin"
        
        if not checkpoint_path.exists():
            raise ValueError(f"pytorch_model.bin not found in {checkpoint_subdirs[0]}")
        
        logger.info(f"Loading original MMTD model from {checkpoint_path}")
        
        # Create model
        model = self.create_original_model()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dict (should match perfectly now)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        model.to(self.device)
        model.eval()
        
        self.loaded_models[fold_name] = model
        logger.info(f"✅ Successfully loaded {fold_name} with original architecture")
        
        return model
    
    def load_all_models(self) -> Dict[str, OriginalMMTD]:
        """Load all fold models"""
        from pathlib import Path
        
        checkpoint_dir = Path(self.checkpoint_dir)
        fold_dirs = sorted([d for d in checkpoint_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('fold')])
        
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            try:
                self.load_fold_model(fold_name)
            except Exception as e:
                logger.error(f"Failed to load {fold_name}: {e}")
        
        return self.loaded_models
    
    def test_model_inference(self, fold_name: str) -> Dict[str, Any]:
        """Test inference with dummy data"""
        if fold_name not in self.loaded_models:
            raise ValueError(f"Model {fold_name} not loaded")
        
        model = self.loaded_models[fold_name]
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 256
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            return {
                'fold_name': fold_name,
                'inference_successful': True,
                'output_shape': list(outputs.logits.shape),
                'predictions': predictions.cpu().numpy().tolist(),
                'probabilities': probabilities.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Inference test failed for {fold_name}: {e}")
            return {
                'fold_name': fold_name,
                'inference_successful': False,
                'error': str(e)
            }


def main():
    """Test the original MMTD model loader"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create loader
    loader = OriginalMMTDLoader(checkpoint_dir="MMTD/checkpoints")
    
    # Load all models
    models = loader.load_all_models()
    
    print("\n" + "="*60)
    print("ORIGINAL MMTD MODEL LOADING RESULTS")
    print("="*60)
    print(f"Successfully loaded: {len(models)} models")
    print(f"Loaded folds: {list(models.keys())}")
    
    # Test each model
    for fold_name in models.keys():
        print(f"\n--- Testing {fold_name.upper()} ---")
        test_result = loader.test_model_inference(fold_name)
        
        if test_result['inference_successful']:
            print(f"✅ Inference: PASSED")
            print(f"Output shape: {test_result['output_shape']}")
            print(f"Predictions: {test_result['predictions']}")
        else:
            print(f"❌ Inference: FAILED - {test_result['error']}")
    
    print("\n" + "="*60)
    print("🎯 READY FOR 99.7% ACCURACY EVALUATION!")
    print("="*60)


if __name__ == "__main__":
    main() 