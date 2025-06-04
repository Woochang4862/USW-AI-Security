"""
Interpretable MMTD Model
Integration of interpretable classifiers with MMTD architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.interpretable_classifiers import LogisticRegressionClassifier, DecisionTreeClassifier, SVMClassifier
from evaluation.original_mmtd_model import OriginalMMTD
from transformers.models.bert.modeling_bert import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class InterpretableMMTD(nn.Module):
    """
    Interpretable MMTD Model
    
    Replaces the original MLP classifier with interpretable alternatives
    while maintaining the same feature extraction pipeline
    """
    
    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        bert_pretrain_weight: str = "bert-base-multilingual-cased",
        beit_pretrain_weight: str = "microsoft/dit-base",
        classifier_config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Interpretable MMTD
        
        Args:
            classifier_type: Type of interpretable classifier ('logistic_regression', 'decision_tree', etc.)
            bert_pretrain_weight: BERT model name
            beit_pretrain_weight: BEiT/DiT model name  
            classifier_config: Configuration for the classifier
            device: Device to run on
        """
        super(InterpretableMMTD, self).__init__()
        
        self.classifier_type = classifier_type
        self.device = device or torch.device("cpu")
        
        # Initialize the original MMTD backbone (without classifier)
        self.mmtd_backbone = OriginalMMTD(
            bert_pretrain_weight=bert_pretrain_weight,
            beit_pretrain_weight=beit_pretrain_weight
        )
        
        # Remove the original classifier and pooler
        # We'll extract features before the final classification
        self.text_encoder = self.mmtd_backbone.text_encoder
        self.image_encoder = self.mmtd_backbone.image_encoder
        self.multi_modality_transformer_layer = self.mmtd_backbone.multi_modality_transformer_layer
        
        # Initialize interpretable classifier
        classifier_config = classifier_config or {}
        self.interpretable_classifier = self._create_classifier(
            classifier_type=classifier_type,
            input_size=768,  # MMTD fusion output size
            num_classes=2,
            device=self.device,
            **classifier_config
        )
        
        # Model properties
        self.num_labels = 2
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized InterpretableMMTD:")
        logger.info(f"  Classifier type: {classifier_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _create_classifier(
        self, 
        classifier_type: str, 
        input_size: int, 
        num_classes: int,
        device: torch.device,
        **kwargs
    ) -> nn.Module:
        """Create the specified interpretable classifier"""
        
        if classifier_type == "logistic_regression":
            return LogisticRegressionClassifier(
                input_size=input_size,
                num_classes=num_classes,
                device=device,
                **kwargs
            )
        elif classifier_type == "decision_tree":
            return DecisionTreeClassifier(
                input_size=input_size,
                num_classes=num_classes,
                device=device,
                **kwargs
            )
        elif classifier_type == "svm":
            return SVMClassifier(
                input_size=input_size,
                num_classes=num_classes,
                device=device,
                **kwargs
            )
        elif classifier_type == "attention":
            # Placeholder for future implementation
            raise NotImplementedError("Attention classifier not yet implemented")
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def extract_features(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract fused features from MMTD backbone
        
        Args:
            input_ids: Text input IDs
            token_type_ids: Token type IDs (optional)
            attention_mask: Attention mask
            pixel_values: Image pixel values
            
        Returns:
            Fused feature tensor of shape (batch_size, 768)
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
        
        # Add modal type embeddings (original MMTD approach)
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)
        
        # Concatenate features
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        
        # Multi-modality fusion
        fused_features = self.multi_modality_transformer_layer(fuse_hidden_state)
        
        # Use first token (CLS-like) as the final representation
        pooled_features = fused_features[:, 0, :]  # Shape: (batch_size, 768)
        
        return pooled_features
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None
    ) -> SequenceClassifierOutput:
        """
        Forward pass through interpretable MMTD
        
        Args:
            input_ids: Text input IDs
            token_type_ids: Token type IDs (optional)
            attention_mask: Attention mask
            pixel_values: Image pixel values
            labels: True labels (optional)
            
        Returns:
            SequenceClassifierOutput with logits and loss
        """
        # Extract fused features
        pooled_features = self.extract_features(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # For Decision Tree: Fit incrementally during training
        if (self.classifier_type == "decision_tree" and 
            labels is not None and 
            hasattr(self.interpretable_classifier, 'fit_incremental')):
            self.interpretable_classifier.fit_incremental(pooled_features, labels)
        
        # For SVM: Fit incrementally during training
        if (self.classifier_type == "svm" and 
            labels is not None and 
            hasattr(self.interpretable_classifier, 'fit_incremental')):
            self.interpretable_classifier.fit_incremental(pooled_features, labels)
        
        # Classification through interpretable classifier
        logits = self.interpretable_classifier(pooled_features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if hasattr(self.interpretable_classifier, 'compute_loss'):
                # Use classifier's custom loss (includes regularization)
                loss = self.interpretable_classifier.compute_loss(logits, labels)
            else:
                # Use standard cross-entropy loss
                loss = F.cross_entropy(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    
    def get_feature_importance(self, **kwargs) -> torch.Tensor:
        """Get feature importance from the interpretable classifier"""
        if hasattr(self.interpretable_classifier, 'get_feature_importance'):
            return self.interpretable_classifier.get_feature_importance(**kwargs)
        else:
            logger.warning(f"Feature importance not available for {self.classifier_type}")
            return torch.zeros(768)  # Return zero importance if not available
    
    def get_decision_tree_rules(self) -> List[str]:
        """Get decision tree rules (only for decision tree classifier)"""
        if (self.classifier_type == "decision_tree" and 
            hasattr(self.interpretable_classifier, 'get_decision_tree_rules')):
            return self.interpretable_classifier.get_decision_tree_rules()
        else:
            logger.warning("Decision tree rules only available for decision tree classifier")
            return []
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """Get Decision Tree structure (Decision Tree only)"""
        if hasattr(self.interpretable_classifier, 'get_tree_structure'):
            return self.interpretable_classifier.get_tree_structure()
        return {}
    
    def get_support_vectors(self) -> Dict[str, Any]:
        """Get SVM support vectors information (SVM only)"""
        if hasattr(self.interpretable_classifier, 'get_support_vectors'):
            return self.interpretable_classifier.get_support_vectors()
        return {}
    
    def get_margin_analysis(self) -> Dict[str, Any]:
        """Get SVM margin analysis (SVM only)"""
        if hasattr(self.interpretable_classifier, 'get_margin_analysis'):
            return self.interpretable_classifier.get_margin_analysis()
        return {}
    
    def get_decision_function_values(self, input_ids, attention_mask, pixel_values) -> torch.Tensor:
        """Get SVM decision function values (SVM only)"""
        if hasattr(self.interpretable_classifier, 'get_decision_function_values'):
            features = self.extract_features(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            return self.interpretable_classifier.get_decision_function_values(features)
        return torch.tensor([])
    
    def visualize_feature_importance(self, **kwargs):
        """Visualize feature importance"""
        if hasattr(self.interpretable_classifier, 'visualize_feature_importance'):
            return self.interpretable_classifier.visualize_feature_importance(**kwargs)
        else:
            raise NotImplementedError(f"Feature importance visualization not implemented for {self.classifier_type}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get classifier-specific info
        classifier_info = {}
        if hasattr(self.interpretable_classifier, 'get_model_summary'):
            classifier_info = self.interpretable_classifier.get_model_summary()
        
        return {
            'model_type': 'InterpretableMMTD',
            'classifier_type': self.classifier_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': total_params - classifier_info.get('num_parameters', 0),
            'classifier_parameters': classifier_info.get('num_parameters', 0),
            'device': str(self.device),
            'classifier_info': classifier_info
        }
    
    def load_mmtd_backbone_weights(self, checkpoint_path: str):
        """
        Load pre-trained MMTD backbone weights
        
        Args:
            checkpoint_path: Path to the MMTD checkpoint
        """
        logger.info(f"Loading MMTD backbone weights from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create a temporary original MMTD model to load weights
        temp_mmtd = OriginalMMTD()
        missing_keys, unexpected_keys = temp_mmtd.load_state_dict(checkpoint, strict=False)
        
        # Transfer weights to our backbone components
        self.text_encoder.load_state_dict(temp_mmtd.text_encoder.state_dict())
        self.image_encoder.load_state_dict(temp_mmtd.image_encoder.state_dict())
        self.multi_modality_transformer_layer.load_state_dict(
            temp_mmtd.multi_modality_transformer_layer.state_dict()
        )
        
        logger.info("✅ Successfully loaded MMTD backbone weights")
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")


def create_interpretable_mmtd(
    classifier_type: str = "logistic_regression",
    classifier_config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None
) -> InterpretableMMTD:
    """
    Factory function to create an interpretable MMTD model
    
    Args:
        classifier_type: Type of interpretable classifier
        classifier_config: Configuration for the classifier
        device: Device to run on
        checkpoint_path: Path to pre-trained MMTD weights
        
    Returns:
        Configured InterpretableMMTD model
    """
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    # Default classifier config
    if classifier_config is None:
        if classifier_type == "logistic_regression":
            classifier_config = {
                'l1_lambda': 0.001,
                'l2_lambda': 0.01,
                'dropout_rate': 0.1
            }
        else:
            classifier_config = {}
    
    # Create model
    model = InterpretableMMTD(
        classifier_type=classifier_type,
        classifier_config=classifier_config,
        device=device
    )
    
    # Load pre-trained weights if provided
    if checkpoint_path:
        model.load_mmtd_backbone_weights(checkpoint_path)
    
    logger.info(f"Created InterpretableMMTD with {classifier_type} classifier")
    logger.info(f"Model summary: {model.get_model_summary()}")
    
    return model


def test_interpretable_mmtd():
    """Test the interpretable MMTD model"""
    print("🧪 Testing Interpretable MMTD Model")
    print("="*50)
    
    # Create test data
    batch_size = 4
    seq_len = 256
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Test device detection
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Create model
    model = create_interpretable_mmtd(
        classifier_type="logistic_regression",
        device=device
    )
    
    print(f"✅ Created model: {model.get_model_summary()}")
    
    # Move test data to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)
    
    # Test feature extraction
    features = model.extract_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values
    )
    print(f"✅ Feature extraction: {features.shape}")
    
    # Test forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels
    )
    print(f"✅ Forward pass: logits {outputs.logits.shape}, loss {outputs.loss.item():.4f}")
    
    # Test feature importance
    importance = model.get_feature_importance()
    print(f"✅ Feature importance: {importance.shape}, sum={importance.sum():.4f}")
    
    print("\n🎉 All tests passed!")
    return model


if __name__ == "__main__":
    test_interpretable_mmtd() 