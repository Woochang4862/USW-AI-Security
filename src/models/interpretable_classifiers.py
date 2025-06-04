"""
Interpretable Classifiers for MMTD
Implementation of interpretable classification models to replace MLP classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.tree import export_text
import joblib

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier(nn.Module):
    """
    Interpretable Logistic Regression classifier for MMTD
    
    Features:
    - Linear decision boundary for interpretability
    - L1/L2 regularization for feature selection
    - Feature importance extraction
    - Gradient-based explanations
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 2,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.01,
        dropout_rate: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Logistic Regression classifier
        
        Args:
            input_size: Size of input features (768 for MMTD fusion output)
            num_classes: Number of output classes (2 for spam/ham)
            l1_lambda: L1 regularization strength (Lasso)
            l2_lambda: L2 regularization strength (Ridge)
            dropout_rate: Dropout rate for regularization
            device: Device to run on
        """
        super(LogisticRegressionClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.device = device or torch.device("cpu")
        
        # Linear layer (the core of logistic regression)
        self.linear = nn.Linear(input_size, num_classes)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized LogisticRegressionClassifier:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  L1 lambda: {l1_lambda}")
        logger.info(f"  L2 lambda: {l2_lambda}")
        logger.info(f"  Device: {self.device}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Linear transformation
        logits = self.linear(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        return predictions
    
    def get_feature_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract feature weights for interpretability
        
        Returns:
            Dictionary containing weights and bias
        """
        return {
            'weights': self.linear.weight.detach().cpu(),
            'bias': self.linear.bias.detach().cpu(),
            'weight_magnitude': torch.abs(self.linear.weight).detach().cpu(),
            'weight_l1_norm': torch.norm(self.linear.weight, p=1).detach().cpu(),
            'weight_l2_norm': torch.norm(self.linear.weight, p=2).detach().cpu()
        }
    
    def get_feature_importance(self, normalize: bool = True) -> torch.Tensor:
        """
        Get feature importance scores based on weight magnitudes
        
        Args:
            normalize: Whether to normalize importance scores
            
        Returns:
            Feature importance tensor
        """
        weights = self.linear.weight.detach().cpu()
        
        # Use absolute values of weights as importance
        importance = torch.abs(weights).mean(dim=0)  # Average across classes
        
        if normalize:
            importance = importance / importance.sum()
        
        return importance
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute L1 and L2 regularization losses
        
        Returns:
            Combined regularization loss
        """
        reg_loss = 0.0
        
        # L1 regularization (Lasso)
        if self.l1_lambda > 0:
            l1_loss = torch.norm(self.linear.weight, p=1)
            reg_loss += self.l1_lambda * l1_loss
        
        # L2 regularization (Ridge)
        if self.l2_lambda > 0:
            l2_loss = torch.norm(self.linear.weight, p=2)
            reg_loss += self.l2_lambda * l2_loss
        
        return reg_loss
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss including regularization
        
        Args:
            logits: Model predictions
            labels: True labels
            
        Returns:
            Total loss
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Regularization loss
        reg_loss = self.compute_regularization_loss()
        
        total_loss = ce_loss + reg_loss
        
        return total_loss
    
    def visualize_feature_importance(
        self, 
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize feature importance
        
        Args:
            feature_names: Names of features (optional)
            top_k: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        importance = self.get_feature_importance(normalize=True)
        
        # Get top k features
        top_indices = torch.argsort(importance, descending=True)[:top_k]
        top_importance = importance[top_indices]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        top_names = [feature_names[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_importance)), top_importance.numpy())
        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_k} Feature Importance (Logistic Regression)')
        
        # Color bars by importance
        colors = plt.cm.viridis(top_importance.numpy() / top_importance.max().item())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model
        
        Returns:
            Dictionary with model information
        """
        weights_info = self.get_feature_weights()
        
        return {
            'model_type': 'LogisticRegression',
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'weight_l1_norm': weights_info['weight_l1_norm'].item(),
            'weight_l2_norm': weights_info['weight_l2_norm'].item(),
            'device': str(self.device)
        }


class DecisionTreeClassifier(nn.Module):
    """
    Interpretable Decision Tree classifier for MMTD
    
    Features:
    - Rule-based interpretability
    - Feature importance extraction
    - Tree structure visualization
    - No gradient computation needed for inference
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 2,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Union[str, int, float] = 'sqrt',
        criterion: str = 'gini',
        random_state: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Decision Tree classifier
        
        Args:
            input_size: Size of input features
            num_classes: Number of output classes
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            criterion: Split quality criterion ('gini' or 'entropy')
            random_state: Random seed for reproducibility
            device: Device to run on
        """
        super(DecisionTreeClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        
        # Initialize sklearn DecisionTree
        self.tree = SklearnDecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state
        )
        
        # Training state
        self.is_fitted = False
        
        # Store training data for incremental training
        self.training_features = []
        self.training_labels = []
        
        logger.info(f"Initialized DecisionTreeClassifier:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  Max depth: {max_depth}")
        logger.info(f"  Min samples split: {min_samples_split}")
        logger.info(f"  Min samples leaf: {min_samples_leaf}")
        logger.info(f"  Device: {self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decision tree
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if not self.is_fitted:
            # If not fitted, return random predictions
            batch_size = x.shape[0]
            logits = torch.randn(batch_size, self.num_classes, device=self.device)
            return logits
        
        # Convert to numpy for sklearn
        x_np = x.detach().cpu().numpy()
        
        # Get predictions and probabilities
        probabilities = self.tree.predict_proba(x_np)
        
        # Convert back to torch tensor
        logits = torch.from_numpy(probabilities).float().to(self.device)
        
        # Convert probabilities to logits
        logits = torch.log(logits + 1e-8)  # Add small epsilon to avoid log(0)
        
        return logits
    
    def fit_incremental(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Incrementally collect training data and fit the tree
        
        Args:
            x: Input features
            labels: True labels
        """
        # Collect training data
        x_np = x.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        self.training_features.append(x_np)
        self.training_labels.append(labels_np)
        
        # Fit tree with accumulated data
        if len(self.training_features) > 0:
            all_features = np.vstack(self.training_features)
            all_labels = np.hstack(self.training_labels)
            
            try:
                self.tree.fit(all_features, all_labels)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Failed to fit decision tree: {e}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        return predictions
    
    def get_feature_importance(self, normalize: bool = True) -> torch.Tensor:
        """
        Get feature importance scores from the decision tree
        
        Args:
            normalize: Whether to normalize importance scores
            
        Returns:
            Feature importance tensor
        """
        if not self.is_fitted:
            return torch.zeros(self.input_size)
        
        importance = self.tree.feature_importances_
        importance_tensor = torch.from_numpy(importance).float()
        
        if normalize:
            importance_tensor = importance_tensor / importance_tensor.sum()
        
        return importance_tensor
    
    def get_decision_tree_rules(self) -> List[str]:
        """
        Extract human-readable rules from the decision tree
        
        Returns:
            List of rules as strings
        """
        if not self.is_fitted:
            return []
        
        try:
            # Generate feature names
            feature_names = [f'feature_{i}' for i in range(self.input_size)]
            
            # Export tree rules as text
            tree_rules = export_text(
                self.tree, 
                feature_names=feature_names,
                max_depth=10
            )
            
            # Split into individual rules
            rules = tree_rules.split('\n')
            rules = [rule.strip() for rule in rules if rule.strip()]
            
            return rules
        except Exception as e:
            logger.warning(f"Failed to extract tree rules: {e}")
            return []
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """
        Get tree structure information
        
        Returns:
            Dictionary with tree structure info
        """
        if not self.is_fitted:
            return {}
        
        try:
            return {
                'n_nodes': self.tree.tree_.node_count,
                'n_leaves': self.tree.get_n_leaves(),
                'max_depth': self.tree.get_depth(),
                'n_features': self.tree.n_features_,
                'n_classes': self.tree.n_classes_,
                'n_outputs': self.tree.n_outputs_
            }
        except Exception as e:
            logger.warning(f"Failed to get tree structure: {e}")
            return {}
    
    def visualize_feature_importance(
        self, 
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize feature importance from decision tree
        
        Args:
            feature_names: Names of features (optional)
            top_k: Number of top features to show
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        importance = self.get_feature_importance(normalize=True)
        
        if importance.sum() == 0:
            logger.warning("No feature importance available (tree not fitted)")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'Tree not fitted yet', ha='center', va='center')
            return fig
        
        # Get top-k features
        top_indices = torch.topk(importance, k=min(top_k, len(importance))).indices
        top_importance = importance[top_indices]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        top_names = [feature_names[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(top_importance)), top_importance.numpy())
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title(f'Top {len(top_importance)} Feature Importance (Decision Tree)')
        ax.set_xticks(range(len(top_importance)))
        ax.set_xticklabels(top_names, rotation=45, ha='right')
        
        # Color bars by importance
        colors = plt.cm.viridis(top_importance.numpy() / top_importance.max().item())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_tree(self, path: str):
        """Save the fitted tree to disk"""
        if self.is_fitted:
            joblib.dump(self.tree, path)
            logger.info(f"Decision tree saved to {path}")
    
    def load_tree(self, path: str):
        """Load a fitted tree from disk"""
        try:
            self.tree = joblib.load(path)
            self.is_fitted = True
            logger.info(f"Decision tree loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load tree from {path}: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary for reporting
        
        Returns:
            Dictionary with model information
        """
        summary = {
            'classifier_type': 'decision_tree',
            'total_parameters': 0,  # Decision trees don't have learnable parameters
            'trainable_parameters': 0,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted,
            'device': str(self.device)
        }
        
        # Add tree structure if fitted
        if self.is_fitted:
            summary.update(self.get_tree_structure())
        
        return summary


class InterpretableClassifierFactory:
    """
    Factory class for creating interpretable classifiers
    """
    
    @staticmethod
    def create_logistic_regression(
        input_size: int,
        num_classes: int = 2,
        **kwargs
    ) -> LogisticRegressionClassifier:
        """Create a Logistic Regression classifier"""
        return LogisticRegressionClassifier(
            input_size=input_size,
            num_classes=num_classes,
            **kwargs
        )
    
    @staticmethod
    def create_decision_tree_classifier(
        input_size: int,
        num_classes: int = 2,
        **kwargs
    ) -> DecisionTreeClassifier:
        """Create a Decision Tree classifier"""
        return DecisionTreeClassifier(
            input_size=input_size,
            num_classes=num_classes,
            **kwargs
        )
    
    @staticmethod
    def create_attention_classifier(
        input_size: int,
        num_classes: int = 2,
        **kwargs
    ):
        """Create an Attention-based classifier (placeholder for future implementation)"""
        raise NotImplementedError("Attention classifier not yet implemented")


def test_logistic_regression():
    """Test the Logistic Regression classifier"""
    print("🧪 Testing Logistic Regression Classifier")
    print("="*50)
    
    # Create test data
    batch_size = 32
    input_size = 768  # MMTD fusion output size
    num_classes = 2
    
    # Random test data
    x = torch.randn(batch_size, input_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create classifier
    classifier = LogisticRegressionClassifier(
        input_size=input_size,
        num_classes=num_classes,
        l1_lambda=0.01,
        l2_lambda=0.01
    )
    
    print(f"✅ Created classifier: {classifier.get_model_summary()}")
    
    # Test forward pass
    logits = classifier(x)
    print(f"✅ Forward pass: {logits.shape}")
    
    # Test predictions
    predictions = classifier.predict(x)
    probabilities = classifier.predict_proba(x)
    print(f"✅ Predictions: {predictions.shape}")
    print(f"✅ Probabilities: {probabilities.shape}")
    
    # Test loss computation
    loss = classifier.compute_loss(logits, labels)
    print(f"✅ Loss computation: {loss.item():.4f}")
    
    # Test feature importance
    importance = classifier.get_feature_importance()
    print(f"✅ Feature importance: {importance.shape}, sum={importance.sum():.4f}")
    
    # Test visualization (without saving)
    fig = classifier.visualize_feature_importance(top_k=10)
    plt.close(fig)  # Close to avoid display
    print(f"✅ Visualization test passed")
    
    print("\n🎉 All tests passed!")
    return classifier


def test_decision_tree():
    """Test DecisionTree classifier"""
    print("Testing DecisionTree classifier...")
    
    # Create classifier
    classifier = DecisionTreeClassifier(
        input_size=768,
        num_classes=2,
        max_depth=10
    )
    
    # Test forward pass (before fitting)
    batch_size = 16
    x = torch.randn(batch_size, 768)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass before fitting
    logits = classifier(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (before fitting): {logits.shape}")
    
    # Fit incrementally
    classifier.fit_incremental(x, labels)
    
    # Forward pass after fitting
    logits = classifier(x)
    print(f"Output shape (after fitting): {logits.shape}")
    
    # Test feature importance
    importance = classifier.get_feature_importance()
    print(f"Feature importance shape: {importance.shape}")
    print(f"Top 5 importance values: {importance.topk(5).values}")
    
    # Test tree rules
    rules = classifier.get_decision_tree_rules()
    print(f"Number of rules extracted: {len(rules)}")
    
    # Test tree structure
    structure = classifier.get_tree_structure()
    print(f"Tree structure: {structure}")
    
    print("DecisionTree test completed!\n")


if __name__ == "__main__":
    test_logistic_regression()
    test_decision_tree() 