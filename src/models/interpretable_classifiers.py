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
from sklearn.svm import SVC
import joblib
import xgboost as xgb

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


class SVMClassifier(nn.Module):
    """
    Interpretable Support Vector Machine classifier for MMTD
    
    Features:
    - Support vector analysis for interpretability
    - Decision boundary visualization
    - Margin analysis
    - Multiple kernel support (linear, rbf, poly)
    - Feature importance via coefficients (linear kernel)
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 2,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: Union[str, float] = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        probability: bool = True,
        random_state: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SVM classifier
        
        Args:
            input_size: Size of input features (768 for MMTD fusion output)
            num_classes: Number of output classes (2 for spam/ham)
            C: Regularization parameter (higher C = less regularization)
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            degree: Degree for polynomial kernel
            coef0: Independent term in kernel function
            probability: Whether to enable probability estimates
            random_state: Random seed for reproducibility
            device: Device to run on
        """
        super(SVMClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
        self.random_state = random_state
        self.device = device or torch.device("cpu")
        
        # Initialize SVM model
        self.svm = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            probability=probability,
            random_state=random_state
        )
        
        self.is_fitted = False
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized SVMClassifier:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  C: {C}")
        logger.info(f"  Kernel: {kernel}")
        logger.info(f"  Gamma: {gamma}")
        logger.info(f"  Device: {self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (prediction)
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before prediction")
        
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        # Get predictions
        if self.probability:
            # Get probability estimates
            probs = self.svm.predict_proba(x_np)
            # Convert to logits (approximate)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            logits = np.log(probs / (1 - probs))
        else:
            # Get decision function values
            decision_values = self.svm.decision_function(x_np)
            if decision_values.ndim == 1:
                # Binary classification
                logits = np.column_stack([-decision_values, decision_values])
            else:
                logits = decision_values
        
        return torch.tensor(logits, dtype=torch.float32, device=self.device)
    
    def fit_incremental(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Fit SVM incrementally (collect data and refit)
        
        Args:
            x: Input features
            labels: Target labels
        """
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Collect training data
        if not hasattr(self, '_X_train'):
            self._X_train = x_np
            self._y_train = labels_np
        else:
            self._X_train = np.vstack([self._X_train, x_np])
            self._y_train = np.concatenate([self._y_train, labels_np])
        
        # Fit SVM with all collected data
        try:
            self.svm.fit(self._X_train, self._y_train)
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"SVM fitting failed: {e}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before prediction")
        
        x_np = x.detach().cpu().numpy()
        
        if self.probability:
            probs = self.svm.predict_proba(x_np)
        else:
            # Use decision function and convert to probabilities
            decision_values = self.svm.decision_function(x_np)
            if decision_values.ndim == 1:
                probs = 1 / (1 + np.exp(-decision_values))
                probs = np.column_stack([1 - probs, probs])
            else:
                probs = np.exp(decision_values) / np.sum(np.exp(decision_values), axis=1, keepdims=True)
        
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before prediction")
        
        x_np = x.detach().cpu().numpy()
        predictions = self.svm.predict(x_np)
        
        return torch.tensor(predictions, dtype=torch.long, device=self.device)
    
    def get_support_vectors(self) -> Dict[str, np.ndarray]:
        """
        Get support vectors and related information
        
        Returns:
            Dictionary containing support vector information
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before getting support vectors")
        
        return {
            'support_vectors': self.svm.support_vectors_,
            'support_indices': self.svm.support_,
            'n_support': self.svm.n_support_,
            'dual_coef': self.svm.dual_coef_,
            'intercept': self.svm.intercept_
        }
    
    def get_feature_importance(self, normalize: bool = True) -> Optional[torch.Tensor]:
        """
        Get feature importance (only for linear kernel)
        
        Args:
            normalize: Whether to normalize importance scores
            
        Returns:
            Feature importance tensor (None for non-linear kernels)
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before getting feature importance")
        
        if self.kernel != 'linear':
            logger.warning("Feature importance only available for linear kernel")
            return None
        
        # Get coefficients from linear SVM
        coef = self.svm.coef_[0] if self.svm.coef_.shape[0] == 1 else self.svm.coef_.mean(axis=0)
        importance = np.abs(coef)
        
        if normalize:
            importance = importance / importance.sum()
        
        return torch.tensor(importance, dtype=torch.float32)
    
    def get_decision_function_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get decision function values (distance to separating hyperplane)
        
        Args:
            x: Input tensor
            
        Returns:
            Decision function values
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before getting decision values")
        
        x_np = x.detach().cpu().numpy()
        decision_values = self.svm.decision_function(x_np)
        
        return torch.tensor(decision_values, dtype=torch.float32, device=self.device)
    
    def get_margin_analysis(self) -> Dict[str, Any]:
        """
        Analyze SVM margins and support vectors
        
        Returns:
            Dictionary containing margin analysis
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before margin analysis")
        
        support_info = self.get_support_vectors()
        
        # Calculate margin (for binary classification)
        if self.kernel == 'linear' and self.num_classes == 2:
            w = self.svm.coef_[0]
            margin = 2.0 / np.linalg.norm(w)
        else:
            margin = None
        
        analysis = {
            'num_support_vectors': len(support_info['support_vectors']),
            'support_vector_ratio': len(support_info['support_vectors']) / len(self._X_train) if hasattr(self, '_X_train') else None,
            'margin_width': margin,
            'n_support_per_class': support_info['n_support'],
            'kernel_type': self.kernel,
            'C_parameter': self.C
        }
        
        return analysis
    
    def visualize_decision_boundary_2d(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_indices: Tuple[int, int] = (0, 1),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize decision boundary in 2D (for visualization purposes)
        
        Args:
            X: Feature matrix
            y: Labels
            feature_indices: Which two features to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before visualization")
        
        # Select two features for visualization
        X_2d = X[:, feature_indices]
        
        # Create a mesh
        h = 0.02
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Create a temporary SVM for 2D visualization
        temp_svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        temp_svm.fit(X_2d, y)
        
        # Get decision boundary
        Z = temp_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Decision boundary
        ax.contour(xx, yy, Z, levels=[0], alpha=0.8, linestyles='--', colors='black')
        ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
        
        # Data points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        
        # Support vectors
        if hasattr(temp_svm, 'support_vectors_'):
            ax.scatter(temp_svm.support_vectors_[:, 0], temp_svm.support_vectors_[:, 1],
                      s=100, facecolors='none', edgecolors='black', linewidths=2,
                      label='Support Vectors')
        
        ax.set_xlabel(f'Feature {feature_indices[0]}')
        ax.set_ylabel(f'Feature {feature_indices[1]}')
        ax.set_title(f'SVM Decision Boundary ({self.kernel} kernel)')
        ax.legend()
        plt.colorbar(scatter)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_feature_importance(
        self, 
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        save_path: Optional[Path] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize feature importance (only for linear kernel)
        
        Args:
            feature_names: Names of features
            top_k: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure or None if not linear kernel
        """
        importance = self.get_feature_importance()
        
        if importance is None:
            logger.warning("Feature importance visualization only available for linear kernel")
            return None
        
        # Get top k features
        top_indices = torch.topk(importance, min(top_k, len(importance))).indices
        top_importance = importance[top_indices]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        top_names = [feature_names[i] for i in top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_importance)), top_importance.numpy())
        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Feature Importance (|coefficient|)')
        ax.set_title(f'SVM Feature Importance (Linear Kernel) - Top {len(top_importance)}')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', ha='left', va='center')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_model(self, path: str):
        """Save SVM model"""
        if self.is_fitted:
            joblib.dump(self.svm, path)
            logger.info(f"SVM model saved to {path}")
        else:
            logger.warning("Cannot save unfitted SVM model")
    
    def load_model(self, path: str):
        """Load SVM model"""
        try:
            self.svm = joblib.load(path)
            self.is_fitted = True
            logger.info(f"SVM model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load SVM model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary containing model information
        """
        summary = {
            'classifier_type': 'svm',
            'total_parameters': 0,  # SVM doesn't have traditional parameters
            'trainable_parameters': 0,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted,
            'device': str(self.device),
            'hyperparameters': {
                'C': self.C,
                'kernel': self.kernel,
                'gamma': self.gamma,
                'degree': self.degree if self.kernel == 'poly' else None,
                'coef0': self.coef0,
                'probability': self.probability
            }
        }
        
        if self.is_fitted:
            support_info = self.get_support_vectors()
            summary.update({
                'num_support_vectors': len(support_info['support_vectors']),
                'support_vector_ratio': len(support_info['support_vectors']) / len(self._X_train) if hasattr(self, '_X_train') else None,
                'training_samples': len(self._X_train) if hasattr(self, '_X_train') else None
            })
            
            # Add margin analysis for linear kernel
            if self.kernel == 'linear':
                margin_analysis = self.get_margin_analysis()
                summary['margin_analysis'] = margin_analysis
        
        return summary


class XGBoostClassifier(nn.Module):
    """
    Interpretable XGBoost classifier for MMTD
    
    Features:
    - High performance gradient boosting
    - Feature importance via gain, weight, cover
    - SHAP values for detailed explanations
    - Tree structure visualization
    - Early stopping support
    - Built-in regularization
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 2,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize XGBoost classifier
        
        Args:
            input_size: Size of input features (768 for MMTD fusion output)
            num_classes: Number of output classes (2 for spam/ham)
            max_depth: Maximum depth of trees
            learning_rate: Learning rate for boosting
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed for reproducibility
            device: Device to run on
        """
        super(XGBoostClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.device = device or torch.device("cpu")
        
        # Initialize XGBoost model
        if num_classes == 2:
            objective = 'binary:logistic'
            num_class = None
        else:
            objective = 'multi:softprob'
            num_class = num_classes
        
        self.xgb_params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': objective,
            'eval_metric': 'logloss',
            'verbosity': 0,  # Suppress XGBoost output
            'n_jobs': -1,    # Use all cores
        }
        
        if num_class is not None:
            self.xgb_params['num_class'] = num_class
        
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.is_fitted = False
        
        # Training data storage for incremental fitting
        self.training_features = []
        self.training_labels = []
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized XGBoostClassifier:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  Max depth: {max_depth}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  N estimators: {n_estimators}")
        logger.info(f"  Device: {self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through XGBoost
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if not self.is_fitted:
            # Return random predictions if not fitted
            batch_size = x.shape[0]
            if self.num_classes == 2:
                logits = torch.randn(batch_size, 2, device=self.device)
            else:
                logits = torch.randn(batch_size, self.num_classes, device=self.device)
            return logits
        
        # Convert to numpy for XGBoost
        x_np = x.detach().cpu().numpy()
        
        # Get predictions
        if self.num_classes == 2:
            # Binary classification - get probabilities
            probs = self.xgb_model.predict_proba(x_np)
            # Convert probabilities to logits
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            logits = np.log(probs / (1 - probs + 1e-7))
        else:
            # Multi-class classification
            probs = self.xgb_model.predict_proba(x_np)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            logits = np.log(probs)
        
        return torch.tensor(logits, dtype=torch.float32, device=self.device)
    
    def fit_incremental(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Incrementally collect training data and fit XGBoost
        
        Args:
            x: Input features
            labels: Target labels
        """
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Collect training data
        self.training_features.append(x_np)
        self.training_labels.append(labels_np)
        
        # Fit XGBoost with accumulated data
        if len(self.training_features) > 0:
            all_features = np.vstack(self.training_features)
            all_labels = np.hstack(self.training_labels)
            
            try:
                self.xgb_model.fit(all_features, all_labels)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Failed to fit XGBoost: {e}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost must be fitted before prediction")
        
        x_np = x.detach().cpu().numpy()
        probs = self.xgb_model.predict_proba(x_np)
        
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost must be fitted before prediction")
        
        x_np = x.detach().cpu().numpy()
        predictions = self.xgb_model.predict(x_np)
        
        return torch.tensor(predictions, dtype=torch.long, device=self.device)
    
    def get_feature_importance(self, importance_type: str = 'gain', normalize: bool = True) -> torch.Tensor:
        """
        Get feature importance from XGBoost
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
            normalize: Whether to normalize importance scores
            
        Returns:
            Feature importance tensor
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost must be fitted before getting feature importance")
        
        try:
            # Get feature importance as dictionary
            importance_dict = self.xgb_model.get_booster().get_score(importance_type=importance_type)
            
            # Convert to array (XGBoost may not include all features if they're not used)
            importance_array = np.zeros(self.input_size)
            for feature_name, importance in importance_dict.items():
                feature_idx = int(feature_name.replace('f', ''))
                if feature_idx < self.input_size:
                    importance_array[feature_idx] = importance
            
            importance_tensor = torch.from_numpy(importance_array).float()
            
            if normalize and importance_tensor.sum() > 0:
                importance_tensor = importance_tensor / importance_tensor.sum()
            
            return importance_tensor
            
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return torch.zeros(self.input_size)
    
    def get_shap_values(self, x: torch.Tensor, max_samples: int = 100) -> Optional[np.ndarray]:
        """
        Get SHAP values for model explanations
        
        Args:
            x: Input tensor
            max_samples: Maximum number of samples to compute SHAP for
            
        Returns:
            SHAP values array or None if SHAP not available
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost must be fitted before getting SHAP values")
        
        try:
            import shap
            
            # Convert to numpy
            x_np = x.detach().cpu().numpy()
            
            # Limit samples for computational efficiency
            if len(x_np) > max_samples:
                x_np = x_np[:max_samples]
            
            # Create explainer
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(x_np)
            
            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Use SHAP values for positive class
                shap_values = shap_values[1]
            
            return shap_values
            
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None
        except Exception as e:
            logger.warning(f"Failed to compute SHAP values: {e}")
            return None
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the boosted trees
        
        Returns:
            Dictionary with tree information
        """
        if not self.is_fitted:
            return {}
        
        try:
            booster = self.xgb_model.get_booster()
            
            # Get basic tree info
            tree_info = {
                'n_estimators': self.xgb_model.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'feature_importances_available': hasattr(self.xgb_model, 'feature_importances_'),
                'best_iteration': getattr(self.xgb_model, 'best_iteration', None),
                'best_score': getattr(self.xgb_model, 'best_score', None)
            }
            
            # Try to get more detailed info
            if hasattr(booster, 'trees_to_dataframe'):
                try:
                    trees_df = booster.trees_to_dataframe()
                    tree_info.update({
                        'total_trees': len(trees_df['Tree'].unique()) if 'Tree' in trees_df.columns else None,
                        'total_nodes': len(trees_df) if trees_df is not None else None,
                        'avg_tree_depth': trees_df['Depth'].mean() if 'Depth' in trees_df.columns else None
                    })
                except:
                    pass
            
            return tree_info
            
        except Exception as e:
            logger.warning(f"Failed to get tree info: {e}")
            return {}
    
    def visualize_feature_importance(
        self, 
        feature_names: Optional[List[str]] = None,
        importance_type: str = 'gain',
        top_k: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize feature importance from XGBoost
        
        Args:
            feature_names: Names of features
            importance_type: Type of importance to plot
            top_k: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        importance = self.get_feature_importance(importance_type=importance_type, normalize=True)
        
        if importance.sum() == 0:
            logger.warning("No feature importance available (model not fitted)")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'Model not fitted yet', ha='center', va='center')
            ax.set_title(f'XGBoost Feature Importance ({importance_type})')
            return fig
        
        # Get top-k features
        top_indices = torch.topk(importance, k=min(top_k, len(importance))).indices
        top_importance = importance[top_indices]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        top_names = [feature_names[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_importance)), top_importance.numpy())
        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel(f'Feature Importance ({importance_type})')
        ax.set_title(f'XGBoost Feature Importance ({importance_type}) - Top {len(top_importance)}')
        
        # Color bars by importance
        colors = plt.cm.viridis(top_importance.numpy() / top_importance.max().item())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', ha='left', va='center')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_model(self, path: str):
        """Save XGBoost model"""
        if self.is_fitted:
            self.xgb_model.save_model(path)
            logger.info(f"XGBoost model saved to {path}")
        else:
            logger.warning("Cannot save unfitted XGBoost model")
    
    def load_model(self, path: str):
        """Load XGBoost model"""
        try:
            self.xgb_model.load_model(path)
            self.is_fitted = True
            logger.info(f"XGBoost model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary containing model information
        """
        summary = {
            'classifier_type': 'xgboost',
            'total_parameters': 0,  # XGBoost doesn't have traditional parameters
            'trainable_parameters': 0,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted,
            'device': str(self.device),
            'hyperparameters': {
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda
            }
        }
        
        if self.is_fitted:
            tree_info = self.get_tree_info()
            summary.update({
                'tree_info': tree_info,
                'training_samples': len(self._X_train) if hasattr(self, '_X_train') else None
            })
        
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
    def create_svm_classifier(
        input_size: int,
        num_classes: int = 2,
        **kwargs
    ) -> SVMClassifier:
        """Create an SVM classifier"""
        return SVMClassifier(
            input_size=input_size,
            num_classes=num_classes,
            **kwargs
        )
    
    @staticmethod
    def create_xgboost_classifier(
        input_size: int,
        num_classes: int = 2,
        **kwargs
    ) -> XGBoostClassifier:
        """Create an XGBoost classifier"""
        return XGBoostClassifier(
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