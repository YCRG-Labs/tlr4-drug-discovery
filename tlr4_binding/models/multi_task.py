"""
Multi-Task Model for TLR4 binding affinity and functional classification.

This module implements the MultiTaskModel class that simultaneously predicts
binding affinity (regression) and functional class (classification) using a
shared encoder with task-specific heads.

Requirements: 12.1, 12.2, 12.3
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# PyTorch imports with availability check
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# PyTorch Geometric imports with availability check
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None
    Batch = None
    GeometricDataLoader = None


@dataclass
class MultiTaskConfig:
    """Configuration for Multi-Task model.
    
    Attributes:
        encoder_hidden_dim: Shared encoder hidden dimension (default: 128)
        affinity_head_dim: Affinity head hidden dimension (default: 64)
        class_head_dim: Classification head hidden dimension (default: 64)
        num_classes: Number of functional classes (default: 3 for agonist/antagonist/unknown)
        alpha: Loss weighting parameter (default: 0.6, range: 0.5-0.7)
        dropout: Dropout rate (default: 0.2)
    """
    encoder_hidden_dim: int = 128
    affinity_head_dim: int = 64
    class_head_dim: int = 64
    num_classes: int = 3
    alpha: float = 0.6
    dropout: float = 0.2


class MultiTaskModel(nn.Module):
    """
    Multi-task model for simultaneous binding affinity and functional classification.
    
    Implements a shared encoder with separate task heads:
    - Shared encoder: Processes input features (can be any encoder like GNN, MLP, etc.)
    - Regression head: Predicts binding affinity
    - Classification head: Predicts functional class (agonist/antagonist/unknown)
    
    Requirements: 12.1, 12.2, 12.3
    
    Attributes:
        config: MultiTaskConfig with model hyperparameters
        encoder: Shared encoder module (provided externally)
        affinity_head: Regression head for binding affinity
        classification_head: Classification head for functional class
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        affinity_head_dim: int = 64,
        class_head_dim: int = 64,
        num_classes: int = 3,
        alpha: float = 0.6,
        dropout: float = 0.2
    ):
        """
        Initialize Multi-Task model with shared encoder and task heads.
        
        Args:
            encoder: Shared encoder module (e.g., TLR4GAT, HybridModel encoder, etc.)
            encoder_output_dim: Dimension of encoder output
            affinity_head_dim: Hidden dimension for affinity head (default: 64)
            class_head_dim: Hidden dimension for classification head (default: 64)
            num_classes: Number of functional classes (default: 3)
            alpha: Loss weighting parameter (default: 0.6, range: 0.5-0.7)
            dropout: Dropout rate (default: 0.2)
        
        Requirements: 12.1
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MultiTaskModel")
        
        super().__init__()
        
        # Store configuration
        self.config = MultiTaskConfig(
            encoder_hidden_dim=encoder_output_dim,
            affinity_head_dim=affinity_head_dim,
            class_head_dim=class_head_dim,
            num_classes=num_classes,
            alpha=alpha,
            dropout=dropout
        )
        
        # Validate alpha parameter
        if alpha < 0.5 or alpha > 0.7:
            logger.warning(f"alpha={alpha} outside recommended range [0.5, 0.7]")
        
        # Shared encoder (provided externally)
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        
        # Regression head for binding affinity (Requirements: 12.1)
        self.affinity_head = nn.Sequential(
            nn.Linear(encoder_output_dim, affinity_head_dim),
            nn.BatchNorm1d(affinity_head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(affinity_head_dim, affinity_head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(affinity_head_dim // 2, 1)  # Single output for regression
        )
        
        # Classification head for functional class (Requirements: 12.1)
        self.classification_head = nn.Sequential(
            nn.Linear(encoder_output_dim, class_head_dim),
            nn.BatchNorm1d(class_head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(class_head_dim, class_head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(class_head_dim // 2, num_classes)  # Logits for each class
        )
        
        logger.info(
            f"MultiTaskModel initialized: "
            f"encoder_dim={encoder_output_dim}, "
            f"affinity_head_dim={affinity_head_dim}, "
            f"class_head_dim={class_head_dim}, "
            f"num_classes={num_classes}, "
            f"alpha={alpha}"
        )
    
    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning affinity prediction and functional class logits.
        
        Args:
            x: Input to encoder (format depends on encoder type)
        
        Returns:
            Tuple of (affinity_prediction, class_logits)
            - affinity_prediction: [batch_size, 1]
            - class_logits: [batch_size, num_classes]
        
        Requirements: 12.1
        """
        # Get shared encoding
        encoding = self.encoder(x)
        
        # Affinity prediction (regression)
        affinity_pred = self.affinity_head(encoding)
        
        # Functional class prediction (classification)
        class_logits = self.classification_head(encoding)
        
        return affinity_pred, class_logits
    
    def compute_loss(
        self,
        affinity_pred: torch.Tensor,
        class_pred: torch.Tensor,
        affinity_true: torch.Tensor,
        class_true: torch.Tensor,
        has_class_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss with missing label handling.
        
        Implements L = α·L_affinity + (1-α)·L_function where:
        - L_affinity: MSE loss for binding affinity (all samples)
        - L_function: Cross-entropy loss for functional class (only labeled samples)
        - α: Weighting parameter (0.5-0.7)
        
        Args:
            affinity_pred: Predicted affinities [batch_size, 1]
            class_pred: Predicted class logits [batch_size, num_classes]
            affinity_true: True affinities [batch_size, 1] or [batch_size]
            class_true: True class labels [batch_size] (long tensor)
            has_class_label: Boolean mask [batch_size] indicating which samples have class labels
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        
        Requirements: 12.2, 12.3
        """
        # Ensure affinity_true has correct shape
        if affinity_true.dim() == 1:
            affinity_true = affinity_true.view(-1, 1)
        
        # Affinity loss (MSE) - computed for all samples (Requirements: 12.2)
        affinity_loss = F.mse_loss(affinity_pred, affinity_true)
        
        # Classification loss (Cross-entropy) - only for labeled samples (Requirements: 12.3)
        if has_class_label.any():
            # Filter to only labeled samples
            class_pred_labeled = class_pred[has_class_label]
            class_true_labeled = class_true[has_class_label]
            
            classification_loss = F.cross_entropy(class_pred_labeled, class_true_labeled)
        else:
            # No labeled samples in this batch
            classification_loss = torch.tensor(0.0, device=affinity_pred.device)
        
        # Weighted total loss (Requirements: 12.2)
        alpha = self.config.alpha
        total_loss = alpha * affinity_loss + (1 - alpha) * classification_loss
        
        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'affinity_loss': affinity_loss.item(),
            'classification_loss': classification_loss.item(),
            'alpha': alpha
        }
        
        return total_loss, loss_dict
    
    def predict_affinity(self, x: Any) -> torch.Tensor:
        """
        Predict only binding affinity.
        
        Args:
            x: Input to encoder
        
        Returns:
            Affinity predictions [batch_size, 1]
        """
        self.eval()
        with torch.no_grad():
            encoding = self.encoder(x)
            affinity_pred = self.affinity_head(encoding)
        return affinity_pred
    
    def predict_class(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict only functional class.
        
        Args:
            x: Input to encoder
        
        Returns:
            Tuple of (class_predictions, class_probabilities)
            - class_predictions: Predicted class indices [batch_size]
            - class_probabilities: Class probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            encoding = self.encoder(x)
            class_logits = self.classification_head(encoding)
            class_probs = F.softmax(class_logits, dim=1)
            class_preds = torch.argmax(class_probs, dim=1)
        return class_preds, class_probs
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def freeze_affinity_head(self) -> None:
        """Freeze affinity head parameters."""
        for param in self.affinity_head.parameters():
            param.requires_grad = False
        logger.info("Affinity head frozen")
    
    def unfreeze_affinity_head(self) -> None:
        """Unfreeze affinity head parameters."""
        for param in self.affinity_head.parameters():
            param.requires_grad = True
        logger.info("Affinity head unfrozen")
    
    def freeze_classification_head(self) -> None:
        """Freeze classification head parameters."""
        for param in self.classification_head.parameters():
            param.requires_grad = False
        logger.info("Classification head frozen")
    
    def unfreeze_classification_head(self) -> None:
        """Unfreeze classification head parameters."""
        for param in self.classification_head.parameters():
            param.requires_grad = True
        logger.info("Classification head unfrozen")


class MultiTaskDataset(Dataset):
    """
    PyTorch Dataset for multi-task learning with optional class labels.
    
    Attributes:
        inputs: Input data (format depends on encoder)
        affinity_labels: Binding affinity labels [n_samples]
        class_labels: Functional class labels [n_samples] (may contain -1 for missing)
        has_class_label: Boolean mask indicating which samples have class labels
    """
    
    def __init__(
        self,
        inputs: Any,
        affinity_labels: np.ndarray,
        class_labels: Optional[np.ndarray] = None
    ):
        """
        Initialize multi-task dataset.
        
        Args:
            inputs: Input data (list, array, etc.)
            affinity_labels: Binding affinity labels
            class_labels: Functional class labels (optional, use -1 for missing)
        """
        self.inputs = inputs
        self.affinity_labels = torch.tensor(affinity_labels, dtype=torch.float32)
        
        if class_labels is not None:
            self.class_labels = torch.tensor(class_labels, dtype=torch.long)
            # Create mask for samples with class labels (not -1)
            self.has_class_label = self.class_labels != -1
        else:
            # No class labels provided
            self.class_labels = torch.full((len(affinity_labels),), -1, dtype=torch.long)
            self.has_class_label = torch.zeros(len(affinity_labels), dtype=torch.bool)
    
    def __len__(self) -> int:
        return len(self.affinity_labels)
    
    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get input, affinity label, class label, and class label mask at index.
        
        Returns:
            Tuple of (input, affinity_label, class_label, has_class_label)
        """
        return (
            self.inputs[idx],
            self.affinity_labels[idx],
            self.class_labels[idx],
            self.has_class_label[idx]
        )


@dataclass
class MultiTaskTrainingConfig:
    """Configuration for Multi-Task model training.
    
    Attributes:
        lr: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 1e-4)
        epochs: Maximum training epochs (default: 200)
        patience: Early stopping patience (default: 20)
        batch_size: Training batch size (default: 32)
        t_max: Cosine annealing T_max (default: 50)
        warmup_epochs: Number of epochs to train heads only (default: 0)
    """
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    t_max: int = 50
    warmup_epochs: int = 0



class MultiTaskTrainer:
    """
    Trainer for Multi-Task model with joint optimization.
    
    Implements training loop with:
    - Joint optimization of affinity and classification tasks
    - Weighted loss with α parameter
    - Handling of missing class labels
    - Adam optimizer with cosine annealing
    - Early stopping
    
    Requirements: 12.2, 12.3
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        config: Optional[MultiTaskTrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Multi-Task trainer.
        
        Args:
            model: MultiTaskModel to train
            config: Training configuration (default: MultiTaskTrainingConfig())
            device: Device to use ('cuda', 'cpu', or None for auto)
        
        Requirements: 12.2, 12.3
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MultiTaskTrainer")
        
        self.model = model
        self.config = config or MultiTaskTrainingConfig()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.t_max,
            eta_min=1e-6
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_affinity_loss': [],
            'train_classification_loss': [],
            'val_loss': [],
            'val_affinity_loss': [],
            'val_classification_loss': [],
            'lr': []
        }
        
        logger.info(
            f"MultiTaskTrainer initialized: lr={self.config.lr}, "
            f"alpha={self.model.config.alpha}, "
            f"patience={self.config.patience}, device={self.device}"
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the Multi-Task model with joint optimization.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary
        
        Requirements: 12.2, 12.3
        """
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        
        # Optional warmup phase: train task heads only
        if self.config.warmup_epochs > 0:
            logger.info(f"Warmup phase: training task heads for {self.config.warmup_epochs} epochs")
            self.model.freeze_encoder()
            
            for epoch in range(self.config.warmup_epochs):
                train_losses = self._train_epoch(train_loader)
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Warmup Epoch {epoch + 1}/{self.config.warmup_epochs}: "
                        f"Train Loss={train_losses['total_loss']:.4f}"
                    )
            
            self.model.unfreeze_encoder()
            logger.info("Warmup complete, starting joint training")
        
        # Joint training phase
        for epoch in range(self.config.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_losses['total_loss'])
            self.training_history['train_affinity_loss'].append(train_losses['affinity_loss'])
            self.training_history['train_classification_loss'].append(train_losses['classification_loss'])
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader is not None:
                val_losses = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_losses['total_loss'])
                self.training_history['val_affinity_loss'].append(val_losses['affinity_loss'])
                self.training_history['val_classification_loss'].append(val_losses['classification_loss'])
                
                # Early stopping check
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience={self.config.patience})"
                    )
                    break
            else:
                self.training_history['val_loss'].append(train_losses['total_loss'])
                self.training_history['val_affinity_loss'].append(train_losses['affinity_loss'])
                self.training_history['val_classification_loss'].append(train_losses['classification_loss'])
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = f"{self.training_history['val_loss'][-1]:.4f}" if val_loader else "N/A"
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"Train Loss={train_losses['total_loss']:.4f} "
                    f"(Aff={train_losses['affinity_loss']:.4f}, "
                    f"Class={train_losses['classification_loss']:.4f}), "
                    f"Val Loss={val_loss_str}, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Restore best model if early stopping was used
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_loss={self.best_val_loss:.4f}")
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with joint optimization.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_affinity_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs, affinity_labels, class_labels, has_class_label = batch
            
            # Move to device (handling different input types)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif hasattr(inputs, 'to'):  # PyG Data/Batch
                inputs = inputs.to(self.device)
            
            affinity_labels = affinity_labels.to(self.device)
            class_labels = class_labels.to(self.device)
            has_class_label = has_class_label.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            affinity_pred, class_pred = self.model(inputs)
            
            # Compute weighted loss (Requirements: 12.2, 12.3)
            loss, loss_dict = self.model.compute_loss(
                affinity_pred,
                class_pred,
                affinity_labels,
                class_labels,
                has_class_label
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss']
            total_affinity_loss += loss_dict['affinity_loss']
            total_classification_loss += loss_dict['classification_loss']
            num_batches += 1
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'affinity_loss': total_affinity_loss / max(num_batches, 1),
            'classification_loss': total_classification_loss / max(num_batches, 1)
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        total_affinity_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, affinity_labels, class_labels, has_class_label = batch
                
                # Move to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                
                affinity_labels = affinity_labels.to(self.device)
                class_labels = class_labels.to(self.device)
                has_class_label = has_class_label.to(self.device)
                
                # Forward pass
                affinity_pred, class_pred = self.model(inputs)
                
                # Compute loss
                _, loss_dict = self.model.compute_loss(
                    affinity_pred,
                    class_pred,
                    affinity_labels,
                    class_labels,
                    has_class_label
                )
                
                total_loss += loss_dict['total_loss']
                total_affinity_loss += loss_dict['affinity_loss']
                total_classification_loss += loss_dict['classification_loss']
                num_batches += 1
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'affinity_loss': total_affinity_loss / max(num_batches, 1),
            'classification_loss': total_classification_loss / max(num_batches, 1)
        }
    
    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: DataLoader for prediction data
        
        Returns:
            Tuple of (affinity_predictions, class_predictions, class_probabilities)
        """
        self.model.eval()
        affinity_preds = []
        class_preds = []
        class_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0]  # First element is input
                
                # Move to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                
                # Forward pass
                affinity_pred, class_logits = self.model(inputs)
                class_prob = F.softmax(class_logits, dim=1)
                class_pred = torch.argmax(class_prob, dim=1)
                
                affinity_preds.append(affinity_pred.cpu().numpy())
                class_preds.append(class_pred.cpu().numpy())
                class_probs.append(class_prob.cpu().numpy())
        
        return (
            np.concatenate(affinity_preds, axis=0),
            np.concatenate(class_preds, axis=0),
            np.concatenate(class_probs, axis=0)
        )
    
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
        
        Returns:
            Dictionary with metrics for both tasks
        """
        self.model.eval()
        all_affinity_preds = []
        all_affinity_targets = []
        all_class_preds = []
        all_class_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, affinity_labels, class_labels, has_class_label = batch
                
                # Move to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                
                # Forward pass
                affinity_pred, class_logits = self.model(inputs)
                class_prob = F.softmax(class_logits, dim=1)
                class_pred = torch.argmax(class_prob, dim=1)
                
                all_affinity_preds.append(affinity_pred.cpu().numpy())
                all_affinity_targets.append(affinity_labels.cpu().numpy())
                
                # Only evaluate classification for labeled samples
                if has_class_label.any():
                    labeled_mask = has_class_label.cpu().numpy()
                    all_class_preds.append(class_pred.cpu().numpy()[labeled_mask])
                    all_class_targets.append(class_labels.cpu().numpy()[labeled_mask])
        
        # Affinity metrics
        affinity_preds = np.concatenate(all_affinity_preds, axis=0).flatten()
        affinity_targets = np.concatenate(all_affinity_targets, axis=0).flatten()
        
        mse = np.mean((affinity_preds - affinity_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(affinity_preds - affinity_targets))
        
        ss_res = np.sum((affinity_targets - affinity_preds) ** 2)
        ss_tot = np.sum((affinity_targets - np.mean(affinity_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics = {
            'affinity_r2': r2,
            'affinity_rmse': rmse,
            'affinity_mae': mae,
            'affinity_mse': mse
        }
        
        # Classification metrics (if any labeled samples)
        if all_class_preds:
            class_preds = np.concatenate(all_class_preds, axis=0)
            class_targets = np.concatenate(all_class_targets, axis=0)
            
            accuracy = np.mean(class_preds == class_targets)
            metrics['classification_accuracy'] = accuracy
            metrics['num_classified_samples'] = len(class_preds)
        else:
            metrics['classification_accuracy'] = 0.0
            metrics['num_classified_samples'] = 0
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'model_config': self.model.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Checkpoint loaded from {path}")


def create_multi_task_model(
    encoder: nn.Module,
    encoder_output_dim: int,
    affinity_head_dim: int = 64,
    class_head_dim: int = 64,
    num_classes: int = 3,
    alpha: float = 0.6,
    dropout: float = 0.2
) -> MultiTaskModel:
    """
    Factory function to create a MultiTaskModel.
    
    Args:
        encoder: Shared encoder module
        encoder_output_dim: Dimension of encoder output
        affinity_head_dim: Hidden dimension for affinity head (default: 64)
        class_head_dim: Hidden dimension for classification head (default: 64)
        num_classes: Number of functional classes (default: 3)
        alpha: Loss weighting parameter (default: 0.6)
        dropout: Dropout rate (default: 0.2)
    
    Returns:
        Configured MultiTaskModel
    
    Requirements: 12.1
    """
    return MultiTaskModel(
        encoder=encoder,
        encoder_output_dim=encoder_output_dim,
        affinity_head_dim=affinity_head_dim,
        class_head_dim=class_head_dim,
        num_classes=num_classes,
        alpha=alpha,
        dropout=dropout
    )


def train_multi_task_model(
    model: MultiTaskModel,
    train_data: MultiTaskDataset,
    val_data: Optional[MultiTaskDataset] = None,
    config: Optional[MultiTaskTrainingConfig] = None,
    device: Optional[str] = None
) -> Tuple[MultiTaskModel, Dict[str, List[float]]]:
    """
    Convenience function to train a MultiTaskModel.
    
    Args:
        model: MultiTaskModel to train
        train_data: Training dataset
        val_data: Validation dataset (optional)
        config: Training configuration
        device: Device to use
    
    Returns:
        Tuple of (trained model, training history)
    
    Requirements: 12.2, 12.3
    """
    config = config or MultiTaskTrainingConfig()
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False
        )
    
    # Create trainer and train
    trainer = MultiTaskTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    return model, history
