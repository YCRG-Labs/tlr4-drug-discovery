"""
ChemBERTa Transformer for TLR4 binding affinity prediction.

This module implements the ChemBERTaPredictor class that fine-tunes a pre-trained
ChemBERTa transformer model for binding affinity prediction.

Requirements: 9.1, 9.2, 9.3, 9.4
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
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Transformers imports with availability check
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        RobertaModel,
        RobertaConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None
    RobertaModel = None
    RobertaConfig = None


@dataclass
class ChemBERTaConfig:
    """Configuration for ChemBERTa model.
    
    Attributes:
        model_name: Pre-trained model name (default: "seyonec/ChemBERTa-zinc-base-v1")
        freeze_layers: Fraction of layers to freeze (default: 0.5 for first 50%)
        hidden_dim_1: First hidden layer dimension (default: 256)
        hidden_dim_2: Second hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 1 for regression)
        dropout: Dropout rate (default: 0.1)
        max_length: Maximum SMILES sequence length (default: 512)
    """
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    freeze_layers: float = 0.5
    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    output_dim: int = 1
    dropout: float = 0.1
    max_length: int = 512


class ChemBERTaPredictor(nn.Module):
    """
    ChemBERTa-based predictor for TLR4 binding affinity.
    
    Implements a fine-tuned transformer model with:
    - Pre-trained ChemBERTa weights (seyonec/ChemBERTa-zinc-base-v1)
    - First 50% of layers frozen for transfer learning
    - Regression head (768 → 256 → 128 → 1)
    
    Requirements: 9.1, 9.2, 9.3
    
    Attributes:
        config: ChemBERTaConfig with model hyperparameters
        tokenizer: ChemBERTa tokenizer for SMILES encoding
        encoder: Pre-trained ChemBERTa encoder
        regression_head: Fully connected layers for binding affinity prediction
    """
    
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        freeze_layers: float = 0.5,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize ChemBERTa predictor with pre-trained weights.
        
        Args:
            model_name: Pre-trained model name (default: "seyonec/ChemBERTa-zinc-base-v1")
            freeze_layers: Fraction of layers to freeze (default: 0.5 for first 50%)
            hidden_dim_1: First hidden layer dimension (default: 256)
            hidden_dim_2: Second hidden layer dimension (default: 128)
            output_dim: Output dimension (default: 1 for regression)
            dropout: Dropout rate (default: 0.1)
        
        Requirements: 9.1, 9.2, 9.3
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ChemBERTaPredictor")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library is required for ChemBERTaPredictor")
        
        super().__init__()
        
        # Store configuration
        self.config = ChemBERTaConfig(
            model_name=model_name,
            freeze_layers=freeze_layers,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load pre-trained ChemBERTa encoder (Requirements: 9.1)
        logger.info(f"Loading pre-trained model from {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get encoder hidden size (typically 768 for base models)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Freeze first 50% of layers (Requirements: 9.2)
        self._freeze_layers(freeze_layers)
        
        # Regression head: 768 → 256 → 128 → 1 (Requirements: 9.3)
        self.regression_head = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, output_dim)
        )
        
        logger.info(
            f"ChemBERTaPredictor initialized: {model_name}, "
            f"freeze_layers={freeze_layers}, "
            f"regression_head={self.encoder_hidden_size}→{hidden_dim_1}→{hidden_dim_2}→{output_dim}"
        )
    
    def _freeze_layers(self, freeze_fraction: float) -> None:
        """
        Freeze the first fraction of transformer layers.
        
        Args:
            freeze_fraction: Fraction of layers to freeze (0.0 to 1.0)
        
        Requirements: 9.2
        """
        if freeze_fraction <= 0:
            logger.info("No layers frozen")
            return
        
        # Get encoder layers (RoBERTa structure)
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'roberta') and hasattr(self.encoder.roberta.encoder, 'layer'):
            layers = self.encoder.roberta.encoder.layer
        else:
            logger.warning("Could not find encoder layers to freeze")
            return
        
        num_layers = len(layers)
        num_freeze = int(num_layers * freeze_fraction)
        
        # Freeze embeddings
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(self.encoder, 'roberta') and hasattr(self.encoder.roberta, 'embeddings'):
            for param in self.encoder.roberta.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze first N layers
        for i in range(num_freeze):
            for param in layers[i].parameters():
                param.requires_grad = False
        
        logger.info(f"Frozen {num_freeze}/{num_layers} encoder layers ({freeze_fraction*100:.0f}%)")
    
    def encode_smiles(
        self,
        smiles: Union[str, List[str]],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode SMILES string(s).
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            return_tensors: Format for returned tensors (default: "pt" for PyTorch)
        
        Returns:
            Dictionary with input_ids and attention_mask tensors
        
        Requirements: 9.1
        """
        # Ensure smiles is a list
        if isinstance(smiles, str):
            smiles = [smiles]
        
        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for binding affinity prediction.
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Predicted binding affinity [batch_size, output_dim]
        
        Requirements: 9.3
        """
        # Encode with ChemBERTa
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        
        # Pass through regression head
        output = self.regression_head(cls_embedding)
        
        return output
    
    def predict_smiles(
        self,
        smiles: Union[str, List[str]],
        device: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict binding affinity from SMILES string(s).
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            device: Device to use (default: model's current device)
        
        Returns:
            Numpy array of predictions
        """
        self.eval()
        
        # Encode SMILES
        encoded = self.encode_smiles(smiles)
        
        # Move to device
        if device is None:
            device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            predictions = self.forward(input_ids, attention_mask)
        
        return predictions.cpu().numpy()
    
    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get [CLS] token embedding without regression head.
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            [CLS] embeddings [batch_size, encoder_hidden_size]
        """
        self.eval()
        
        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        
        return cls_embedding
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the [CLS] embedding.
        
        Returns:
            Embedding dimension (typically 768 for base models)
        """
        return self.encoder_hidden_size


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for SMILES strings and binding affinities.
    
    Attributes:
        smiles_list: List of SMILES strings
        labels: Numpy array of binding affinities
        tokenizer: ChemBERTa tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """
        Initialize SMILES dataset.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Numpy array of binding affinities
            tokenizer: ChemBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


@dataclass
class ChemBERTaTrainingConfig:
    """Configuration for ChemBERTa training.
    
    Attributes:
        lr: Learning rate (default: 1e-4, range: 1e-4 to 1e-5)
        weight_decay: L2 regularization (default: 1e-5)
        epochs: Maximum training epochs (default: 100)
        patience: Early stopping patience (default: 15)
        batch_size: Training batch size (default: 16)
        warmup_steps: Number of warmup steps (default: 100)
        gradient_accumulation_steps: Gradient accumulation (default: 1)
    """
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    batch_size: int = 16
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1


class ChemBERTaTrainer:
    """
    Trainer for ChemBERTa model with lower learning rate for fine-tuning.
    
    Implements training loop with:
    - Lower learning rate (1e-4 to 1e-5) for fine-tuning
    - AdamW optimizer
    - Early stopping
    
    Requirements: 9.4
    """
    
    def __init__(
        self,
        model: ChemBERTaPredictor,
        config: Optional[ChemBERTaTrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize ChemBERTa trainer.
        
        Args:
            model: ChemBERTaPredictor model to train
            config: Training configuration (default: ChemBERTaTrainingConfig())
            device: Device to use ('cuda', 'cpu', or None for auto)
        
        Requirements: 9.4
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ChemBERTaTrainer")
        
        self.model = model
        self.config = config or ChemBERTaTrainingConfig()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer with lower learning rate (Requirements: 9.4)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Will be initialized in train()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        logger.info(
            f"ChemBERTaTrainer initialized: lr={self.config.lr}, "
            f"patience={self.config.patience}, device={self.device}"
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the ChemBERTa model with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary with train_loss, val_loss, lr
        
        Requirements: 9.4
        """
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        
        # Initialize scheduler with total steps
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
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
                self.training_history['val_loss'].append(train_loss)
            
            # Logging
            if verbose and (epoch + 1) % 5 == 0:
                val_loss_str = f"{self.training_history['val_loss'][-1]:.4f}" if val_loader else "N/A"
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss_str}, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Restore best model if early stopping was used
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_loss={self.best_val_loss:.4f}")
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device).view(-1, 1)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            
            # Compute MSE loss
            loss = F.mse_loss(predictions, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).view(-1, 1)
                
                predictions = self.model(input_ids, attention_mask)
                loss = F.mse_loss(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: DataLoader for prediction data
        
        Returns:
            Numpy array of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pred = self.model(input_ids, attention_mask)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
        
        Returns:
            Dictionary with R², RMSE, MAE metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                pred = self.model(input_ids, attention_mask)
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse
        }
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_config': self.model.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Checkpoint loaded from {path}")


def create_chemberta_model(
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    freeze_layers: float = 0.5,
    hidden_dim_1: int = 256,
    hidden_dim_2: int = 128
) -> ChemBERTaPredictor:
    """
    Factory function to create a ChemBERTaPredictor model.
    
    Args:
        model_name: Pre-trained model name (default: "seyonec/ChemBERTa-zinc-base-v1")
        freeze_layers: Fraction of layers to freeze (default: 0.5)
        hidden_dim_1: First hidden layer dimension (default: 256)
        hidden_dim_2: Second hidden layer dimension (default: 128)
    
    Returns:
        Configured ChemBERTaPredictor model
    
    Requirements: 9.1, 9.2, 9.3
    """
    return ChemBERTaPredictor(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2
    )


def train_chemberta_model(
    model: ChemBERTaPredictor,
    train_smiles: List[str],
    train_labels: np.ndarray,
    val_smiles: Optional[List[str]] = None,
    val_labels: Optional[np.ndarray] = None,
    config: Optional[ChemBERTaTrainingConfig] = None,
    device: Optional[str] = None
) -> Tuple[ChemBERTaPredictor, Dict[str, List[float]]]:
    """
    Convenience function to train a ChemBERTa model.
    
    Args:
        model: ChemBERTaPredictor model to train
        train_smiles: List of training SMILES strings
        train_labels: Training labels
        val_smiles: List of validation SMILES strings (optional)
        val_labels: Validation labels (optional)
        config: Training configuration
        device: Device to use
    
    Returns:
        Tuple of (trained model, training history)
    
    Requirements: 9.4
    """
    config = config or ChemBERTaTrainingConfig()
    
    # Create datasets
    train_dataset = SMILESDataset(
        train_smiles,
        train_labels,
        model.tokenizer,
        model.config.max_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = None
    if val_smiles is not None and val_labels is not None:
        val_dataset = SMILESDataset(
            val_smiles,
            val_labels,
            model.tokenizer,
            model.config.max_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
    
    # Create trainer and train
    trainer = ChemBERTaTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    return model, history
