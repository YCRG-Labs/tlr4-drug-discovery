"""
Graph Attention Network (GAT) for TLR4 binding affinity prediction.

This module implements the TLR4GAT class with attention-based message passing
for learning molecular representations and predicting binding affinities.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 18.1
"""

from __future__ import annotations

import logging
import math
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# PyTorch Geometric imports with availability check
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None
    Batch = None
    GATConv = None
    global_mean_pool = None
    global_max_pool = None
    DataLoader = None


@dataclass
class GATConfig:
    """Configuration for TLR4GAT model.
    
    Attributes:
        node_features: Dimension of input node features
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 1 for regression)
        num_layers: Number of GAT layers (default: 4, range: 3-4)
        num_heads: Number of attention heads per layer (default: 8, range: 4-8)
        dropout: Dropout rate (default: 0.2, range: 0.2-0.3)
        use_edge_attr: Whether to use edge attributes (default: False)
    """
    node_features: int
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.2
    use_edge_attr: bool = False


class TLR4GAT(nn.Module):
    """
    Graph Attention Network for TLR4 binding affinity prediction.
    
    Implements a GAT architecture with:
    - 3-4 GAT layers with 4-8 attention heads per layer
    - Batch normalization and dropout (0.2-0.3) for regularization
    - Mean + max global pooling for graph-level representation
    - Attention weight extraction for interpretability
    
    Requirements: 8.1, 8.2, 8.3, 18.1
    
    Attributes:
        config: GATConfig with model hyperparameters
        gat_layers: ModuleList of GATConv layers
        batch_norms: ModuleList of BatchNorm1d layers
        fc_layers: Final fully connected layers for prediction
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        Initialize TLR4GAT model.
        
        Args:
            node_features: Dimension of input node features
            hidden_dim: Hidden layer dimension (default: 128)
            output_dim: Output dimension (default: 1 for regression)
            num_layers: Number of GAT layers (default: 4, range: 3-4)
            num_heads: Number of attention heads per layer (default: 8, range: 4-8)
            dropout: Dropout rate (default: 0.2, range: 0.2-0.3)
        
        Requirements: 8.1, 8.2
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TLR4GAT")
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is required for TLR4GAT")
        
        super().__init__()
        
        # Store configuration
        self.config = GATConfig(
            node_features=node_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Validate parameters
        if num_layers < 3 or num_layers > 4:
            logger.warning(f"num_layers={num_layers} outside recommended range [3,4]")
        if num_heads < 4 or num_heads > 8:
            logger.warning(f"num_heads={num_heads} outside recommended range [4,8]")
        if dropout < 0.2 or dropout > 0.3:
            logger.warning(f"dropout={dropout} outside recommended range [0.2,0.3]")
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First GAT layer: node_features -> hidden_dim
        # GATConv output dim = hidden_dim * num_heads when concat=True
        # We use concat=True for intermediate layers, concat=False for last layer
        self.gat_layers.append(
            GATConv(
                in_channels=node_features,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Intermediate GAT layers
        for i in range(1, num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Last GAT layer: concat=False to get single output per node
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim * num_heads,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=False,  # Average attention heads
                dropout=dropout
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Fully connected layers after pooling
        # Mean + max pooling doubles the dimension
        pooled_dim = hidden_dim * 2
        self.fc_layers = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Store attention weights for interpretability
        self._attention_weights: List[torch.Tensor] = []
        self._return_attention = False
        
        logger.info(
            f"TLR4GAT initialized: {num_layers} layers, {num_heads} heads, "
            f"hidden_dim={hidden_dim}, dropout={dropout}"
        )
    
    def forward(
        self,
        data: Union[Data, Batch]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the GAT model.
        
        Args:
            data: PyTorch Geometric Data or Batch object with:
                - x: Node feature matrix [num_nodes, node_features]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment vector (for batched graphs)
        
        Returns:
            If return_attention is False:
                Predicted binding affinity [batch_size, output_dim]
            If return_attention is True:
                Tuple of (predictions, attention_weights_list)
        
        Requirements: 8.3
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Clear previous attention weights
        self._attention_weights = []
        
        # Apply GAT layers with batch normalization and dropout
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            # GAT layer with attention
            x, attention = gat_layer(x, edge_index, return_attention_weights=True)
            
            # Store attention weights for interpretability
            if self._return_attention:
                self._attention_weights.append(attention)
            
            # Batch normalization
            x = batch_norm(x)
            
            # ReLU activation (except for last layer before pooling)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Global pooling: mean + max (Requirements: 8.3)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        # Final prediction layers
        output = self.fc_layers(x_pooled)
        
        if self._return_attention:
            return output, self._attention_weights
        return output

    
    def get_attention_weights(self, data: Union[Data, Batch]) -> Dict[int, float]:
        """
        Extract attention weights mapped to atom indices for interpretability.
        
        Performs a forward pass with attention extraction enabled and aggregates
        attention weights across all layers and heads to produce per-atom importance.
        
        Args:
            data: PyTorch Geometric Data object for a single molecule
        
        Returns:
            Dictionary mapping atom index to aggregated attention weight
        
        Requirements: 18.1
        """
        self.eval()
        self._return_attention = True
        
        with torch.no_grad():
            _, attention_list = self.forward(data)
        
        self._return_attention = False
        
        # Aggregate attention weights across layers and heads
        num_atoms = data.x.size(0)
        atom_attention = {i: 0.0 for i in range(num_atoms)}
        
        for layer_idx, (edge_index, attention) in enumerate(attention_list):
            # attention shape: [num_edges, num_heads] or [num_edges]
            if attention.dim() == 2:
                # Average across heads
                attention = attention.mean(dim=1)
            
            # Aggregate attention by target node (incoming attention)
            edge_index_np = edge_index.cpu().numpy()
            attention_np = attention.cpu().numpy()
            
            for edge_idx in range(edge_index_np.shape[1]):
                target_node = edge_index_np[1, edge_idx]
                atom_attention[target_node] += attention_np[edge_idx]
        
        # Normalize attention weights
        total_attention = sum(atom_attention.values())
        if total_attention > 0:
            atom_attention = {k: v / total_attention for k, v in atom_attention.items()}
        
        return atom_attention
    
    def get_node_embeddings(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get node-level embeddings after GAT layers (before pooling).
        
        Args:
            data: PyTorch Geometric Data or Batch object
        
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        self.eval()
        
        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            
            for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
                x, _ = gat_layer(x, edge_index, return_attention_weights=True)
                x = batch_norm(x)
                if i < len(self.gat_layers) - 1:
                    x = F.relu(x)
        
        return x
    
    def get_graph_embedding(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get graph-level embedding after pooling (before final FC layers).
        
        Args:
            data: PyTorch Geometric Data or Batch object
        
        Returns:
            Graph embedding [batch_size, hidden_dim * 2]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x, _ = gat_layer(x, edge_index, return_attention_weights=True)
            x = batch_norm(x)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        return x_pooled
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the graph-level embedding.
        
        Returns:
            Embedding dimension (hidden_dim * 2 due to mean+max pooling)
        
        Requirements: 8.3
        """
        return self.config.hidden_dim * 2


@dataclass
class TrainingConfig:
    """Configuration for GAT training.
    
    Attributes:
        lr: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 1e-4)
        epochs: Maximum training epochs (default: 200)
        patience: Early stopping patience (default: 20)
        batch_size: Training batch size (default: 32)
        t_max: Cosine annealing T_max (default: 50)
    """
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    t_max: int = 50


class GATTrainer:
    """
    Trainer for TLR4GAT model with Adam optimizer and cosine annealing.
    
    Implements training loop with:
    - Adam optimizer with lr=1e-3
    - Cosine annealing learning rate schedule
    - Early stopping with patience=20
    
    Requirements: 8.4, 8.5
    """
    
    def __init__(
        self,
        model: TLR4GAT,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize GAT trainer.
        
        Args:
            model: TLR4GAT model to train
            config: Training configuration (default: TrainingConfig())
            device: Device to use ('cuda', 'cpu', or None for auto)
        
        Requirements: 8.4, 8.5
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GATTrainer")
        
        self.model = model
        self.config = config or TrainingConfig()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer with Adam and lr=1e-3 (Requirements: 8.4)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing scheduler (Requirements: 8.4)
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
            'val_loss': [],
            'lr': []
        }
        
        logger.info(
            f"GATTrainer initialized: lr={self.config.lr}, "
            f"patience={self.config.patience}, device={self.device}"
        )

    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the GAT model with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary with train_loss, val_loss, lr
        
        Requirements: 8.4, 8.5
        """
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                
                # Early stopping check (Requirements: 8.5)
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
            
            # Update learning rate with cosine annealing
            self.scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
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
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch)
            
            # Compute MSE loss
            targets = batch.y.view(-1, 1) if batch.y.dim() == 1 else batch.y
            loss = F.mse_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
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
                batch = batch.to(self.device)
                
                predictions = self.model(batch)
                targets = batch.y.view(-1, 1) if batch.y.dim() == 1 else batch.y
                loss = F.mse_loss(predictions, targets)
                
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
                batch = batch.to(self.device)
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
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
                batch = batch.to(self.device)
                pred = self.model(batch)
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
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
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Checkpoint loaded from {path}")


def create_gat_model(
    node_features: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.2
) -> TLR4GAT:
    """
    Factory function to create a TLR4GAT model.
    
    Args:
        node_features: Dimension of input node features
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of GAT layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.2)
    
    Returns:
        Configured TLR4GAT model
    
    Requirements: 8.1, 8.2, 8.3
    """
    return TLR4GAT(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )


def train_gat_model(
    model: TLR4GAT,
    train_data: List[Data],
    val_data: Optional[List[Data]] = None,
    config: Optional[TrainingConfig] = None,
    device: Optional[str] = None
) -> Tuple[TLR4GAT, Dict[str, List[float]]]:
    """
    Convenience function to train a TLR4GAT model.
    
    Args:
        model: TLR4GAT model to train
        train_data: List of training Data objects
        val_data: List of validation Data objects (optional)
        config: Training configuration
        device: Device to use
    
    Returns:
        Tuple of (trained model, training history)
    
    Requirements: 8.4, 8.5
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is required for training")
    
    config = config or TrainingConfig()
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False) if val_data else None
    
    # Create trainer and train
    trainer = GATTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    return model, history
