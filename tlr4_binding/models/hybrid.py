"""
Hybrid Model for TLR4 binding affinity prediction.

This module implements the HybridModel class that combines Graph Neural Network (GNN)
and traditional descriptor branches for enhanced prediction performance.

Requirements: 10.1, 10.2, 10.3
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

# Import TLR4GAT model
try:
    from tlr4_binding.models.gat import TLR4GAT
    GAT_AVAILABLE = True
except ImportError:
    GAT_AVAILABLE = False
    TLR4GAT = None


@dataclass
class HybridConfig:
    """Configuration for Hybrid model.
    
    Attributes:
        gnn_hidden_dim: GNN branch hidden dimension (default: 128)
        descriptor_hidden_dim: Descriptor branch hidden dimension (default: 128)
        fusion_hidden_dim: Fusion layer hidden dimension (default: 128)
        output_dim: Output dimension (default: 1 for regression)
        dropout: Dropout rate (default: 0.2)
    """
    gnn_hidden_dim: int = 128
    descriptor_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    output_dim: int = 1
    dropout: float = 0.2


class HybridModel(nn.Module):
    """
    Hybrid model combining GNN and descriptor-based predictions.
    
    Implements a dual-branch architecture with:
    - GNN branch: TLR4GAT for learning from molecular graphs
    - Descriptor branch: MLP for traditional molecular descriptors
    - Fusion layers: Concatenation and joint prediction
    
    Requirements: 10.1, 10.2, 10.3
    
    Attributes:
        config: HybridConfig with model hyperparameters
        gnn_model: TLR4GAT model for graph branch
        descriptor_branch: MLP for descriptor branch
        fusion_layers: Fully connected layers for joint prediction
    """
    
    def __init__(
        self,
        gnn_model: TLR4GAT,
        descriptor_dim: int,
        descriptor_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize Hybrid model with dual-branch architecture.
        
        Args:
            gnn_model: Pre-configured TLR4GAT model for graph branch
            descriptor_dim: Dimension of input descriptor features
            descriptor_hidden_dim: Hidden dimension for descriptor branch (default: 128)
            fusion_hidden_dim: Hidden dimension for fusion layers (default: 128)
            output_dim: Output dimension (default: 1 for regression)
            dropout: Dropout rate (default: 0.2)
        
        Requirements: 10.1, 10.2
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for HybridModel")
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is required for HybridModel")
        if not GAT_AVAILABLE or gnn_model is None:
            raise RuntimeError("TLR4GAT model is required for HybridModel")
        
        super().__init__()
        
        # Store configuration
        self.config = HybridConfig(
            gnn_hidden_dim=gnn_model.config.hidden_dim,
            descriptor_hidden_dim=descriptor_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.descriptor_dim = descriptor_dim
        
        # GNN branch: Use provided TLR4GAT model
        self.gnn_model = gnn_model
        
        # Get GNN embedding dimension (hidden_dim * 2 due to mean+max pooling)
        self.gnn_embedding_dim = self.gnn_model.get_embedding_dim()
        
        # Descriptor branch: MLP for traditional descriptors
        self.descriptor_branch = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_hidden_dim * 2),
            nn.BatchNorm1d(descriptor_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(descriptor_hidden_dim * 2, descriptor_hidden_dim),
            nn.BatchNorm1d(descriptor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layers: Concatenate embeddings and predict
        # Total embedding dimension = GNN embedding + descriptor embedding
        total_embedding_dim = self.gnn_embedding_dim + descriptor_hidden_dim
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_embedding_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, output_dim)
        )
        
        logger.info(
            f"HybridModel initialized: "
            f"GNN embedding={self.gnn_embedding_dim}, "
            f"Descriptor embedding={descriptor_hidden_dim}, "
            f"Total embedding={total_embedding_dim}, "
            f"Fusion hidden={fusion_hidden_dim}"
        )
    
    def forward(
        self,
        graph_data: Union[Data, Batch],
        descriptors: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass combining both branches.
        
        Args:
            graph_data: PyTorch Geometric Data or Batch object for GNN branch
            descriptors: Descriptor feature tensor [batch_size, descriptor_dim]
        
        Returns:
            Predicted binding affinity [batch_size, output_dim]
        
        Requirements: 10.2, 10.3
        """
        # GNN branch: Get graph-level embedding
        gnn_embedding = self.gnn_model.get_graph_embedding(graph_data)
        
        # Descriptor branch: Process traditional descriptors
        descriptor_embedding = self.descriptor_branch(descriptors)
        
        # Concatenate embeddings (Requirements: 10.2)
        fused_embedding = torch.cat([gnn_embedding, descriptor_embedding], dim=1)
        
        # Fusion layers for final prediction
        output = self.fusion_layers(fused_embedding)
        
        return output
    
    def get_embeddings(
        self,
        graph_data: Union[Data, Batch],
        descriptors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings from both branches and fused embedding.
        
        Args:
            graph_data: PyTorch Geometric Data or Batch object
            descriptors: Descriptor feature tensor [batch_size, descriptor_dim]
        
        Returns:
            Tuple of (gnn_embedding, descriptor_embedding, fused_embedding)
        """
        self.eval()
        
        with torch.no_grad():
            # GNN branch embedding
            gnn_embedding = self.gnn_model.get_graph_embedding(graph_data)
            
            # Descriptor branch embedding
            descriptor_embedding = self.descriptor_branch(descriptors)
            
            # Fused embedding
            fused_embedding = torch.cat([gnn_embedding, descriptor_embedding], dim=1)
        
        return gnn_embedding, descriptor_embedding, fused_embedding
    
    def get_gnn_attention(
        self,
        graph_data: Union[Data, Batch]
    ) -> Dict[int, float]:
        """
        Extract attention weights from GNN branch for interpretability.
        
        Args:
            graph_data: PyTorch Geometric Data object for a single molecule
        
        Returns:
            Dictionary mapping atom index to aggregated attention weight
        """
        return self.gnn_model.get_attention_weights(graph_data)
    
    def freeze_gnn_branch(self) -> None:
        """
        Freeze GNN branch parameters for transfer learning.
        """
        for param in self.gnn_model.parameters():
            param.requires_grad = False
        logger.info("GNN branch frozen")
    
    def unfreeze_gnn_branch(self) -> None:
        """
        Unfreeze GNN branch parameters for joint training.
        """
        for param in self.gnn_model.parameters():
            param.requires_grad = True
        logger.info("GNN branch unfrozen")
    
    def freeze_descriptor_branch(self) -> None:
        """
        Freeze descriptor branch parameters.
        """
        for param in self.descriptor_branch.parameters():
            param.requires_grad = False
        logger.info("Descriptor branch frozen")
    
    def unfreeze_descriptor_branch(self) -> None:
        """
        Unfreeze descriptor branch parameters.
        """
        for param in self.descriptor_branch.parameters():
            param.requires_grad = True
        logger.info("Descriptor branch unfrozen")


class HybridDataset(Dataset):
    """
    PyTorch Dataset for hybrid model combining graphs and descriptors.
    
    Attributes:
        graph_data_list: List of PyTorch Geometric Data objects
        descriptors: Numpy array of descriptor features [n_samples, descriptor_dim]
        labels: Numpy array of binding affinities [n_samples]
    """
    
    def __init__(
        self,
        graph_data_list: List[Data],
        descriptors: np.ndarray,
        labels: np.ndarray
    ):
        """
        Initialize hybrid dataset.
        
        Args:
            graph_data_list: List of PyTorch Geometric Data objects
            descriptors: Numpy array of descriptor features
            labels: Numpy array of binding affinities
        """
        if len(graph_data_list) != len(descriptors) or len(descriptors) != len(labels):
            raise ValueError(
                f"Mismatched lengths: graphs={len(graph_data_list)}, "
                f"descriptors={len(descriptors)}, labels={len(labels)}"
            )
        
        self.graph_data_list = graph_data_list
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.graph_data_list)
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Get graph, descriptors, and label at index.
        
        Returns:
            Tuple of (graph_data, descriptors, label)
        """
        return (
            self.graph_data_list[idx],
            self.descriptors[idx],
            self.labels[idx]
        )


def hybrid_collate_fn(batch: List[Tuple[Data, torch.Tensor, torch.Tensor]]) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for HybridDataset.
    
    Args:
        batch: List of (graph_data, descriptors, label) tuples
    
    Returns:
        Tuple of (batched_graphs, batched_descriptors, batched_labels)
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is required for batching")
    
    graphs, descriptors, labels = zip(*batch)
    
    # Batch graphs using PyTorch Geometric
    batched_graphs = Batch.from_data_list(list(graphs))
    
    # Stack descriptors and labels
    batched_descriptors = torch.stack(descriptors)
    batched_labels = torch.stack(labels)
    
    return batched_graphs, batched_descriptors, batched_labels


@dataclass
class HybridTrainingConfig:
    """Configuration for Hybrid model training.
    
    Attributes:
        lr: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 1e-4)
        epochs: Maximum training epochs (default: 200)
        patience: Early stopping patience (default: 20)
        batch_size: Training batch size (default: 32)
        t_max: Cosine annealing T_max (default: 50)
        warmup_epochs: Number of epochs to train descriptor branch only (default: 0)
    """
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    t_max: int = 50
    warmup_epochs: int = 0


class HybridTrainer:
    """
    Trainer for Hybrid model with end-to-end joint optimization.
    
    Implements training loop with:
    - Joint optimization of both GNN and descriptor branches
    - Adam optimizer with cosine annealing
    - Early stopping
    - Optional warmup phase for descriptor branch
    
    Requirements: 10.3
    """
    
    def __init__(
        self,
        model: HybridModel,
        config: Optional[HybridTrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Hybrid trainer.
        
        Args:
            model: HybridModel to train
            config: Training configuration (default: HybridTrainingConfig())
            device: Device to use ('cuda', 'cpu', or None for auto)
        
        Requirements: 10.3
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for HybridTrainer")
        
        self.model = model
        self.config = config or HybridTrainingConfig()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer for joint training (Requirements: 10.3)
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
            'val_loss': [],
            'lr': []
        }
        
        logger.info(
            f"HybridTrainer initialized: lr={self.config.lr}, "
            f"patience={self.config.patience}, device={self.device}"
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the Hybrid model with end-to-end optimization.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary with train_loss, val_loss, lr
        
        Requirements: 10.3
        """
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        
        # Optional warmup phase: train descriptor branch only
        if self.config.warmup_epochs > 0:
            logger.info(f"Warmup phase: training descriptor branch for {self.config.warmup_epochs} epochs")
            self.model.freeze_gnn_branch()
            
            for epoch in range(self.config.warmup_epochs):
                train_loss = self._train_epoch(train_loader)
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(f"Warmup Epoch {epoch + 1}/{self.config.warmup_epochs}: Train Loss={train_loss:.4f}")
            
            self.model.unfreeze_gnn_branch()
            logger.info("Warmup complete, starting joint training")
        
        # Joint training phase (Requirements: 10.3)
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
        Train for one epoch with joint optimization.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            graphs, descriptors, labels = batch
            graphs = graphs.to(self.device)
            descriptors = descriptors.to(self.device)
            labels = labels.to(self.device).view(-1, 1)
            
            self.optimizer.zero_grad()
            
            # Forward pass through both branches
            predictions = self.model(graphs, descriptors)
            
            # Compute MSE loss
            loss = F.mse_loss(predictions, labels)
            
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
                graphs, descriptors, labels = batch
                graphs = graphs.to(self.device)
                descriptors = descriptors.to(self.device)
                labels = labels.to(self.device).view(-1, 1)
                
                predictions = self.model(graphs, descriptors)
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
                graphs, descriptors, _ = batch
                graphs = graphs.to(self.device)
                descriptors = descriptors.to(self.device)
                
                pred = self.model(graphs, descriptors)
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
                graphs, descriptors, labels = batch
                graphs = graphs.to(self.device)
                descriptors = descriptors.to(self.device)
                
                pred = self.model(graphs, descriptors)
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


def create_hybrid_model(
    gnn_model: TLR4GAT,
    descriptor_dim: int,
    descriptor_hidden_dim: int = 128,
    fusion_hidden_dim: int = 128,
    dropout: float = 0.2
) -> HybridModel:
    """
    Factory function to create a HybridModel.
    
    Args:
        gnn_model: Pre-configured TLR4GAT model
        descriptor_dim: Dimension of input descriptor features
        descriptor_hidden_dim: Hidden dimension for descriptor branch (default: 128)
        fusion_hidden_dim: Hidden dimension for fusion layers (default: 128)
        dropout: Dropout rate (default: 0.2)
    
    Returns:
        Configured HybridModel
    
    Requirements: 10.1, 10.2
    """
    return HybridModel(
        gnn_model=gnn_model,
        descriptor_dim=descriptor_dim,
        descriptor_hidden_dim=descriptor_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout
    )


def train_hybrid_model(
    model: HybridModel,
    train_data: HybridDataset,
    val_data: Optional[HybridDataset] = None,
    config: Optional[HybridTrainingConfig] = None,
    device: Optional[str] = None
) -> Tuple[HybridModel, Dict[str, List[float]]]:
    """
    Convenience function to train a HybridModel.
    
    Args:
        model: HybridModel to train
        train_data: Training dataset
        val_data: Validation dataset (optional)
        config: Training configuration
        device: Device to use
    
    Returns:
        Tuple of (trained model, training history)
    
    Requirements: 10.3
    """
    config = config or HybridTrainingConfig()
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=hybrid_collate_fn
    )
    
    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=hybrid_collate_fn
        )
    
    # Create trainer and train
    trainer = HybridTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    return model, history
