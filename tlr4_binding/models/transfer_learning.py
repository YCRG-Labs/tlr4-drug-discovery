"""
Transfer Learning Manager for TLR4 binding affinity prediction.

This module implements transfer learning capabilities by pre-training models
on related TLR targets (TLR2, TLR7, TLR8, TLR9) and fine-tuning on TLR4 data.

Requirements: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import logging
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# PyTorch imports with availability check
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# PyTorch Geometric imports with availability check
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None
    DataLoader = None


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning.
    
    Attributes:
        pretrain_lr: Learning rate for pre-training (default: 1e-3)
        finetune_lr: Learning rate for fine-tuning (default: 1e-4)
        pretrain_epochs: Maximum pre-training epochs (default: 100)
        finetune_epochs: Maximum fine-tuning epochs (default: 100)
        pretrain_patience: Early stopping patience for pre-training (default: 20)
        finetune_patience: Early stopping patience for fine-tuning (default: 20)
        freeze_layers: Number of layers to freeze during fine-tuning (default: 0)
        batch_size: Batch size for training (default: 32)
        weight_decay: L2 regularization (default: 1e-4)
    """
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    pretrain_epochs: int = 100
    finetune_epochs: int = 100
    pretrain_patience: int = 20
    finetune_patience: int = 20
    freeze_layers: int = 0
    batch_size: int = 32
    weight_decay: float = 1e-4


class TransferLearningManager:
    """
    Manager for transfer learning from related TLR targets to TLR4.
    
    Implements:
    - Pre-training on combined TLR2/7/8/9 data (500-1000 compounds)
    - Fine-tuning on TLR4-specific data with optional layer freezing
    - Comparison of transfer learning vs training from scratch
    
    Requirements: 11.1, 11.2, 11.3, 11.4
    
    Attributes:
        config: TransferLearningConfig with hyperparameters
        device: Device to use for training ('cuda' or 'cpu')
        pretrained_model: Model after pre-training
        finetuned_model: Model after fine-tuning
    """
    
    def __init__(
        self,
        config: Optional[TransferLearningConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize TransferLearningManager.
        
        Args:
            config: Transfer learning configuration (default: TransferLearningConfig())
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TransferLearningManager")
        
        self.config = config or TransferLearningConfig()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.pretrained_model: Optional[nn.Module] = None
        self.finetuned_model: Optional[nn.Module] = None
        
        # Training history
        self.pretrain_history: Dict[str, List[float]] = {}
        self.finetune_history: Dict[str, List[float]] = {}
        
        logger.info(
            f"TransferLearningManager initialized: "
            f"pretrain_lr={self.config.pretrain_lr}, "
            f"finetune_lr={self.config.finetune_lr}, "
            f"device={self.device}"
        )
    
    def pretrain(
        self,
        model: nn.Module,
        related_data: Union[DataLoader, List[Data]],
        val_data: Optional[Union[DataLoader, List[Data]]] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> nn.Module:
        """
        Pre-train model on combined TLR2/7/8/9 data.
        
        Trains the model on a larger dataset from related TLR targets to learn
        general TLR binding patterns before fine-tuning on TLR4-specific data.
        
        Args:
            model: Neural network model to pre-train (e.g., TLR4GAT)
            related_data: DataLoader or list of Data objects for TLR2/7/8/9 compounds
            val_data: Optional validation data
            epochs: Number of pre-training epochs (default: from config)
            verbose: Whether to print training progress
        
        Returns:
            Pre-trained model
        
        Requirements: 11.1
        """
        logger.info("Starting pre-training on related TLR data")
        
        # Convert to DataLoader if needed
        if not isinstance(related_data, DataLoader):
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("PyTorch Geometric is required for graph data")
            related_data = DataLoader(
                related_data,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        if val_data is not None and not isinstance(val_data, DataLoader):
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("PyTorch Geometric is required for graph data")
            val_data = DataLoader(
                val_data,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer with standard learning rate for pre-training
        optimizer = Adam(
            model.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Setup learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # Training loop with early stopping
        epochs = epochs or self.config.pretrain_epochs
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        self.pretrain_history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(model, related_data, optimizer)
            self.pretrain_history['train_loss'].append(train_loss)
            self.pretrain_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_data is not None:
                val_loss = self._validate_epoch(model, val_data)
                self.pretrain_history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.pretrain_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience={self.config.pretrain_patience})"
                    )
                    break
            else:
                self.pretrain_history['val_loss'].append(train_loss)
            
            # Update learning rate
            scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = f"{self.pretrain_history['val_loss'][-1]:.4f}" if val_data else "N/A"
                logger.info(
                    f"Pre-train Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss_str}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best pre-trained model with val_loss={best_val_loss:.4f}")
        
        self.pretrained_model = model
        logger.info("Pre-training completed")
        
        return model

    
    def finetune(
        self,
        model: nn.Module,
        tlr4_data: Union[DataLoader, List[Data]],
        val_data: Optional[Union[DataLoader, List[Data]]] = None,
        freeze_layers: Optional[int] = None,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> nn.Module:
        """
        Fine-tune pre-trained model on TLR4-specific data.
        
        Fine-tunes a pre-trained model on TLR4 data with optional layer freezing
        and lower learning rate to preserve learned representations while adapting
        to TLR4-specific patterns.
        
        Args:
            model: Pre-trained neural network model
            tlr4_data: DataLoader or list of Data objects for TLR4 compounds
            val_data: Optional validation data
            freeze_layers: Number of early layers to freeze (default: from config)
            lr: Learning rate for fine-tuning (default: 1e-4 from config)
            epochs: Number of fine-tuning epochs (default: from config)
            verbose: Whether to print training progress
        
        Returns:
            Fine-tuned model
        
        Requirements: 11.2, 11.3
        """
        logger.info("Starting fine-tuning on TLR4 data")
        
        # Convert to DataLoader if needed
        if not isinstance(tlr4_data, DataLoader):
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("PyTorch Geometric is required for graph data")
            tlr4_data = DataLoader(
                tlr4_data,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        if val_data is not None and not isinstance(val_data, DataLoader):
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("PyTorch Geometric is required for graph data")
            val_data = DataLoader(
                val_data,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Move model to device
        model = model.to(self.device)
        
        # Freeze layers if specified (Requirements: 11.2)
        freeze_layers = freeze_layers if freeze_layers is not None else self.config.freeze_layers
        if freeze_layers > 0:
            self._freeze_layers(model, freeze_layers)
            logger.info(f"Froze first {freeze_layers} layers for fine-tuning")
        
        # Setup optimizer with lower learning rate (Requirements: 11.3)
        lr = lr if lr is not None else self.config.finetune_lr
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        logger.info(f"Fine-tuning with lr={lr} (lower than pre-training)")
        
        # Setup learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-7
        )
        
        # Training loop with early stopping
        epochs = epochs or self.config.finetune_epochs
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        self.finetune_history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(model, tlr4_data, optimizer)
            self.finetune_history['train_loss'].append(train_loss)
            self.finetune_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_data is not None:
                val_loss = self._validate_epoch(model, val_data)
                self.finetune_history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.finetune_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience={self.config.finetune_patience})"
                    )
                    break
            else:
                self.finetune_history['val_loss'].append(train_loss)
            
            # Update learning rate
            scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = f"{self.finetune_history['val_loss'][-1]:.4f}" if val_data else "N/A"
                logger.info(
                    f"Fine-tune Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss_str}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best fine-tuned model with val_loss={best_val_loss:.4f}")
        
        # Unfreeze all layers after fine-tuning
        if freeze_layers > 0:
            self._unfreeze_all_layers(model)
        
        self.finetuned_model = model
        logger.info("Fine-tuning completed")
        
        return model
    
    def compare_transfer_vs_scratch(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        related_data: Union[DataLoader, List[Data]],
        tlr4_train_data: Union[DataLoader, List[Data]],
        tlr4_val_data: Optional[Union[DataLoader, List[Data]]] = None,
        tlr4_test_data: Optional[Union[DataLoader, List[Data]]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compare transfer learning vs training from scratch.
        
        Trains two models: one with transfer learning (pre-train + fine-tune) and
        one from scratch on TLR4 data only. Evaluates both on test data to assess
        the benefit of transfer learning.
        
        Args:
            model_class: Model class to instantiate (e.g., TLR4GAT)
            model_kwargs: Keyword arguments for model initialization
            related_data: DataLoader or list for TLR2/7/8/9 pre-training data
            tlr4_train_data: DataLoader or list for TLR4 training data
            tlr4_val_data: Optional TLR4 validation data
            tlr4_test_data: Optional TLR4 test data for evaluation
            verbose: Whether to print progress
        
        Returns:
            Dictionary with comparison results:
                - 'transfer_model': Fine-tuned model
                - 'scratch_model': Model trained from scratch
                - 'transfer_metrics': Test metrics for transfer model
                - 'scratch_metrics': Test metrics for scratch model
                - 'transfer_history': Training history for transfer learning
                - 'scratch_history': Training history for scratch model
                - 'improvement': Percentage improvement in R² (if test data provided)
        
        Requirements: 11.4
        """
        logger.info("Starting comparison: Transfer Learning vs Training from Scratch")
        
        # ===== Transfer Learning Path =====
        logger.info("\n=== Transfer Learning Path ===")
        
        # Create model for transfer learning
        transfer_model = model_class(**model_kwargs)
        
        # Pre-train on related TLR data
        logger.info("Step 1: Pre-training on related TLR data")
        transfer_model = self.pretrain(
            transfer_model,
            related_data,
            val_data=None,  # No validation during pre-training for fair comparison
            verbose=verbose
        )
        
        # Fine-tune on TLR4 data
        logger.info("Step 2: Fine-tuning on TLR4 data")
        transfer_model = self.finetune(
            transfer_model,
            tlr4_train_data,
            val_data=tlr4_val_data,
            verbose=verbose
        )
        
        transfer_history = {
            'pretrain': self.pretrain_history,
            'finetune': self.finetune_history
        }
        
        # ===== Training from Scratch Path =====
        logger.info("\n=== Training from Scratch Path ===")
        
        # Create fresh model for training from scratch
        scratch_model = model_class(**model_kwargs)
        scratch_model = scratch_model.to(self.device)
        
        # Convert to DataLoader if needed
        if not isinstance(tlr4_train_data, DataLoader):
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("PyTorch Geometric is required for graph data")
            tlr4_train_loader = DataLoader(
                tlr4_train_data,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        else:
            tlr4_train_loader = tlr4_train_data
        
        if tlr4_val_data is not None and not isinstance(tlr4_val_data, DataLoader):
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("PyTorch Geometric is required for graph data")
            tlr4_val_loader = DataLoader(
                tlr4_val_data,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        else:
            tlr4_val_loader = tlr4_val_data
        
        # Train from scratch with same total epochs as transfer learning
        total_transfer_epochs = len(self.pretrain_history['train_loss']) + len(self.finetune_history['train_loss'])
        scratch_epochs = total_transfer_epochs
        
        logger.info(f"Training from scratch for {scratch_epochs} epochs (matching transfer learning)")
        
        optimizer = Adam(
            scratch_model.parameters(),
            lr=self.config.pretrain_lr,  # Use same initial LR as pre-training
            weight_decay=self.config.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        scratch_history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(scratch_epochs):
            # Training phase
            train_loss = self._train_epoch(scratch_model, tlr4_train_loader, optimizer)
            scratch_history['train_loss'].append(train_loss)
            scratch_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if tlr4_val_loader is not None:
                val_loss = self._validate_epoch(scratch_model, tlr4_val_loader)
                scratch_history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in scratch_model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.finetune_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience={self.config.finetune_patience})"
                    )
                    break
            else:
                scratch_history['val_loss'].append(train_loss)
            
            # Update learning rate
            scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = f"{scratch_history['val_loss'][-1]:.4f}" if tlr4_val_loader else "N/A"
                logger.info(
                    f"Scratch Epoch {epoch + 1}/{scratch_epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss_str}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            scratch_model.load_state_dict(best_model_state)
            logger.info(f"Restored best scratch model with val_loss={best_val_loss:.4f}")
        
        # ===== Evaluation =====
        results = {
            'transfer_model': transfer_model,
            'scratch_model': scratch_model,
            'transfer_history': transfer_history,
            'scratch_history': scratch_history
        }
        
        if tlr4_test_data is not None:
            logger.info("\n=== Evaluation on Test Data ===")
            
            # Convert to DataLoader if needed
            if not isinstance(tlr4_test_data, DataLoader):
                if not TORCH_GEOMETRIC_AVAILABLE:
                    raise RuntimeError("PyTorch Geometric is required for graph data")
                tlr4_test_loader = DataLoader(
                    tlr4_test_data,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
            else:
                tlr4_test_loader = tlr4_test_data
            
            # Evaluate both models
            transfer_metrics = self._evaluate_model(transfer_model, tlr4_test_loader)
            scratch_metrics = self._evaluate_model(scratch_model, tlr4_test_loader)
            
            results['transfer_metrics'] = transfer_metrics
            results['scratch_metrics'] = scratch_metrics
            
            # Calculate improvement
            if 'r2' in transfer_metrics and 'r2' in scratch_metrics:
                r2_improvement = ((transfer_metrics['r2'] - scratch_metrics['r2']) / 
                                 max(abs(scratch_metrics['r2']), 1e-6)) * 100
                results['improvement'] = {
                    'r2_percent': r2_improvement,
                    'r2_absolute': transfer_metrics['r2'] - scratch_metrics['r2']
                }
                
                logger.info(f"\nTransfer Learning R²: {transfer_metrics['r2']:.4f}")
                logger.info(f"Training from Scratch R²: {scratch_metrics['r2']:.4f}")
                logger.info(f"Improvement: {r2_improvement:+.2f}%")
            
            logger.info("\nTransfer Learning Metrics:")
            for metric, value in transfer_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("\nTraining from Scratch Metrics:")
            for metric, value in scratch_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nComparison completed")
        return results
    
    def _train_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            data_loader: DataLoader for training data
            optimizer: Optimizer
        
        Returns:
            Average training loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch)
            
            # Compute MSE loss
            targets = batch.y.view(-1, 1) if batch.y.dim() == 1 else batch.y
            loss = torch.nn.functional.mse_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> float:
        """
        Validate for one epoch.
        
        Args:
            model: Model to validate
            data_loader: DataLoader for validation data
        
        Returns:
            Average validation loss for the epoch
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                predictions = model(batch)
                targets = batch.y.view(-1, 1) if batch.y.dim() == 1 else batch.y
                loss = torch.nn.functional.mse_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
        
        Returns:
            Dictionary with R², RMSE, MAE metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                pred = model(batch)
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
    
    def _freeze_layers(self, model: nn.Module, num_layers: int) -> None:
        """
        Freeze the first num_layers of the model.
        
        Args:
            model: Model to freeze layers in
            num_layers: Number of layers to freeze
        """
        # For GAT models, freeze GAT layers
        if hasattr(model, 'gat_layers'):
            for i, layer in enumerate(model.gat_layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    logger.debug(f"Froze GAT layer {i}")
        
        # For other models, freeze by parameter groups
        else:
            param_groups = list(model.named_parameters())
            total_params = len(param_groups)
            freeze_count = min(num_layers, total_params)
            
            for i, (name, param) in enumerate(param_groups):
                if i < freeze_count:
                    param.requires_grad = False
                    logger.debug(f"Froze parameter: {name}")
    
    def _unfreeze_all_layers(self, model: nn.Module) -> None:
        """
        Unfreeze all layers in the model.
        
        Args:
            model: Model to unfreeze
        """
        for param in model.parameters():
            param.requires_grad = True
        logger.debug("Unfroze all layers")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save transfer learning checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'config': self.config,
            'pretrain_history': self.pretrain_history,
            'finetune_history': self.finetune_history,
        }
        
        if self.pretrained_model is not None:
            checkpoint['pretrained_model_state'] = self.pretrained_model.state_dict()
        
        if self.finetuned_model is not None:
            checkpoint['finetuned_model_state'] = self.finetuned_model.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Transfer learning checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, model: Optional[nn.Module] = None) -> None:
        """
        Load transfer learning checkpoint.
        
        Args:
            path: Path to load checkpoint from
            model: Optional model to load state into
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.pretrain_history = checkpoint.get('pretrain_history', {})
        self.finetune_history = checkpoint.get('finetune_history', {})
        
        if model is not None and 'finetuned_model_state' in checkpoint:
            model.load_state_dict(checkpoint['finetuned_model_state'])
            self.finetuned_model = model
            logger.info("Loaded fine-tuned model state")
        elif model is not None and 'pretrained_model_state' in checkpoint:
            model.load_state_dict(checkpoint['pretrained_model_state'])
            self.pretrained_model = model
            logger.info("Loaded pre-trained model state")
        
        logger.info(f"Transfer learning checkpoint loaded from {path}")


def create_transfer_learning_manager(
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    freeze_layers: int = 0,
    device: Optional[str] = None
) -> TransferLearningManager:
    """
    Factory function to create a TransferLearningManager.
    
    Args:
        pretrain_lr: Learning rate for pre-training (default: 1e-3)
        finetune_lr: Learning rate for fine-tuning (default: 1e-4)
        freeze_layers: Number of layers to freeze during fine-tuning (default: 0)
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        Configured TransferLearningManager
    
    Requirements: 11.1, 11.2, 11.3
    """
    config = TransferLearningConfig(
        pretrain_lr=pretrain_lr,
        finetune_lr=finetune_lr,
        freeze_layers=freeze_layers
    )
    
    return TransferLearningManager(config=config, device=device)
