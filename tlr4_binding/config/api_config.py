"""
API configuration and credentials management.

This module provides secure management of API keys and credentials
for external data sources (ChEMBL, PubChem) and model services.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """
    Configuration for external API access.
    
    Manages API keys, rate limits, and connection settings for
    ChEMBL, PubChem, and other external services.
    """
    # ChEMBL API settings
    chembl_base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    chembl_timeout: int = 30
    chembl_max_retries: int = 3
    chembl_rate_limit: float = 0.5  # seconds between requests
    
    # PubChem API settings
    pubchem_base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    pubchem_timeout: int = 30
    pubchem_max_retries: int = 3
    pubchem_rate_limit: float = 0.2  # seconds between requests
    
    # HuggingFace settings (for ChemBERTa)
    huggingface_token: Optional[str] = None
    huggingface_cache_dir: str = ".cache/huggingface"
    
    # General settings
    user_agent: str = "TLR4-Binding-Predictor/1.0"
    
    def __post_init__(self):
        """Load API keys from environment variables."""
        self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load sensitive credentials from environment variables."""
        # HuggingFace token
        if self.huggingface_token is None:
            self.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
        
        # Log status (without revealing keys)
        if self.huggingface_token:
            logger.info("HuggingFace token loaded from environment")
    
    def get_chembl_headers(self) -> Dict[str, str]:
        """Get headers for ChEMBL API requests."""
        return {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
    
    def get_pubchem_headers(self) -> Dict[str, str]:
        """Get headers for PubChem API requests."""
        return {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "chembl_base_url": self.chembl_base_url,
            "chembl_timeout": self.chembl_timeout,
            "chembl_max_retries": self.chembl_max_retries,
            "chembl_rate_limit": self.chembl_rate_limit,
            "pubchem_base_url": self.pubchem_base_url,
            "pubchem_timeout": self.pubchem_timeout,
            "pubchem_max_retries": self.pubchem_max_retries,
            "pubchem_rate_limit": self.pubchem_rate_limit,
            "huggingface_cache_dir": self.huggingface_cache_dir,
            "user_agent": self.user_agent,
            "has_huggingface_token": self.huggingface_token is not None,
        }


@dataclass
class HyperparameterConfig:
    """
    Hyperparameter configuration for all model types.
    
    Centralizes hyperparameter settings for GNN, transformer,
    hybrid, and traditional ML models.
    """
    # General training settings
    random_seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Graph Attention Network (GAT) hyperparameters
    gat_hidden_dim: int = 128
    gat_num_layers: int = 4
    gat_num_heads: int = 8
    gat_dropout: float = 0.2
    gat_batch_norm: bool = True
    gat_pooling: str = "mean_max"  # "mean", "max", "mean_max", "attention"
    
    # GAT training hyperparameters
    gat_learning_rate: float = 1e-3
    gat_weight_decay: float = 1e-5
    gat_batch_size: int = 32
    gat_epochs: int = 200
    gat_early_stopping_patience: int = 20
    gat_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    
    # ChemBERTa hyperparameters
    chemberta_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    chemberta_freeze_fraction: float = 0.5
    chemberta_hidden_dims: tuple = (256, 128)
    chemberta_dropout: float = 0.1
    
    # ChemBERTa training hyperparameters
    chemberta_learning_rate: float = 1e-4
    chemberta_weight_decay: float = 1e-5
    chemberta_batch_size: int = 16
    chemberta_epochs: int = 50
    chemberta_warmup_steps: int = 100
    
    # Hybrid model hyperparameters
    hybrid_fusion_dim: int = 128
    hybrid_fusion_layers: int = 2
    hybrid_fusion_dropout: float = 0.2
    
    # Transfer learning hyperparameters
    transfer_pretrain_epochs: int = 100
    transfer_finetune_epochs: int = 50
    transfer_finetune_lr: float = 1e-4
    transfer_freeze_layers: int = 0
    
    # Multi-task learning hyperparameters
    multitask_alpha: float = 0.6  # Weight for affinity loss
    multitask_classification_weight: float = 0.4
    
    # Traditional ML hyperparameters (for ensemble baseline)
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 20
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Validation hyperparameters
    cv_outer_folds: int = 5
    cv_inner_folds: int = 3
    y_scrambling_permutations: int = 100
    test_size: float = 0.2
    
    # Applicability domain hyperparameters
    ad_leverage_multiplier: float = 3.0  # h* = 3p/n
    ad_similarity_threshold: float = 0.3
    
    def get_gat_config(self) -> Dict[str, Any]:
        """Get GAT-specific configuration."""
        return {
            "hidden_dim": self.gat_hidden_dim,
            "num_layers": self.gat_num_layers,
            "num_heads": self.gat_num_heads,
            "dropout": self.gat_dropout,
            "batch_norm": self.gat_batch_norm,
            "pooling": self.gat_pooling,
            "learning_rate": self.gat_learning_rate,
            "weight_decay": self.gat_weight_decay,
            "batch_size": self.gat_batch_size,
            "epochs": self.gat_epochs,
            "early_stopping_patience": self.gat_early_stopping_patience,
            "scheduler": self.gat_scheduler,
        }
    
    def get_chemberta_config(self) -> Dict[str, Any]:
        """Get ChemBERTa-specific configuration."""
        return {
            "model_name": self.chemberta_model_name,
            "freeze_fraction": self.chemberta_freeze_fraction,
            "hidden_dims": self.chemberta_hidden_dims,
            "dropout": self.chemberta_dropout,
            "learning_rate": self.chemberta_learning_rate,
            "weight_decay": self.chemberta_weight_decay,
            "batch_size": self.chemberta_batch_size,
            "epochs": self.chemberta_epochs,
            "warmup_steps": self.chemberta_warmup_steps,
        }
    
    def get_transfer_config(self) -> Dict[str, Any]:
        """Get transfer learning configuration."""
        return {
            "pretrain_epochs": self.transfer_pretrain_epochs,
            "finetune_epochs": self.transfer_finetune_epochs,
            "finetune_lr": self.transfer_finetune_lr,
            "freeze_layers": self.transfer_freeze_layers,
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return {
            "outer_folds": self.cv_outer_folds,
            "inner_folds": self.cv_inner_folds,
            "y_scrambling_permutations": self.y_scrambling_permutations,
            "test_size": self.test_size,
            "ad_leverage_multiplier": self.ad_leverage_multiplier,
            "ad_similarity_threshold": self.ad_similarity_threshold,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all hyperparameters to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterConfig':
        """Create instance from dictionary."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def save(self, filepath: Path) -> None:
        """Save hyperparameters to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Hyperparameters saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'HyperparameterConfig':
        """Load hyperparameters from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Hyperparameters loaded from {filepath}")
        return cls.from_dict(data)


# Global instances
api_config = APIConfig()
hyperparameter_config = HyperparameterConfig()


def get_api_config() -> APIConfig:
    """Get global API configuration."""
    return api_config


def get_hyperparameter_config() -> HyperparameterConfig:
    """Get global hyperparameter configuration."""
    return hyperparameter_config


def update_hyperparameters(updates: Dict[str, Any]) -> None:
    """Update global hyperparameter configuration."""
    global hyperparameter_config
    for key, value in updates.items():
        if hasattr(hyperparameter_config, key):
            setattr(hyperparameter_config, key, value)
        else:
            logger.warning(f"Unknown hyperparameter: {key}")
