"""
TLR4 Binding Affinity Prediction System

A machine learning pipeline for predicting binding free energies of small molecules
to the TLR4 receptor using molecular structure analysis and feature extraction.
"""

__version__ = "0.1.0"
__author__ = "Brandon Yee & Maximilian Rutowski"
__email__ = "b.yee@ycrg-labs.org"

# Legacy imports (for backward compatibility)
from .molecular_analysis.features import MolecularFeatures as LegacyMolecularFeatures
from .molecular_analysis.features import BindingData, PredictionResult
from .molecular_analysis.extractor import MolecularFeatureExtractor
from .data_processing.preprocessor import DataPreprocessor

# Lazy import to avoid dependency issues during testing
try:
    from .ml_components.trainer import MLModelTrainer
except (ImportError, AttributeError) as e:
    import warnings
    warnings.warn(f"MLModelTrainer import failed: {e}. Some functionality may be limited.")
    MLModelTrainer = None

# New data models for methodology enhancement
from .data.models import CompoundRecord, FunctionalClass, AffinitySource
from .features.models import MolecularFeatures
from .models.models import ModelPrediction
from .validation.models import ValidationResult

# Configuration
from .config import (
    Config,
    APIConfig,
    HyperparameterConfig,
    get_api_config,
    get_hyperparameter_config,
)

__all__ = [
    # Legacy exports
    "LegacyMolecularFeatures",
    "BindingData", 
    "PredictionResult",
    "MolecularFeatureExtractor",
    "DataPreprocessor",
    "MLModelTrainer",
    # New data models
    "CompoundRecord",
    "FunctionalClass",
    "AffinitySource",
    "MolecularFeatures",
    "ModelPrediction",
    "ValidationResult",
    # Configuration
    "Config",
    "APIConfig",
    "HyperparameterConfig",
    "get_api_config",
    "get_hyperparameter_config",
]
