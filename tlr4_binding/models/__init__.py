"""
Machine learning models module for TLR4 binding prediction.

This module provides components for:
- Graph Neural Networks (GAT, AttentiveFP)
- Transformer models (ChemBERTa)
- Hybrid architectures
- Transfer learning and multi-task learning
"""

from .models import ModelPrediction

# Import GAT components with availability check
try:
    from .gat import (
        TLR4GAT,
        GATConfig,
        GATTrainer,
        TrainingConfig,
        create_gat_model,
        train_gat_model,
    )
    GAT_AVAILABLE = True
except ImportError:
    GAT_AVAILABLE = False
    TLR4GAT = None
    GATConfig = None
    GATTrainer = None
    TrainingConfig = None
    create_gat_model = None
    train_gat_model = None

# Import ChemBERTa components with availability check
try:
    from .chemberta import (
        ChemBERTaPredictor,
        ChemBERTaConfig,
        ChemBERTaTrainer,
        ChemBERTaTrainingConfig,
        SMILESDataset,
        create_chemberta_model,
        train_chemberta_model,
    )
    CHEMBERTA_AVAILABLE = True
except ImportError:
    CHEMBERTA_AVAILABLE = False
    ChemBERTaPredictor = None
    ChemBERTaConfig = None
    ChemBERTaTrainer = None
    ChemBERTaTrainingConfig = None
    SMILESDataset = None
    create_chemberta_model = None
    train_chemberta_model = None

# Import Hybrid model components with availability check
try:
    from .hybrid import (
        HybridModel,
        HybridConfig,
        HybridTrainer,
        HybridTrainingConfig,
        HybridDataset,
        hybrid_collate_fn,
        create_hybrid_model,
        train_hybrid_model,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    HybridModel = None
    HybridConfig = None
    HybridTrainer = None
    HybridTrainingConfig = None
    HybridDataset = None
    hybrid_collate_fn = None
    create_hybrid_model = None
    train_hybrid_model = None

# Import Transfer Learning components with availability check
try:
    from .transfer_learning import (
        TransferLearningManager,
        TransferLearningConfig,
        create_transfer_learning_manager,
    )
    TRANSFER_LEARNING_AVAILABLE = True
except ImportError:
    TRANSFER_LEARNING_AVAILABLE = False
    TransferLearningManager = None
    TransferLearningConfig = None
    create_transfer_learning_manager = None

# Import Multi-Task components with availability check
try:
    from .multi_task import (
        MultiTaskModel,
        MultiTaskConfig,
        MultiTaskTrainer,
        MultiTaskTrainingConfig,
        MultiTaskDataset,
        create_multi_task_model,
        train_multi_task_model,
    )
    MULTI_TASK_AVAILABLE = True
except ImportError:
    MULTI_TASK_AVAILABLE = False
    MultiTaskModel = None
    MultiTaskConfig = None
    MultiTaskTrainer = None
    MultiTaskTrainingConfig = None
    MultiTaskDataset = None
    create_multi_task_model = None
    train_multi_task_model = None

__all__ = [
    "ModelPrediction",
    "TLR4GAT",
    "GATConfig",
    "GATTrainer",
    "TrainingConfig",
    "create_gat_model",
    "train_gat_model",
    "GAT_AVAILABLE",
    "ChemBERTaPredictor",
    "ChemBERTaConfig",
    "ChemBERTaTrainer",
    "ChemBERTaTrainingConfig",
    "SMILESDataset",
    "create_chemberta_model",
    "train_chemberta_model",
    "CHEMBERTA_AVAILABLE",
    "HybridModel",
    "HybridConfig",
    "HybridTrainer",
    "HybridTrainingConfig",
    "HybridDataset",
    "hybrid_collate_fn",
    "create_hybrid_model",
    "train_hybrid_model",
    "HYBRID_AVAILABLE",
    "TransferLearningManager",
    "TransferLearningConfig",
    "create_transfer_learning_manager",
    "TRANSFER_LEARNING_AVAILABLE",
    "MultiTaskModel",
    "MultiTaskConfig",
    "MultiTaskTrainer",
    "MultiTaskTrainingConfig",
    "MultiTaskDataset",
    "create_multi_task_model",
    "train_multi_task_model",
    "MULTI_TASK_AVAILABLE",
]
