"""
Demo: Transfer Learning for TLR4 Binding Affinity Prediction

This example demonstrates:
1. Pre-training a GAT model on related TLR data (TLR2/7/8/9)
2. Fine-tuning the pre-trained model on TLR4-specific data
3. Comparing transfer learning vs training from scratch

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import torch
    from torch_geometric.data import Data
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

if DEPENDENCIES_AVAILABLE:
    from tlr4_binding.models import (
        TLR4GAT,
        TransferLearningManager,
        TransferLearningConfig,
        create_transfer_learning_manager,
        TRANSFER_LEARNING_AVAILABLE
    )


def create_synthetic_graph_data(
    num_samples: int = 100,
    num_nodes_range: tuple = (10, 30),
    node_features: int = 9,
    seed: int = 42
) -> list:
    """
    Create synthetic molecular graph data for demonstration.
    
    Args:
        num_samples: Number of graph samples to generate
        num_nodes_range: Range of number of nodes per graph
        node_features: Dimension of node features
        seed: Random seed
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data_list = []
    
    for i in range(num_samples):
        # Random number of nodes
        num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1])
        
        # Random node features
        x = torch.randn(num_nodes, node_features)
        
        # Create random edges (ensuring connectivity)
        edge_list = []
        for node in range(1, num_nodes):
            # Connect to previous node (ensures connectivity)
            edge_list.append([node - 1, node])
            edge_list.append([node, node - 1])
        
        # Add some random edges
        num_random_edges = np.random.randint(0, num_nodes)
        for _ in range(num_random_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edge_list.append([src, dst])
                edge_list.append([dst, src])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Random binding affinity (kcal/mol, typical range: -15 to -5)
        y = torch.tensor([np.random.uniform(-15, -5)], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list


def demo_basic_transfer_learning():
    """
    Demonstrate basic transfer learning workflow.
    
    Requirements: 11.1, 11.2, 11.3
    """
    logger.info("=" * 80)
    logger.info("Demo: Basic Transfer Learning Workflow")
    logger.info("=" * 80)
    
    # Create synthetic data
    logger.info("\n1. Creating synthetic data...")
    
    # Related TLR data (TLR2/7/8/9) - larger dataset
    related_train_data = create_synthetic_graph_data(num_samples=500, seed=42)
    related_val_data = create_synthetic_graph_data(num_samples=100, seed=43)
    
    # TLR4 data - smaller dataset
    tlr4_train_data = create_synthetic_graph_data(num_samples=100, seed=44)
    tlr4_val_data = create_synthetic_graph_data(num_samples=20, seed=45)
    tlr4_test_data = create_synthetic_graph_data(num_samples=30, seed=46)
    
    logger.info(f"  Related TLR training samples: {len(related_train_data)}")
    logger.info(f"  Related TLR validation samples: {len(related_val_data)}")
    logger.info(f"  TLR4 training samples: {len(tlr4_train_data)}")
    logger.info(f"  TLR4 validation samples: {len(tlr4_val_data)}")
    logger.info(f"  TLR4 test samples: {len(tlr4_test_data)}")
    
    # Create model
    logger.info("\n2. Creating GAT model...")
    model = TLR4GAT(
        node_features=9,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        dropout=0.2
    )
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create transfer learning manager
    logger.info("\n3. Creating Transfer Learning Manager...")
    config = TransferLearningConfig(
        pretrain_lr=1e-3,
        finetune_lr=1e-4,
        pretrain_epochs=50,
        finetune_epochs=50,
        pretrain_patience=10,
        finetune_patience=10,
        freeze_layers=0,
        batch_size=32
    )
    
    manager = TransferLearningManager(config=config)
    logger.info(f"  Pre-training LR: {config.pretrain_lr}")
    logger.info(f"  Fine-tuning LR: {config.finetune_lr}")
    logger.info(f"  Device: {manager.device}")
    
    # Pre-train on related TLR data
    logger.info("\n4. Pre-training on related TLR data...")
    logger.info("  (This demonstrates learning general TLR binding patterns)")
    
    pretrained_model = manager.pretrain(
        model=model,
        related_data=related_train_data,
        val_data=related_val_data,
        epochs=20,  # Reduced for demo
        verbose=True
    )
    
    logger.info(f"  Pre-training completed in {len(manager.pretrain_history['train_loss'])} epochs")
    logger.info(f"  Final training loss: {manager.pretrain_history['train_loss'][-1]:.4f}")
    if manager.pretrain_history['val_loss']:
        logger.info(f"  Final validation loss: {manager.pretrain_history['val_loss'][-1]:.4f}")
    
    # Fine-tune on TLR4 data
    logger.info("\n5. Fine-tuning on TLR4-specific data...")
    logger.info("  (This adapts the model to TLR4-specific patterns)")
    
    finetuned_model = manager.finetune(
        model=pretrained_model,
        tlr4_data=tlr4_train_data,
        val_data=tlr4_val_data,
        freeze_layers=0,  # No layer freezing for this demo
        epochs=20,  # Reduced for demo
        verbose=True
    )
    
    logger.info(f"  Fine-tuning completed in {len(manager.finetune_history['train_loss'])} epochs")
    logger.info(f"  Final training loss: {manager.finetune_history['train_loss'][-1]:.4f}")
    if manager.finetune_history['val_loss']:
        logger.info(f"  Final validation loss: {manager.finetune_history['val_loss'][-1]:.4f}")
    
    # Evaluate on test data
    logger.info("\n6. Evaluating on TLR4 test data...")
    from torch_geometric.loader import DataLoader
    
    test_loader = DataLoader(tlr4_test_data, batch_size=32, shuffle=False)
    test_metrics = manager._evaluate_model(finetuned_model, test_loader)
    
    logger.info("  Test Metrics:")
    logger.info(f"    R²: {test_metrics['r2']:.4f}")
    logger.info(f"    RMSE: {test_metrics['rmse']:.4f} kcal/mol")
    logger.info(f"    MAE: {test_metrics['mae']:.4f} kcal/mol")
    
    logger.info("\n✓ Basic transfer learning workflow completed successfully!")
    
    return manager, finetuned_model, test_metrics


def demo_layer_freezing():
    """
    Demonstrate fine-tuning with layer freezing.
    
    Requirements: 11.2
    """
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Fine-tuning with Layer Freezing")
    logger.info("=" * 80)
    
    # Create synthetic data
    logger.info("\n1. Creating synthetic data...")
    related_train_data = create_synthetic_graph_data(num_samples=300, seed=50)
    tlr4_train_data = create_synthetic_graph_data(num_samples=80, seed=51)
    tlr4_val_data = create_synthetic_graph_data(num_samples=20, seed=52)
    
    # Create and pre-train model
    logger.info("\n2. Pre-training model...")
    model = TLR4GAT(node_features=9, hidden_dim=64, num_layers=3, num_heads=4)
    
    manager = create_transfer_learning_manager(
        pretrain_lr=1e-3,
        finetune_lr=1e-4,
        freeze_layers=2  # Freeze first 2 layers
    )
    
    pretrained_model = manager.pretrain(
        model=model,
        related_data=related_train_data,
        epochs=15,
        verbose=False
    )
    
    logger.info(f"  Pre-training completed: {len(manager.pretrain_history['train_loss'])} epochs")
    
    # Fine-tune with layer freezing
    logger.info("\n3. Fine-tuning with first 2 layers frozen...")
    logger.info("  (Frozen layers preserve general TLR patterns)")
    logger.info("  (Unfrozen layers adapt to TLR4-specific patterns)")
    
    finetuned_model = manager.finetune(
        model=pretrained_model,
        tlr4_data=tlr4_train_data,
        val_data=tlr4_val_data,
        freeze_layers=2,  # Freeze first 2 GAT layers
        epochs=15,
        verbose=True
    )
    
    logger.info(f"  Fine-tuning completed: {len(manager.finetune_history['train_loss'])} epochs")
    logger.info(f"  Final training loss: {manager.finetune_history['train_loss'][-1]:.4f}")
    
    logger.info("\n✓ Layer freezing demonstration completed!")
    
    return manager, finetuned_model


def demo_transfer_vs_scratch():
    """
    Demonstrate comparison of transfer learning vs training from scratch.
    
    Requirements: 11.4
    """
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Transfer Learning vs Training from Scratch")
    logger.info("=" * 80)
    
    # Create synthetic data
    logger.info("\n1. Creating synthetic data...")
    
    # Related TLR data for pre-training
    related_train_data = create_synthetic_graph_data(num_samples=400, seed=60)
    
    # TLR4 data
    tlr4_train_data = create_synthetic_graph_data(num_samples=80, seed=61)
    tlr4_val_data = create_synthetic_graph_data(num_samples=20, seed=62)
    tlr4_test_data = create_synthetic_graph_data(num_samples=30, seed=63)
    
    logger.info(f"  Related TLR samples: {len(related_train_data)}")
    logger.info(f"  TLR4 training samples: {len(tlr4_train_data)}")
    logger.info(f"  TLR4 validation samples: {len(tlr4_val_data)}")
    logger.info(f"  TLR4 test samples: {len(tlr4_test_data)}")
    
    # Create transfer learning manager
    logger.info("\n2. Setting up comparison...")
    config = TransferLearningConfig(
        pretrain_lr=1e-3,
        finetune_lr=1e-4,
        pretrain_epochs=20,
        finetune_epochs=20,
        pretrain_patience=10,
        finetune_patience=10,
        batch_size=32
    )
    
    manager = TransferLearningManager(config=config)
    
    # Run comparison
    logger.info("\n3. Running comparison (this may take a few minutes)...")
    logger.info("  Training two models:")
    logger.info("    A) Transfer Learning: Pre-train on related TLRs → Fine-tune on TLR4")
    logger.info("    B) From Scratch: Train directly on TLR4 data only")
    
    results = manager.compare_transfer_vs_scratch(
        model_class=TLR4GAT,
        model_kwargs={
            'node_features': 9,
            'hidden_dim': 64,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.2
        },
        related_data=related_train_data,
        tlr4_train_data=tlr4_train_data,
        tlr4_val_data=tlr4_val_data,
        tlr4_test_data=tlr4_test_data,
        verbose=False
    )
    
    # Display results
    logger.info("\n4. Comparison Results:")
    logger.info("=" * 60)
    
    if 'transfer_metrics' in results and 'scratch_metrics' in results:
        transfer_metrics = results['transfer_metrics']
        scratch_metrics = results['scratch_metrics']
        
        logger.info("\nTransfer Learning Performance:")
        logger.info(f"  R²:   {transfer_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {transfer_metrics['rmse']:.4f} kcal/mol")
        logger.info(f"  MAE:  {transfer_metrics['mae']:.4f} kcal/mol")
        
        logger.info("\nTraining from Scratch Performance:")
        logger.info(f"  R²:   {scratch_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {scratch_metrics['rmse']:.4f} kcal/mol")
        logger.info(f"  MAE:  {scratch_metrics['mae']:.4f} kcal/mol")
        
        if 'improvement' in results:
            improvement = results['improvement']
            logger.info("\nImprovement from Transfer Learning:")
            logger.info(f"  R² improvement: {improvement['r2_percent']:+.2f}%")
            logger.info(f"  R² absolute: {improvement['r2_absolute']:+.4f}")
            
            if improvement['r2_percent'] > 0:
                logger.info("\n✓ Transfer learning shows improvement over training from scratch!")
            else:
                logger.info("\n⚠ Transfer learning did not improve over training from scratch")
                logger.info("  (This can happen with synthetic data or insufficient pre-training)")
    
    # Training efficiency comparison
    transfer_epochs = (len(results['transfer_history']['pretrain']['train_loss']) +
                      len(results['transfer_history']['finetune']['train_loss']))
    scratch_epochs = len(results['scratch_history']['train_loss'])
    
    logger.info("\nTraining Efficiency:")
    logger.info(f"  Transfer learning total epochs: {transfer_epochs}")
    logger.info(f"  From scratch total epochs: {scratch_epochs}")
    
    logger.info("\n✓ Comparison completed successfully!")
    
    return results


def main():
    """Run all transfer learning demonstrations."""
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies (PyTorch, PyTorch Geometric) not available")
        logger.error("Please install: pip install torch torch-geometric")
        return
    
    if not TRANSFER_LEARNING_AVAILABLE:
        logger.error("Transfer learning module not available")
        return
    
    logger.info("Transfer Learning Demonstrations for TLR4 Binding Prediction")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Basic transfer learning workflow
        manager1, model1, metrics1 = demo_basic_transfer_learning()
        
        # Demo 2: Layer freezing
        manager2, model2 = demo_layer_freezing()
        
        # Demo 3: Transfer vs scratch comparison
        results = demo_transfer_vs_scratch()
        
        logger.info("\n" + "=" * 80)
        logger.info("All demonstrations completed successfully!")
        logger.info("=" * 80)
        
        logger.info("\nKey Takeaways:")
        logger.info("1. Transfer learning leverages related TLR data to improve TLR4 predictions")
        logger.info("2. Pre-training learns general TLR binding patterns")
        logger.info("3. Fine-tuning adapts the model to TLR4-specific patterns")
        logger.info("4. Layer freezing can preserve general patterns while adapting to specifics")
        logger.info("5. Transfer learning is especially beneficial with limited TLR4 data")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
