"""
Demo script for Hybrid Model combining GNN and descriptor branches.

This script demonstrates:
1. Creating a hybrid model with TLR4GAT and descriptor branches
2. Preparing hybrid datasets with graphs and descriptors
3. Training the hybrid model with joint optimization
4. Making predictions and extracting embeddings

Requirements: 10.1, 10.2, 10.3
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

if DEPENDENCIES_AVAILABLE:
    from tlr4_binding.models.gat import create_gat_model
    from tlr4_binding.models.hybrid import (
        create_hybrid_model,
        HybridDataset,
        HybridTrainingConfig,
        train_hybrid_model
    )


def create_synthetic_graph_data(num_samples: int = 50) -> list:
    """
    Create synthetic molecular graph data for demonstration.
    
    Args:
        num_samples: Number of synthetic graphs to create
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []
    
    for i in range(num_samples):
        # Random number of atoms (10-30)
        num_atoms = np.random.randint(10, 31)
        
        # Random node features (9 features per atom)
        node_features = torch.randn(num_atoms, 9)
        
        # Create edges (simple ring + random connections)
        edge_list = []
        for j in range(num_atoms):
            # Ring connections
            edge_list.append([j, (j + 1) % num_atoms])
            edge_list.append([(j + 1) % num_atoms, j])
            
            # Random additional connections
            if np.random.random() > 0.5:
                target = np.random.randint(0, num_atoms)
                if target != j:
                    edge_list.append([j, target])
                    edge_list.append([target, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=num_atoms
        )
        
        graphs.append(data)
    
    return graphs


def create_synthetic_descriptors(num_samples: int = 50, descriptor_dim: int = 53) -> np.ndarray:
    """
    Create synthetic molecular descriptors for demonstration.
    
    Args:
        num_samples: Number of samples
        descriptor_dim: Dimension of descriptor features
    
    Returns:
        Numpy array of descriptors [num_samples, descriptor_dim]
    """
    # Generate random descriptors with some structure
    descriptors = np.random.randn(num_samples, descriptor_dim)
    
    # Normalize to reasonable ranges
    descriptors = (descriptors - descriptors.mean(axis=0)) / (descriptors.std(axis=0) + 1e-8)
    
    return descriptors


def create_synthetic_labels(num_samples: int = 50) -> np.ndarray:
    """
    Create synthetic binding affinity labels for demonstration.
    
    Args:
        num_samples: Number of samples
    
    Returns:
        Numpy array of binding affinities
    """
    # Generate labels in typical binding affinity range (-12 to -6 kcal/mol)
    labels = np.random.uniform(-12, -6, num_samples)
    
    return labels


def demo_hybrid_model():
    """
    Demonstrate hybrid model creation, training, and prediction.
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Cannot run demo: required dependencies not available")
        return
    
    logger.info("=" * 80)
    logger.info("Hybrid Model Demo")
    logger.info("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    num_samples = 100
    descriptor_dim = 53
    train_split = 0.8
    
    logger.info(f"\nGenerating synthetic data: {num_samples} samples")
    
    # Create synthetic data
    graphs = create_synthetic_graph_data(num_samples)
    descriptors = create_synthetic_descriptors(num_samples, descriptor_dim)
    labels = create_synthetic_labels(num_samples)
    
    logger.info(f"  - Graphs: {len(graphs)} molecular graphs")
    logger.info(f"  - Descriptors: {descriptors.shape}")
    logger.info(f"  - Labels: {labels.shape}")
    
    # Split into train/val
    split_idx = int(num_samples * train_split)
    train_graphs = graphs[:split_idx]
    train_descriptors = descriptors[:split_idx]
    train_labels = labels[:split_idx]
    
    val_graphs = graphs[split_idx:]
    val_descriptors = descriptors[split_idx:]
    val_labels = labels[split_idx:]
    
    logger.info(f"\nData split:")
    logger.info(f"  - Training: {len(train_graphs)} samples")
    logger.info(f"  - Validation: {len(val_graphs)} samples")
    
    # Step 1: Create GNN model (TLR4GAT)
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Creating GNN model (TLR4GAT)")
    logger.info("=" * 80)
    
    node_features = 9  # From synthetic data
    gnn_model = create_gat_model(
        node_features=node_features,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        dropout=0.2
    )
    
    logger.info(f"GNN model created:")
    logger.info(f"  - Node features: {node_features}")
    logger.info(f"  - Hidden dim: 64")
    logger.info(f"  - Num layers: 3")
    logger.info(f"  - Num heads: 4")
    logger.info(f"  - Embedding dim: {gnn_model.get_embedding_dim()}")
    
    # Step 2: Create Hybrid model
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating Hybrid model")
    logger.info("=" * 80)
    
    hybrid_model = create_hybrid_model(
        gnn_model=gnn_model,
        descriptor_dim=descriptor_dim,
        descriptor_hidden_dim=64,
        fusion_hidden_dim=64,
        dropout=0.2
    )
    
    logger.info(f"Hybrid model created:")
    logger.info(f"  - GNN embedding: {hybrid_model.gnn_embedding_dim}")
    logger.info(f"  - Descriptor input: {descriptor_dim}")
    logger.info(f"  - Descriptor embedding: 64")
    logger.info(f"  - Total embedding: {hybrid_model.gnn_embedding_dim + 64}")
    logger.info(f"  - Fusion hidden: 64")
    
    # Count parameters
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    
    # Step 3: Create datasets
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating hybrid datasets")
    logger.info("=" * 80)
    
    train_dataset = HybridDataset(
        graph_data_list=train_graphs,
        descriptors=train_descriptors,
        labels=train_labels
    )
    
    val_dataset = HybridDataset(
        graph_data_list=val_graphs,
        descriptors=val_descriptors,
        labels=val_labels
    )
    
    logger.info(f"Datasets created:")
    logger.info(f"  - Training dataset: {len(train_dataset)} samples")
    logger.info(f"  - Validation dataset: {len(val_dataset)} samples")
    
    # Step 4: Train hybrid model
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training hybrid model with joint optimization")
    logger.info("=" * 80)
    
    training_config = HybridTrainingConfig(
        lr=1e-3,
        epochs=50,
        patience=10,
        batch_size=16,
        warmup_epochs=0  # No warmup for demo
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Learning rate: {training_config.lr}")
    logger.info(f"  - Epochs: {training_config.epochs}")
    logger.info(f"  - Patience: {training_config.patience}")
    logger.info(f"  - Batch size: {training_config.batch_size}")
    logger.info(f"  - Warmup epochs: {training_config.warmup_epochs}")
    
    logger.info("\nStarting training...")
    
    trained_model, history = train_hybrid_model(
        model=hybrid_model,
        train_data=train_dataset,
        val_data=val_dataset,
        config=training_config,
        device='cpu'  # Use CPU for demo
    )
    
    logger.info("\nTraining complete!")
    logger.info(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  - Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  - Best val loss: {min(history['val_loss']):.4f}")
    logger.info(f"  - Epochs trained: {len(history['train_loss'])}")
    
    # Step 5: Make predictions
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Making predictions")
    logger.info("=" * 80)
    
    # Test on validation set
    from torch.utils.data import DataLoader
    from tlr4_binding.models.hybrid import hybrid_collate_fn
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=hybrid_collate_fn
    )
    
    trained_model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            graphs, descriptors, labels = batch
            pred = trained_model(graphs, descriptors)
            predictions.append(pred.numpy())
            targets.append(labels.numpy())
    
    predictions = np.concatenate(predictions, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    logger.info(f"Validation metrics:")
    logger.info(f"  - R²: {r2:.4f}")
    logger.info(f"  - RMSE: {rmse:.4f}")
    logger.info(f"  - MAE: {mae:.4f}")
    
    # Step 6: Extract embeddings
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Extracting embeddings from both branches")
    logger.info("=" * 80)
    
    # Get embeddings for first validation sample
    sample_graph = val_graphs[0]
    sample_descriptors = torch.tensor(val_descriptors[0:1], dtype=torch.float32)
    
    gnn_emb, desc_emb, fused_emb = trained_model.get_embeddings(
        sample_graph,
        sample_descriptors
    )
    
    logger.info(f"Embeddings for sample 0:")
    logger.info(f"  - GNN embedding shape: {gnn_emb.shape}")
    logger.info(f"  - Descriptor embedding shape: {desc_emb.shape}")
    logger.info(f"  - Fused embedding shape: {fused_emb.shape}")
    logger.info(f"  - GNN embedding norm: {torch.norm(gnn_emb).item():.4f}")
    logger.info(f"  - Descriptor embedding norm: {torch.norm(desc_emb).item():.4f}")
    logger.info(f"  - Fused embedding norm: {torch.norm(fused_emb).item():.4f}")
    
    # Step 7: Test attention extraction
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Extracting attention weights from GNN branch")
    logger.info("=" * 80)
    
    attention_weights = trained_model.get_gnn_attention(sample_graph)
    
    logger.info(f"Attention weights for sample 0:")
    logger.info(f"  - Number of atoms: {len(attention_weights)}")
    logger.info(f"  - Total attention: {sum(attention_weights.values()):.4f}")
    logger.info(f"  - Top 5 atoms by attention:")
    
    sorted_atoms = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)
    for atom_idx, weight in sorted_atoms[:5]:
        logger.info(f"    - Atom {atom_idx}: {weight:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo complete!")
    logger.info("=" * 80)
    logger.info("\nKey features demonstrated:")
    logger.info("  ✓ Dual-branch architecture (GNN + descriptors)")
    logger.info("  ✓ Embedding concatenation and fusion")
    logger.info("  ✓ Joint end-to-end optimization")
    logger.info("  ✓ Prediction and evaluation")
    logger.info("  ✓ Embedding extraction from both branches")
    logger.info("  ✓ Attention weight extraction for interpretability")


if __name__ == "__main__":
    demo_hybrid_model()
