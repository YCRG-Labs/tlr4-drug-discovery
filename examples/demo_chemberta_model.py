"""
Demo: ChemBERTa Transformer Model for TLR4 Binding Affinity Prediction

This example demonstrates how to use the ChemBERTa transformer model for
predicting TLR4 binding affinities from SMILES strings.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import directly from module to avoid package init issues
    sys.path.insert(0, str(Path(__file__).parent.parent / "tlr4_binding" / "models"))
    from chemberta import (
        ChemBERTaPredictor,
        ChemBERTaConfig,
        ChemBERTaTrainer,
        ChemBERTaTrainingConfig,
        SMILESDataset,
        create_chemberta_model,
        train_chemberta_model
    )
    import torch
    from torch.utils.data import DataLoader
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Please install: pip install torch transformers")
    DEPENDENCIES_AVAILABLE = False
    sys.exit(1)


def demo_model_creation():
    """Demonstrate creating a ChemBERTa model."""
    print("=" * 70)
    print("1. Creating ChemBERTa Model")
    print("=" * 70)
    
    # Create model with default settings
    print("\nCreating model with default settings...")
    model = create_chemberta_model(
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        freeze_layers=0.5,  # Freeze first 50% of layers
        hidden_dim_1=256,
        hidden_dim_2=128
    )
    
    print(f"✓ Model created successfully")
    print(f"  - Model: {model.config.model_name}")
    print(f"  - Frozen layers: {model.config.freeze_layers * 100:.0f}%")
    print(f"  - Encoder hidden size: {model.encoder_hidden_size}")
    print(f"  - Regression head: {model.encoder_hidden_size} → {model.config.hidden_dim_1} → {model.config.hidden_dim_2} → {model.config.output_dim}")
    
    return model


def demo_smiles_encoding(model):
    """Demonstrate SMILES encoding."""
    print("\n" + "=" * 70)
    print("2. SMILES Encoding")
    print("=" * 70)
    
    # Sample SMILES strings
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    ]
    
    print("\nEncoding SMILES strings...")
    for smiles in smiles_list:
        print(f"  - {smiles}")
    
    # Encode SMILES
    encoded = model.encode_smiles(smiles_list)
    
    print(f"\n✓ Encoded successfully")
    print(f"  - Input IDs shape: {encoded['input_ids'].shape}")
    print(f"  - Attention mask shape: {encoded['attention_mask'].shape}")
    
    return encoded


def demo_prediction(model):
    """Demonstrate making predictions."""
    print("\n" + "=" * 70)
    print("3. Making Predictions")
    print("=" * 70)
    
    # Sample SMILES and their known activities (example values)
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
    ]
    
    print("\nPredicting binding affinities...")
    model.eval()
    
    # Make predictions
    predictions = model.predict_smiles(smiles_list)
    
    print(f"\n✓ Predictions completed")
    print("\nResults:")
    for smiles, pred in zip(smiles_list, predictions):
        print(f"  {smiles:20s} → {pred[0]:.3f} kcal/mol")
    
    return predictions


def demo_embeddings(model):
    """Demonstrate extracting embeddings."""
    print("\n" + "=" * 70)
    print("4. Extracting Embeddings")
    print("=" * 70)
    
    smiles = "c1ccccc1"  # Benzene
    print(f"\nExtracting embedding for: {smiles}")
    
    # Encode and get embedding
    encoded = model.encode_smiles(smiles)
    embedding = model.get_embedding(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"\n✓ Embedding extracted")
    print(f"  - Shape: {embedding.shape}")
    print(f"  - Dimension: {model.get_embedding_dim()}")
    print(f"  - Mean: {embedding.mean().item():.4f}")
    print(f"  - Std: {embedding.std().item():.4f}")
    
    return embedding


def demo_training_setup():
    """Demonstrate setting up training."""
    print("\n" + "=" * 70)
    print("5. Training Setup")
    print("=" * 70)
    
    # Sample training data
    train_smiles = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    ]
    train_labels = np.array([-5.2, -6.1, -4.8, -7.3])
    
    print("\nTraining data:")
    print(f"  - Number of samples: {len(train_smiles)}")
    print(f"  - Label range: [{train_labels.min():.2f}, {train_labels.max():.2f}] kcal/mol")
    
    # Create model
    model = create_chemberta_model()
    
    # Create training configuration
    config = ChemBERTaTrainingConfig(
        lr=1e-4,  # Lower learning rate for fine-tuning
        batch_size=2,
        epochs=5,
        patience=3
    )
    
    print(f"\nTraining configuration:")
    print(f"  - Learning rate: {config.lr} (fine-tuning range: 1e-4 to 1e-5)")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max epochs: {config.epochs}")
    print(f"  - Early stopping patience: {config.patience}")
    
    # Create dataset
    dataset = SMILESDataset(
        train_smiles,
        train_labels,
        model.tokenizer,
        max_length=512
    )
    
    print(f"\n✓ Dataset created")
    print(f"  - Size: {len(dataset)}")
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    print(f"✓ DataLoader created")
    print(f"  - Batches: {len(train_loader)}")
    
    # Note: Actual training would be done with:
    # trainer = ChemBERTaTrainer(model, config)
    # history = trainer.train(train_loader)
    
    print("\n✓ Training setup complete")
    print("  (Actual training not performed in this demo)")
    
    return model, config, train_loader


def demo_architecture_details(model):
    """Show detailed architecture information."""
    print("\n" + "=" * 70)
    print("6. Model Architecture Details")
    print("=" * 70)
    
    print("\nChemBERTa Architecture:")
    print(f"  - Base model: {model.config.model_name}")
    print(f"  - Encoder layers: {len(model.encoder.encoder.layer) if hasattr(model.encoder, 'encoder') else 'N/A'}")
    print(f"  - Frozen layers: {model.config.freeze_layers * 100:.0f}%")
    print(f"  - Encoder output: {model.encoder_hidden_size}")
    
    print("\nRegression Head:")
    print(f"  - Layer 1: {model.encoder_hidden_size} → {model.config.hidden_dim_1} (ReLU + Dropout)")
    print(f"  - Layer 2: {model.config.hidden_dim_1} → {model.config.hidden_dim_2} (ReLU + Dropout)")
    print(f"  - Layer 3: {model.config.hidden_dim_2} → {model.config.output_dim} (Output)")
    print(f"  - Dropout rate: {model.config.dropout}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  - Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("ChemBERTa Transformer Model Demo")
    print("TLR4 Binding Affinity Prediction")
    print("=" * 70)
    
    try:
        # 1. Create model
        model = demo_model_creation()
        
        # 2. Encode SMILES
        demo_smiles_encoding(model)
        
        # 3. Make predictions
        demo_prediction(model)
        
        # 4. Extract embeddings
        demo_embeddings(model)
        
        # 5. Training setup
        demo_training_setup()
        
        # 6. Architecture details
        demo_architecture_details(model)
        
        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        print("\n✓ All ChemBERTa components demonstrated:")
        print("  - Model creation with pre-trained weights (Requirement 9.1)")
        print("  - Layer freezing (first 50%) (Requirement 9.2)")
        print("  - Regression head (768 → 256 → 128 → 1) (Requirement 9.3)")
        print("  - SMILES encoding with tokenization (Requirement 9.1)")
        print("  - Fine-tuning with lower learning rate (1e-4 to 1e-5) (Requirement 9.4)")
        print("  - Prediction and embedding extraction")
        print("  - Training setup with early stopping")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print("Please install required dependencies:")
        print("  pip install torch transformers")
        sys.exit(1)
    
    sys.exit(main())
