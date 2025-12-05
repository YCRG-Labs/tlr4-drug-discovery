"""
Demo script for the InterpretabilityAnalyzer module.

This script demonstrates:
1. Extracting attention weights from a GAT model
2. Visualizing attention on molecular structures
3. Calculating SHAP values for traditional models
4. Plotting feature importance

Requirements: 18.1, 18.2, 18.3, 18.4
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tlr4_binding.interpretability import (
    InterpretabilityAnalyzer,
    create_interpretability_analyzer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_attention_extraction():
    """Demonstrate attention weight extraction from GAT model."""
    logger.info("=" * 60)
    logger.info("Demo: Attention Weight Extraction")
    logger.info("=" * 60)
    
    try:
        from tlr4_binding.models import TLR4GAT, GAT_AVAILABLE
        from tlr4_binding.features import MolecularGraphGenerator
        import torch
        
        if not GAT_AVAILABLE:
            logger.warning("GAT not available - skipping attention extraction demo")
            return
        
        # Create analyzer
        analyzer = create_interpretability_analyzer()
        
        # Create graph generator
        graph_gen = MolecularGraphGenerator()
        node_feature_dim = graph_gen.get_node_feature_dim()
        
        # Create a simple GAT model with correct feature dimension
        model = TLR4GAT(
            node_features=node_feature_dim,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            dropout=0.2
        )
        model.eval()
        
        # Example molecule: aspirin
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        
        # Generate graph
        graph_data = graph_gen.mol_to_graph(smiles)
        
        # Extract attention weights
        attention = analyzer.extract_attention(model, graph_data, smiles)
        
        logger.info(f"Extracted attention for {len(attention)} atoms")
        
        # Get top attention atoms
        top_atoms = analyzer.get_top_attention_atoms(attention, n=5)
        logger.info("Top 5 atoms by attention:")
        for atom_idx, weight in top_atoms:
            logger.info(f"  Atom {atom_idx}: {weight:.4f}")
        
        # Visualize attention
        output_path = Path("results/figures/attention_aspirin.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img = analyzer.visualize_attention(
            smiles=smiles,
            attention=attention,
            output_path=output_path,
            title="Attention Weights - Aspirin",
            highlight_threshold=0.1
        )
        
        if img:
            logger.info(f"Attention visualization saved to {output_path}")
        
        logger.info("✓ Attention extraction demo completed successfully")
        
    except ImportError as e:
        logger.warning(f"Required dependencies not available: {e}")
    except Exception as e:
        logger.error(f"Attention extraction demo failed: {e}", exc_info=True)


def demo_shap_analysis():
    """Demonstrate SHAP analysis for traditional models."""
    logger.info("=" * 60)
    logger.info("Demo: SHAP Analysis")
    logger.info("=" * 60)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Create analyzer
        analyzer = create_interpretability_analyzer()
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        
        # Create target with known feature importance
        # Features 0, 5, 10 are important
        y = 2.0 * X[:, 0] + 1.5 * X[:, 5] - 1.0 * X[:, 10] + 0.5 * np.random.randn(n_samples)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Create feature names
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Calculate SHAP values
        logger.info("Computing SHAP values...")
        shap_values = analyzer.calculate_shap(
            model=model,
            X=X[:20],  # Use subset for speed
            feature_names=feature_names,
            background_samples=50,
            max_evals=500
        )
        
        logger.info(f"SHAP values shape: {shap_values.shape}")
        
        # Get top features
        top_features = analyzer.get_top_features(shap_values, feature_names, n=10)
        logger.info("Top 10 features by SHAP importance:")
        for feature_name, importance in top_features:
            logger.info(f"  {feature_name}: {importance:.4f}")
        
        # Plot feature importance
        output_path = Path("results/figures/shap_feature_importance.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig = analyzer.plot_feature_importance(
            shap_values=shap_values,
            feature_names=feature_names,
            output_path=output_path,
            title="Feature Importance (SHAP)",
            top_n=15
        )
        
        if fig:
            logger.info(f"Feature importance plot saved to {output_path}")
        
        # Create SHAP summary plot
        summary_path = Path("results/figures/shap_summary.png")
        analyzer.create_shap_summary_plot(
            shap_values=shap_values,
            X=X[:20],
            feature_names=feature_names,
            output_path=summary_path,
            plot_type="dot",
            max_display=15
        )
        logger.info(f"SHAP summary plot saved to {summary_path}")
        
        logger.info("✓ SHAP analysis demo completed successfully")
        
    except ImportError as e:
        logger.warning(f"Required dependencies not available: {e}")
    except Exception as e:
        logger.error(f"SHAP analysis demo failed: {e}", exc_info=True)


def demo_combined_interpretability():
    """Demonstrate combined interpretability analysis."""
    logger.info("=" * 60)
    logger.info("Demo: Combined Interpretability Analysis")
    logger.info("=" * 60)
    
    try:
        # Create analyzer
        analyzer = create_interpretability_analyzer()
        
        logger.info("InterpretabilityAnalyzer created successfully")
        logger.info("Available methods:")
        logger.info("  - extract_attention(): Extract attention from GNN models")
        logger.info("  - visualize_attention(): Visualize attention on molecules")
        logger.info("  - calculate_shap(): Calculate SHAP values")
        logger.info("  - plot_feature_importance(): Plot feature importance")
        logger.info("  - create_shap_summary_plot(): Create SHAP summary plots")
        logger.info("  - get_top_attention_atoms(): Get top attention atoms")
        logger.info("  - get_top_features(): Get top SHAP features")
        
        logger.info("✓ Combined interpretability demo completed successfully")
        
    except Exception as e:
        logger.error(f"Combined interpretability demo failed: {e}", exc_info=True)


def main():
    """Run all interpretability demos."""
    logger.info("Starting InterpretabilityAnalyzer Demo")
    logger.info("=" * 60)
    
    # Run demos
    demo_attention_extraction()
    print()
    
    demo_shap_analysis()
    print()
    
    demo_combined_interpretability()
    print()
    
    logger.info("=" * 60)
    logger.info("All demos completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
