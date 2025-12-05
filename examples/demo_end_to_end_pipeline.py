#!/usr/bin/env python3
"""
Demo script for end-to-end TLR4 binding affinity prediction pipeline.

This script demonstrates the complete pipeline integration including:
- Data collection and quality control
- Feature engineering
- Model training
- Validation
- Interpretability analysis
- Benchmarking and reporting
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tlr4_binding.pipeline import TLR4Pipeline, PipelineConfig, create_pipeline


def demo_minimal_pipeline():
    """Demonstrate minimal pipeline with synthetic data."""
    print("="*80)
    print("DEMO 1: Minimal Pipeline with Synthetic Data")
    print("="*80)
    
    # Create minimal configuration
    config = PipelineConfig(
        collect_data=False,  # Skip data collection for demo
        collect_transfer_data=False,
        calculate_3d_descriptors=False,  # Skip for speed
        calculate_electrostatic=False,
        generate_graphs=False,
        train_traditional_ml=True,
        train_gnn=False,
        train_transformer=False,
        train_hybrid=False,
        train_transfer_learning=False,
        train_multi_task=False,
        nested_cv_outer_folds=3,
        nested_cv_inner_folds=2,
        y_scrambling_iterations=10,
        run_scaffold_validation=False,
        calculate_applicability_domain=False,
        generate_attention_viz=False,
        generate_shap_analysis=False,
        output_dir="./results/demo_minimal_pipeline",
        generate_report=True
    )
    
    # Create pipeline
    pipeline = create_pipeline(config)
    
    print("\nPipeline configuration:")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Models to train: Traditional ML only")
    print(f"  Validation: Minimal (3x2 nested CV, 10 Y-scrambling iterations)")
    
    # Create synthetic data for demo
    print("\nCreating synthetic TLR4 binding data...")
    n_compounds = 100
    smiles_list = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "CCN",  # Ethylamine
    ] * 20  # Repeat to get 100 compounds
    
    binding_affinities = np.random.uniform(-12, -5, n_compounds)
    
    synthetic_data = pd.DataFrame({
        'smiles': smiles_list,
        'canonical_smiles': smiles_list,
        'binding_affinity': binding_affinities,
        'functional_class': np.random.choice(['agonist', 'antagonist', 'unknown'], n_compounds)
    })
    
    # Save synthetic data
    os.makedirs(config.output_dir, exist_ok=True)
    synthetic_data.to_csv(f"{config.output_dir}/tlr4_dataset.csv", index=False)
    
    print(f"Created synthetic dataset: {len(synthetic_data)} compounds")
    print(f"Binding affinity range: {binding_affinities.min():.2f} to {binding_affinities.max():.2f} kcal/mol")
    
    # Note: Full pipeline run would require all components to be properly implemented
    print("\nNote: This is a configuration demo. Full pipeline execution requires")
    print("all components to be properly implemented and integrated.")
    
    return pipeline


def demo_feature_engineering_pipeline():
    """Demonstrate pipeline with focus on feature engineering."""
    print("\n" + "="*80)
    print("DEMO 2: Feature Engineering Pipeline")
    print("="*80)
    
    config = PipelineConfig(
        collect_data=False,
        calculate_3d_descriptors=True,
        calculate_electrostatic=True,
        generate_graphs=True,
        train_traditional_ml=False,
        train_gnn=False,
        train_transformer=False,
        train_hybrid=False,
        train_transfer_learning=False,
        train_multi_task=False,
        output_dir="./results/demo_feature_engineering",
        generate_report=False
    )
    
    pipeline = create_pipeline(config)
    
    print("\nPipeline configuration:")
    print(f"  3D descriptors: {config.calculate_3d_descriptors}")
    print(f"  Electrostatic: {config.calculate_electrostatic}")
    print(f"  Graph generation: {config.generate_graphs}")
    print(f"  Model training: Disabled (focus on features)")
    
    print("\nThis configuration focuses on feature engineering:")
    print("  - 3D conformer generation and shape descriptors")
    print("  - Electrostatic properties (charges, dipole)")
    print("  - Graph representations for GNN models")
    
    return pipeline


def demo_validation_pipeline():
    """Demonstrate pipeline with comprehensive validation."""
    print("\n" + "="*80)
    print("DEMO 3: Comprehensive Validation Pipeline")
    print("="*80)
    
    config = PipelineConfig(
        collect_data=False,
        calculate_3d_descriptors=False,
        calculate_electrostatic=False,
        generate_graphs=False,
        train_traditional_ml=True,
        train_gnn=False,
        train_transformer=False,
        train_hybrid=False,
        train_transfer_learning=False,
        train_multi_task=False,
        external_test_size=0.2,
        nested_cv_outer_folds=5,
        nested_cv_inner_folds=3,
        y_scrambling_iterations=100,
        run_scaffold_validation=True,
        calculate_applicability_domain=True,
        output_dir="./results/demo_validation",
        generate_report=True
    )
    
    pipeline = create_pipeline(config)
    
    print("\nPipeline configuration:")
    print(f"  External test set: {config.external_test_size * 100}%")
    print(f"  Nested CV: {config.nested_cv_outer_folds}x{config.nested_cv_inner_folds}")
    print(f"  Y-scrambling: {config.y_scrambling_iterations} iterations")
    print(f"  Scaffold validation: {config.run_scaffold_validation}")
    print(f"  Applicability domain: {config.calculate_applicability_domain}")
    
    print("\nThis configuration provides rigorous validation:")
    print("  - External test set for unbiased evaluation")
    print("  - Nested CV to avoid hyperparameter leakage")
    print("  - Y-scrambling to assess model robustness")
    print("  - Scaffold validation for generalization assessment")
    print("  - Applicability domain for prediction reliability")
    
    return pipeline


def demo_full_pipeline():
    """Demonstrate full pipeline with all components."""
    print("\n" + "="*80)
    print("DEMO 4: Full Pipeline (All Components)")
    print("="*80)
    
    config = PipelineConfig(
        collect_data=True,
        collect_transfer_data=True,
        calculate_3d_descriptors=True,
        calculate_electrostatic=True,
        generate_graphs=True,
        train_traditional_ml=True,
        train_gnn=True,
        train_transformer=True,
        train_hybrid=True,
        train_transfer_learning=True,
        train_multi_task=True,
        external_test_size=0.2,
        nested_cv_outer_folds=5,
        nested_cv_inner_folds=3,
        y_scrambling_iterations=100,
        run_scaffold_validation=True,
        calculate_applicability_domain=True,
        generate_attention_viz=True,
        generate_shap_analysis=True,
        top_compounds_for_viz=10,
        output_dir="./results/demo_full_pipeline",
        save_models=True,
        generate_report=True
    )
    
    pipeline = create_pipeline(config)
    
    print("\nPipeline configuration: FULL")
    print("\nData Collection:")
    print(f"  - TLR4 targets: {config.chembl_targets}")
    print(f"  - PubChem assays: {config.pubchem_assays}")
    print(f"  - Transfer learning targets: {config.related_tlr_targets}")
    
    print("\nFeature Engineering:")
    print("  - 2D molecular descriptors")
    print("  - 3D conformational descriptors")
    print("  - Electrostatic properties")
    print("  - Graph representations")
    
    print("\nModels:")
    print("  - Traditional ML ensemble")
    print("  - Graph Attention Network (GAT)")
    print("  - ChemBERTa transformer")
    print("  - Hybrid (GNN + descriptors)")
    print("  - Transfer learning (pre-trained on related TLRs)")
    print("  - Multi-task (affinity + function)")
    
    print("\nValidation:")
    print(f"  - External test set: {config.external_test_size * 100}%")
    print(f"  - Nested CV: {config.nested_cv_outer_folds}x{config.nested_cv_inner_folds}")
    print(f"  - Y-scrambling: {config.y_scrambling_iterations} iterations")
    print(f"  - Scaffold validation: Yes")
    print(f"  - Applicability domain: Yes")
    
    print("\nInterpretability:")
    print(f"  - Attention visualization: Yes (top {config.top_compounds_for_viz} compounds)")
    print(f"  - SHAP analysis: Yes")
    
    print("\nOutput:")
    print(f"  - Directory: {config.output_dir}")
    print(f"  - Save models: {config.save_models}")
    print(f"  - Generate report: {config.generate_report}")
    
    print("\nNote: This is the complete pipeline configuration.")
    print("Running this would execute all methodology enhancements.")
    
    return pipeline


def demo_custom_pipeline():
    """Demonstrate custom pipeline configuration."""
    print("\n" + "="*80)
    print("DEMO 5: Custom Pipeline Configuration")
    print("="*80)
    
    # Custom configuration for specific research question
    config = PipelineConfig(
        collect_data=False,
        calculate_3d_descriptors=True,
        calculate_electrostatic=True,
        generate_graphs=True,
        train_traditional_ml=True,
        train_gnn=True,
        train_transformer=False,  # Skip transformer
        train_hybrid=True,
        train_transfer_learning=False,  # Skip transfer learning
        train_multi_task=True,
        nested_cv_outer_folds=5,
        nested_cv_inner_folds=3,
        y_scrambling_iterations=100,
        run_scaffold_validation=True,
        calculate_applicability_domain=True,
        generate_attention_viz=True,
        generate_shap_analysis=True,
        output_dir="./results/demo_custom_pipeline",
        generate_report=True
    )
    
    pipeline = create_pipeline(config)
    
    print("\nCustom configuration example:")
    print("Research question: Compare GNN vs hybrid models with multi-task learning")
    print("\nEnabled components:")
    print("  - Traditional ML (baseline)")
    print("  - GNN (Graph Attention Network)")
    print("  - Hybrid (GNN + descriptors)")
    print("  - Multi-task learning")
    print("  - Full validation suite")
    print("  - Interpretability analysis")
    
    print("\nDisabled components:")
    print("  - Transformer models (not needed for this comparison)")
    print("  - Transfer learning (focusing on TLR4 only)")
    
    print("\nThis demonstrates how to customize the pipeline for specific research needs.")
    
    return pipeline


def main():
    """Run all pipeline demonstrations."""
    print("TLR4 Binding Affinity Prediction - End-to-End Pipeline Demos")
    print("="*80)
    
    try:
        # Demo 1: Minimal pipeline
        demo_minimal_pipeline()
        
        # Demo 2: Feature engineering focus
        demo_feature_engineering_pipeline()
        
        # Demo 3: Validation focus
        demo_validation_pipeline()
        
        # Demo 4: Full pipeline
        demo_full_pipeline()
        
        # Demo 5: Custom configuration
        demo_custom_pipeline()
        
        print("\n" + "="*80)
        print("All demonstrations completed!")
        print("="*80)
        
        print("\nNext steps:")
        print("1. Review the pipeline configurations above")
        print("2. Create your own configuration file (JSON)")
        print("3. Run the pipeline: python run_pipeline.py --config your_config.json")
        print("4. Or use quick mode: python run_pipeline.py --quick")
        print("5. Or use full mode: python run_pipeline.py --full")
        
        print("\nFor configuration template:")
        print("  python run_pipeline.py --save-template")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
