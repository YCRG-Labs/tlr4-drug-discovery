#!/usr/bin/env python3
"""
Demo script for interpretability output generation.

This script demonstrates comprehensive interpretability output generation including:
- Attention weight visualization for GNN models
- SHAP analysis and feature importance
- Batch processing for multiple compounds
- Report generation
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tlr4_binding.interpretability.output_generator import (
    InterpretabilityOutputGenerator,
    InterpretabilityConfig,
    create_output_generator
)


def create_synthetic_data(n_samples=200, n_features=50):
    """Create synthetic molecular data for demonstration."""
    print("Creating synthetic TLR4 binding data...")
    
    # Generate features with meaningful names
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Generate binding affinities
    y = (
        -0.5 * X[:, 0] +  # molecular_weight effect
        -0.3 * X[:, 1] +  # logP effect
        -0.2 * X[:, 2] +  # TPSA effect
        -0.15 * X[:, 5] +  # hydrogen bond donors
        -0.1 * X[:, 10] +  # aromatic rings
        np.random.normal(0, 1, n_samples)
    )
    
    # Create feature names
    feature_names = [
        'molecular_weight', 'logP', 'TPSA', 'rotatable_bonds', 'hbd', 'hba',
        'radius_of_gyration', 'molecular_volume', 'surface_area', 'asphericity',
        'aromatic_rings', 'aliphatic_rings', 'formal_charge', 'polar_surface_area',
        'molecular_flexibility', 'hydrogen_bond_donors', 'hydrogen_bond_acceptors',
        'topological_polar_surface_area', 'molecular_connectivity', 'electrotopological_state'
    ]
    
    # Extend feature names if needed
    if n_features > len(feature_names):
        feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
    
    feature_names = feature_names[:n_features]
    
    # Generate SMILES
    smiles_templates = [
        "CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN",
        "CCCC", "c1ccc(O)cc1", "CC(C)C", "CCCN", "c1ccncc1"
    ]
    smiles_list = [smiles_templates[i % len(smiles_templates)] + f"_{i}" 
                   for i in range(n_samples)]
    
    print(f"Generated {n_samples} compounds with {n_features} features")
    print(f"Binding affinity range: {y.min():.2f} to {y.max():.2f} kcal/mol")
    
    return X, y, feature_names, smiles_list


def demo_shap_analysis():
    """Demonstrate SHAP analysis and feature importance."""
    print("="*80)
    print("DEMO 1: SHAP Analysis and Feature Importance")
    print("="*80)
    
    # Create data
    X, y, feature_names, smiles_list = create_synthetic_data(n_samples=200, n_features=30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    print(f"Model R²: {model.score(X_test, y_test):.4f}")
    
    # Configure output generator
    config = InterpretabilityConfig(
        generate_attention_viz=False,  # No GNN model
        generate_shap_analysis=True,
        shap_top_features=20,
        shap_sample_size=100,  # Use subset for speed
        generate_feature_importance=True,
        feature_importance_top_k=20,
        output_dir="./results/demo_shap_analysis",
        save_summary_plots=True,
        generate_report=True
    )
    
    # Create generator
    generator = create_output_generator(config)
    
    # Generate SHAP analysis
    print("\nGenerating SHAP analysis...")
    shap_values, importance_df = generator.generate_shap_analysis(
        model=model,
        X=X_test,
        feature_names=feature_names
    )
    
    # Generate feature importance
    print("\nGenerating feature importance...")
    feature_importance = generator.generate_feature_importance_plot(
        model=model,
        feature_names=feature_names
    )
    
    # Generate report
    print("\nGenerating report...")
    report = generator.generate_report()
    
    print("\n" + "="*80)
    print("SHAP Analysis Complete")
    print("="*80)
    print(f"\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\nOutputs saved to: {config.output_dir}")
    print("  - shap_analysis/shap_feature_importance.csv")
    print("  - shap_analysis/shap_summary.png")
    print("  - feature_importance/feature_importance.csv")
    print("  - feature_importance/feature_importance.png")
    print("  - interpretability_report.md")
    
    return generator.outputs


def demo_comprehensive_outputs():
    """Demonstrate comprehensive interpretability output generation."""
    print("\n" + "="*80)
    print("DEMO 2: Comprehensive Interpretability Outputs")
    print("="*80)
    
    # Create data
    X, y, feature_names, smiles_list = create_synthetic_data(n_samples=300, n_features=50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get corresponding SMILES for test set
    train_idx = np.arange(len(X_train))
    test_idx = np.arange(len(X_train), len(X_train) + len(X_test))
    smiles_test = [smiles_list[i] for i in test_idx]
    
    # Train model
    print("\nTraining model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    print(f"Model R²: {model.score(X_test, y_test):.4f}")
    print(f"Test RMSE: {np.sqrt(np.mean((predictions - y_test)**2)):.4f}")
    
    # Configure output generator (comprehensive)
    config = InterpretabilityConfig(
        generate_attention_viz=False,  # Would need GNN model
        generate_shap_analysis=True,
        shap_top_features=30,
        shap_sample_size=None,  # Use all samples
        generate_feature_importance=True,
        feature_importance_top_k=30,
        output_dir="./results/demo_comprehensive_interpretability",
        save_individual_plots=True,
        save_summary_plots=True,
        generate_report=True,
        figure_dpi=300,
        figure_format="png"
    )
    
    # Create generator
    generator = create_output_generator(config)
    
    # Generate all outputs
    print("\nGenerating all interpretability outputs...")
    outputs = generator.generate_all_outputs(
        model=model,
        X=X_test,
        feature_names=feature_names,
        smiles_list=smiles_test,
        predictions=predictions,
        true_values=y_test
    )
    
    print("\n" + "="*80)
    print("Comprehensive Interpretability Analysis Complete")
    print("="*80)
    
    print(f"\nAnalysis Summary:")
    print(f"  Compounds analyzed: {outputs.n_compounds_analyzed}")
    print(f"  Features analyzed: {outputs.n_features_analyzed}")
    print(f"  Timestamp: {outputs.timestamp}")
    
    if outputs.shap_feature_importance is not None:
        print(f"\nTop 5 SHAP features:")
        for idx, row in outputs.shap_feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    if outputs.feature_importance is not None:
        print(f"\nTop 5 model features:")
        for idx, row in outputs.feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nReport saved to: {outputs.report_path}")
    
    return outputs


def demo_batch_processing():
    """Demonstrate batch processing for multiple models."""
    print("\n" + "="*80)
    print("DEMO 3: Batch Processing for Multiple Models")
    print("="*80)
    
    # Create data
    X, y, feature_names, smiles_list = create_synthetic_data(n_samples=250, n_features=40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train multiple models
    models = {
        'RF_50': RandomForestRegressor(n_estimators=50, random_state=42),
        'RF_100': RandomForestRegressor(n_estimators=100, random_state=42),
        'RF_200': RandomForestRegressor(n_estimators=200, random_state=42)
    }
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"  {name}: R² = {score:.4f}")
    
    # Generate interpretability for each model
    print("\nGenerating interpretability outputs for each model...")
    
    all_outputs = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        
        config = InterpretabilityConfig(
            generate_attention_viz=False,
            generate_shap_analysis=True,
            shap_top_features=15,
            shap_sample_size=50,
            generate_feature_importance=True,
            feature_importance_top_k=15,
            output_dir=f"./results/demo_batch_{model_name}",
            save_summary_plots=True,
            generate_report=True
        )
        
        generator = create_output_generator(config)
        
        outputs = generator.generate_all_outputs(
            model=model,
            X=X_test,
            feature_names=feature_names
        )
        
        all_outputs[model_name] = outputs
        print(f"  Outputs saved to: {config.output_dir}")
    
    print("\n" + "="*80)
    print("Batch Processing Complete")
    print("="*80)
    
    # Compare feature importance across models
    print("\nFeature Importance Comparison:")
    print("Top 5 features per model:")
    
    for model_name, outputs in all_outputs.items():
        print(f"\n{model_name}:")
        if outputs.feature_importance is not None:
            for idx, row in outputs.feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return all_outputs


def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n" + "="*80)
    print("DEMO 4: Custom Configuration Options")
    print("="*80)
    
    # Show different configuration options
    configs = {
        'Minimal': InterpretabilityConfig(
            generate_shap_analysis=True,
            shap_top_features=10,
            generate_feature_importance=False,
            save_summary_plots=True,
            generate_report=False
        ),
        'Standard': InterpretabilityConfig(
            generate_shap_analysis=True,
            shap_top_features=20,
            generate_feature_importance=True,
            feature_importance_top_k=20,
            save_summary_plots=True,
            generate_report=True
        ),
        'Comprehensive': InterpretabilityConfig(
            generate_attention_viz=True,
            attention_top_k=20,
            generate_shap_analysis=True,
            shap_top_features=30,
            generate_feature_importance=True,
            feature_importance_top_k=30,
            save_individual_plots=True,
            save_summary_plots=True,
            generate_report=True,
            figure_dpi=300
        )
    }
    
    print("\nConfiguration Options:")
    for config_name, config in configs.items():
        print(f"\n{config_name} Configuration:")
        print(f"  Attention viz: {config.generate_attention_viz}")
        print(f"  SHAP analysis: {config.generate_shap_analysis}")
        if config.generate_shap_analysis:
            print(f"    Top features: {config.shap_top_features}")
        print(f"  Feature importance: {config.generate_feature_importance}")
        if config.generate_feature_importance:
            print(f"    Top features: {config.feature_importance_top_k}")
        print(f"  Generate report: {config.generate_report}")
    
    print("\nChoose configuration based on your needs:")
    print("  - Minimal: Quick analysis, essential outputs only")
    print("  - Standard: Balanced analysis for most use cases")
    print("  - Comprehensive: Full analysis with all visualizations")
    
    return configs


def main():
    """Run all interpretability output demonstrations."""
    print("TLR4 Binding Prediction - Interpretability Output Generation Demos")
    print("="*80)
    
    # Create results directory
    os.makedirs("./results", exist_ok=True)
    
    try:
        # Demo 1: SHAP analysis
        demo_shap_analysis()
        
        # Demo 2: Comprehensive outputs
        demo_comprehensive_outputs()
        
        # Demo 3: Batch processing
        demo_batch_processing()
        
        # Demo 4: Custom configuration
        demo_custom_configuration()
        
        print("\n" + "="*80)
        print("All demonstrations completed!")
        print("="*80)
        
        print("\nGenerated outputs:")
        print("  - results/demo_shap_analysis/")
        print("  - results/demo_comprehensive_interpretability/")
        print("  - results/demo_batch_*/")
        print("  - SHAP feature importance (CSV)")
        print("  - SHAP summary plots (PNG)")
        print("  - Feature importance plots (PNG)")
        print("  - Interpretability reports (MD)")
        
        print("\nNext steps:")
        print("1. Review interpretability reports in output directories")
        print("2. Examine SHAP feature importance to identify key descriptors")
        print("3. Compare feature importance across different models")
        print("4. Use insights to guide compound optimization")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
