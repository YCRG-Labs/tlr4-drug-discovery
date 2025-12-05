#!/usr/bin/env python3
"""
Demo script for full validation suite.

This script demonstrates the comprehensive validation suite including:
- Nested cross-validation
- Y-scrambling validation
- Scaffold-based validation
- Applicability domain analysis
- Model comparison with statistical testing
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tlr4_binding.validation.full_validation_suite import (
    FullValidationSuite,
    ValidationSuiteConfig,
    create_validation_suite
)


def create_synthetic_data(n_samples=500, n_features=50):
    """Create synthetic molecular data for demonstration."""
    print("Creating synthetic TLR4 binding data...")
    
    # Generate features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Generate binding affinities with some structure
    y = (
        -0.5 * X[:, 0] +
        -0.3 * X[:, 1] +
        -0.2 * X[:, 2] +
        np.random.normal(0, 1, n_samples)
    )
    
    # Generate SMILES (simplified for demo)
    smiles_templates = [
        "CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN",
        "CCCC", "c1ccc(O)cc1", "CC(C)C", "CCCN", "c1ccncc1"
    ]
    smiles_list = [smiles_templates[i % len(smiles_templates)] + f"_{i}" 
                   for i in range(n_samples)]
    
    print(f"Generated {n_samples} compounds with {n_features} features")
    print(f"Binding affinity range: {y.min():.2f} to {y.max():.2f} kcal/mol")
    
    return X, y, smiles_list


def demo_minimal_validation():
    """Demonstrate minimal validation suite."""
    print("="*80)
    print("DEMO 1: Minimal Validation Suite")
    print("="*80)
    
    # Create data
    X, y, smiles_list = create_synthetic_data(n_samples=200, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    
    # Train models
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  {name} trained")
    
    # Configure validation suite (minimal)
    config = ValidationSuiteConfig(
        run_nested_cv=True,
        nested_cv_outer_folds=3,
        nested_cv_inner_folds=2,
        run_y_scrambling=True,
        y_scrambling_iterations=50,
        run_scaffold_validation=False,
        run_applicability_domain=True,
        run_model_comparison=True,
        output_dir="./results/demo_minimal_validation",
        save_intermediate=True
    )
    
    # Create validation suite
    suite = create_validation_suite(config)
    
    # Run validation
    print("\nRunning validation suite...")
    results = suite.run_full_suite(
        models=models,
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Validation Summary")
    print("="*80)
    print(f"Best model: {results.best_model}")
    print(f"Best R²: {results.best_model_r2:.4f}")
    print(f"Validation passed: {results.validation_passed}")
    if results.validation_warnings:
        print(f"Warnings: {len(results.validation_warnings)}")
    
    return results


def demo_comprehensive_validation():
    """Demonstrate comprehensive validation suite."""
    print("\n" + "="*80)
    print("DEMO 2: Comprehensive Validation Suite")
    print("="*80)
    
    # Create data
    X, y, smiles_list = create_synthetic_data(n_samples=500, n_features=50)
    
    # Split data
    train_idx = np.arange(400)
    test_idx = np.arange(400, 500)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    smiles_train = [smiles_list[i] for i in train_idx]
    smiles_test = [smiles_list[i] for i in test_idx]
    
    # Create multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    # Train models
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  {name} trained")
    
    # Configure validation suite (comprehensive)
    config = ValidationSuiteConfig(
        run_nested_cv=True,
        nested_cv_outer_folds=5,
        nested_cv_inner_folds=3,
        run_y_scrambling=True,
        y_scrambling_iterations=100,
        y_scrambling_threshold=0.5,
        run_scaffold_validation=True,
        scaffold_test_size=0.2,
        run_applicability_domain=True,
        leverage_threshold_multiplier=3.0,
        run_model_comparison=True,
        statistical_test="wilcoxon",
        multiple_comparison_correction="bonferroni",
        output_dir="./results/demo_comprehensive_validation",
        save_intermediate=True,
        generate_plots=True
    )
    
    # Create validation suite
    suite = create_validation_suite(config)
    
    # Run full validation
    print("\nRunning comprehensive validation suite...")
    results = suite.run_full_suite(
        models=models,
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        smiles_list=smiles_list,
        smiles_train=smiles_train,
        smiles_test=smiles_test
    )
    
    # Print detailed summary
    print("\n" + "="*80)
    print("Comprehensive Validation Summary")
    print("="*80)
    
    print("\nNested Cross-Validation Results:")
    if results.nested_cv_results:
        for model_name, cv_result in results.nested_cv_results.items():
            if 'error' not in cv_result:
                print(f"  {model_name}:")
                print(f"    R²: {cv_result['r2_mean']:.4f} ± {cv_result['r2_std']:.4f}")
                print(f"    RMSE: {cv_result['rmse_mean']:.4f} ± {cv_result['rmse_std']:.4f}")
    
    print("\nY-Scrambling Results:")
    if results.y_scrambling_results:
        for model_name, scrambling_result in results.y_scrambling_results.items():
            if 'error' not in scrambling_result:
                print(f"  {model_name}:")
                print(f"    Original R²: {scrambling_result['original_r2']:.4f}")
                print(f"    cR²p: {scrambling_result['cr2p']:.4f}")
                passed = scrambling_result['cr2p'] > config.y_scrambling_threshold
                print(f"    Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    print("\nScaffold Validation Results:")
    if results.scaffold_validation_results:
        for model_name, scaffold_result in results.scaffold_validation_results.items():
            if 'error' not in scaffold_result:
                print(f"  {model_name}:")
                print(f"    R²: {scaffold_result['r2']:.4f}")
                print(f"    RMSE: {scaffold_result['rmse']:.4f}")
    
    print("\nApplicability Domain:")
    if results.applicability_domain_results:
        ad_results = results.applicability_domain_results
        print(f"  Compounds in domain: {ad_results['n_in_domain']} ({ad_results['fraction_in_domain']*100:.1f}%)")
        print(f"  Compounds out of domain: {ad_results['n_out_domain']}")
        print(f"  Leverage threshold: {ad_results['threshold']:.4f}")
    
    print("\nBest Model:")
    print(f"  Model: {results.best_model}")
    print(f"  R²: {results.best_model_r2:.4f}")
    
    print("\nValidation Status:")
    print(f"  Overall: {'✓ PASSED' if results.validation_passed else '✗ FAILED'}")
    if results.validation_warnings:
        print(f"  Warnings ({len(results.validation_warnings)}):")
        for warning in results.validation_warnings:
            print(f"    - {warning}")
    
    print(f"\nResults saved to: {config.output_dir}")
    
    return results


def demo_validation_comparison():
    """Demonstrate validation comparison across different configurations."""
    print("\n" + "="*80)
    print("DEMO 3: Validation Configuration Comparison")
    print("="*80)
    
    # Create data
    X, y, smiles_list = create_synthetic_data(n_samples=300, n_features=30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    models = {'Random Forest': model}
    
    # Test different validation configurations
    configs = {
        'Quick': ValidationSuiteConfig(
            nested_cv_outer_folds=3,
            nested_cv_inner_folds=2,
            y_scrambling_iterations=50,
            run_scaffold_validation=False,
            output_dir="./results/demo_quick_validation"
        ),
        'Standard': ValidationSuiteConfig(
            nested_cv_outer_folds=5,
            nested_cv_inner_folds=3,
            y_scrambling_iterations=100,
            run_scaffold_validation=True,
            output_dir="./results/demo_standard_validation"
        ),
        'Rigorous': ValidationSuiteConfig(
            nested_cv_outer_folds=10,
            nested_cv_inner_folds=5,
            y_scrambling_iterations=1000,
            run_scaffold_validation=True,
            output_dir="./results/demo_rigorous_validation"
        )
    }
    
    print("\nComparing validation configurations:")
    for config_name, config in configs.items():
        print(f"\n{config_name} Configuration:")
        print(f"  Nested CV: {config.nested_cv_outer_folds}x{config.nested_cv_inner_folds}")
        print(f"  Y-scrambling: {config.y_scrambling_iterations} iterations")
        print(f"  Scaffold validation: {config.run_scaffold_validation}")
    
    print("\nNote: This demonstrates different validation rigor levels.")
    print("Choose based on your computational budget and validation requirements.")
    
    return configs


def main():
    """Run all validation suite demonstrations."""
    print("TLR4 Binding Prediction - Full Validation Suite Demos")
    print("="*80)
    
    # Create results directory
    os.makedirs("./results", exist_ok=True)
    
    try:
        # Demo 1: Minimal validation
        demo_minimal_validation()
        
        # Demo 2: Comprehensive validation
        demo_comprehensive_validation()
        
        # Demo 3: Configuration comparison
        demo_validation_comparison()
        
        print("\n" + "="*80)
        print("All demonstrations completed!")
        print("="*80)
        
        print("\nGenerated outputs:")
        print("  - results/demo_minimal_validation/")
        print("  - results/demo_comprehensive_validation/")
        print("  - Nested CV results (JSON)")
        print("  - Y-scrambling results (JSON)")
        print("  - Scaffold validation results (JSON)")
        print("  - Applicability domain results (JSON)")
        print("  - Model comparison tables (CSV)")
        print("  - Comprehensive comparison table (CSV)")
        
        print("\nNext steps:")
        print("1. Review validation results in the output directories")
        print("2. Check comprehensive_comparison.csv for model rankings")
        print("3. Examine Y-scrambling results to assess model robustness")
        print("4. Review applicability domain to understand prediction reliability")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
