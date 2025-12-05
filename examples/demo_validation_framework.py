"""
Demonstration of the ValidationFramework for TLR4 Binding Affinity Prediction

This example shows how to use the comprehensive validation framework including:
- Stratified splitting by affinity quartiles
- Nested cross-validation with hyperparameter optimization
- Y-scrambling validation for robustness assessment
- Scaffold-based splitting for generalization testing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import logging

from tlr4_binding.validation import ValidationFramework

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples=200):
    """Generate sample molecular data for demonstration."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, 20)
    
    # Generate target with some structure
    y = (2 * X[:, 0] + 
         0.5 * X[:, 1] - 
         1.5 * X[:, 2] + 
         np.random.randn(n_samples) * 0.5)
    
    # Generate sample SMILES (simplified for demo)
    smiles = [
        'CCO', 'CCCO', 'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC',
        'CC(C)O', 'CCC(C)O', 'c1ccc(O)cc1', 'c1ccc(O)cc1C',
        'CCCC', 'CC(=O)O', 'c1ccncc1', 'CC(=O)C', 'c1cccnc1',
        'c1ccc2ccccc2c1', 'CC(C)C', 'c1ccc(C)cc1', 'CCCCO',
        'c1ccc(CC)cc1', 'CC(C)CC'
    ] * (n_samples // 20 + 1)
    smiles = smiles[:n_samples]
    
    return X, y, smiles


def demo_stratified_split():
    """Demonstrate stratified splitting by affinity quartiles."""
    print("\n" + "="*70)
    print("DEMO 1: Stratified Splitting by Affinity Quartiles")
    print("="*70)
    
    X, y, _ = generate_sample_data()
    
    framework = ValidationFramework(random_state=42)
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = framework.stratified_split(
        X, y, test_size=0.2, n_bins=4
    )
    
    print(f"\nResults:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Train affinity: {y_train.mean():.3f} ± {y_train.std():.3f}")
    print(f"  Test affinity: {y_test.mean():.3f} ± {y_test.std():.3f}")
    
    # Train a simple model
    model = Ridge()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")


def demo_nested_cv():
    """Demonstrate nested cross-validation with hyperparameter optimization."""
    print("\n" + "="*70)
    print("DEMO 2: Nested Cross-Validation")
    print("="*70)
    
    X, y, _ = generate_sample_data()
    
    framework = ValidationFramework(random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
    
    # Perform nested CV
    print("\nRunning nested CV (this may take a moment)...")
    results = framework.nested_cv(
        model_class=Ridge(),
        X=X,
        y=y,
        param_grid=param_grid,
        outer_folds=5,
        inner_folds=3
    )
    
    print(f"\nResults:")
    print(f"  Mean R²: {results['mean_score']:.4f}")
    print(f"  Std R²: {results['std_score']:.4f}")
    print(f"  Outer fold scores: {[f'{s:.4f}' for s in results['outer_scores']]}")
    
    print(f"\nBest hyperparameters per fold:")
    for i, params in enumerate(results['best_params_per_fold']):
        print(f"  Fold {i+1}: {params}")


def demo_y_scrambling():
    """Demonstrate Y-scrambling validation."""
    print("\n" + "="*70)
    print("DEMO 3: Y-Scrambling Validation")
    print("="*70)
    
    X, y, _ = generate_sample_data()
    
    framework = ValidationFramework(random_state=42)
    
    # Perform Y-scrambling
    print("\nRunning Y-scrambling with 50 permutations...")
    results = framework.y_scrambling(
        model_class=Ridge(),
        X=X,
        y=y,
        n_permutations=50
    )
    
    print(f"\nResults:")
    print(f"  Real data R²: {results['r2_real']:.4f}")
    print(f"  Scrambled R² (mean): {results['r2_scrambled_mean']:.4f}")
    print(f"  Scrambled R² (std): {results['r2_scrambled_std']:.4f}")
    print(f"  cR²p metric: {results['cr2p']:.4f}")
    print(f"  Potentially overfit: {results['is_potentially_overfit']}")
    
    if results['is_potentially_overfit']:
        print("\n  ⚠️  Warning: Model may be overfit (cR²p ≤ 0.5)")
    else:
        print("\n  ✓ Model appears robust (cR²p > 0.5)")


def demo_scaffold_split():
    """Demonstrate scaffold-based splitting."""
    print("\n" + "="*70)
    print("DEMO 4: Scaffold-Based Splitting")
    print("="*70)
    
    X, y, smiles = generate_sample_data()
    
    framework = ValidationFramework(random_state=42)
    
    # Perform scaffold split
    train_indices, test_indices = framework.scaffold_split(
        smiles=smiles,
        y=y,
        test_size=0.2
    )
    
    print(f"\nResults:")
    print(f"  Training set: {len(train_indices)} compounds")
    print(f"  Test set: {len(test_indices)} compounds")
    
    # Get scaffolds
    scaffolds = framework._generate_scaffolds(smiles)
    train_scaffolds = set(scaffolds[i] for i in train_indices)
    test_scaffolds = set(scaffolds[i] for i in test_indices)
    
    print(f"  Unique scaffolds in train: {len(train_scaffolds)}")
    print(f"  Unique scaffolds in test: {len(test_scaffolds)}")
    print(f"  Scaffold overlap: {len(train_scaffolds.intersection(test_scaffolds))}")
    
    # Train and evaluate
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    model = Ridge()
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R² (novel scaffolds): {test_score:.4f}")


def demo_complete_validation():
    """Demonstrate complete validation pipeline."""
    print("\n" + "="*70)
    print("DEMO 5: Complete Validation Pipeline")
    print("="*70)
    
    X, y, smiles = generate_sample_data()
    
    framework = ValidationFramework(random_state=42)
    
    print("\n1. Stratified Split")
    X_train, X_test, y_train, y_test = framework.stratified_split(X, y)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    print("\n2. Nested CV on Training Data")
    cv_results = framework.nested_cv(
        model_class=Ridge(),
        X=X_train,
        y=y_train,
        param_grid={'alpha': [0.1, 1.0, 10.0]},
        outer_folds=3,
        inner_folds=2
    )
    print(f"   CV R²: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    print("\n3. Y-Scrambling Validation")
    scrambling_results = framework.y_scrambling(
        model_class=Ridge(),
        X=X_train,
        y=y_train,
        n_permutations=30
    )
    print(f"   Real R²: {scrambling_results['r2_real']:.4f}")
    print(f"   Scrambled R²: {scrambling_results['r2_scrambled_mean']:.4f}")
    print(f"   cR²p: {scrambling_results['cr2p']:.4f}")
    
    print("\n4. Scaffold-Based Validation")
    train_indices, test_indices = framework.scaffold_split(smiles, y)
    print(f"   Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    print("\n5. Final Model Evaluation")
    model = Ridge(alpha=cv_results['best_params_per_fold'][0].get('alpha', 1.0))
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = framework.calculate_metrics(y_test, y_pred)
    
    print(f"   Test R²: {metrics['r2']:.4f}")
    print(f"   Test RMSE: {metrics['rmse']:.4f}")
    print(f"   Test MAE: {metrics['mae']:.4f}")
    
    print("\n" + "="*70)
    print("Validation Complete!")
    print("="*70)


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("VALIDATION FRAMEWORK DEMONSTRATION")
    print("TLR4 Binding Affinity Prediction")
    print("="*70)
    
    try:
        demo_stratified_split()
        demo_nested_cv()
        demo_y_scrambling()
        demo_scaffold_split()
        demo_complete_validation()
        
        print("\n✓ All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
