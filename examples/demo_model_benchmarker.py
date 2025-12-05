"""
Demo: Model Benchmarker for TLR4 Binding Prediction

This example demonstrates how to use the ModelBenchmarker to:
1. Evaluate individual models
2. Compare multiple model architectures
3. Perform statistical significance testing
4. Run ablation studies on feature groups
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from tlr4_binding.validation.benchmarker import ModelBenchmarker


def generate_synthetic_data(n_samples=200, n_features=20, noise_level=0.5):
    """Generate synthetic molecular descriptor data"""
    np.random.seed(42)
    
    # Create feature groups
    feature_names = []
    feature_names.extend([f'2D_desc_{i}' for i in range(8)])
    feature_names.extend([f'3D_desc_{i}' for i in range(6)])
    feature_names.extend([f'electrostatic_{i}' for i in range(4)])
    feature_names.extend([f'graph_feat_{i}' for i in range(2)])
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Create target with some structure
    y = (
        2.0 * X['2D_desc_0'] +
        1.5 * X['3D_desc_0'] +
        1.0 * X['electrostatic_0'] +
        0.5 * X['graph_feat_0'] +
        np.random.randn(n_samples) * noise_level
    )
    
    return X, y.values


def demo_basic_evaluation():
    """Demo 1: Basic model evaluation"""
    print("=" * 70)
    print("DEMO 1: Basic Model Evaluation")
    print("=" * 70)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    benchmarker = ModelBenchmarker()
    result = benchmarker.evaluate_model(
        y_test, y_pred,
        model_name="RandomForest",
        metadata={"n_estimators": 100, "max_depth": None}
    )
    
    print(f"\nModel: {result.model_name}")
    print(f"R² Score: {result.r2:.4f}")
    print(f"RMSE: {result.rmse:.4f}")
    print(f"MAE: {result.mae:.4f}")
    print(f"Samples: {result.n_samples}")
    print(f"Metadata: {result.metadata}")


def demo_model_comparison():
    """Demo 2: Compare multiple models"""
    print("\n" + "=" * 70)
    print("DEMO 2: Model Comparison")
    print("=" * 70)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train multiple models
    models = {
        'Baseline_Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    # Get predictions for all models
    predictions = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = (y_test, y_pred)
    
    # Compare models
    benchmarker = ModelBenchmarker()
    comparison = benchmarker.compare_models(predictions)
    
    print("\n" + "-" * 70)
    print("Model Comparison Results:")
    print("-" * 70)
    print(comparison.model_comparisons.to_string(index=False))
    
    print(f"\nBest Model: {comparison.best_model}")
    
    print("\nModel Rankings:")
    for model_name, ranks in comparison.rankings.items():
        avg_rank = ranks.get('average_rank', 0)
        print(f"  {model_name}: Average Rank = {avg_rank:.2f}")


def demo_statistical_comparison():
    """Demo 3: Statistical significance testing"""
    print("\n" + "=" * 70)
    print("DEMO 3: Statistical Significance Testing")
    print("=" * 70)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train models
    models = {
        'Baseline': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = (y_test, y_pred)
    
    # Statistical comparison with Bonferroni correction
    benchmarker = ModelBenchmarker(alpha=0.05)
    tests = benchmarker.statistical_comparison(
        predictions,
        correction_method='bonferroni'
    )
    
    print("\n" + "-" * 70)
    print("Pairwise Statistical Tests (Wilcoxon signed-rank):")
    print("-" * 70)
    
    for comparison_name, test_result in tests.items():
        print(f"\n{comparison_name}:")
        print(f"  Raw p-value: {test_result['p_value_raw']:.6f}")
        print(f"  Corrected p-value: {test_result['p_value_corrected']:.6f}")
        print(f"  Significant: {test_result['significant']}")
        print(f"  Correction method: {test_result['correction_method']}")


def demo_ablation_study():
    """Demo 4: Feature group ablation study"""
    print("\n" + "=" * 70)
    print("DEMO 4: Feature Group Ablation Study")
    print("=" * 70)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=150)
    
    # Define feature groups
    feature_groups = {
        '2D_descriptors': [col for col in X.columns if '2D_desc' in col],
        '3D_descriptors': [col for col in X.columns if '3D_desc' in col],
        'electrostatic': [col for col in X.columns if 'electrostatic' in col],
        'graph_features': [col for col in X.columns if 'graph_feat' in col]
    }
    
    print("\nFeature Groups:")
    for group_name, features in feature_groups.items():
        print(f"  {group_name}: {len(features)} features")
    
    # Run ablation study
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    benchmarker = ModelBenchmarker()
    
    print("\nRunning ablation study (this may take a moment)...")
    ablation_results = benchmarker.run_ablation(
        X, y, model, feature_groups, cv_folds=5
    )
    
    print("\n" + "-" * 70)
    print("Ablation Study Results (sorted by impact):")
    print("-" * 70)
    
    for result in ablation_results:
        print(f"\n{result.feature_group}:")
        print(f"  Baseline R²: {result.baseline_r2:.4f}")
        print(f"  Ablated R²: {result.ablated_r2:.4f}")
        print(f"  ΔR²: {result.r2_difference:+.4f}")
        print(f"  Relative Impact: {result.relative_impact:+.1%}")
        if result.p_value is not None:
            print(f"  p-value: {result.p_value:.4f}")
            significance = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else "ns"
            print(f"  Significance: {significance}")


def demo_comprehensive_report():
    """Demo 5: Generate comprehensive benchmarking report"""
    print("\n" + "=" * 70)
    print("DEMO 5: Comprehensive Benchmarking Report")
    print("=" * 70)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train models
    models = {
        'Baseline': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = (y_test, y_pred)
    
    # Run all analyses
    benchmarker = ModelBenchmarker()
    
    comparison_result = benchmarker.compare_models(predictions)
    statistical_tests = benchmarker.statistical_comparison(predictions)
    
    feature_groups = {
        '2D_descriptors': [col for col in X.columns if '2D_desc' in col],
        '3D_descriptors': [col for col in X.columns if '3D_desc' in col]
    }
    ablation_results = benchmarker.run_ablation(
        X, y, models['RandomForest'], feature_groups, cv_folds=3
    )
    
    # Generate report
    report = benchmarker.generate_report(
        comparison_result=comparison_result,
        statistical_tests=statistical_tests,
        ablation_results=ablation_results
    )
    
    print("\n" + "-" * 70)
    print("Comprehensive Benchmarking Report:")
    print("-" * 70)
    
    print("\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    print("\nBest Model Details:")
    best_model_data = report['model_comparison']['model_comparisons'][0]
    print(f"  Model: {best_model_data['model_name']}")
    print(f"  R²: {best_model_data['r2']:.4f}")
    print(f"  RMSE: {best_model_data['rmse']:.4f}")
    print(f"  MAE: {best_model_data['mae']:.4f}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("MODEL BENCHMARKER DEMONSTRATION")
    print("TLR4 Binding Prediction - Comprehensive Model Evaluation")
    print("=" * 70)
    
    demo_basic_evaluation()
    demo_model_comparison()
    demo_statistical_comparison()
    demo_ablation_study()
    demo_comprehensive_report()
    
    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
