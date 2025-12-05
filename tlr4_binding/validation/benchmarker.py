"""
Model Benchmarker for TLR4 Binding Prediction

This module implements comprehensive model benchmarking functionality including:
- Model evaluation with R², RMSE, MAE metrics
- Systematic comparison of multiple model architectures
- Statistical significance testing with Wilcoxon tests and multiple comparison correction
- Ablation studies for feature group analysis

Implements Requirements 19.1, 19.2, 19.3 from the TLR4 Methodology Enhancement specification.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator
import logging
from dataclasses import dataclass, asdict
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    """Container for model evaluation metrics"""
    model_name: str
    r2: float
    rmse: float
    mae: float
    n_samples: int
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Container for model comparison results"""
    model_comparisons: pd.DataFrame
    statistical_tests: Dict[str, Dict[str, Any]]
    best_model: str
    rankings: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_comparisons': self.model_comparisons.to_dict('records'),
            'statistical_tests': self.statistical_tests,
            'best_model': self.best_model,
            'rankings': self.rankings
        }


@dataclass
class AblationResult:
    """Container for ablation study results"""
    feature_group: str
    baseline_r2: float
    ablated_r2: float
    r2_difference: float
    relative_impact: float
    p_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ModelBenchmarker:
    """
    Comprehensive model benchmarking for TLR4 binding prediction.
    
    Provides systematic evaluation and comparison of multiple model architectures
    including baseline ensemble, GNN, transformer, hybrid, and transfer learning models.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize ModelBenchmarker.
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.alpha = alpha
        self.evaluation_history: List[ModelEvaluationResult] = []
        logger.info(f"ModelBenchmarker initialized with alpha={alpha}")
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      model_name: str = "model",
                      metadata: Optional[Dict[str, Any]] = None) -> ModelEvaluationResult:
        """
        Evaluate a single model returning R², RMSE, MAE.
        
        Implements Requirement 19.2: Model evaluation with standard regression metrics.
        
        Args:
            y_true: True target values (shape: [n_samples])
            y_pred: Predicted target values (shape: [n_samples])
            model_name: Name identifier for the model
            metadata: Optional additional metadata about the model
            
        Returns:
            ModelEvaluationResult containing R², RMSE, MAE, and sample count
            
        Raises:
            ValueError: If input arrays have mismatched shapes or invalid values
            
        Example:
            >>> benchmarker = ModelBenchmarker()
            >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
            >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2])
            >>> result = benchmarker.evaluate_model(y_true, y_pred, "test_model")
            >>> print(f"R²: {result.r2:.3f}, RMSE: {result.rmse:.3f}")
        """
        # Input validation
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, "
                f"y_pred has shape {y_pred.shape}"
            )
        
        if len(y_true) == 0:
            raise ValueError("Cannot evaluate model with empty arrays")
        
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input arrays contain NaN values")
        
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Input arrays contain infinite values")
        
        # Calculate metrics
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        n_samples = len(y_true)
        
        result = ModelEvaluationResult(
            model_name=model_name,
            r2=r2,
            rmse=rmse,
            mae=mae,
            n_samples=n_samples,
            metadata=metadata or {}
        )
        
        # Store in history
        self.evaluation_history.append(result)
        
        logger.info(
            f"Evaluated {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}, "
            f"MAE={mae:.4f} (n={n_samples})"
        )
        
        return result
    
    def compare_models(self,
                      predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      model_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> ComparisonResult:
        """
        Systematic comparison of multiple model architectures.
        
        Implements Requirement 19.1: Compare baseline, GNN, transformer, hybrid, 
        and transfer learning models on the same test set.
        
        Args:
            predictions: Dictionary mapping model names to (y_true, y_pred) tuples.
                        All models must have predictions for the same test set.
            model_metadata: Optional metadata for each model
            
        Returns:
            ComparisonResult containing comparison DataFrame, rankings, and best model
            
        Raises:
            ValueError: If predictions dict is empty or models have different test sets
            
        Example:
            >>> predictions = {
            ...     'baseline': (y_true, y_pred_baseline),
            ...     'gnn': (y_true, y_pred_gnn),
            ...     'hybrid': (y_true, y_pred_hybrid)
            ... }
            >>> result = benchmarker.compare_models(predictions)
            >>> print(result.best_model)
            >>> print(result.model_comparisons)
        """
        if not predictions:
            raise ValueError("predictions dictionary cannot be empty")
        
        # Validate that all models use the same test set
        y_true_reference = None
        for model_name, (y_true, y_pred) in predictions.items():
            if y_true_reference is None:
                y_true_reference = np.asarray(y_true).flatten()
            else:
                y_true_current = np.asarray(y_true).flatten()
                if not np.array_equal(y_true_reference, y_true_current):
                    raise ValueError(
                        f"Model {model_name} has different test set than reference. "
                        "All models must be evaluated on the same test set."
                    )
        
        logger.info(f"Comparing {len(predictions)} models")
        
        # Evaluate each model
        evaluation_results = []
        for model_name, (y_true, y_pred) in predictions.items():
            metadata = model_metadata.get(model_name, {}) if model_metadata else {}
            result = self.evaluate_model(y_true, y_pred, model_name, metadata)
            evaluation_results.append(result)
        
        # Create comparison DataFrame
        comparison_data = [result.to_dict() for result in evaluation_results]
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by R² (descending)
        comparison_df = comparison_df.sort_values('r2', ascending=False).reset_index(drop=True)
        
        # Determine rankings
        rankings = {}
        for metric in ['r2', 'rmse', 'mae']:
            if metric == 'r2':
                # Higher is better
                sorted_df = comparison_df.sort_values(metric, ascending=False)
            else:
                # Lower is better
                sorted_df = comparison_df.sort_values(metric, ascending=True)
            
            for rank, model_name in enumerate(sorted_df['model_name'], start=1):
                if model_name not in rankings:
                    rankings[model_name] = {}
                rankings[model_name][f'{metric}_rank'] = rank
        
        # Calculate average rank
        for model_name in rankings:
            ranks = [rankings[model_name][f'{m}_rank'] for m in ['r2', 'rmse', 'mae']]
            rankings[model_name]['average_rank'] = np.mean(ranks)
        
        # Best model is the one with highest R²
        best_model = comparison_df.iloc[0]['model_name']
        
        # Placeholder for statistical tests (will be implemented in statistical_comparison)
        statistical_tests = {}
        
        result = ComparisonResult(
            model_comparisons=comparison_df,
            statistical_tests=statistical_tests,
            best_model=best_model,
            rankings=rankings
        )
        
        logger.info(f"Model comparison complete. Best model: {best_model}")
        
        return result
    
    def statistical_comparison(self,
                              predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              correction_method: str = 'bonferroni') -> Dict[str, Dict[str, Any]]:
        """
        Perform statistical comparison with Wilcoxon tests and multiple comparison correction.
        
        Implements Requirement 19.3: Statistical comparison with Wilcoxon signed-rank tests
        and multiple comparison correction.
        
        Args:
            predictions: Dictionary mapping model names to (y_true, y_pred) tuples
            correction_method: Method for multiple comparison correction
                             ('bonferroni', 'holm', 'hochberg', or 'none')
            
        Returns:
            Dictionary containing pairwise test results with corrected p-values
            
        Raises:
            ValueError: If fewer than 2 models provided or invalid correction method
            
        Example:
            >>> predictions = {
            ...     'baseline': (y_true, y_pred_baseline),
            ...     'gnn': (y_true, y_pred_gnn)
            ... }
            >>> tests = benchmarker.statistical_comparison(predictions)
            >>> print(tests['baseline_vs_gnn']['p_value'])
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 models for statistical comparison")
        
        valid_methods = ['bonferroni', 'holm', 'hochberg', 'none']
        if correction_method not in valid_methods:
            raise ValueError(
                f"Invalid correction_method '{correction_method}'. "
                f"Must be one of {valid_methods}"
            )
        
        logger.info(
            f"Performing statistical comparison of {len(predictions)} models "
            f"with {correction_method} correction"
        )
        
        # Validate all models use same test set
        y_true_reference = None
        for model_name, (y_true, y_pred) in predictions.items():
            if y_true_reference is None:
                y_true_reference = np.asarray(y_true).flatten()
            else:
                y_true_current = np.asarray(y_true).flatten()
                if not np.array_equal(y_true_reference, y_true_current):
                    raise ValueError(
                        f"Model {model_name} has different test set. "
                        "All models must use the same test set for paired tests."
                    )
        
        # Calculate residuals for each model
        residuals = {}
        for model_name, (y_true, y_pred) in predictions.items():
            y_true = np.asarray(y_true).flatten()
            y_pred = np.asarray(y_pred).flatten()
            residuals[model_name] = np.abs(y_true - y_pred)
        
        # Perform pairwise Wilcoxon signed-rank tests
        model_names = list(predictions.keys())
        pairwise_tests = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # Wilcoxon signed-rank test on absolute residuals
                # Tests if model1 has significantly different errors than model2
                try:
                    statistic, p_value = stats.wilcoxon(
                        residuals[model1],
                        residuals[model2],
                        alternative='two-sided'
                    )
                except ValueError as e:
                    # Handle case where differences are all zero
                    logger.warning(
                        f"Wilcoxon test failed for {model1} vs {model2}: {e}"
                    )
                    statistic, p_value = np.nan, 1.0
                
                pairwise_tests.append({
                    'comparison': f'{model1}_vs_{model2}',
                    'model1': model1,
                    'model2': model2,
                    'statistic': float(statistic) if not np.isnan(statistic) else None,
                    'p_value_raw': float(p_value)
                })
        
        # Apply multiple comparison correction
        p_values = [test['p_value_raw'] for test in pairwise_tests]
        
        if correction_method == 'bonferroni':
            corrected_p_values = self._bonferroni_correction(p_values)
        elif correction_method == 'holm':
            corrected_p_values = self._holm_correction(p_values)
        elif correction_method == 'hochberg':
            corrected_p_values = self._hochberg_correction(p_values)
        else:  # 'none'
            corrected_p_values = p_values
        
        # Add corrected p-values and significance to results
        for test, p_corrected in zip(pairwise_tests, corrected_p_values):
            test['p_value_corrected'] = float(p_corrected)
            test['significant'] = p_corrected < self.alpha
            test['correction_method'] = correction_method
        
        # Convert to dictionary keyed by comparison name
        results = {test['comparison']: test for test in pairwise_tests}
        
        # Log significant differences
        significant_tests = [t for t in pairwise_tests if t['significant']]
        logger.info(
            f"Found {len(significant_tests)} significant differences "
            f"out of {len(pairwise_tests)} comparisons (alpha={self.alpha})"
        )
        
        return results
    
    def run_ablation(self,
                    X: pd.DataFrame,
                    y: np.ndarray,
                    model: BaseEstimator,
                    feature_groups: Dict[str, List[str]],
                    cv_folds: int = 5) -> List[AblationResult]:
        """
        Run ablation study for feature group analysis.
        
        Implements Requirement 19.1: Ablation studies to understand feature group contributions.
        
        Args:
            X: Feature matrix (DataFrame with named columns)
            y: Target values
            model: Trained model object (must have fit/predict methods)
            feature_groups: Dictionary mapping group names to lists of feature names
            cv_folds: Number of cross-validation folds (default: 5)
            
        Returns:
            List of AblationResult objects showing impact of removing each feature group
            
        Raises:
            ValueError: If feature groups reference non-existent columns
            
        Example:
            >>> feature_groups = {
            ...     '3D_descriptors': ['PMI1', 'PMI2', 'spherocity'],
            ...     'electrostatic': ['gasteiger_charge', 'dipole']
            ... }
            >>> results = benchmarker.run_ablation(X, y, model, feature_groups)
            >>> for result in results:
            ...     print(f"{result.feature_group}: {result.r2_difference:.3f}")
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        
        # Validate feature groups
        for group_name, features in feature_groups.items():
            missing_features = set(features) - set(X.columns)
            if missing_features:
                raise ValueError(
                    f"Feature group '{group_name}' references non-existent columns: "
                    f"{missing_features}"
                )
        
        logger.info(
            f"Running ablation study with {len(feature_groups)} feature groups "
            f"using {cv_folds}-fold CV"
        )
        
        # Get baseline performance with all features
        baseline_scores = cross_val_score(
            clone(model), X, y,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1
        )
        baseline_r2 = float(np.mean(baseline_scores))
        
        logger.info(f"Baseline R² with all features: {baseline_r2:.4f}")
        
        # Test each feature group ablation
        ablation_results = []
        
        for group_name, features in feature_groups.items():
            logger.info(f"Ablating feature group: {group_name} ({len(features)} features)")
            
            # Remove feature group
            X_ablated = X.drop(columns=features)
            
            if X_ablated.shape[1] == 0:
                logger.warning(
                    f"Skipping {group_name} - would remove all features"
                )
                continue
            
            # Evaluate without this feature group
            ablated_scores = cross_val_score(
                clone(model), X_ablated, y,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            ablated_r2 = float(np.mean(ablated_scores))
            
            # Calculate impact
            r2_difference = baseline_r2 - ablated_r2
            relative_impact = (r2_difference / abs(baseline_r2)) if baseline_r2 != 0 else 0.0
            
            # Statistical test: paired t-test between baseline and ablated scores
            try:
                _, p_value = stats.ttest_rel(baseline_scores, ablated_scores)
                p_value = float(p_value)
            except Exception as e:
                logger.warning(f"Statistical test failed for {group_name}: {e}")
                p_value = None
            
            result = AblationResult(
                feature_group=group_name,
                baseline_r2=baseline_r2,
                ablated_r2=ablated_r2,
                r2_difference=r2_difference,
                relative_impact=relative_impact,
                p_value=p_value
            )
            
            ablation_results.append(result)
            
            p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            logger.info(
                f"  {group_name}: ΔR² = {r2_difference:+.4f} "
                f"({relative_impact:+.1%}), p = {p_value_str}"
            )
        
        # Sort by absolute impact
        ablation_results.sort(key=lambda x: abs(x.r2_difference), reverse=True)
        
        logger.info(f"Ablation study complete. Tested {len(ablation_results)} groups.")
        
        return ablation_results
    
    def _bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons"""
        n_tests = len(p_values)
        return [min(p * n_tests, 1.0) for p in p_values]
    
    def _holm_correction(self, p_values: List[float]) -> List[float]:
        """Apply Holm-Bonferroni correction for multiple comparisons"""
        n_tests = len(p_values)
        
        # Sort p-values with original indices
        indexed_p = [(i, p) for i, p in enumerate(p_values)]
        indexed_p.sort(key=lambda x: x[1])
        
        # Apply Holm correction
        corrected = [0.0] * n_tests
        for rank, (original_idx, p_value) in enumerate(indexed_p, start=1):
            corrected_p = min(p_value * (n_tests - rank + 1), 1.0)
            corrected[original_idx] = corrected_p
        
        # Enforce monotonicity
        indexed_corrected = [(i, p) for i, p in enumerate(corrected)]
        indexed_corrected.sort(key=lambda x: p_values[x[0]])
        
        max_so_far = 0.0
        for original_idx, _ in indexed_corrected:
            corrected[original_idx] = max(corrected[original_idx], max_so_far)
            max_so_far = corrected[original_idx]
        
        return corrected
    
    def _hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Hochberg correction for multiple comparisons"""
        n_tests = len(p_values)
        
        # Sort p-values with original indices (descending order for Hochberg)
        indexed_p = [(i, p) for i, p in enumerate(p_values)]
        indexed_p.sort(key=lambda x: x[1], reverse=True)
        
        # Apply Hochberg correction
        corrected = [0.0] * n_tests
        for rank, (original_idx, p_value) in enumerate(indexed_p):
            corrected_p = min(p_value * (rank + 1), 1.0)
            corrected[original_idx] = corrected_p
        
        # Enforce monotonicity (decreasing)
        indexed_corrected = [(i, p) for i, p in enumerate(corrected)]
        indexed_corrected.sort(key=lambda x: p_values[x[0]], reverse=True)
        
        min_so_far = 1.0
        for original_idx, _ in indexed_corrected:
            corrected[original_idx] = min(corrected[original_idx], min_so_far)
            min_so_far = corrected[original_idx]
        
        return corrected
    
    def generate_report(self,
                       comparison_result: Optional[ComparisonResult] = None,
                       statistical_tests: Optional[Dict[str, Dict[str, Any]]] = None,
                       ablation_results: Optional[List[AblationResult]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmarking report.
        
        Args:
            comparison_result: Results from compare_models()
            statistical_tests: Results from statistical_comparison()
            ablation_results: Results from run_ablation()
            
        Returns:
            Dictionary containing comprehensive benchmarking report
        """
        report = {
            'summary': {},
            'model_comparison': None,
            'statistical_tests': None,
            'ablation_study': None
        }
        
        if comparison_result:
            report['model_comparison'] = comparison_result.to_dict()
            report['summary']['best_model'] = comparison_result.best_model
            report['summary']['n_models'] = len(comparison_result.model_comparisons)
        
        if statistical_tests:
            report['statistical_tests'] = statistical_tests
            significant_count = sum(
                1 for test in statistical_tests.values() if test['significant']
            )
            report['summary']['significant_differences'] = significant_count
            report['summary']['total_comparisons'] = len(statistical_tests)
        
        if ablation_results:
            report['ablation_study'] = [r.to_dict() for r in ablation_results]
            report['summary']['n_feature_groups'] = len(ablation_results)
            if ablation_results:
                most_important = max(ablation_results, key=lambda x: abs(x.r2_difference))
                report['summary']['most_important_group'] = most_important.feature_group
        
        return report
