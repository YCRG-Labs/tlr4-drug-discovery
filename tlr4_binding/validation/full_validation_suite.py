"""
Full validation suite for TLR4 binding affinity prediction models.

This module provides a comprehensive validation suite that executes:
- Nested cross-validation
- Y-scrambling validation
- Scaffold-based validation
- Applicability domain analysis
- Model comparison with statistical testing

Requirements: 13.1, 14.1, 15.1, 17.1, 19.1
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from .framework import ValidationFramework
from .applicability_domain import ApplicabilityDomainAnalyzer
from .benchmarker import ModelBenchmarker, ModelEvaluationResult
from .models import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationSuiteConfig:
    """Configuration for the full validation suite."""
    
    # Nested CV
    run_nested_cv: bool = True
    nested_cv_outer_folds: int = 5
    nested_cv_inner_folds: int = 3
    
    # Y-scrambling
    run_y_scrambling: bool = True
    y_scrambling_iterations: int = 100
    y_scrambling_threshold: float = 0.5  # cR²p threshold
    
    # Scaffold validation
    run_scaffold_validation: bool = True
    scaffold_test_size: float = 0.2
    
    # Applicability domain
    run_applicability_domain: bool = True
    leverage_threshold_multiplier: float = 3.0
    
    # Model comparison
    run_model_comparison: bool = True
    statistical_test: str = "wilcoxon"  # or "t-test"
    multiple_comparison_correction: str = "bonferroni"  # or "fdr"
    
    # Output
    output_dir: str = "./results/validation_suite"
    save_intermediate: bool = True
    generate_plots: bool = True


@dataclass
class ValidationSuiteResults:
    """Results from the full validation suite."""
    
    # Nested CV results
    nested_cv_results: Optional[Dict[str, Dict[str, float]]] = None
    
    # Y-scrambling results
    y_scrambling_results: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Scaffold validation results
    scaffold_validation_results: Optional[Dict[str, Dict[str, float]]] = None
    
    # Applicability domain results
    applicability_domain_results: Optional[Dict[str, Any]] = None
    
    # Model comparison results
    model_comparison_results: Optional[pd.DataFrame] = None
    statistical_comparison_results: Optional[pd.DataFrame] = None
    
    # Summary
    best_model: Optional[str] = None
    best_model_r2: Optional[float] = None
    validation_passed: bool = False
    validation_warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize list fields."""
        if self.validation_warnings is None:
            self.validation_warnings = []


class FullValidationSuite:
    """
    Comprehensive validation suite for TLR4 binding prediction models.
    
    This class orchestrates all validation procedures including:
    - Nested cross-validation for unbiased performance estimation
    - Y-scrambling to assess model robustness
    - Scaffold-based validation for generalization assessment
    - Applicability domain analysis for prediction reliability
    - Statistical model comparison
    """
    
    def __init__(self, config: Optional[ValidationSuiteConfig] = None):
        """
        Initialize the validation suite.
        
        Args:
            config: Validation suite configuration
        """
        self.config = config or ValidationSuiteConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validation_framework = ValidationFramework()
        self.applicability_domain = ApplicabilityDomainAnalyzer()
        self.benchmarker = ModelBenchmarker()
        
        # Results storage
        self.results = ValidationSuiteResults()
        
        logger.info(f"Validation suite initialized with output directory: {self.output_dir}")
    
    def run_nested_cv_validation(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run nested cross-validation for all models.
        
        Args:
            models: Dictionary of model instances
            X: Feature matrix
            y: Target values
            feature_names: Optional feature names
            
        Returns:
            Dictionary of nested CV results per model
        """
        logger.info("="*80)
        logger.info("Running Nested Cross-Validation")
        logger.info("="*80)
        logger.info(f"Outer folds: {self.config.nested_cv_outer_folds}")
        logger.info(f"Inner folds: {self.config.nested_cv_inner_folds}")
        
        nested_cv_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            try:
                # Run nested CV
                cv_result = self.validation_framework.nested_cv(
                    model=model,
                    X=X,
                    y=y,
                    outer_folds=self.config.nested_cv_outer_folds,
                    inner_folds=self.config.nested_cv_inner_folds
                )
                
                nested_cv_results[model_name] = cv_result
                
                logger.info(f"  Mean R²: {cv_result['r2_mean']:.4f} ± {cv_result['r2_std']:.4f}")
                logger.info(f"  Mean RMSE: {cv_result['rmse_mean']:.4f} ± {cv_result['rmse_std']:.4f}")
                logger.info(f"  Mean MAE: {cv_result['mae_mean']:.4f} ± {cv_result['mae_std']:.4f}")
                
            except Exception as e:
                logger.error(f"  Nested CV failed for {model_name}: {e}")
                nested_cv_results[model_name] = {'error': str(e)}
        
        # Save results
        if self.config.save_intermediate:
            output_path = self.output_dir / "nested_cv_results.json"
            with open(output_path, 'w') as f:
                json.dump(nested_cv_results, f, indent=2)
            logger.info(f"\nNested CV results saved to: {output_path}")
        
        self.results.nested_cv_results = nested_cv_results
        return nested_cv_results
    
    def run_y_scrambling_validation(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run Y-scrambling validation for all models.
        
        Args:
            models: Dictionary of model instances
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of Y-scrambling results per model
        """
        logger.info("\n" + "="*80)
        logger.info("Running Y-Scrambling Validation")
        logger.info("="*80)
        logger.info(f"Iterations: {self.config.y_scrambling_iterations}")
        logger.info(f"cR²p threshold: {self.config.y_scrambling_threshold}")
        
        y_scrambling_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            try:
                # Run Y-scrambling
                scrambling_result = self.validation_framework.y_scrambling(
                    model=model,
                    X=X,
                    y=y,
                    n_permutations=self.config.y_scrambling_iterations
                )
                
                y_scrambling_results[model_name] = scrambling_result
                
                logger.info(f"  Original R²: {scrambling_result['original_r2']:.4f}")
                logger.info(f"  Mean scrambled R²: {scrambling_result['mean_scrambled_r2']:.4f}")
                logger.info(f"  cR²p: {scrambling_result['cr2p']:.4f}")
                
                # Check if model passes validation
                if scrambling_result['cr2p'] <= self.config.y_scrambling_threshold:
                    warning = f"{model_name} may be overfit (cR²p = {scrambling_result['cr2p']:.4f} ≤ {self.config.y_scrambling_threshold})"
                    logger.warning(f"  WARNING: {warning}")
                    self.results.validation_warnings.append(warning)
                else:
                    logger.info(f"  ✓ Passed Y-scrambling validation")
                
            except Exception as e:
                logger.error(f"  Y-scrambling failed for {model_name}: {e}")
                y_scrambling_results[model_name] = {'error': str(e)}
        
        # Save results
        if self.config.save_intermediate:
            output_path = self.output_dir / "y_scrambling_results.json"
            with open(output_path, 'w') as f:
                json.dump(y_scrambling_results, f, indent=2)
            logger.info(f"\nY-scrambling results saved to: {output_path}")
        
        self.results.y_scrambling_results = y_scrambling_results
        return y_scrambling_results
    
    def run_scaffold_validation(
        self,
        models: Dict[str, Any],
        smiles_list: List[str],
        y: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run scaffold-based validation for all models.
        
        Args:
            models: Dictionary of model instances
            smiles_list: List of SMILES strings
            y: Target values
            X: Optional feature matrix (if None, will be generated)
            
        Returns:
            Dictionary of scaffold validation results per model
        """
        logger.info("\n" + "="*80)
        logger.info("Running Scaffold-Based Validation")
        logger.info("="*80)
        logger.info(f"Test size: {self.config.scaffold_test_size * 100}%")
        
        scaffold_results = {}
        
        # Split by scaffold
        train_idx, test_idx = self.validation_framework.scaffold_split(
            smiles_list=smiles_list,
            test_size=self.config.scaffold_test_size
        )
        
        logger.info(f"Train set: {len(train_idx)} compounds")
        logger.info(f"Test set: {len(test_idx)} compounds")
        
        if X is not None:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for model_name, model in models.items():
                logger.info(f"\nEvaluating {model_name}...")
                
                try:
                    # Train on scaffold train set
                    model.fit(X_train, y_train)
                    
                    # Predict on scaffold test set
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    scaffold_results[model_name] = {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'n_train': len(train_idx),
                        'n_test': len(test_idx)
                    }
                    
                    logger.info(f"  R²: {r2:.4f}")
                    logger.info(f"  RMSE: {rmse:.4f}")
                    logger.info(f"  MAE: {mae:.4f}")
                    
                except Exception as e:
                    logger.error(f"  Scaffold validation failed for {model_name}: {e}")
                    scaffold_results[model_name] = {'error': str(e)}
        
        # Save results
        if self.config.save_intermediate:
            output_path = self.output_dir / "scaffold_validation_results.json"
            with open(output_path, 'w') as f:
                json.dump(scaffold_results, f, indent=2)
            logger.info(f"\nScaffold validation results saved to: {output_path}")
        
        self.results.scaffold_validation_results = scaffold_results
        return scaffold_results
    
    def run_applicability_domain_analysis(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        smiles_train: Optional[List[str]] = None,
        smiles_test: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run applicability domain analysis.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix
            smiles_train: Optional training SMILES
            smiles_test: Optional test SMILES
            
        Returns:
            Dictionary of applicability domain results
        """
        logger.info("\n" + "="*80)
        logger.info("Running Applicability Domain Analysis")
        logger.info("="*80)
        
        # Fit applicability domain on training data
        self.applicability_domain.fit(X_train)
        
        # Calculate leverage for test set
        leverage = self.applicability_domain.calculate_leverage(X_test)
        
        # Determine threshold
        n, p = X_train.shape
        threshold = self.config.leverage_threshold_multiplier * p / n
        
        # Identify compounds in/out of domain
        in_domain = leverage <= threshold
        n_in_domain = np.sum(in_domain)
        n_out_domain = np.sum(~in_domain)
        
        logger.info(f"Leverage threshold: {threshold:.4f}")
        logger.info(f"Compounds in domain: {n_in_domain} ({n_in_domain/len(leverage)*100:.1f}%)")
        logger.info(f"Compounds out of domain: {n_out_domain} ({n_out_domain/len(leverage)*100:.1f}%)")
        
        # Calculate similarity if SMILES provided
        similarity_results = None
        if smiles_train is not None and smiles_test is not None:
            logger.info("\nCalculating Tanimoto similarity to training set...")
            similarities = []
            for test_smiles in smiles_test:
                try:
                    sim = self.applicability_domain.calculate_similarity(test_smiles, smiles_train)
                    similarities.append(sim)
                except:
                    similarities.append(0.0)
            
            similarities = np.array(similarities)
            logger.info(f"Mean similarity: {np.mean(similarities):.4f}")
            logger.info(f"Min similarity: {np.min(similarities):.4f}")
            logger.info(f"Max similarity: {np.max(similarities):.4f}")
            
            similarity_results = {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            }
        
        results = {
            'threshold': float(threshold),
            'n_in_domain': int(n_in_domain),
            'n_out_domain': int(n_out_domain),
            'fraction_in_domain': float(n_in_domain / len(leverage)),
            'leverage_mean': float(np.mean(leverage)),
            'leverage_std': float(np.std(leverage)),
            'leverage_max': float(np.max(leverage)),
            'similarity': similarity_results
        }
        
        # Save results
        if self.config.save_intermediate:
            output_path = self.output_dir / "applicability_domain_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nApplicability domain results saved to: {output_path}")
        
        self.results.applicability_domain_results = results
        return results
    
    def run_model_comparison(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run comprehensive model comparison with statistical testing.
        
        Args:
            models: Dictionary of trained models
            X_test: Test feature matrix
            y_test: Test target values
            
        Returns:
            Tuple of (comparison_df, statistical_comparison_df)
        """
        logger.info("\n" + "="*80)
        logger.info("Running Model Comparison")
        logger.info("="*80)
        
        # Evaluate all models
        evaluation_results = {}
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            try:
                result = self.benchmarker.evaluate_model(model, X_test, y_test)
                evaluation_results[model_name] = result
                logger.info(f"  R²: {result.r2:.4f}")
                logger.info(f"  RMSE: {result.rmse:.4f}")
                logger.info(f"  MAE: {result.mae:.4f}")
            except Exception as e:
                logger.error(f"  Evaluation failed: {e}")
        
        # Create comparison table
        comparison_df = self.benchmarker.compare_models(evaluation_results, X_test, y_test)
        
        # Statistical comparison
        logger.info("\nRunning statistical comparison...")
        statistical_df = self.benchmarker.statistical_comparison(
            evaluation_results,
            test=self.config.statistical_test,
            correction=self.config.multiple_comparison_correction
        )
        
        # Identify best model
        if len(comparison_df) > 0:
            best_idx = comparison_df['r2'].idxmax()
            self.results.best_model = comparison_df.loc[best_idx, 'model']
            self.results.best_model_r2 = comparison_df.loc[best_idx, 'r2']
            logger.info(f"\nBest model: {self.results.best_model} (R² = {self.results.best_model_r2:.4f})")
        
        # Save results
        if self.config.save_intermediate:
            comparison_df.to_csv(self.output_dir / "model_comparison.csv", index=False)
            statistical_df.to_csv(self.output_dir / "statistical_comparison.csv", index=False)
            logger.info(f"\nComparison results saved to: {self.output_dir}")
        
        self.results.model_comparison_results = comparison_df
        self.results.statistical_comparison_results = statistical_df
        
        return comparison_df, statistical_df
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comprehensive comparison table combining all validation results.
        
        Returns:
            DataFrame with all validation metrics per model
        """
        logger.info("\n" + "="*80)
        logger.info("Generating Comprehensive Comparison Table")
        logger.info("="*80)
        
        # Collect all model names
        model_names = set()
        if self.results.nested_cv_results:
            model_names.update(self.results.nested_cv_results.keys())
        if self.results.y_scrambling_results:
            model_names.update(self.results.y_scrambling_results.keys())
        if self.results.scaffold_validation_results:
            model_names.update(self.results.scaffold_validation_results.keys())
        
        # Build comparison table
        rows = []
        for model_name in sorted(model_names):
            row = {'model': model_name}
            
            # Nested CV metrics
            if self.results.nested_cv_results and model_name in self.results.nested_cv_results:
                cv_result = self.results.nested_cv_results[model_name]
                if 'error' not in cv_result:
                    row['cv_r2_mean'] = cv_result.get('r2_mean', np.nan)
                    row['cv_r2_std'] = cv_result.get('r2_std', np.nan)
                    row['cv_rmse_mean'] = cv_result.get('rmse_mean', np.nan)
                    row['cv_rmse_std'] = cv_result.get('rmse_std', np.nan)
            
            # Y-scrambling metrics
            if self.results.y_scrambling_results and model_name in self.results.y_scrambling_results:
                scrambling_result = self.results.y_scrambling_results[model_name]
                if 'error' not in scrambling_result:
                    row['original_r2'] = scrambling_result.get('original_r2', np.nan)
                    row['cr2p'] = scrambling_result.get('cr2p', np.nan)
                    row['y_scrambling_passed'] = scrambling_result.get('cr2p', 0) > self.config.y_scrambling_threshold
            
            # Scaffold validation metrics
            if self.results.scaffold_validation_results and model_name in self.results.scaffold_validation_results:
                scaffold_result = self.results.scaffold_validation_results[model_name]
                if 'error' not in scaffold_result:
                    row['scaffold_r2'] = scaffold_result.get('r2', np.nan)
                    row['scaffold_rmse'] = scaffold_result.get('rmse', np.nan)
            
            rows.append(row)
        
        comparison_table = pd.DataFrame(rows)
        
        # Save table
        output_path = self.output_dir / "comprehensive_comparison.csv"
        comparison_table.to_csv(output_path, index=False)
        logger.info(f"Comprehensive comparison table saved to: {output_path}")
        
        # Print summary
        logger.info("\nValidation Summary:")
        logger.info(comparison_table.to_string(index=False))
        
        return comparison_table
    
    def run_full_suite(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        smiles_list: Optional[List[str]] = None,
        smiles_train: Optional[List[str]] = None,
        smiles_test: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> ValidationSuiteResults:
        """
        Run the complete validation suite.
        
        Args:
            models: Dictionary of model instances
            X: Feature matrix (training data for CV)
            y: Target values (training data for CV)
            X_test: Optional test feature matrix
            y_test: Optional test target values
            smiles_list: Optional SMILES for scaffold validation
            smiles_train: Optional training SMILES for applicability domain
            smiles_test: Optional test SMILES for applicability domain
            feature_names: Optional feature names
            
        Returns:
            ValidationSuiteResults object
        """
        logger.info("="*80)
        logger.info("Starting Full Validation Suite")
        logger.info("="*80)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Number of models: {len(models)}")
        logger.info(f"Training samples: {len(y)}")
        if y_test is not None:
            logger.info(f"Test samples: {len(y_test)}")
        
        # 1. Nested Cross-Validation
        if self.config.run_nested_cv:
            self.run_nested_cv_validation(models, X, y, feature_names)
        
        # 2. Y-Scrambling Validation
        if self.config.run_y_scrambling:
            self.run_y_scrambling_validation(models, X, y)
        
        # 3. Scaffold Validation
        if self.config.run_scaffold_validation and smiles_list is not None:
            self.run_scaffold_validation(models, smiles_list, y, X)
        
        # 4. Applicability Domain Analysis
        if self.config.run_applicability_domain and X_test is not None:
            self.run_applicability_domain_analysis(X, X_test, smiles_train, smiles_test)
        
        # 5. Model Comparison
        if self.config.run_model_comparison and X_test is not None and y_test is not None:
            self.run_model_comparison(models, X_test, y_test)
        
        # 6. Generate comprehensive comparison table
        comparison_table = self.generate_comparison_table()
        
        # Determine if validation passed
        self.results.validation_passed = len(self.results.validation_warnings) == 0
        
        logger.info("\n" + "="*80)
        logger.info("Full Validation Suite Completed")
        logger.info("="*80)
        logger.info(f"Validation passed: {self.results.validation_passed}")
        if self.results.validation_warnings:
            logger.warning(f"Warnings: {len(self.results.validation_warnings)}")
            for warning in self.results.validation_warnings:
                logger.warning(f"  - {warning}")
        
        return self.results


def create_validation_suite(config: Optional[ValidationSuiteConfig] = None) -> FullValidationSuite:
    """
    Create a validation suite instance.
    
    Args:
        config: Validation suite configuration
        
    Returns:
        FullValidationSuite instance
    """
    return FullValidationSuite(config)
