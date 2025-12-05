"""
Core data models for the validation framework module.

This module defines the ValidationResult dataclass for representing
comprehensive model validation results including multiple validation strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


@dataclass
class ValidationResult:
    """
    Comprehensive validation result for model evaluation.
    
    Contains metrics from multiple validation strategies including
    external test set, nested CV, Y-scrambling, and scaffold validation.
    
    Attributes:
        model_name: Name of the evaluated model
        r2_test: R² on external test set
        rmse_test: RMSE on external test set
        mae_test: MAE on external test set
        r2_cv_mean: Mean R² from cross-validation
        r2_cv_std: Standard deviation of R² from cross-validation
        cr2p: Y-scrambling cR²p metric
        scaffold_r2: R² on scaffold-based validation
        n_in_domain: Number of test compounds within applicability domain
        n_out_domain: Number of test compounds outside applicability domain
    """
    model_name: str
    r2_test: float = 0.0
    rmse_test: float = 0.0
    mae_test: float = 0.0
    r2_cv_mean: float = 0.0
    r2_cv_std: float = 0.0
    cr2p: float = 0.0
    scaffold_r2: float = 0.0
    n_in_domain: int = 0
    n_out_domain: int = 0
    
    # Additional metrics
    pearson_r: Optional[float] = None
    spearman_rho: Optional[float] = None
    kendall_tau: Optional[float] = None
    
    # Nested CV details
    outer_fold_r2: List[float] = field(default_factory=list)
    inner_fold_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Y-scrambling details
    scrambled_r2_values: List[float] = field(default_factory=list)
    scrambled_r2_mean: Optional[float] = None
    scrambled_r2_std: Optional[float] = None
    is_potentially_overfit: bool = False
    
    # Scaffold validation details
    scaffold_fold_r2: List[float] = field(default_factory=list)
    n_unique_scaffolds: int = 0
    
    # Applicability domain details
    ad_coverage: float = 0.0  # Fraction of test set in domain
    ad_r2_in_domain: Optional[float] = None
    ad_r2_out_domain: Optional[float] = None
    
    # Statistical comparison
    p_value_vs_baseline: Optional[float] = None
    effect_size: Optional[float] = None
    
    # Metadata
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    validation_timestamp: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        # Calculate AD coverage
        total = self.n_in_domain + self.n_out_domain
        if total > 0:
            self.ad_coverage = self.n_in_domain / total
        
        # Check for potential overfitting based on cR²p
        if self.cr2p <= 0.5:
            self.is_potentially_overfit = True
    
    def is_valid_model(self, 
                       min_r2: float = 0.5,
                       min_cr2p: float = 0.5,
                       max_rmse: float = 2.0) -> bool:
        """
        Check if model meets minimum validation criteria.
        
        Args:
            min_r2: Minimum acceptable R² on test set
            min_cr2p: Minimum acceptable cR²p (Y-scrambling)
            max_rmse: Maximum acceptable RMSE
        
        Returns:
            True if model meets all criteria
        """
        return (
            self.r2_test >= min_r2 and
            self.cr2p >= min_cr2p and
            self.rmse_test <= max_rmse and
            not self.is_potentially_overfit
        )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of key metrics."""
        return {
            'r2_test': self.r2_test,
            'rmse_test': self.rmse_test,
            'mae_test': self.mae_test,
            'r2_cv_mean': self.r2_cv_mean,
            'r2_cv_std': self.r2_cv_std,
            'cr2p': self.cr2p,
            'scaffold_r2': self.scaffold_r2,
            'ad_coverage': self.ad_coverage,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'r2_test': self.r2_test,
            'rmse_test': self.rmse_test,
            'mae_test': self.mae_test,
            'r2_cv_mean': self.r2_cv_mean,
            'r2_cv_std': self.r2_cv_std,
            'cr2p': self.cr2p,
            'scaffold_r2': self.scaffold_r2,
            'n_in_domain': self.n_in_domain,
            'n_out_domain': self.n_out_domain,
            'pearson_r': self.pearson_r,
            'spearman_rho': self.spearman_rho,
            'kendall_tau': self.kendall_tau,
            'outer_fold_r2': self.outer_fold_r2,
            'inner_fold_results': self.inner_fold_results,
            'scrambled_r2_values': self.scrambled_r2_values,
            'scrambled_r2_mean': self.scrambled_r2_mean,
            'scrambled_r2_std': self.scrambled_r2_std,
            'is_potentially_overfit': self.is_potentially_overfit,
            'scaffold_fold_r2': self.scaffold_fold_r2,
            'n_unique_scaffolds': self.n_unique_scaffolds,
            'ad_coverage': self.ad_coverage,
            'ad_r2_in_domain': self.ad_r2_in_domain,
            'ad_r2_out_domain': self.ad_r2_out_domain,
            'p_value_vs_baseline': self.p_value_vs_baseline,
            'effect_size': self.effect_size,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_features': self.n_features,
            'validation_timestamp': self.validation_timestamp,
            'hyperparameters': self.hyperparameters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create instance from dictionary."""
        return cls(
            model_name=data.get('model_name', ''),
            r2_test=data.get('r2_test', 0.0),
            rmse_test=data.get('rmse_test', 0.0),
            mae_test=data.get('mae_test', 0.0),
            r2_cv_mean=data.get('r2_cv_mean', 0.0),
            r2_cv_std=data.get('r2_cv_std', 0.0),
            cr2p=data.get('cr2p', 0.0),
            scaffold_r2=data.get('scaffold_r2', 0.0),
            n_in_domain=data.get('n_in_domain', 0),
            n_out_domain=data.get('n_out_domain', 0),
            pearson_r=data.get('pearson_r'),
            spearman_rho=data.get('spearman_rho'),
            kendall_tau=data.get('kendall_tau'),
            outer_fold_r2=data.get('outer_fold_r2', []),
            inner_fold_results=data.get('inner_fold_results', []),
            scrambled_r2_values=data.get('scrambled_r2_values', []),
            scrambled_r2_mean=data.get('scrambled_r2_mean'),
            scrambled_r2_std=data.get('scrambled_r2_std'),
            is_potentially_overfit=data.get('is_potentially_overfit', False),
            scaffold_fold_r2=data.get('scaffold_fold_r2', []),
            n_unique_scaffolds=data.get('n_unique_scaffolds', 0),
            ad_coverage=data.get('ad_coverage', 0.0),
            ad_r2_in_domain=data.get('ad_r2_in_domain'),
            ad_r2_out_domain=data.get('ad_r2_out_domain'),
            p_value_vs_baseline=data.get('p_value_vs_baseline'),
            effect_size=data.get('effect_size'),
            n_train=data.get('n_train', 0),
            n_test=data.get('n_test', 0),
            n_features=data.get('n_features', 0),
            validation_timestamp=data.get('validation_timestamp'),
            hyperparameters=data.get('hyperparameters', {}),
        )
    
    def compare_to(self, other: 'ValidationResult') -> Dict[str, float]:
        """
        Compare this result to another validation result.
        
        Args:
            other: Another ValidationResult to compare against
        
        Returns:
            Dictionary of metric differences (self - other)
        """
        return {
            'r2_test_diff': self.r2_test - other.r2_test,
            'rmse_test_diff': self.rmse_test - other.rmse_test,
            'mae_test_diff': self.mae_test - other.mae_test,
            'cr2p_diff': self.cr2p - other.cr2p,
            'scaffold_r2_diff': self.scaffold_r2 - other.scaffold_r2,
        }
