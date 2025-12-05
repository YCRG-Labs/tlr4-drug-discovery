"""
Validation framework module for TLR4 binding prediction.

This module provides components for:
- Stratified data splitting
- Nested cross-validation
- Y-scrambling validation
- Applicability domain analysis
- Scaffold-based validation
- Model benchmarking and comparison
"""

from .models import ValidationResult
from .framework import ValidationFramework
from .applicability_domain import ApplicabilityDomainAnalyzer
from .benchmarker import (
    ModelBenchmarker,
    ModelEvaluationResult,
    ComparisonResult,
    AblationResult
)

__all__ = [
    "ValidationResult",
    "ValidationFramework",
    "ApplicabilityDomainAnalyzer",
    "ModelBenchmarker",
    "ModelEvaluationResult",
    "ComparisonResult",
    "AblationResult",
]
