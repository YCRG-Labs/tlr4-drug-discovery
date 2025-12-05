"""
Data collection and processing module for TLR4 binding prediction.

This module provides components for:
- Collecting binding data from ChEMBL and PubChem
- Standardizing activity values
- Quality control and filtering
- Functional classification of compounds
"""

from .models import CompoundRecord, FunctionalClass, AffinitySource
from .collector import (
    DataCollector,
    ActivityMeasurement,
    kd_to_delta_g,
    delta_g_to_kd,
    ic50_to_ki,
    TLR_TARGETS,
    PUBCHEM_ASSAYS,
    R,
    T,
)
from .quality_control import QualityController
from .functional_classifier import (
    FunctionalClassifier,
    FunctionalEvidence,
    AssayPriority,
)

__all__ = [
    "CompoundRecord",
    "FunctionalClass",
    "AffinitySource",
    "DataCollector",
    "ActivityMeasurement",
    "kd_to_delta_g",
    "delta_g_to_kd",
    "ic50_to_ki",
    "TLR_TARGETS",
    "PUBCHEM_ASSAYS",
    "R",
    "T",
    "QualityController",
    "FunctionalClassifier",
    "FunctionalEvidence",
    "AssayPriority",
]
