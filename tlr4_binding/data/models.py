"""
Core data models for the data collection module.

This module defines the CompoundRecord dataclass for representing
TLR4 ligand compounds with binding affinity data.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class FunctionalClass(str, Enum):
    """Functional classification of TLR4 ligands."""
    AGONIST = "agonist"
    ANTAGONIST = "antagonist"
    UNKNOWN = "unknown"


class AffinitySource(str, Enum):
    """Source of binding affinity data."""
    CHEMBL = "ChEMBL"
    PUBCHEM = "PubChem"
    LITERATURE = "Literature"


@dataclass
class CompoundRecord:
    """
    Comprehensive compound record for TLR4 binding data.
    
    Represents a single compound with its binding affinity data,
    quality metrics, and functional classification.
    
    Attributes:
        smiles: Original SMILES string from data source
        canonical_smiles: Canonicalized SMILES representation
        binding_affinity: Binding free energy in kcal/mol
        affinity_source: Data source (ChEMBL, PubChem, Literature)
        functional_class: Agonist, antagonist, or unknown
        assay_type: Type of assay used to measure binding
        quality_score: Data quality score (0-1)
        is_pains: Whether compound matches PAINS patterns
        scaffold: Murcko scaffold SMILES
    """
    smiles: str
    canonical_smiles: str
    binding_affinity: float  # kcal/mol
    affinity_source: str = AffinitySource.CHEMBL.value
    functional_class: str = FunctionalClass.UNKNOWN.value
    assay_type: str = ""
    quality_score: float = 1.0
    is_pains: bool = False
    scaffold: str = ""
    
    # Additional metadata
    compound_id: Optional[str] = None
    compound_name: Optional[str] = None
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    
    # Original activity data before standardization
    original_value: Optional[float] = None
    original_unit: Optional[str] = None
    original_type: Optional[str] = None  # IC50, EC50, Ki, Kd
    
    # Conflict tracking for merged data
    has_conflict: bool = False
    conflict_sources: list = field(default_factory=list)
    
    def __post_init__(self):
        """Validate compound record after initialization."""
        # Validate functional class
        valid_classes = [fc.value for fc in FunctionalClass]
        if self.functional_class not in valid_classes:
            raise ValueError(
                f"Invalid functional_class: {self.functional_class}. "
                f"Must be one of: {valid_classes}"
            )
        
        # Validate affinity source
        valid_sources = [src.value for src in AffinitySource]
        if self.affinity_source not in valid_sources:
            raise ValueError(
                f"Invalid affinity_source: {self.affinity_source}. "
                f"Must be one of: {valid_sources}"
            )
        
        # Validate quality score
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                f"quality_score must be between 0 and 1, got {self.quality_score}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'smiles': self.smiles,
            'canonical_smiles': self.canonical_smiles,
            'binding_affinity': self.binding_affinity,
            'affinity_source': self.affinity_source,
            'functional_class': self.functional_class,
            'assay_type': self.assay_type,
            'quality_score': self.quality_score,
            'is_pains': self.is_pains,
            'scaffold': self.scaffold,
            'compound_id': self.compound_id,
            'compound_name': self.compound_name,
            'target_id': self.target_id,
            'target_name': self.target_name,
            'original_value': self.original_value,
            'original_unit': self.original_unit,
            'original_type': self.original_type,
            'has_conflict': self.has_conflict,
            'conflict_sources': self.conflict_sources,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompoundRecord':
        """Create instance from dictionary."""
        # Filter to valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if compound meets quality threshold."""
        return self.quality_score >= threshold and not self.is_pains
    
    def is_strong_binder(self, threshold: float = -7.0) -> bool:
        """Check if compound is a strong binder (more negative = stronger)."""
        return self.binding_affinity <= threshold
