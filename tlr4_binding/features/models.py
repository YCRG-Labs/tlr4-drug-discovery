"""
Core data models for the feature engineering module.

This module defines the MolecularFeatures dataclass for representing
comprehensive molecular descriptors including 2D, 3D, and graph features.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import numpy as np


@dataclass
class MolecularFeatures:
    """
    Comprehensive molecular feature representation for binding prediction.
    
    Contains 2D descriptors, 3D conformational descriptors, electrostatic
    properties, and graph-based representations for GNN input.
    
    Attributes:
        smiles: SMILES string of the molecule
        descriptors_2d: Dictionary of 2D molecular descriptors (~53 features)
        descriptors_3d: Dictionary of 3D conformational descriptors (~40-50 features)
        electrostatic: Dictionary of electrostatic/electronic properties
        graph: PyTorch Geometric Data object for GNN input (stored as dict for serialization)
    """
    smiles: str
    descriptors_2d: Dict[str, float] = field(default_factory=dict)
    descriptors_3d: Dict[str, float] = field(default_factory=dict)
    electrostatic: Dict[str, float] = field(default_factory=dict)
    graph: Optional[Dict[str, Any]] = None  # Serializable graph representation
    
    # Metadata
    compound_id: Optional[str] = None
    conformer_count: int = 0
    has_3d: bool = False
    calculation_errors: List[str] = field(default_factory=list)
    
    def get_all_descriptors(self) -> Dict[str, float]:
        """Get all descriptors as a single dictionary."""
        all_desc = {}
        all_desc.update(self.descriptors_2d)
        all_desc.update(self.descriptors_3d)
        all_desc.update(self.electrostatic)
        return all_desc
    
    def get_feature_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get feature values as numpy array.
        
        Args:
            feature_names: Optional list of feature names to include.
                          If None, includes all features.
        
        Returns:
            Numpy array of feature values
        """
        all_desc = self.get_all_descriptors()
        
        if feature_names is None:
            feature_names = sorted(all_desc.keys())
        
        return np.array([all_desc.get(name, np.nan) for name in feature_names])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'smiles': self.smiles,
            'descriptors_2d': self.descriptors_2d,
            'descriptors_3d': self.descriptors_3d,
            'electrostatic': self.electrostatic,
            'graph': self.graph,
            'compound_id': self.compound_id,
            'conformer_count': self.conformer_count,
            'has_3d': self.has_3d,
            'calculation_errors': self.calculation_errors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MolecularFeatures':
        """Create instance from dictionary."""
        return cls(
            smiles=data.get('smiles', ''),
            descriptors_2d=data.get('descriptors_2d', {}),
            descriptors_3d=data.get('descriptors_3d', {}),
            electrostatic=data.get('electrostatic', {}),
            graph=data.get('graph'),
            compound_id=data.get('compound_id'),
            conformer_count=data.get('conformer_count', 0),
            has_3d=data.get('has_3d', False),
            calculation_errors=data.get('calculation_errors', []),
        )
    
    def get_descriptor_count(self) -> Dict[str, int]:
        """Get count of descriptors by category."""
        return {
            '2d': len(self.descriptors_2d),
            '3d': len(self.descriptors_3d),
            'electrostatic': len(self.electrostatic),
            'total': len(self.get_all_descriptors()),
        }
    
    def has_missing_values(self) -> bool:
        """Check if any descriptor has NaN or None values."""
        all_desc = self.get_all_descriptors()
        for value in all_desc.values():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return True
        return False
    
    def get_missing_features(self) -> List[str]:
        """Get list of features with missing values."""
        missing = []
        all_desc = self.get_all_descriptors()
        for name, value in all_desc.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing.append(name)
        return missing
