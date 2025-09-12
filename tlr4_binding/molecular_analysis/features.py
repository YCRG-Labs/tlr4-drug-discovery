"""
Core data models and feature definitions for TLR4 binding prediction.

This module defines the fundamental data structures used throughout the pipeline
for representing molecular features, binding data, and prediction results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class MolecularFeatures:
    """
    Comprehensive molecular feature representation for binding prediction.
    
    Contains both 2D and 3D molecular descriptors extracted from PDBQT files
    and calculated using RDKit and PyMOL.
    """
    # Basic identification
    compound_name: str
    
    # 2D Molecular Descriptors (RDKit)
    molecular_weight: float
    logp: float  # Lipophilicity
    tpsa: float  # Topological Polar Surface Area
    rotatable_bonds: int
    hbd: int  # H-bond donors
    hba: int  # H-bond acceptors
    formal_charge: int
    
    # 3D Structural Descriptors (PyMOL)
    radius_of_gyration: float
    molecular_volume: float
    surface_area: float
    asphericity: float
    
    # Topological Features
    ring_count: int
    aromatic_rings: int
    branching_index: float
    
    # Electronic Properties
    dipole_moment: float
    polarizability: float
    
    # Additional metadata
    pdbqt_file: Optional[str] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'compound_name': self.compound_name,
            'molecular_weight': self.molecular_weight,
            'logp': self.logp,
            'tpsa': self.tpsa,
            'rotatable_bonds': self.rotatable_bonds,
            'hbd': self.hbd,
            'hba': self.hba,
            'formal_charge': self.formal_charge,
            'radius_of_gyration': self.radius_of_gyration,
            'molecular_volume': self.molecular_volume,
            'surface_area': self.surface_area,
            'asphericity': self.asphericity,
            'ring_count': self.ring_count,
            'aromatic_rings': self.aromatic_rings,
            'branching_index': self.branching_index,
            'dipole_moment': self.dipole_moment,
            'polarizability': self.polarizability,
            'pdbqt_file': self.pdbqt_file,
            'smiles': self.smiles,
            'inchi': self.inchi
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, int, str]]) -> 'MolecularFeatures':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class BindingData:
    """
    Binding affinity data from AutoDock Vina results.
    
    Represents the binding interaction between a ligand and TLR4 receptor
    with associated confidence metrics.
    """
    ligand: str
    mode: int
    affinity: float  # kcal/mol (lower values = stronger binding)
    rmsd_lb: float  # RMSD lower bound
    rmsd_ub: float  # RMSD upper bound
    
    # Additional binding metrics
    binding_energy: Optional[float] = None
    interaction_energy: Optional[float] = None
    vina_score: Optional[float] = None
    
    def is_strong_binding(self, threshold: float = -7.0) -> bool:
        """
        Check if binding is considered strong based on affinity threshold.
        
        Args:
            threshold: Affinity threshold in kcal/mol (default: -7.0)
            
        Returns:
            True if affinity is below threshold (stronger binding)
        """
        return self.affinity <= threshold
    
    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'ligand': self.ligand,
            'mode': self.mode,
            'affinity': self.affinity,
            'rmsd_lb': self.rmsd_lb,
            'rmsd_ub': self.rmsd_ub,
            'binding_energy': self.binding_energy,
            'interaction_energy': self.interaction_energy,
            'vina_score': self.vina_score
        }


@dataclass
class PredictionResult:
    """
    Prediction result with uncertainty quantification.
    
    Contains the predicted binding affinity along with confidence intervals
    and feature importance information.
    """
    compound_name: str
    predicted_affinity: float  # kcal/mol
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_used: str
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Additional prediction metadata
    prediction_uncertainty: Optional[float] = None
    model_confidence: Optional[float] = None
    prediction_timestamp: Optional[str] = None
    
    def is_strong_binding(self, threshold: float = -7.0) -> bool:
        """
        Check if predicted binding is strong based on affinity threshold.
        
        Args:
            threshold: Affinity threshold in kcal/mol (default: -7.0)
            
        Returns:
            True if predicted affinity is below threshold
        """
        return self.predicted_affinity <= threshold
    
    def get_confidence_interval_width(self) -> float:
        """Get the width of the confidence interval."""
        return self.confidence_interval_upper - self.confidence_interval_lower
    
    def to_dict(self) -> Dict[str, Union[float, str, Dict]]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'compound_name': self.compound_name,
            'predicted_affinity': self.predicted_affinity,
            'confidence_interval_lower': self.confidence_interval_lower,
            'confidence_interval_upper': self.confidence_interval_upper,
            'model_used': self.model_used,
            'feature_contributions': self.feature_contributions,
            'prediction_uncertainty': self.prediction_uncertainty,
            'model_confidence': self.model_confidence,
            'prediction_timestamp': self.prediction_timestamp
        }


class FeatureSet:
    """
    Container for molecular feature sets with validation and analysis methods.
    
    Provides utilities for feature validation, correlation analysis, and
    feature selection operations.
    """
    
    def __init__(self, features: List[MolecularFeatures]):
        """
        Initialize with list of molecular features.
        
        Args:
            features: List of MolecularFeatures objects
        """
        self.features = features
        self._dataframe = None
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get features as pandas DataFrame."""
        if self._dataframe is None:
            self._dataframe = pd.DataFrame([f.to_dict() for f in self.features])
        return self._dataframe
    
    def get_numerical_features(self) -> List[str]:
        """Get list of numerical feature column names."""
        numerical_cols = []
        for col in self.dataframe.columns:
            if col not in ['compound_name', 'pdbqt_file', 'smiles', 'inchi']:
                if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                    numerical_cols.append(col)
        return numerical_cols
    
    def validate_features(self) -> Dict[str, List[str]]:
        """
        Validate feature data quality.
        
        Returns:
            Dictionary with validation results and issues found
        """
        issues = {
            'missing_values': [],
            'infinite_values': [],
            'outliers': [],
            'invalid_ranges': []
        }
        
        df = self.dataframe
        numerical_cols = self.get_numerical_features()
        
        # Check for missing values
        for col in numerical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'].append(f"{col}: {missing_count} missing values")
        
        # Check for infinite values
        for col in numerical_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues['infinite_values'].append(f"{col}: {inf_count} infinite values")
        
        # Check for outliers (using IQR method)
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                issues['outliers'].append(f"{col}: {len(outliers)} outliers detected")
        
        return issues
    
    def get_feature_correlations(self, threshold: float = 0.95) -> pd.DataFrame:
        """
        Find highly correlated feature pairs.
        
        Args:
            threshold: Correlation threshold for flagging high correlations
            
        Returns:
            DataFrame with highly correlated feature pairs
        """
        df = self.dataframe[self.get_numerical_features()]
        corr_matrix = df.corr().abs()
        
        # Find pairs above threshold
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val >= threshold:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return pd.DataFrame(high_corr_pairs)
