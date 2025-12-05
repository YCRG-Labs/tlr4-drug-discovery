"""
Core data models for the machine learning models module.

This module defines the ModelPrediction dataclass for representing
model predictions with uncertainty quantification and interpretability data.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List


@dataclass
class ModelPrediction:
    """
    Prediction result with uncertainty quantification and interpretability.
    
    Contains the predicted binding affinity along with confidence intervals,
    applicability domain status, and attention weights for interpretability.
    
    Attributes:
        smiles: SMILES string of the predicted molecule
        predicted_affinity: Predicted binding free energy in kcal/mol
        confidence_interval: Tuple of (lower, upper) bounds
        functional_class: Predicted functional class (if multi-task model)
        functional_probability: Class probabilities for functional prediction
        in_applicability_domain: Whether compound is within model's AD
        leverage: Leverage value for AD assessment
        nearest_training_similarity: Max Tanimoto similarity to training set
        attention_weights: Atom-level attention weights from GNN
    """
    smiles: str
    predicted_affinity: float  # kcal/mol
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    functional_class: Optional[str] = None
    functional_probability: Optional[Dict[str, float]] = None
    in_applicability_domain: bool = True
    leverage: float = 0.0
    nearest_training_similarity: float = 0.0
    attention_weights: Optional[Dict[int, float]] = None
    
    # Model metadata
    model_name: str = ""
    model_version: str = ""
    prediction_timestamp: Optional[str] = None
    
    # Uncertainty metrics
    prediction_std: Optional[float] = None
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    
    # Feature contributions (for interpretability)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[Dict[str, float]] = None
    
    def get_confidence_interval_width(self) -> float:
        """Get the width of the confidence interval."""
        return self.confidence_interval[1] - self.confidence_interval[0]
    
    def is_reliable_prediction(self, 
                               max_ci_width: float = 2.0,
                               min_similarity: float = 0.3) -> bool:
        """
        Check if prediction is considered reliable.
        
        Args:
            max_ci_width: Maximum acceptable confidence interval width
            min_similarity: Minimum similarity to training set
        
        Returns:
            True if prediction meets reliability criteria
        """
        ci_ok = self.get_confidence_interval_width() <= max_ci_width
        ad_ok = self.in_applicability_domain
        sim_ok = self.nearest_training_similarity >= min_similarity
        return ci_ok and ad_ok and sim_ok
    
    def is_strong_binder(self, threshold: float = -7.0) -> bool:
        """Check if predicted binding is strong."""
        return self.predicted_affinity <= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'smiles': self.smiles,
            'predicted_affinity': self.predicted_affinity,
            'confidence_interval': self.confidence_interval,
            'functional_class': self.functional_class,
            'functional_probability': self.functional_probability,
            'in_applicability_domain': self.in_applicability_domain,
            'leverage': self.leverage,
            'nearest_training_similarity': self.nearest_training_similarity,
            'attention_weights': self.attention_weights,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction_timestamp': self.prediction_timestamp,
            'prediction_std': self.prediction_std,
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'aleatoric_uncertainty': self.aleatoric_uncertainty,
            'feature_contributions': self.feature_contributions,
            'shap_values': self.shap_values,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPrediction':
        """Create instance from dictionary."""
        # Handle confidence_interval as tuple
        ci = data.get('confidence_interval', (0.0, 0.0))
        if isinstance(ci, list):
            ci = tuple(ci)
        
        return cls(
            smiles=data.get('smiles', ''),
            predicted_affinity=data.get('predicted_affinity', 0.0),
            confidence_interval=ci,
            functional_class=data.get('functional_class'),
            functional_probability=data.get('functional_probability'),
            in_applicability_domain=data.get('in_applicability_domain', True),
            leverage=data.get('leverage', 0.0),
            nearest_training_similarity=data.get('nearest_training_similarity', 0.0),
            attention_weights=data.get('attention_weights'),
            model_name=data.get('model_name', ''),
            model_version=data.get('model_version', ''),
            prediction_timestamp=data.get('prediction_timestamp'),
            prediction_std=data.get('prediction_std'),
            epistemic_uncertainty=data.get('epistemic_uncertainty'),
            aleatoric_uncertainty=data.get('aleatoric_uncertainty'),
            feature_contributions=data.get('feature_contributions', {}),
            shap_values=data.get('shap_values'),
        )
    
    def get_top_contributing_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N contributing features by absolute value."""
        if not self.feature_contributions:
            return []
        
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]
    
    def get_top_attention_atoms(self, n: int = 5) -> List[Tuple[int, float]]:
        """Get top N atoms by attention weight."""
        if not self.attention_weights:
            return []
        
        sorted_atoms = sorted(
            self.attention_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_atoms[:n]
