"""
Interpretability module for TLR4 binding prediction models.

This module provides tools for extracting and visualizing model explanations:
- Attention weight extraction from GNN models
- Attention visualization overlaid on molecular structures
- SHAP analysis for traditional models
- Feature importance visualization

Requirements: 18.1, 18.2, 18.3, 18.4
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# RDKit imports with availability check
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Draw = None
    AllChem = None

# PyTorch imports with availability check
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Data = None

# Matplotlib imports with availability check
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    cm = None
    Normalize = None

# SHAP imports with availability check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


class InterpretabilityAnalyzer:
    """
    Analyzer for extracting and visualizing model explanations.
    
    Provides methods for:
    - Extracting attention weights from GNN models
    - Visualizing attention on molecular structures
    - Calculating SHAP values for traditional models
    - Plotting feature importance
    
    Requirements: 18.1, 18.2, 18.3, 18.4
    """
    
    def __init__(self):
        """Initialize the InterpretabilityAnalyzer."""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - molecular visualization will be limited")
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - plotting will be limited")
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - SHAP analysis will not be available")
        
        logger.info("InterpretabilityAnalyzer initialized")
    
    def extract_attention(
        self,
        model: Any,
        data: Union[Data, Any],
        smiles: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Extract attention weights mapped to atom indices for interpretability.
        
        Performs a forward pass with attention extraction enabled and aggregates
        attention weights across all layers and heads to produce per-atom importance.
        
        Args:
            model: TLR4GAT model with get_attention_weights method
            data: PyTorch Geometric Data object for a single molecule
            smiles: Optional SMILES string for validation
        
        Returns:
            Dictionary mapping atom index to aggregated attention weight
        
        Requirements: 18.1
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for attention extraction")
        
        # Check if model has get_attention_weights method
        if not hasattr(model, 'get_attention_weights'):
            raise ValueError(
                "Model must have get_attention_weights method for attention extraction"
            )
        
        # Extract attention weights using model's method
        try:
            attention_weights = model.get_attention_weights(data)
        except Exception as e:
            logger.error(f"Failed to extract attention weights: {e}")
            raise
        
        # Validate attention weights
        if not attention_weights:
            logger.warning("No attention weights extracted")
            return {}
        
        # Validate that attention weights sum to approximately 1.0
        total_attention = sum(attention_weights.values())
        if not (0.99 <= total_attention <= 1.01):
            logger.warning(
                f"Attention weights sum to {total_attention:.4f}, expected ~1.0"
            )
        
        logger.info(
            f"Extracted attention weights for {len(attention_weights)} atoms"
        )
        
        return attention_weights
    
    def visualize_attention(
        self,
        smiles: str,
        attention: Dict[int, float],
        output_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        highlight_threshold: float = 0.1,
        img_size: Tuple[int, int] = (600, 600)
    ) -> Optional[Any]:
        """
        Overlay attention weights on molecular structure.
        
        Creates a visualization of the molecule with atoms colored by their
        attention weights. High attention atoms are highlighted.
        
        Args:
            smiles: SMILES string of the molecule
            attention: Dictionary mapping atom index to attention weight
            output_path: Path to save the image (optional)
            title: Title for the visualization (optional)
            highlight_threshold: Minimum attention weight to highlight (default: 0.1)
            img_size: Image size as (width, height) tuple (default: (600, 600))
        
        Returns:
            PIL Image object if successful, None otherwise
        
        Requirements: 18.2, 18.4
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for molecular visualization")
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Generate 2D coordinates if not present
        if not mol.GetNumConformers():
            AllChem.Compute2DCoords(mol)
        
        # Validate attention weights match molecule
        num_atoms = mol.GetNumAtoms()
        if len(attention) != num_atoms:
            logger.warning(
                f"Attention weights ({len(attention)}) don't match "
                f"number of atoms ({num_atoms})"
            )
        
        # Prepare atom colors based on attention weights
        atom_colors = {}
        highlight_atoms = []
        
        # Use a colormap for attention weights
        max_attention = max(attention.values()) if attention else 1.0
        
        for atom_idx in range(num_atoms):
            attn_weight = attention.get(atom_idx, 0.0)
            
            # Normalize attention weight to [0, 1] for coloring
            normalized_attn = attn_weight / max_attention if max_attention > 0 else 0.0
            
            # Color: red (high attention) to blue (low attention)
            # RGB values: (1-normalized, 0, normalized) gives red->purple->blue
            r = normalized_attn
            g = 0.0
            b = 1.0 - normalized_attn
            atom_colors[atom_idx] = (r, g, b)
            
            # Highlight high-attention atoms
            if attn_weight >= highlight_threshold:
                highlight_atoms.append(atom_idx)
        
        # Create drawing options
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
        
        # Set drawing options
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        
        # Draw molecule with atom colors
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors
        )
        
        drawer.FinishDrawing()
        
        # Get image
        img_data = drawer.GetDrawingText()
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(img_data)
            logger.info(f"Attention visualization saved to {output_path}")
        
        # Convert to PIL Image for return
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_data))
            return img
        except ImportError:
            logger.warning("PIL not available - returning raw image data")
            return img_data
    
    def calculate_shap(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        background_samples: int = 100,
        max_evals: int = 1000
    ) -> np.ndarray:
        """
        Calculate SHAP values for feature importance in traditional models.
        
        Uses SHAP (SHapley Additive exPlanations) to compute feature contributions
        to model predictions. Works with scikit-learn style models.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix [n_samples, n_features]
            feature_names: Optional list of feature names
            background_samples: Number of background samples for SHAP (default: 100)
            max_evals: Maximum evaluations for SHAP computation (default: 1000)
        
        Returns:
            SHAP values array [n_samples, n_features]
        
        Requirements: 18.3
        """
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library is required for SHAP analysis")
        
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method for SHAP analysis")
        
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        logger.info(
            f"Computing SHAP values for {n_samples} samples with {n_features} features"
        )
        
        # Select background data (subset of X for efficiency)
        if n_samples > background_samples:
            background_indices = np.random.choice(
                n_samples, size=background_samples, replace=False
            )
            background_data = X[background_indices]
        else:
            background_data = X
        
        # Create SHAP explainer
        # Use KernelExplainer for model-agnostic explanations
        try:
            explainer = shap.KernelExplainer(
                model.predict,
                background_data,
                link="identity"
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(
                X,
                nsamples=min(max_evals, 2 * n_features + 2048)
            )
            
            # Convert to numpy array if needed
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values[0])
            
            logger.info(f"SHAP values computed: shape {shap_values.shape}")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            raise
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Feature Importance (SHAP)",
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[Any]:
        """
        Generate feature importance plot from SHAP values.
        
        Creates a bar plot showing the mean absolute SHAP value for each feature,
        indicating feature importance for model predictions.
        
        Args:
            shap_values: SHAP values array [n_samples, n_features]
            feature_names: List of feature names
            output_path: Path to save the plot (optional)
            title: Plot title (default: "Feature Importance (SHAP)")
            top_n: Number of top features to display (default: 20)
            figsize: Figure size as (width, height) tuple (default: (10, 8))
        
        Returns:
            Matplotlib figure object if successful, None otherwise
        
        Requirements: 18.3
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib is required for plotting")
        
        # Validate inputs
        if shap_values.ndim != 2:
            raise ValueError(f"shap_values must be 2D array, got shape {shap_values.shape}")
        
        n_samples, n_features = shap_values.shape
        
        if len(feature_names) != n_features:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) doesn't match "
                f"number of features ({n_features})"
            )
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        # Select top N features
        top_indices = sorted_indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = mean_abs_shap[top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importance, align='center', color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        return fig
    
    def create_shap_summary_plot(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
        plot_type: str = "dot",
        max_display: int = 20
    ) -> None:
        """
        Create SHAP summary plot showing feature effects.
        
        Args:
            shap_values: SHAP values array [n_samples, n_features]
            X: Feature matrix [n_samples, n_features]
            feature_names: List of feature names
            output_path: Path to save the plot (optional)
            plot_type: Type of plot ("dot", "bar", "violin") (default: "dot")
            max_display: Maximum number of features to display (default: 20)
        
        Requirements: 18.3
        """
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library is required for summary plots")
        
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib is required for plotting")
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {output_path}")
            plt.close()
        else:
            plt.show()
    
    def get_top_attention_atoms(
        self,
        attention: Dict[int, float],
        n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top N atoms by attention weight.
        
        Args:
            attention: Dictionary mapping atom index to attention weight
            n: Number of top atoms to return (default: 5)
        
        Returns:
            List of (atom_index, attention_weight) tuples sorted by weight
        
        Requirements: 18.2
        """
        sorted_atoms = sorted(
            attention.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_atoms[:n]
    
    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N features by mean absolute SHAP value.
        
        Args:
            shap_values: SHAP values array [n_samples, n_features]
            feature_names: List of feature names
            n: Number of top features to return (default: 10)
        
        Returns:
            List of (feature_name, mean_abs_shap) tuples sorted by importance
        
        Requirements: 18.3
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        top_features = [
            (feature_names[i], mean_abs_shap[i])
            for i in sorted_indices[:n]
        ]
        
        return top_features


def create_interpretability_analyzer() -> InterpretabilityAnalyzer:
    """
    Factory function to create an InterpretabilityAnalyzer.
    
    Returns:
        Configured InterpretabilityAnalyzer instance
    
    Requirements: 18.1, 18.2, 18.3
    """
    return InterpretabilityAnalyzer()
