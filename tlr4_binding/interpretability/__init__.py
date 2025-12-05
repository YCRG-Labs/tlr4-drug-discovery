"""
Interpretability module for TLR4 binding prediction models.

This module provides tools for extracting and visualizing model explanations:
- Attention weight extraction from GNN models
- Attention visualization overlaid on molecular structures
- SHAP analysis for traditional models
- Feature importance visualization

Requirements: 18.1, 18.2, 18.3, 18.4
"""

from .analyzer import (
    InterpretabilityAnalyzer,
    create_interpretability_analyzer,
)

__all__ = [
    "InterpretabilityAnalyzer",
    "create_interpretability_analyzer",
]
