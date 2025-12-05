"""
Feature engineering module for TLR4 binding prediction.

This module provides components for:
- 2D molecular descriptor calculation
- 3D conformational descriptor calculation
- Electrostatic property calculation
- Graph-based molecular representations
"""

from .models import MolecularFeatures

__all__ = [
    "MolecularFeatures",
]

# Conditionally import Descriptor3DCalculator (requires RDKit)
try:
    from .descriptor_3d import Descriptor3DCalculator, RDKIT_AVAILABLE
    __all__.append("Descriptor3DCalculator")
except ImportError:
    Descriptor3DCalculator = None
    RDKIT_AVAILABLE = False

# Conditionally import ElectrostaticCalculator (requires RDKit)
try:
    from .electrostatic import ElectrostaticCalculator
    __all__.append("ElectrostaticCalculator")
except ImportError:
    ElectrostaticCalculator = None

# Conditionally import MolecularGraphGenerator (requires RDKit, PyTorch, PyTorch Geometric)
try:
    from .graph_generator import MolecularGraphGenerator
    __all__.append("MolecularGraphGenerator")
except ImportError:
    MolecularGraphGenerator = None
