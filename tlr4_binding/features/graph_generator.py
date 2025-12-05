"""
Molecular Graph Generator for TLR4 binding prediction.

This module provides the MolecularGraphGenerator class for converting molecules
to graph representations compatible with PyTorch Geometric for GNN input.

Requirements: 7.1, 7.2, 7.3
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

logger = logging.getLogger(__name__)

# RDKit imports with availability check
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

# PyTorch and PyTorch Geometric imports with availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None

# Type checking imports
if TYPE_CHECKING:
    from rdkit.Chem import Mol, Atom, Bond


# Atom feature constants
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # Atomic numbers 1-118
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP if RDKIT_AVAILABLE else 0,
        Chem.rdchem.HybridizationType.SP2 if RDKIT_AVAILABLE else 1,
        Chem.rdchem.HybridizationType.SP3 if RDKIT_AVAILABLE else 2,
        Chem.rdchem.HybridizationType.SP3D if RDKIT_AVAILABLE else 3,
        Chem.rdchem.HybridizationType.SP3D2 if RDKIT_AVAILABLE else 4,
        Chem.rdchem.HybridizationType.UNSPECIFIED if RDKIT_AVAILABLE else 5,
    ],
    'num_hs': [0, 1, 2, 3, 4],
    'num_radical_electrons': [0, 1, 2],
}

# Bond feature constants
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE if RDKIT_AVAILABLE else 0,
        Chem.rdchem.BondType.DOUBLE if RDKIT_AVAILABLE else 1,
        Chem.rdchem.BondType.TRIPLE if RDKIT_AVAILABLE else 2,
        Chem.rdchem.BondType.AROMATIC if RDKIT_AVAILABLE else 3,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE if RDKIT_AVAILABLE else 0,
        Chem.rdchem.BondStereo.STEREOZ if RDKIT_AVAILABLE else 1,
        Chem.rdchem.BondStereo.STEREOE if RDKIT_AVAILABLE else 2,
        Chem.rdchem.BondStereo.STEREOCIS if RDKIT_AVAILABLE else 3,
        Chem.rdchem.BondStereo.STEREOTRANS if RDKIT_AVAILABLE else 4,
    ],
}


def one_hot_encode(value: Any, allowable_set: List[Any], include_unknown: bool = True) -> List[int]:
    """
    One-hot encode a value based on an allowable set.
    
    Args:
        value: Value to encode
        allowable_set: List of allowable values
        include_unknown: Whether to include an unknown category
    
    Returns:
        One-hot encoded list
    """
    if include_unknown:
        if value not in allowable_set:
            return [0] * len(allowable_set) + [1]
        return [1 if v == value else 0 for v in allowable_set] + [0]
    else:
        if value not in allowable_set:
            raise ValueError(f"Value {value} not in allowable set {allowable_set}")
        return [1 if v == value else 0 for v in allowable_set]


class MolecularGraphGenerator:
    """
    Generator for molecular graph representations compatible with PyTorch Geometric.
    
    Converts SMILES strings to graph Data objects with atom features as nodes
    and bond features as edges, suitable for Graph Neural Network input.
    
    Attributes:
        include_hydrogens: Whether to include hydrogen atoms in the graph
        use_chirality: Whether to include chirality information in features
    """
    
    def __init__(
        self,
        include_hydrogens: bool = False,
        use_chirality: bool = True
    ):
        """
        Initialize the molecular graph generator.
        
        Args:
            include_hydrogens: Whether to include explicit hydrogens (default: False)
            use_chirality: Whether to include chirality in atom features (default: True)
        """
        self.include_hydrogens = include_hydrogens
        self.use_chirality = use_chirality
    
    def get_atom_features(self, atom: Any) -> List[float]:
        """
        Extract node features from an RDKit atom.
        
        Features include:
        - Atomic number (one-hot encoded)
        - Degree (one-hot encoded)
        - Formal charge (one-hot encoded)
        - Hybridization (one-hot encoded)
        - Aromaticity (binary)
        - Ring membership (binary)
        - Number of hydrogens (one-hot encoded)
        - Chirality (if enabled)
        
        Args:
            atom: RDKit Atom object
        
        Returns:
            List of float features for the atom
        
        Requirements: 7.1
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for atom feature extraction")
        
        features = []
        
        # Atomic number (one-hot, 118 elements + unknown)
        features.extend(one_hot_encode(
            atom.GetAtomicNum(),
            list(range(1, 119)),
            include_unknown=True
        ))
        
        # Degree (one-hot, 0-6 + unknown)
        features.extend(one_hot_encode(
            atom.GetDegree(),
            [0, 1, 2, 3, 4, 5, 6],
            include_unknown=True
        ))
        
        # Formal charge (one-hot, -3 to +3 + unknown)
        features.extend(one_hot_encode(
            atom.GetFormalCharge(),
            [-3, -2, -1, 0, 1, 2, 3],
            include_unknown=True
        ))
        
        # Hybridization (one-hot)
        features.extend(one_hot_encode(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
            include_unknown=True
        ))
        
        # Aromaticity (binary)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # Ring membership (binary)
        features.append(1.0 if atom.IsInRing() else 0.0)
        
        # Number of hydrogens (one-hot, 0-4 + unknown)
        features.extend(one_hot_encode(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4],
            include_unknown=True
        ))
        
        # Chirality (if enabled)
        if self.use_chirality:
            try:
                chirality = atom.GetChiralTag()
                features.extend(one_hot_encode(
                    chirality,
                    [
                        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                        Chem.rdchem.ChiralType.CHI_OTHER,
                    ],
                    include_unknown=True
                ))
            except Exception:
                # Default to unspecified if chirality extraction fails
                features.extend([1, 0, 0, 0, 0])
        
        return features
    
    def get_bond_features(self, bond: Any) -> List[float]:
        """
        Extract edge features from an RDKit bond.
        
        Features include:
        - Bond type (one-hot: single, double, triple, aromatic)
        - Conjugation (binary)
        - Ring membership (binary)
        - Stereo configuration (one-hot)
        
        Args:
            bond: RDKit Bond object
        
        Returns:
            List of float features for the bond
        
        Requirements: 7.2
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for bond feature extraction")
        
        features = []
        
        # Bond type (one-hot)
        features.extend(one_hot_encode(
            bond.GetBondType(),
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
            include_unknown=True
        ))
        
        # Conjugation (binary)
        features.append(1.0 if bond.GetIsConjugated() else 0.0)
        
        # Ring membership (binary)
        features.append(1.0 if bond.IsInRing() else 0.0)
        
        # Stereo configuration (one-hot)
        features.extend(one_hot_encode(
            bond.GetStereo(),
            [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOTRANS,
            ],
            include_unknown=True
        ))
        
        return features
    
    def mol_to_graph(self, smiles: str) -> Any:
        """
        Convert a SMILES string to a PyTorch Geometric Data object.
        
        Creates a graph representation where:
        - Nodes represent atoms with features (atomic number, degree, charge, etc.)
        - Edges represent bonds with features (bond type, conjugation, ring membership)
        - The adjacency matrix is symmetric (undirected graph)
        
        Args:
            smiles: SMILES string of the molecule
        
        Returns:
            PyTorch Geometric Data object with:
                - x: Node feature matrix [num_nodes, num_node_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge feature matrix [num_edges, num_edge_features]
                - smiles: Original SMILES string
                - num_nodes: Number of atoms
        
        Raises:
            ValueError: If SMILES string is invalid
            RuntimeError: If required libraries are not available
        
        Requirements: 7.1, 7.2, 7.3
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for molecular graph generation")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for molecular graph generation")
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is required for molecular graph generation")
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add hydrogens if requested
        if self.include_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Get atom features (node features)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        # Convert to tensor
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Get bond features (edge features) and edge indices
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_feat = self.get_bond_features(bond)
            
            # Add both directions for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        # Handle molecules with no bonds (single atoms)
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(self.get_bond_features(None)) if mol.GetNumBonds() > 0 else 13), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles,
            num_nodes=mol.GetNumAtoms()
        )
        
        return data

    def _get_empty_bond_features_size(self) -> int:
        """Get the size of bond features for empty edge cases."""
        # Bond type (5) + conjugation (1) + ring (1) + stereo (6) = 13
        return 13
    
    def create_dataset(
        self,
        smiles_list: List[str],
        labels: Optional[np.ndarray] = None
    ) -> List[Any]:
        """
        Create a dataset of graph representations for batch processing.
        
        Converts a list of SMILES strings to PyTorch Geometric Data objects,
        optionally including target labels for supervised learning.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Optional numpy array of target values (e.g., binding affinities)
        
        Returns:
            List of PyTorch Geometric Data objects, each with:
                - x: Node feature matrix
                - edge_index: Edge connectivity
                - edge_attr: Edge feature matrix
                - y: Target value (if labels provided)
                - smiles: Original SMILES string
        
        Raises:
            RuntimeError: If required libraries are not available
        
        Requirements: 7.3
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dataset creation")
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is required for dataset creation")
        
        dataset = []
        failed_count = 0
        
        for idx, smiles in enumerate(smiles_list):
            try:
                data = self.mol_to_graph(smiles)
                
                # Add target label if provided
                if labels is not None:
                    data.y = torch.tensor([labels[idx]], dtype=torch.float)
                
                dataset.append(data)
                
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to convert SMILES at index {idx}: {smiles}. Error: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            logger.info(f"Successfully converted {len(dataset)}/{len(smiles_list)} molecules. "
                       f"Failed: {failed_count}")
        else:
            logger.debug(f"Successfully converted all {len(dataset)} molecules to graphs")
        
        return dataset
    
    def get_node_feature_dim(self) -> int:
        """
        Get the dimension of node features.
        
        Returns:
            Integer dimension of node feature vector
        """
        # Atomic number: 119 (118 + unknown)
        # Degree: 8 (7 + unknown)
        # Formal charge: 8 (7 + unknown)
        # Hybridization: 6 (5 + unknown)
        # Aromaticity: 1
        # Ring membership: 1
        # Num hydrogens: 6 (5 + unknown)
        # Chirality (if enabled): 5 (4 + unknown)
        
        dim = 119 + 8 + 8 + 6 + 1 + 1 + 6
        if self.use_chirality:
            dim += 5
        return dim
    
    def get_edge_feature_dim(self) -> int:
        """
        Get the dimension of edge features.
        
        Returns:
            Integer dimension of edge feature vector
        """
        # Bond type: 5 (4 + unknown)
        # Conjugation: 1
        # Ring membership: 1
        # Stereo: 6 (5 + unknown)
        return 5 + 1 + 1 + 6
    
    def to_dict(self, data: Any) -> Dict[str, Any]:
        """
        Convert a PyTorch Geometric Data object to a serializable dictionary.
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            Dictionary representation of the graph
        """
        result = {
            'x': data.x.numpy().tolist() if hasattr(data.x, 'numpy') else data.x,
            'edge_index': data.edge_index.numpy().tolist() if hasattr(data.edge_index, 'numpy') else data.edge_index,
            'edge_attr': data.edge_attr.numpy().tolist() if hasattr(data.edge_attr, 'numpy') else data.edge_attr,
            'smiles': data.smiles if hasattr(data, 'smiles') else None,
            'num_nodes': data.num_nodes if hasattr(data, 'num_nodes') else None,
        }
        
        if hasattr(data, 'y') and data.y is not None:
            result['y'] = data.y.numpy().tolist() if hasattr(data.y, 'numpy') else data.y
        
        return result
    
    def from_dict(self, data_dict: Dict[str, Any]) -> Any:
        """
        Create a PyTorch Geometric Data object from a dictionary.
        
        Args:
            data_dict: Dictionary representation of the graph
        
        Returns:
            PyTorch Geometric Data object
        """
        if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("PyTorch and PyTorch Geometric are required")
        
        data = Data(
            x=torch.tensor(data_dict['x'], dtype=torch.float),
            edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(data_dict['edge_attr'], dtype=torch.float),
            smiles=data_dict.get('smiles'),
            num_nodes=data_dict.get('num_nodes'),
        )
        
        if 'y' in data_dict and data_dict['y'] is not None:
            data.y = torch.tensor(data_dict['y'], dtype=torch.float)
        
        return data
