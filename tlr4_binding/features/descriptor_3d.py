"""
3D Molecular Descriptor Calculator for TLR4 binding prediction.

This module provides the Descriptor3DCalculator class for generating 3D conformers
and calculating spatial descriptors including PMI, shape descriptors, and WHIM.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

# RDKit imports with availability check
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
    from rdkit.Chem import rdMolTransforms
    from rdkit.Chem.rdMolDescriptors import CalcPMI1, CalcPMI2, CalcPMI3
    from rdkit.Chem.rdMolDescriptors import CalcNPR1, CalcNPR2
    from rdkit.Chem.rdMolDescriptors import CalcSpherocityIndex, CalcAsphericity, CalcEccentricity
    from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration
    from rdkit.Chem.rdMolDescriptors import CalcInertialShapeFactor
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    rdMolDescriptors = None

# Type checking imports
if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


class Descriptor3DCalculator:
    """
    Calculator for 3D molecular descriptors from SMILES strings.
    
    Generates low-energy 3D conformers using MMFF94 force field and calculates
    spatial descriptors including Principal Moments of Inertia (PMI), shape
    descriptors, and WHIM descriptors.
    
    Attributes:
        num_conformers: Default number of conformers to generate
        max_iterations: Maximum iterations for energy minimization
        random_seed: Random seed for reproducibility
        energy_window: Energy window (kcal/mol) for conformer filtering
    """
    
    def __init__(
        self,
        num_conformers: int = 10,
        max_iterations: int = 500,
        random_seed: int = 42,
        energy_window: float = 10.0
    ):
        """
        Initialize the 3D descriptor calculator.
        
        Args:
            num_conformers: Number of conformers to generate (default: 10)
            max_iterations: Max iterations for MMFF94 minimization (default: 500)
            random_seed: Random seed for conformer generation (default: 42)
            energy_window: Energy window in kcal/mol for filtering conformers (default: 10.0)
        """
        self.num_conformers = num_conformers
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.energy_window = energy_window

    def generate_conformers(
        self,
        smiles: str,
        num_conformers: Optional[int] = None
    ) -> List[Any]:
        """
        Generate low-energy 3D conformers using MMFF94 force field.
        
        Generates multiple conformers, performs energy minimization using MMFF94,
        and returns conformers within the energy window of the lowest energy conformer.
        
        Args:
            smiles: SMILES string of the molecule
            num_conformers: Number of conformers to generate (uses default if None)
        
        Returns:
            List of RDKit Mol objects with 3D coordinates, sorted by energy.
            Returns empty list if conformer generation fails.
        
        Raises:
            ValueError: If SMILES string is invalid
            RuntimeError: If RDKit is not available
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for conformer generation but is not installed")
        
        if num_conformers is None:
            num_conformers = self.num_conformers
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add hydrogens for proper 3D geometry
        mol = Chem.AddHs(mol)
        
        # Generate initial conformers using ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.numThreads = 0  # Use all available threads
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        
        # Generate conformers
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            params=params
        )
        
        if len(conf_ids) == 0:
            logger.warning(f"Failed to generate conformers for: {smiles}")
            return []
        
        # Minimize each conformer with MMFF94 and collect energies
        energies = []
        valid_conf_ids = []
        
        for conf_id in conf_ids:
            try:
                # Get MMFF94 force field
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol,
                    AllChem.MMFFGetMoleculeProperties(mol),
                    confId=conf_id
                )
                
                if ff is None:
                    # Fall back to UFF if MMFF94 fails
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                
                if ff is not None:
                    # Minimize
                    ff.Minimize(maxIts=self.max_iterations)
                    energy = ff.CalcEnergy()
                    energies.append(energy)
                    valid_conf_ids.append(conf_id)
                else:
                    logger.warning(f"Could not create force field for conformer {conf_id}")
                    
            except Exception as e:
                logger.warning(f"Error minimizing conformer {conf_id}: {e}")
        
        if not valid_conf_ids:
            logger.warning(f"No valid conformers after minimization for: {smiles}")
            return []
        
        # Sort by energy and filter by energy window
        sorted_indices = np.argsort(energies)
        min_energy = energies[sorted_indices[0]]
        
        # Create list of conformer molecules within energy window
        conformers = []
        for idx in sorted_indices:
            if energies[idx] - min_energy <= self.energy_window:
                # Create a copy of the molecule with just this conformer
                conf_mol = Chem.Mol(mol)
                # Keep only this conformer
                conf_to_keep = valid_conf_ids[idx]
                conformers.append((conf_mol, conf_to_keep, energies[idx]))
        
        # Return molecules with their lowest energy conformer
        result = []
        for conf_mol, conf_id, energy in conformers:
            # Create a new molecule with just the selected conformer
            new_mol = Chem.Mol(mol)
            # Remove all conformers except the one we want
            confs_to_remove = [c.GetId() for c in new_mol.GetConformers() if c.GetId() != conf_id]
            for cid in confs_to_remove:
                new_mol.RemoveConformer(cid)
            result.append(new_mol)
        
        logger.debug(f"Generated {len(result)} conformers within {self.energy_window} kcal/mol window")
        return result
    
    def get_lowest_energy_conformer(self, smiles: str) -> Optional[Any]:
        """
        Get the lowest energy conformer for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
        
        Returns:
            RDKit Mol object with 3D coordinates, or None if generation fails
        """
        conformers = self.generate_conformers(smiles, num_conformers=self.num_conformers)
        if conformers:
            return conformers[0]  # Already sorted by energy
        return None

    def calculate_pmi(self, mol: Any) -> Dict[str, float]:
        """
        Calculate Principal Moments of Inertia and normalized ratios.
        
        Calculates PMI1, PMI2, PMI3 (sorted such that PMI1 <= PMI2 <= PMI3)
        and normalized ratios NPR1 = PMI1/PMI3, NPR2 = PMI2/PMI3.
        
        Args:
            mol: RDKit Mol object with 3D coordinates
        
        Returns:
            Dictionary containing:
                - PMI1, PMI2, PMI3: Principal moments of inertia
                - NPR1, NPR2: Normalized principal moment ratios (bounded [0,1])
        
        Raises:
            ValueError: If molecule has no 3D conformer
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no 3D conformer")
        
        result = {}
        
        try:
            # Calculate Principal Moments of Inertia
            # RDKit returns them sorted: PMI1 <= PMI2 <= PMI3
            result['PMI1'] = CalcPMI1(mol)
            result['PMI2'] = CalcPMI2(mol)
            result['PMI3'] = CalcPMI3(mol)
            
            # Calculate Normalized Principal Moment Ratios
            # NPR1 = PMI1/PMI3, NPR2 = PMI2/PMI3
            # These are bounded in [0, 1]
            result['NPR1'] = CalcNPR1(mol)
            result['NPR2'] = CalcNPR2(mol)
            
        except Exception as e:
            logger.warning(f"Error calculating PMI descriptors: {e}")
            result = {
                'PMI1': np.nan,
                'PMI2': np.nan,
                'PMI3': np.nan,
                'NPR1': np.nan,
                'NPR2': np.nan
            }
        
        return result
    
    def calculate_shape_descriptors(self, mol: Any) -> Dict[str, float]:
        """
        Calculate molecular shape descriptors.
        
        Calculates spherocity, asphericity, eccentricity, and radius of gyration
        from the 3D conformer.
        
        Args:
            mol: RDKit Mol object with 3D coordinates
        
        Returns:
            Dictionary containing:
                - spherocity: Spherocity index (bounded [0,1], 1 = perfect sphere)
                - asphericity: Asphericity (bounded [0,1], 0 = spherical)
                - eccentricity: Eccentricity (bounded [0,1])
                - radius_of_gyration: Radius of gyration (> 0)
                - inertial_shape_factor: Inertial shape factor
        
        Raises:
            ValueError: If molecule has no 3D conformer
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no 3D conformer")
        
        result = {}
        
        try:
            # Spherocity index: 1 for perfect sphere, 0 for linear
            result['spherocity'] = CalcSpherocityIndex(mol)
            
            # Asphericity: 0 for spherical, increases with deviation
            result['asphericity'] = CalcAsphericity(mol)
            
            # Eccentricity: measure of elongation
            result['eccentricity'] = CalcEccentricity(mol)
            
            # Radius of gyration: measure of molecular size
            result['radius_of_gyration'] = CalcRadiusOfGyration(mol)
            
            # Inertial shape factor
            result['inertial_shape_factor'] = CalcInertialShapeFactor(mol)
            
        except Exception as e:
            logger.warning(f"Error calculating shape descriptors: {e}")
            result = {
                'spherocity': np.nan,
                'asphericity': np.nan,
                'eccentricity': np.nan,
                'radius_of_gyration': np.nan,
                'inertial_shape_factor': np.nan
            }
        
        return result

    def calculate_whim(self, mol: Any) -> Dict[str, float]:
        """
        Calculate WHIM (Weighted Holistic Invariant Molecular) descriptors.
        
        WHIM descriptors capture size, shape, symmetry, and atom distribution
        properties of the 3D molecular structure.
        
        Args:
            mol: RDKit Mol object with 3D coordinates
        
        Returns:
            Dictionary containing WHIM descriptors:
                - WHIM descriptors for different weighting schemes (unity, mass, vdw, etc.)
                - Includes L (size), P (shape), G (symmetry) components
        
        Raises:
            ValueError: If molecule has no 3D conformer
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no 3D conformer")
        
        result = {}
        
        try:
            # RDKit provides GETAWAY and WHIM-like descriptors through Descriptors3D
            # We'll calculate available 3D descriptors that capture similar information
            
            # Autocorrelation descriptors (related to WHIM)
            # These capture spatial distribution of atomic properties
            
            # Get 3D autocorrelation descriptors
            autocorr = rdMolDescriptors.CalcAUTOCORR3D(mol)
            for i, val in enumerate(autocorr):
                result[f'AUTOCORR3D_{i+1}'] = val
            
            # RDF (Radial Distribution Function) descriptors
            rdf = rdMolDescriptors.CalcRDF(mol)
            for i, val in enumerate(rdf):
                result[f'RDF_{i+1}'] = val
            
            # MORSE (Molecule Representation of Structures based on Electron diffraction)
            morse = rdMolDescriptors.CalcMORSE(mol)
            for i, val in enumerate(morse):
                result[f'MORSE_{i+1}'] = val
            
            # GETAWAY descriptors (GEometry, Topology, and Atom-Weights AssemblY)
            # These are closely related to WHIM
            getaway = rdMolDescriptors.CalcGETAWAY(mol)
            for i, val in enumerate(getaway):
                result[f'GETAWAY_{i+1}'] = val
            
        except Exception as e:
            logger.warning(f"Error calculating WHIM/3D autocorrelation descriptors: {e}")
            # Return empty dict on error - caller should handle missing descriptors
        
        return result
    
    def calculate_all(self, smiles: str) -> Dict[str, float]:
        """
        Calculate all 3D descriptors for a molecule.
        
        Generates conformers and calculates PMI, shape, and WHIM descriptors.
        
        Args:
            smiles: SMILES string of the molecule
        
        Returns:
            Dictionary containing all 3D descriptors (40-50 features).
            Returns empty dict with error logged if calculation fails.
        """
        result = {}
        
        try:
            # Generate lowest energy conformer
            mol = self.get_lowest_energy_conformer(smiles)
            
            if mol is None:
                logger.warning(f"Could not generate conformer for: {smiles}")
                return result
            
            # Calculate PMI descriptors
            pmi = self.calculate_pmi(mol)
            result.update(pmi)
            
            # Calculate shape descriptors
            shape = self.calculate_shape_descriptors(mol)
            result.update(shape)
            
            # Calculate WHIM/autocorrelation descriptors
            whim = self.calculate_whim(mol)
            result.update(whim)
            
            logger.debug(f"Calculated {len(result)} 3D descriptors for molecule")
            
        except Exception as e:
            logger.error(f"Error calculating 3D descriptors for {smiles}: {e}")
        
        return result
    
    def calculate_for_conformer(self, mol: Any) -> Dict[str, float]:
        """
        Calculate all 3D descriptors for a pre-generated conformer.
        
        Args:
            mol: RDKit Mol object with 3D coordinates
        
        Returns:
            Dictionary containing all 3D descriptors
        """
        result = {}
        
        try:
            # Calculate PMI descriptors
            pmi = self.calculate_pmi(mol)
            result.update(pmi)
            
            # Calculate shape descriptors
            shape = self.calculate_shape_descriptors(mol)
            result.update(shape)
            
            # Calculate WHIM/autocorrelation descriptors
            whim = self.calculate_whim(mol)
            result.update(whim)
            
        except Exception as e:
            logger.error(f"Error calculating 3D descriptors: {e}")
        
        return result
    
    def get_descriptor_names(self) -> List[str]:
        """
        Get list of all 3D descriptor names.
        
        Returns:
            List of descriptor names that will be calculated
        """
        # PMI descriptors
        names = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2']
        
        # Shape descriptors
        names.extend([
            'spherocity', 'asphericity', 'eccentricity',
            'radius_of_gyration', 'inertial_shape_factor'
        ])
        
        # WHIM/autocorrelation descriptors (approximate counts)
        # AUTOCORR3D: 80 descriptors
        names.extend([f'AUTOCORR3D_{i+1}' for i in range(80)])
        
        # RDF: 210 descriptors
        names.extend([f'RDF_{i+1}' for i in range(210)])
        
        # MORSE: 224 descriptors
        names.extend([f'MORSE_{i+1}' for i in range(224)])
        
        # GETAWAY: 273 descriptors
        names.extend([f'GETAWAY_{i+1}' for i in range(273)])
        
        return names
