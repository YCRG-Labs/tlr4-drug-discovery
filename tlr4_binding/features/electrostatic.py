"""
Electrostatic and Electronic Property Calculator for TLR4 binding prediction.

This module provides the ElectrostaticCalculator class for calculating
charge distributions and electronic properties relevant to TLR4 binding.

Requirements: 6.1, 6.2, 6.3
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

# RDKit imports with availability check
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdPartialCharges
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    Descriptors = None
    rdMolDescriptors = None
    rdPartialCharges = None

# Type checking imports
if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


class ElectrostaticCalculator:
    """
    Calculator for electrostatic and electronic molecular properties.
    
    Calculates Gasteiger partial charges, PEOE-VSA descriptors, molecular
    dipole moment, and polarizability for TLR4 binding prediction.
    
    Attributes:
        num_iterations: Number of iterations for Gasteiger charge calculation
    """
    
    def __init__(self, num_iterations: int = 12):
        """
        Initialize the electrostatic calculator.
        
        Args:
            num_iterations: Number of iterations for Gasteiger charge calculation (default: 12)
        """
        self.num_iterations = num_iterations

    def calculate_gasteiger_charges(self, mol: Any) -> np.ndarray:
        """
        Calculate Gasteiger partial charges for all atoms.
        
        Gasteiger charges are electronegativity equalization charges that
        model the distribution of electrons in a molecule. The sum of all
        charges equals the formal molecular charge.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Numpy array of partial charges for each atom.
            Returns empty array if calculation fails.
        
        Raises:
            RuntimeError: If RDKit is not available
            ValueError: If mol is None or invalid
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for Gasteiger charge calculation but is not installed")
        
        if mol is None:
            raise ValueError("Molecule cannot be None")
        
        try:
            # Compute Gasteiger charges
            AllChem.ComputeGasteigerCharges(mol, nIter=self.num_iterations)
            
            # Extract charges from atoms
            charges = []
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp('_GasteigerCharge')
                # Handle NaN values that can occur for certain atoms
                if np.isnan(charge):
                    charge = 0.0
                charges.append(charge)
            
            return np.array(charges)
            
        except Exception as e:
            logger.warning(f"Error calculating Gasteiger charges: {e}")
            return np.array([])

    def calculate_peoe_vsa(self, mol: Any) -> Dict[str, float]:
        """
        Calculate PEOE-VSA (Partial Equalization of Orbital Electronegativity - 
        van der Waals Surface Area) descriptors.
        
        PEOE-VSA descriptors partition the van der Waals surface area by
        partial charge ranges, capturing the distribution of charge over
        the molecular surface.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Dictionary containing PEOE-VSA descriptors:
                - PEOE_VSA1 through PEOE_VSA14: Surface area in different charge bins
                - SMR_VSA1 through SMR_VSA10: Molar refractivity surface area bins
                - SlogP_VSA1 through SlogP_VSA12: LogP surface area bins
        
        Raises:
            RuntimeError: If RDKit is not available
            ValueError: If mol is None or invalid
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for PEOE-VSA calculation but is not installed")
        
        if mol is None:
            raise ValueError("Molecule cannot be None")
        
        result = {}
        
        try:
            # PEOE_VSA descriptors (14 bins based on partial charge)
            peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
            for i, val in enumerate(peoe_vsa):
                result[f'PEOE_VSA{i+1}'] = val if not np.isnan(val) else 0.0
            
            # SMR_VSA descriptors (10 bins based on molar refractivity)
            smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
            for i, val in enumerate(smr_vsa):
                result[f'SMR_VSA{i+1}'] = val if not np.isnan(val) else 0.0
            
            # SlogP_VSA descriptors (12 bins based on logP contribution)
            slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
            for i, val in enumerate(slogp_vsa):
                result[f'SlogP_VSA{i+1}'] = val if not np.isnan(val) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating PEOE-VSA descriptors: {e}")
        
        return result


    def calculate_dipole(self, mol: Any) -> float:
        """
        Calculate molecular dipole moment.
        
        The dipole moment is calculated from Gasteiger partial charges and
        atomic coordinates. For 2D molecules, an approximate value is computed
        based on charge separation.
        
        Args:
            mol: RDKit Mol object (with or without 3D coordinates)
        
        Returns:
            Molecular dipole moment in Debye units.
            Returns 0.0 if calculation fails.
        
        Raises:
            RuntimeError: If RDKit is not available
            ValueError: If mol is None or invalid
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for dipole calculation but is not installed")
        
        if mol is None:
            raise ValueError("Molecule cannot be None")
        
        try:
            # Ensure Gasteiger charges are computed
            AllChem.ComputeGasteigerCharges(mol, nIter=self.num_iterations)
            
            # Check if molecule has 3D coordinates
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                
                # Calculate dipole moment from charges and positions
                dipole_vector = np.zeros(3)
                
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    charge = atom.GetDoubleProp('_GasteigerCharge')
                    if np.isnan(charge):
                        charge = 0.0
                    
                    pos = conf.GetAtomPosition(idx)
                    dipole_vector += charge * np.array([pos.x, pos.y, pos.z])
                
                # Convert to Debye (1 e*Å = 4.803 Debye)
                dipole_magnitude = np.linalg.norm(dipole_vector) * 4.803
                return float(dipole_magnitude)
            else:
                # For 2D molecules, use a simplified estimate based on charge separation
                # This is an approximation using topological distance
                charges = []
                for atom in mol.GetAtoms():
                    charge = atom.GetDoubleProp('_GasteigerCharge')
                    if np.isnan(charge):
                        charge = 0.0
                    charges.append(charge)
                
                # Estimate based on charge variance (rough approximation)
                if len(charges) > 0:
                    charge_variance = np.var(charges)
                    # Scale factor to approximate Debye units
                    return float(np.sqrt(charge_variance) * len(charges) * 0.5)
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating dipole moment: {e}")
            return 0.0

    def calculate_polarizability(self, mol: Any) -> float:
        """
        Calculate molecular polarizability.
        
        Uses the Wildman-Crippen method to estimate molecular polarizability
        based on atom contributions (related to molar refractivity).
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Estimated molecular polarizability (related to molar refractivity).
            Returns 0.0 if calculation fails.
        
        Raises:
            RuntimeError: If RDKit is not available
            ValueError: If mol is None or invalid
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for polarizability calculation but is not installed")
        
        if mol is None:
            raise ValueError("Molecule cannot be None")
        
        try:
            # Molar refractivity is related to polarizability
            # MR = (4π/3) * N_A * α, where α is polarizability
            # We use MR as a proxy for polarizability
            mr = Descriptors.MolMR(mol)
            return float(mr) if not np.isnan(mr) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating polarizability: {e}")
            return 0.0


    def calculate_charge_statistics(self, mol: Any) -> Dict[str, float]:
        """
        Calculate statistical summaries of atomic charges.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Dictionary containing charge statistics:
                - charge_sum: Sum of all partial charges (should equal formal charge)
                - charge_mean: Mean partial charge
                - charge_std: Standard deviation of charges
                - charge_min: Minimum partial charge
                - charge_max: Maximum partial charge
                - charge_range: Range of charges (max - min)
                - positive_charge_sum: Sum of positive charges
                - negative_charge_sum: Sum of negative charges
        """
        result = {}
        
        try:
            charges = self.calculate_gasteiger_charges(mol)
            
            if len(charges) > 0:
                result['charge_sum'] = float(np.sum(charges))
                result['charge_mean'] = float(np.mean(charges))
                result['charge_std'] = float(np.std(charges))
                result['charge_min'] = float(np.min(charges))
                result['charge_max'] = float(np.max(charges))
                result['charge_range'] = float(np.max(charges) - np.min(charges))
                result['positive_charge_sum'] = float(np.sum(charges[charges > 0]))
                result['negative_charge_sum'] = float(np.sum(charges[charges < 0]))
            else:
                # Default values if charge calculation fails
                for key in ['charge_sum', 'charge_mean', 'charge_std', 'charge_min',
                           'charge_max', 'charge_range', 'positive_charge_sum', 
                           'negative_charge_sum']:
                    result[key] = 0.0
                    
        except Exception as e:
            logger.warning(f"Error calculating charge statistics: {e}")
            for key in ['charge_sum', 'charge_mean', 'charge_std', 'charge_min',
                       'charge_max', 'charge_range', 'positive_charge_sum', 
                       'negative_charge_sum']:
                result[key] = 0.0
        
        return result

    def calculate_all(self, smiles: str) -> Dict[str, float]:
        """
        Calculate all electrostatic descriptors for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
        
        Returns:
            Dictionary containing all electrostatic descriptors.
            Returns empty dict if calculation fails.
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required but is not installed")
        
        result = {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES string: {smiles}")
                return result
            
            # Add hydrogens for more accurate charge calculation
            mol = Chem.AddHs(mol)
            
            # Calculate PEOE-VSA descriptors
            peoe_vsa = self.calculate_peoe_vsa(mol)
            result.update(peoe_vsa)
            
            # Calculate charge statistics
            charge_stats = self.calculate_charge_statistics(mol)
            result.update(charge_stats)
            
            # Calculate dipole moment
            result['dipole_moment'] = self.calculate_dipole(mol)
            
            # Calculate polarizability
            result['polarizability'] = self.calculate_polarizability(mol)
            
            logger.debug(f"Calculated {len(result)} electrostatic descriptors for molecule")
            
        except Exception as e:
            logger.error(f"Error calculating electrostatic descriptors for {smiles}: {e}")
        
        return result

    def calculate_from_mol(self, mol: Any) -> Dict[str, float]:
        """
        Calculate all electrostatic descriptors from an RDKit Mol object.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Dictionary containing all electrostatic descriptors.
        """
        result = {}
        
        try:
            # Calculate PEOE-VSA descriptors
            peoe_vsa = self.calculate_peoe_vsa(mol)
            result.update(peoe_vsa)
            
            # Calculate charge statistics
            charge_stats = self.calculate_charge_statistics(mol)
            result.update(charge_stats)
            
            # Calculate dipole moment
            result['dipole_moment'] = self.calculate_dipole(mol)
            
            # Calculate polarizability
            result['polarizability'] = self.calculate_polarizability(mol)
            
        except Exception as e:
            logger.error(f"Error calculating electrostatic descriptors: {e}")
        
        return result

    def get_descriptor_names(self) -> List[str]:
        """
        Get list of all electrostatic descriptor names.
        
        Returns:
            List of descriptor names that will be calculated
        """
        names = []
        
        # PEOE_VSA descriptors (14)
        names.extend([f'PEOE_VSA{i+1}' for i in range(14)])
        
        # SMR_VSA descriptors (10)
        names.extend([f'SMR_VSA{i+1}' for i in range(10)])
        
        # SlogP_VSA descriptors (12)
        names.extend([f'SlogP_VSA{i+1}' for i in range(12)])
        
        # Charge statistics (8)
        names.extend([
            'charge_sum', 'charge_mean', 'charge_std', 'charge_min',
            'charge_max', 'charge_range', 'positive_charge_sum', 
            'negative_charge_sum'
        ])
        
        # Dipole and polarizability (2)
        names.extend(['dipole_moment', 'polarizability'])
        
        return names

    def get_formal_charge(self, mol: Any) -> int:
        """
        Get the formal molecular charge.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Formal molecular charge as integer
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required but is not installed")
        
        if mol is None:
            raise ValueError("Molecule cannot be None")
        
        return Chem.GetFormalCharge(mol)
