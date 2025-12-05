"""
Quality control module for TLR4 binding data.

This module provides the QualityController class for:
- Removing PAINS (Pan-Assay Interference Compounds)
- Canonicalizing SMILES strings
- Calculating chemical diversity using Morgan fingerprints
- Flagging outliers in activity measurements

Requirements: 4.1, 4.2, 4.3, 4.4
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityController:
    """
    Quality controller for molecular data validation and filtering.
    
    This class provides methods to:
    - Remove Pan-Assay Interference Compounds (PAINS)
    - Canonicalize SMILES representations
    - Validate stereochemistry consistency
    - Calculate pairwise Tanimoto similarity for diversity assessment
    - Flag compounds with inconsistent activity values
    
    Attributes:
        pains_catalog: RDKit FilterCatalog for PAINS detection
    """
    
    def __init__(self):
        """Initialize the QualityController with PAINS filter catalog."""
        self._pains_catalog = None
        self._rdkit_available = self._check_rdkit()
    
    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            logger.warning("RDKit not available. Some quality control features will be limited.")
            return False
    
    @property
    def pains_catalog(self):
        """Lazy-load PAINS filter catalog."""
        if self._pains_catalog is None and self._rdkit_available:
            self._pains_catalog = self._create_pains_catalog()
        return self._pains_catalog
    
    def _create_pains_catalog(self):
        """Create RDKit FilterCatalog for PAINS detection."""
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
            
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(params)
            
            logger.info("PAINS filter catalog initialized successfully")
            return catalog
            
        except ImportError as e:
            logger.error(f"Could not create PAINS catalog: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating PAINS catalog: {e}")
            return None

    def remove_pains(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Remove Pan-Assay Interference Compounds from the dataset.
        
        Uses RDKit's PAINS filter catalog to identify and remove compounds
        that match known PAINS substructure patterns.
        
        Args:
            df: DataFrame containing compound data
            smiles_column: Name of column containing SMILES strings
        
        Returns:
            DataFrame with PAINS compounds removed and is_pains column added
        
        Requirements: 4.1
        """
        if df.empty:
            return df
        
        if not self._rdkit_available:
            logger.warning("RDKit not available. Skipping PAINS filtering.")
            df = df.copy()
            df['is_pains'] = False
            return df
        
        from rdkit import Chem
        
        df = df.copy()
        is_pains_list = []
        pains_matches = []
        
        catalog = self.pains_catalog
        
        for smiles in df[smiles_column]:
            if pd.isna(smiles) or not smiles:
                is_pains_list.append(False)
                pains_matches.append(None)
                continue
            
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                
                if mol is None:
                    is_pains_list.append(False)
                    pains_matches.append(None)
                    continue
                
                if catalog is not None:
                    entry = catalog.GetFirstMatch(mol)
                    if entry is not None:
                        is_pains_list.append(True)
                        pains_matches.append(entry.GetDescription())
                    else:
                        is_pains_list.append(False)
                        pains_matches.append(None)
                else:
                    is_pains_list.append(False)
                    pains_matches.append(None)
                    
            except Exception as e:
                logger.warning(f"Error checking PAINS for SMILES '{smiles}': {e}")
                is_pains_list.append(False)
                pains_matches.append(None)
        
        df['is_pains'] = is_pains_list
        df['pains_match'] = pains_matches
        
        # Count and log PAINS compounds
        n_pains = sum(is_pains_list)
        logger.info(f"Found {n_pains}/{len(df)} PAINS compounds")
        
        # Filter out PAINS compounds
        filtered_df = df[~df['is_pains']].copy()
        
        logger.info(f"Removed {n_pains} PAINS compounds. {len(filtered_df)} compounds remaining.")
        
        return filtered_df
    
    def check_pains(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a single SMILES matches PAINS patterns.
        
        Args:
            smiles: SMILES string to check
        
        Returns:
            Tuple of (is_pains, match_description)
        """
        if not self._rdkit_available:
            return False, None
        
        from rdkit import Chem
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, None
            
            catalog = self.pains_catalog
            if catalog is not None:
                entry = catalog.GetFirstMatch(mol)
                if entry is not None:
                    return True, entry.GetDescription()
            
            return False, None
            
        except Exception as e:
            logger.warning(f"Error checking PAINS: {e}")
            return False, None

    def canonicalize_smiles(self, smiles: str) -> str:
        """
        Canonicalize a SMILES string using RDKit.
        
        Standardizes the SMILES representation and validates stereochemistry.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Canonicalized SMILES string, or original if canonicalization fails
        
        Requirements: 4.2
        """
        if not smiles or pd.isna(smiles):
            return smiles
        
        if not self._rdkit_available:
            logger.warning("RDKit not available. Returning original SMILES.")
            return smiles
        
        from rdkit import Chem
        
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}")
                return smiles
            
            # Generate canonical SMILES with stereochemistry
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            
            return canonical
            
        except Exception as e:
            logger.warning(f"Error canonicalizing SMILES '{smiles}': {e}")
            return smiles
    
    def canonicalize_dataframe(
        self, 
        df: pd.DataFrame, 
        smiles_column: str = 'smiles',
        output_column: str = 'canonical_smiles'
    ) -> pd.DataFrame:
        """
        Canonicalize all SMILES in a DataFrame.
        
        Args:
            df: DataFrame containing SMILES data
            smiles_column: Name of column containing SMILES strings
            output_column: Name of column for canonicalized SMILES
        
        Returns:
            DataFrame with canonicalized SMILES column added
        """
        if df.empty:
            return df
        
        df = df.copy()
        df[output_column] = df[smiles_column].apply(self.canonicalize_smiles)
        
        # Count successful canonicalizations
        n_changed = (df[smiles_column] != df[output_column]).sum()
        logger.info(f"Canonicalized {n_changed}/{len(df)} SMILES strings")
        
        return df
    
    def check_stereochemistry(self, smiles: str) -> bool:
        """
        Validate stereochemistry consistency of a SMILES string.
        
        Checks if the molecule has valid stereochemistry that can be
        parsed and regenerated consistently.
        
        Args:
            smiles: SMILES string to validate
        
        Returns:
            True if stereochemistry is valid and consistent
        
        Requirements: 4.2
        """
        if not smiles or pd.isna(smiles):
            return False
        
        if not self._rdkit_available:
            return True  # Cannot validate without RDKit
        
        from rdkit import Chem
        
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            
            if mol is None:
                return False
            
            # Check for stereocenters
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            
            # Generate canonical SMILES with stereochemistry
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            
            # Re-parse and compare
            mol2 = Chem.MolFromSmiles(canonical)
            if mol2 is None:
                return False
            
            canonical2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=True)
            
            # Stereochemistry is consistent if round-trip produces same SMILES
            return canonical == canonical2
            
        except Exception as e:
            logger.warning(f"Error validating stereochemistry: {e}")
            return False

    def calculate_diversity(
        self, 
        smiles_list: List[str],
        radius: int = 2,
        n_bits: int = 2048
    ) -> np.ndarray:
        """
        Calculate pairwise Tanimoto similarity matrix using Morgan fingerprints.
        
        Computes the chemical diversity of a set of molecules by calculating
        pairwise Tanimoto similarity coefficients.
        
        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius (default 2)
            n_bits: Number of bits in fingerprint (default 2048)
        
        Returns:
            Symmetric numpy array of pairwise Tanimoto similarities
        
        Requirements: 4.4
        """
        n = len(smiles_list)
        
        if n == 0:
            return np.array([])
        
        if n == 1:
            return np.array([[1.0]])
        
        if not self._rdkit_available:
            logger.warning("RDKit not available. Returning identity matrix.")
            return np.eye(n)
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        # Generate fingerprints
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            if pd.isna(smiles) or not smiles:
                fingerprints.append(None)
                continue
            
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    fingerprints.append(fp)
                    valid_indices.append(i)
                else:
                    fingerprints.append(None)
            except Exception as e:
                logger.warning(f"Error generating fingerprint for SMILES '{smiles}': {e}")
                fingerprints.append(None)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n, n))
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i + 1, n):
                if fingerprints[i] is not None and fingerprints[j] is not None:
                    similarity = DataStructs.TanimotoSimilarity(
                        fingerprints[i], fingerprints[j]
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                else:
                    # Set to 0 if either fingerprint is invalid
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
        
        logger.info(
            f"Calculated diversity matrix for {len(valid_indices)}/{n} valid molecules. "
            f"Mean similarity: {similarity_matrix[np.triu_indices(n, k=1)].mean():.3f}"
        )
        
        return similarity_matrix
    
    def calculate_tanimoto_similarity(
        self, 
        smiles1: str, 
        smiles2: str,
        radius: int = 2,
        n_bits: int = 2048
    ) -> float:
        """
        Calculate Tanimoto similarity between two molecules.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint
        
        Returns:
            Tanimoto similarity coefficient (0-1)
        """
        if not self._rdkit_available:
            return 0.0
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        try:
            mol1 = Chem.MolFromSmiles(str(smiles1))
            mol2 = Chem.MolFromSmiles(str(smiles2))
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
            
        except Exception as e:
            logger.warning(f"Error calculating Tanimoto similarity: {e}")
            return 0.0
    
    def get_diversity_statistics(
        self, 
        similarity_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate diversity statistics from a similarity matrix.
        
        Args:
            similarity_matrix: Pairwise Tanimoto similarity matrix
        
        Returns:
            Dictionary with diversity statistics
        """
        n = similarity_matrix.shape[0]
        
        if n < 2:
            return {
                'mean_similarity': 1.0,
                'min_similarity': 1.0,
                'max_similarity': 1.0,
                'std_similarity': 0.0,
                'diversity_score': 0.0,
            }
        
        # Get upper triangle (excluding diagonal)
        upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
        
        return {
            'mean_similarity': float(np.mean(upper_tri)),
            'min_similarity': float(np.min(upper_tri)),
            'max_similarity': float(np.max(upper_tri)),
            'std_similarity': float(np.std(upper_tri)),
            'diversity_score': float(1.0 - np.mean(upper_tri)),  # Higher = more diverse
        }

    def flag_outliers(
        self, 
        df: pd.DataFrame,
        column: str = 'binding_affinity',
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Flag compounds with inconsistent or outlier activity values.
        
        Identifies outliers using IQR or Z-score methods and flags them
        for manual review.
        
        Args:
            df: DataFrame containing activity data
            column: Name of column to check for outliers
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
                      - For IQR: multiplier (default 1.5)
                      - For Z-score: number of standard deviations (default 3.0)
        
        Returns:
            DataFrame with is_outlier and outlier_reason columns added
        
        Requirements: 4.3
        """
        if df.empty:
            return df
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            df = df.copy()
            df['is_outlier'] = False
            df['outlier_reason'] = None
            return df
        
        df = df.copy()
        values = df[column].dropna()
        
        if len(values) < 3:
            logger.warning("Not enough values for outlier detection")
            df['is_outlier'] = False
            df['outlier_reason'] = None
            return df
        
        if method == 'iqr':
            is_outlier, reasons = self._detect_iqr_outliers(df, column, threshold)
        elif method == 'zscore':
            is_outlier, reasons = self._detect_zscore_outliers(df, column, threshold)
        else:
            logger.warning(f"Unknown outlier method: {method}. Using IQR.")
            is_outlier, reasons = self._detect_iqr_outliers(df, column, threshold)
        
        df['is_outlier'] = is_outlier
        df['outlier_reason'] = reasons
        
        n_outliers = sum(is_outlier)
        logger.info(f"Flagged {n_outliers}/{len(df)} compounds as outliers using {method} method")
        
        return df
    
    def _detect_iqr_outliers(
        self, 
        df: pd.DataFrame, 
        column: str, 
        threshold: float
    ) -> Tuple[List[bool], List[Optional[str]]]:
        """Detect outliers using IQR method."""
        values = df[column]
        
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        is_outlier = []
        reasons = []
        
        for val in values:
            if pd.isna(val):
                is_outlier.append(False)
                reasons.append(None)
            elif val < lower_bound:
                is_outlier.append(True)
                reasons.append(f"Below IQR lower bound ({lower_bound:.2f})")
            elif val > upper_bound:
                is_outlier.append(True)
                reasons.append(f"Above IQR upper bound ({upper_bound:.2f})")
            else:
                is_outlier.append(False)
                reasons.append(None)
        
        return is_outlier, reasons
    
    def _detect_zscore_outliers(
        self, 
        df: pd.DataFrame, 
        column: str, 
        threshold: float
    ) -> Tuple[List[bool], List[Optional[str]]]:
        """Detect outliers using Z-score method."""
        values = df[column]
        
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return [False] * len(values), [None] * len(values)
        
        is_outlier = []
        reasons = []
        
        for val in values:
            if pd.isna(val):
                is_outlier.append(False)
                reasons.append(None)
            else:
                z_score = abs((val - mean) / std)
                if z_score > threshold:
                    is_outlier.append(True)
                    reasons.append(f"Z-score {z_score:.2f} exceeds threshold {threshold}")
                else:
                    is_outlier.append(False)
                    reasons.append(None)
        
        return is_outlier, reasons
    
    def check_structure_activity_consistency(
        self,
        df: pd.DataFrame,
        smiles_column: str = 'canonical_smiles',
        activity_column: str = 'binding_affinity',
        similarity_threshold: float = 0.9,
        activity_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Check for structure-activity consistency.
        
        Identifies compounds that are structurally similar but have
        very different activity values, which may indicate data quality issues.
        
        Args:
            df: DataFrame with compound data
            smiles_column: Column containing SMILES strings
            activity_column: Column containing activity values
            similarity_threshold: Minimum Tanimoto similarity to consider similar
            activity_threshold: Maximum activity difference for consistent pairs
        
        Returns:
            DataFrame with consistency flags added
        
        Requirements: 4.3
        """
        if df.empty or len(df) < 2:
            df = df.copy()
            df['has_inconsistent_pair'] = False
            df['inconsistent_pairs'] = [[] for _ in range(len(df))]
            return df
        
        df = df.copy()
        smiles_list = df[smiles_column].tolist()
        activities = df[activity_column].tolist()
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_diversity(smiles_list)
        
        # Find inconsistent pairs
        inconsistent_flags = [False] * len(df)
        inconsistent_pairs = [[] for _ in range(len(df))]
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                similarity = similarity_matrix[i, j]
                
                if similarity >= similarity_threshold:
                    # Check activity difference
                    act_i = activities[i]
                    act_j = activities[j]
                    
                    if pd.notna(act_i) and pd.notna(act_j):
                        activity_diff = abs(act_i - act_j)
                        
                        if activity_diff > activity_threshold:
                            inconsistent_flags[i] = True
                            inconsistent_flags[j] = True
                            inconsistent_pairs[i].append({
                                'index': j,
                                'similarity': similarity,
                                'activity_diff': activity_diff
                            })
                            inconsistent_pairs[j].append({
                                'index': i,
                                'similarity': similarity,
                                'activity_diff': activity_diff
                            })
        
        df['has_inconsistent_pair'] = inconsistent_flags
        df['inconsistent_pairs'] = inconsistent_pairs
        
        n_inconsistent = sum(inconsistent_flags)
        logger.info(f"Found {n_inconsistent}/{len(df)} compounds with inconsistent structure-activity relationships")
        
        return df

    def run_full_quality_control(
        self,
        df: pd.DataFrame,
        smiles_column: str = 'smiles',
        activity_column: str = 'binding_affinity',
        remove_pains: bool = True,
        remove_outliers: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run full quality control pipeline on a dataset.
        
        Performs canonicalization, PAINS filtering, outlier detection,
        and diversity calculation.
        
        Args:
            df: Input DataFrame
            smiles_column: Column containing SMILES strings
            activity_column: Column containing activity values
            remove_pains: Whether to remove PAINS compounds
            remove_outliers: Whether to remove outliers (default False, just flag)
        
        Returns:
            Tuple of (processed DataFrame, quality report dictionary)
        """
        report = {
            'initial_count': len(df),
            'steps': [],
        }
        
        if df.empty:
            report['final_count'] = 0
            return df, report
        
        # Step 1: Canonicalize SMILES
        df = self.canonicalize_dataframe(df, smiles_column, 'canonical_smiles')
        report['steps'].append({
            'step': 'canonicalization',
            'count_after': len(df),
        })
        
        # Step 2: PAINS filtering
        if remove_pains:
            df = self.remove_pains(df, smiles_column)
            n_pains = report['initial_count'] - len(df)
            report['steps'].append({
                'step': 'pains_filtering',
                'removed': n_pains,
                'count_after': len(df),
            })
        else:
            # Just check PAINS without removing
            df = self._check_pains_only(df, smiles_column)
            n_pains = df['is_pains'].sum() if 'is_pains' in df.columns else 0
            report['steps'].append({
                'step': 'pains_check',
                'flagged': n_pains,
                'count_after': len(df),
            })
        
        # Step 3: Outlier detection
        if activity_column in df.columns:
            df = self.flag_outliers(df, activity_column)
            n_outliers = df['is_outlier'].sum() if 'is_outlier' in df.columns else 0
            
            if remove_outliers:
                df = df[~df['is_outlier']].copy()
                report['steps'].append({
                    'step': 'outlier_removal',
                    'removed': n_outliers,
                    'count_after': len(df),
                })
            else:
                report['steps'].append({
                    'step': 'outlier_detection',
                    'flagged': n_outliers,
                    'count_after': len(df),
                })
        
        # Step 4: Calculate diversity
        if len(df) > 1:
            smiles_col = 'canonical_smiles' if 'canonical_smiles' in df.columns else smiles_column
            similarity_matrix = self.calculate_diversity(df[smiles_col].tolist())
            diversity_stats = self.get_diversity_statistics(similarity_matrix)
            report['diversity'] = diversity_stats
        
        report['final_count'] = len(df)
        
        return df, report
    
    def _check_pains_only(
        self, 
        df: pd.DataFrame, 
        smiles_column: str
    ) -> pd.DataFrame:
        """Check PAINS without removing compounds."""
        if df.empty:
            return df
        
        if not self._rdkit_available:
            df = df.copy()
            df['is_pains'] = False
            return df
        
        from rdkit import Chem
        
        df = df.copy()
        is_pains_list = []
        
        catalog = self.pains_catalog
        
        for smiles in df[smiles_column]:
            if pd.isna(smiles) or not smiles:
                is_pains_list.append(False)
                continue
            
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                
                if mol is None:
                    is_pains_list.append(False)
                    continue
                
                if catalog is not None:
                    entry = catalog.GetFirstMatch(mol)
                    is_pains_list.append(entry is not None)
                else:
                    is_pains_list.append(False)
                    
            except Exception as e:
                logger.warning(f"Error checking PAINS: {e}")
                is_pains_list.append(False)
        
        df['is_pains'] = is_pains_list
        
        return df
