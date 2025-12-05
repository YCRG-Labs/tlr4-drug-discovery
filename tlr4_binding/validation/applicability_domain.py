"""
Applicability Domain Analyzer for TLR4 Binding Affinity Prediction

This module implements applicability domain analysis to define the chemical space
where model predictions are reliable. It uses:
- Mahalanobis distance to measure distance from training set centroid
- Leverage values with threshold h* = 3p/n
- Tanimoto similarity for structural similarity assessment

Requirements: 16.1, 16.2, 16.3, 16.4
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
import logging

logger = logging.getLogger(__name__)

# RDKit imports for similarity calculations
try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Similarity calculations will be limited.")
    RDKIT_AVAILABLE = False


class ApplicabilityDomainAnalyzer:
    """
    Applicability Domain Analyzer for assessing prediction reliability.
    
    Defines the chemical space where model predictions are reliable using:
    - Mahalanobis distance from training set centroid
    - Leverage values with warning threshold h* = 3p/n
    - Tanimoto similarity to nearest training compounds
    
    Compounds outside the applicability domain should be flagged as having
    uncertain predictions.
    """
    
    def __init__(self, threshold_multiplier: float = 3.0):
        """
        Initialize the applicability domain analyzer.
        
        Args:
            threshold_multiplier: Multiplier for leverage threshold (default: 3.0)
                                 Threshold h* = threshold_multiplier * p / n
        """
        self.threshold_multiplier = threshold_multiplier
        self.X_train = None
        self.mean = None
        self.cov_inv = None
        self.n_samples = None
        self.n_features = None
        self.leverage_threshold = None
        self.train_smiles = None
        self.train_fingerprints = None
        
    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], 
            train_smiles: Optional[List[str]] = None) -> None:
        """
        Fit the applicability domain based on training data.
        
        Calculates the training set centroid and covariance matrix for
        Mahalanobis distance computation. Optionally stores training SMILES
        for similarity calculations.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            train_smiles: Optional list of training SMILES for similarity calculations
            
        Requirements: 16.1, 16.2
        """
        logger.info("Fitting applicability domain analyzer...")
        
        # Convert to numpy array if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        
        self.X_train = X_train
        self.n_samples, self.n_features = X_train.shape
        
        # Calculate mean and covariance
        self.mean = np.mean(X_train, axis=0)
        
        # Handle edge case of single sample
        if self.n_samples == 1:
            logger.warning("Only one training sample. Using identity covariance.")
            self.cov_inv = np.eye(self.n_features)
        else:
            cov = np.cov(X_train, rowvar=False)
            
            # Use pseudo-inverse for numerical stability
            # This handles singular or near-singular covariance matrices
            try:
                self.cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                logger.warning("Covariance matrix is singular. Using pseudo-inverse.")
                self.cov_inv = pinv(cov)
        
        # Calculate leverage threshold: h* = 3p/n
        self.leverage_threshold = (self.threshold_multiplier * self.n_features) / self.n_samples
        
        logger.info(f"Applicability domain fitted:")
        logger.info(f"  Training samples: {self.n_samples}")
        logger.info(f"  Features: {self.n_features}")
        logger.info(f"  Leverage threshold h*: {self.leverage_threshold:.4f}")
        
        # Store SMILES and generate fingerprints if provided
        if train_smiles is not None and RDKIT_AVAILABLE:
            self.train_smiles = train_smiles
            self.train_fingerprints = self._generate_fingerprints(train_smiles)
            logger.info(f"  Stored {len(train_smiles)} training SMILES for similarity calculations")
    
    def calculate_leverage(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculate leverage values for test compounds.
        
        Leverage is based on Mahalanobis distance from the training set centroid.
        Higher leverage indicates compounds further from the training distribution.
        
        The leverage h_i for compound i is calculated as:
        h_i = (x_i - mean)^T * Σ^(-1) * (x_i - mean) / n
        
        where Σ is the covariance matrix of the training set.
        
        Args:
            X: Test feature matrix (n_samples, n_features)
            
        Returns:
            Array of leverage values (n_samples,)
            
        Requirements: 16.1, 16.2
        """
        if self.mean is None or self.cov_inv is None:
            raise ValueError("Applicability domain not fitted. Call fit() first.")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check feature dimensions match
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {X.shape[1]}"
            )
        
        # Calculate Mahalanobis distance for each sample
        leverages = np.zeros(len(X))
        
        for i in range(len(X)):
            diff = X[i] - self.mean
            try:
                # Mahalanobis distance squared
                mahal_dist_sq = diff @ self.cov_inv @ diff
                # Normalize by number of samples
                leverages[i] = mahal_dist_sq / self.n_samples
            except Exception as e:
                logger.warning(f"Error calculating leverage for sample {i}: {e}")
                leverages[i] = np.inf
        
        return leverages
    
    def calculate_mahalanobis_distance(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculate Mahalanobis distance from test compounds to training set centroid.
        
        Mahalanobis distance accounts for correlations between features and
        provides a scale-invariant measure of distance.
        
        Args:
            X: Test feature matrix (n_samples, n_features)
            
        Returns:
            Array of Mahalanobis distances (n_samples,)
            
        Requirements: 16.1
        """
        if self.mean is None or self.cov_inv is None:
            raise ValueError("Applicability domain not fitted. Call fit() first.")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check feature dimensions match
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {X.shape[1]}"
            )
        
        # Calculate Mahalanobis distance for each sample
        distances = np.zeros(len(X))
        
        for i in range(len(X)):
            diff = X[i] - self.mean
            try:
                # Mahalanobis distance
                mahal_dist_sq = diff @ self.cov_inv @ diff
                distances[i] = np.sqrt(mahal_dist_sq)
            except Exception as e:
                logger.warning(f"Error calculating Mahalanobis distance for sample {i}: {e}")
                distances[i] = np.inf
        
        return distances
    
    def is_in_domain(self, 
                     X: Union[pd.DataFrame, np.ndarray],
                     threshold: Optional[float] = None) -> np.ndarray:
        """
        Determine if compounds are within the applicability domain.
        
        Compounds with leverage exceeding the threshold h* = 3p/n are
        considered outside the applicability domain.
        
        Args:
            X: Test feature matrix (n_samples, n_features)
            threshold: Custom leverage threshold (optional, uses h* if None)
            
        Returns:
            Boolean array indicating domain membership (n_samples,)
            True = in domain, False = out of domain
            
        Requirements: 16.3
        """
        leverages = self.calculate_leverage(X)
        
        # Use default threshold if not provided
        if threshold is None:
            threshold = self.leverage_threshold
        
        # Compounds are in domain if leverage <= threshold
        in_domain = leverages <= threshold
        
        n_in = np.sum(in_domain)
        n_out = len(in_domain) - n_in
        
        logger.info(f"Domain membership: {n_in} in domain, {n_out} out of domain")
        
        return in_domain
    
    def get_confidence(self, 
                      X: Union[pd.DataFrame, np.ndarray],
                      method: str = 'leverage') -> np.ndarray:
        """
        Calculate confidence scores for predictions based on domain membership.
        
        Confidence decreases as compounds move further from the training distribution.
        
        Args:
            X: Test feature matrix (n_samples, n_features)
            method: Method for confidence calculation ('leverage' or 'distance')
            
        Returns:
            Array of confidence scores in [0, 1] (n_samples,)
            1.0 = high confidence (well within domain)
            0.0 = low confidence (far outside domain)
            
        Requirements: 16.4
        """
        if method == 'leverage':
            leverages = self.calculate_leverage(X)
            
            # Convert leverage to confidence score
            # Confidence = 1 when leverage = 0
            # Confidence = 0.5 when leverage = threshold
            # Confidence approaches 0 as leverage increases
            confidence = 1.0 / (1.0 + leverages / self.leverage_threshold)
            
        elif method == 'distance':
            distances = self.calculate_mahalanobis_distance(X)
            
            # Normalize distances to [0, 1] range
            # Use median distance as reference point
            median_dist = np.median(distances[np.isfinite(distances)])
            
            # Confidence decreases exponentially with distance
            confidence = np.exp(-distances / median_dist)
            
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        # Clip to [0, 1] range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def calculate_similarity(self, 
                           smiles: Union[str, List[str]],
                           method: str = 'tanimoto',
                           radius: int = 2,
                           n_bits: int = 2048) -> Union[float, np.ndarray]:
        """
        Calculate maximum Tanimoto similarity to training set.
        
        For each test compound, finds the maximum similarity to any compound
        in the training set. Higher similarity indicates the compound is more
        similar to known training examples.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            method: Similarity method ('tanimoto' for Morgan fingerprints)
            radius: Morgan fingerprint radius (default: 2)
            n_bits: Fingerprint size (default: 2048)
            
        Returns:
            Maximum similarity score(s) in [0, 1]
            Single float if input is single SMILES, array if list
            
        Requirements: 16.4
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Cannot calculate similarity.")
            if isinstance(smiles, str):
                return 0.0
            else:
                return np.zeros(len(smiles))
        
        if self.train_fingerprints is None:
            logger.warning("Training fingerprints not available. Call fit() with train_smiles.")
            if isinstance(smiles, str):
                return 0.0
            else:
                return np.zeros(len(smiles))
        
        # Handle single SMILES
        single_input = isinstance(smiles, str)
        if single_input:
            smiles = [smiles]
        
        # Generate fingerprints for test compounds
        test_fps = self._generate_fingerprints(smiles, radius=radius, n_bits=n_bits)
        
        # Calculate maximum similarity to training set
        max_similarities = []
        
        for test_fp in test_fps:
            if test_fp is None:
                max_similarities.append(0.0)
                continue
            
            # Calculate similarity to all training compounds
            similarities = [
                DataStructs.TanimotoSimilarity(test_fp, train_fp)
                for train_fp in self.train_fingerprints
                if train_fp is not None
            ]
            
            if len(similarities) > 0:
                max_similarities.append(max(similarities))
            else:
                max_similarities.append(0.0)
        
        # Return single value or array
        if single_input:
            return max_similarities[0]
        else:
            return np.array(max_similarities)
    
    def _generate_fingerprints(self,
                              smiles_list: List[str],
                              radius: int = 2,
                              n_bits: int = 2048) -> List[Optional[object]]:
        """
        Generate Morgan fingerprints from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius
            n_bits: Fingerprint size
            
        Returns:
            List of RDKit fingerprint objects (None for invalid SMILES)
        """
        fingerprints = []
        
        for smi in smiles_list:
            try:
                if smi is None or smi == '' or pd.isna(smi):
                    fingerprints.append(None)
                    continue
                
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    fingerprints.append(fp)
                else:
                    fingerprints.append(None)
            except Exception as e:
                logger.warning(f"Error generating fingerprint for SMILES '{smi}': {e}")
                fingerprints.append(None)
        
        return fingerprints
    
    def get_domain_statistics(self, X: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Get comprehensive statistics about domain membership.
        
        Args:
            X: Test feature matrix (n_samples, n_features)
            
        Returns:
            Dictionary with domain statistics
        """
        leverages = self.calculate_leverage(X)
        distances = self.calculate_mahalanobis_distance(X)
        in_domain = self.is_in_domain(X)
        confidence = self.get_confidence(X)
        
        stats = {
            'n_samples': len(X),
            'n_in_domain': int(np.sum(in_domain)),
            'n_out_domain': int(len(in_domain) - np.sum(in_domain)),
            'leverage_threshold': self.leverage_threshold,
            'leverage_mean': float(np.mean(leverages)),
            'leverage_std': float(np.std(leverages)),
            'leverage_max': float(np.max(leverages)),
            'distance_mean': float(np.mean(distances[np.isfinite(distances)])),
            'distance_std': float(np.std(distances[np.isfinite(distances)])),
            'distance_max': float(np.max(distances[np.isfinite(distances)])),
            'confidence_mean': float(np.mean(confidence)),
            'confidence_std': float(np.std(confidence)),
            'confidence_min': float(np.min(confidence))
        }
        
        return stats
