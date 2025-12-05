"""
Validation Framework for TLR4 Binding Affinity Prediction

This module implements comprehensive validation strategies including:
- Stratified data splitting by affinity quartiles
- Nested cross-validation with hyperparameter optimization
- Y-scrambling validation for model robustness assessment
- Scaffold-based validation for generalization testing

Requirements: 13.1, 14.1, 14.2, 15.1, 15.2, 15.3, 17.1, 17.2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
from dataclasses import dataclass
import warnings

from .models import ValidationResult

logger = logging.getLogger(__name__)

# RDKit imports for scaffold splitting
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Scaffold splitting will use fallback methods.")
    RDKIT_AVAILABLE = False


class ValidationFramework:
    """
    Comprehensive validation framework for TLR4 binding affinity prediction.
    
    Implements multiple validation strategies to ensure robust model evaluation:
    - Stratified splitting by binding affinity quartiles
    - Nested cross-validation for unbiased performance estimation
    - Y-scrambling for statistical validation
    - Scaffold-based splitting for generalization assessment
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the validation framework.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.validation_results = {}
        
    def stratified_split(self,
                        X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray],
                        test_size: float = 0.2,
                        n_bins: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data stratified by binding affinity quartiles.
        
        Creates stratification bins based on binding affinity values to ensure
        balanced representation of different affinity ranges in train and test sets.
        Reserves 20% for external test set by default.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (binding affinities in kcal/mol)
            test_size: Fraction of data to reserve for test set (default: 0.2)
            n_bins: Number of stratification bins (default: 4 for quartiles)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Requirements: 13.1
        """
        logger.info(f"Performing stratified split with {n_bins} bins (test_size={test_size})")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Create stratification bins based on target quartiles
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy='quantile',
            subsample=None
        )
        
        # Fit and transform to create stratification labels
        y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
        
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y_binned,
            random_state=self.random_state
        )
        
        # Validate stratification quality
        self._validate_stratification(y_train, y_test, y_binned, n_bins)
        
        logger.info(f"Stratified split completed: {len(X_train)} train, {len(X_test)} test")
        logger.info(f"Train affinity range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        logger.info(f"Test affinity range: [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        return X_train, X_test, y_train, y_test
    
    def _validate_stratification(self,
                                 y_train: np.ndarray,
                                 y_test: np.ndarray,
                                 y_binned: np.ndarray,
                                 n_bins: int) -> None:
        """
        Validate that stratification produced balanced splits.
        
        Checks that each quartile has approximately equal representation
        (within 5%) in both train and test sets.
        """
        # Recreate bins for train and test
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy='quantile',
            subsample=None
        )
        
        # Fit on combined data to get consistent bins
        y_combined = np.concatenate([y_train, y_test])
        discretizer.fit(y_combined.reshape(-1, 1))
        
        train_bins = discretizer.transform(y_train.reshape(-1, 1)).flatten()
        test_bins = discretizer.transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate distribution in each split
        train_dist = np.bincount(train_bins.astype(int), minlength=n_bins) / len(train_bins)
        test_dist = np.bincount(test_bins.astype(int), minlength=n_bins) / len(test_bins)
        
        # Check if distributions are within 5% tolerance
        tolerance = 0.05
        max_diff = np.max(np.abs(train_dist - test_dist))
        
        if max_diff > tolerance:
            logger.warning(f"Stratification imbalance detected: max difference = {max_diff:.3f}")
        else:
            logger.info(f"Stratification validated: max difference = {max_diff:.3f} (within {tolerance} tolerance)")
    
    def nested_cv(self,
                  model_class: Any,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  param_grid: Optional[Dict[str, List]] = None,
                  outer_folds: int = 5,
                  inner_folds: int = 3,
                  scoring: str = 'r2') -> Dict[str, Any]:
        """
        Perform nested cross-validation with hyperparameter optimization.
        
        Implements nested CV with:
        - Outer loop (5-fold): Unbiased performance estimation
        - Inner loop (3-fold): Hyperparameter optimization
        
        This prevents information leakage from hyperparameter tuning into
        performance estimates.
        
        Args:
            model_class: Model class or instance to evaluate
            X: Feature matrix (n_samples, n_features)
            y: Target values (binding affinities)
            param_grid: Hyperparameter grid for optimization (optional)
            outer_folds: Number of outer CV folds (default: 5)
            inner_folds: Number of inner CV folds (default: 3)
            scoring: Scoring metric (default: 'r2')
            
        Returns:
            Dictionary containing:
            - outer_scores: List of R² scores from outer folds
            - mean_score: Mean R² across outer folds
            - std_score: Standard deviation of R² across outer folds
            - best_params_per_fold: Best hyperparameters from each outer fold
            - inner_scores: Inner fold scores for each outer fold
            
        Requirements: 14.1, 14.2
        """
        logger.info(f"Starting nested CV: {outer_folds} outer folds, {inner_folds} inner folds")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Setup outer cross-validation
        outer_cv = KFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Setup inner cross-validation
        inner_cv = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=self.random_state + 1
        )
        
        outer_scores = []
        best_params_per_fold = []
        inner_scores_per_fold = []
        
        # Outer loop: Performance estimation
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            logger.info(f"Processing outer fold {fold_idx + 1}/{outer_folds}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner loop: Hyperparameter optimization
            if param_grid is not None and len(param_grid) > 0:
                # Use GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=model_class,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_outer, y_train_outer)
                
                # Get best model from inner CV
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                inner_scores = grid_search.cv_results_['mean_test_score']
                
                best_params_per_fold.append(best_params)
                inner_scores_per_fold.append({
                    'mean': grid_search.best_score_,
                    'std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                    'all_scores': inner_scores.tolist()
                })
                
                logger.info(f"  Best params: {best_params}")
                logger.info(f"  Inner CV score: {grid_search.best_score_:.4f}")
            else:
                # No hyperparameter tuning, just train model
                if hasattr(model_class, 'fit'):
                    best_model = model_class
                else:
                    best_model = model_class()
                    
                best_model.fit(X_train_outer, y_train_outer)
                best_params_per_fold.append({})
                inner_scores_per_fold.append({'mean': None, 'std': None})
            
            # Evaluate on outer test fold
            y_pred_outer = best_model.predict(X_test_outer)
            outer_score = r2_score(y_test_outer, y_pred_outer)
            outer_scores.append(outer_score)
            
            logger.info(f"  Outer fold score: {outer_score:.4f}")
        
        # Calculate summary statistics
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        results = {
            'outer_scores': outer_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'best_params_per_fold': best_params_per_fold,
            'inner_scores': inner_scores_per_fold,
            'n_outer_folds': outer_folds,
            'n_inner_folds': inner_folds
        }
        
        logger.info(f"Nested CV completed: R² = {mean_score:.4f} ± {std_score:.4f}")
        
        return results

    def y_scrambling(self,
                     model_class: Any,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     n_permutations: int = 100,
                     test_size: float = 0.2,
                     return_models: bool = False) -> Dict[str, Any]:
        """
        Perform Y-scrambling validation to assess model robustness.
        
        Randomly permutes target labels multiple times and trains models on
        scrambled data. A robust model should perform poorly on scrambled data.
        Calculates the cR²p metric: cR²p = R² × sqrt(R² - R²_scrambled_mean)
        
        Models with cR²p ≤ 0.5 are flagged as potentially overfit.
        
        Args:
            model_class: Model class or instance to evaluate
            X: Feature matrix (n_samples, n_features)
            y: Target values (binding affinities)
            n_permutations: Number of random permutations (default: 100, range: 100-1000)
            test_size: Fraction of data for test set (default: 0.2)
            return_models: Whether to return trained models (default: False)
            
        Returns:
            Dictionary containing:
            - r2_real: R² score on real data
            - r2_scrambled_values: List of R² scores on scrambled data
            - r2_scrambled_mean: Mean R² on scrambled data
            - r2_scrambled_std: Std dev of R² on scrambled data
            - cr2p: cR²p metric value
            - is_potentially_overfit: Boolean flag (True if cR²p ≤ 0.5)
            - models: List of trained models (if return_models=True)
            
        Requirements: 15.1, 15.2, 15.3
        """
        logger.info(f"Starting Y-scrambling validation with {n_permutations} permutations")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data for consistent evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state
        )
        
        # Train model on real data
        logger.info("Training model on real data...")
        if hasattr(model_class, 'fit'):
            real_model = model_class
        else:
            real_model = model_class()
        
        real_model.fit(X_train, y_train)
        y_pred_real = real_model.predict(X_test)
        r2_real = r2_score(y_test, y_pred_real)
        
        logger.info(f"Real data R²: {r2_real:.4f}")
        
        # Perform Y-scrambling
        r2_scrambled_values = []
        scrambled_models = [] if return_models else None
        
        np.random.seed(self.random_state)
        
        for i in range(n_permutations):
            if (i + 1) % 20 == 0:
                logger.info(f"  Permutation {i + 1}/{n_permutations}")
            
            # Randomly permute training labels
            y_train_scrambled = np.random.permutation(y_train)
            
            # Train model on scrambled data
            if hasattr(model_class, 'fit'):
                scrambled_model = model_class
            else:
                scrambled_model = model_class()
            
            try:
                scrambled_model.fit(X_train, y_train_scrambled)
                y_pred_scrambled = scrambled_model.predict(X_test)
                r2_scrambled = r2_score(y_test, y_pred_scrambled)
                r2_scrambled_values.append(r2_scrambled)
                
                if return_models:
                    scrambled_models.append(scrambled_model)
            except Exception as e:
                logger.warning(f"Error in permutation {i + 1}: {e}")
                continue
        
        # Calculate statistics
        r2_scrambled_mean = np.mean(r2_scrambled_values)
        r2_scrambled_std = np.std(r2_scrambled_values)
        
        # Calculate cR²p metric
        # cR²p = R² × sqrt(R² - R²_scrambled_mean)
        # Handle negative values under sqrt
        diff = r2_real - r2_scrambled_mean
        if diff < 0:
            cr2p = 0.0
            logger.warning("R² on real data is less than mean scrambled R². Setting cR²p = 0")
        else:
            cr2p = r2_real * np.sqrt(diff)
        
        # Check for potential overfitting
        is_potentially_overfit = cr2p <= 0.5
        
        results = {
            'r2_real': r2_real,
            'r2_scrambled_values': r2_scrambled_values,
            'r2_scrambled_mean': r2_scrambled_mean,
            'r2_scrambled_std': r2_scrambled_std,
            'cr2p': cr2p,
            'is_potentially_overfit': is_potentially_overfit,
            'n_permutations': len(r2_scrambled_values)
        }
        
        if return_models:
            results['models'] = scrambled_models
        
        logger.info(f"Y-scrambling completed:")
        logger.info(f"  Real R²: {r2_real:.4f}")
        logger.info(f"  Scrambled R²: {r2_scrambled_mean:.4f} ± {r2_scrambled_std:.4f}")
        logger.info(f"  cR²p: {cr2p:.4f}")
        logger.info(f"  Potentially overfit: {is_potentially_overfit}")
        
        return results
    
    def scaffold_split(self,
                      smiles: List[str],
                      y: Union[pd.Series, np.ndarray],
                      test_size: float = 0.2,
                      include_chirality: bool = False) -> Tuple[List[int], List[int]]:
        """
        Split data by Murcko scaffold to assess generalization to novel scaffolds.
        
        Extracts Murcko scaffolds from SMILES and ensures that no scaffold
        appears in both training and test sets. This tests the model's ability
        to generalize to compounds with novel core structures.
        
        Args:
            smiles: List of SMILES strings
            y: Target values (for size balancing)
            test_size: Fraction of data for test set (default: 0.2)
            include_chirality: Whether to include stereochemistry in scaffolds
            
        Returns:
            Tuple of (train_indices, test_indices)
            
        Requirements: 17.1, 17.2
        """
        logger.info(f"Performing scaffold-based split (test_size={test_size})")
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Using random split as fallback.")
            return self._fallback_random_split(len(smiles), test_size)
        
        # Convert y to numpy array if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        # Generate Murcko scaffolds
        scaffolds = self._generate_scaffolds(smiles, include_chirality)
        
        # Group indices by scaffold
        scaffold_to_indices = {}
        for idx, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = []
            scaffold_to_indices[scaffold].append(idx)
        
        # Check for sufficient scaffold diversity
        n_unique_scaffolds = len(scaffold_to_indices)
        logger.info(f"Found {n_unique_scaffolds} unique scaffolds")
        
        if n_unique_scaffolds < 3:
            logger.warning(f"Only {n_unique_scaffolds} unique scaffolds. Using random split.")
            return self._fallback_random_split(len(smiles), test_size)
        
        # Sort scaffolds by size (largest first) for balanced splitting
        sorted_scaffolds = sorted(
            scaffold_to_indices.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Allocate scaffolds to train/test splits
        train_indices, test_indices = self._allocate_scaffolds_to_splits(
            sorted_scaffolds,
            len(smiles),
            test_size
        )
        
        # Validate scaffold disjointness
        self._validate_scaffold_split(scaffolds, train_indices, test_indices)
        
        logger.info(f"Scaffold split completed:")
        logger.info(f"  Train: {len(train_indices)} compounds")
        logger.info(f"  Test: {len(test_indices)} compounds")
        logger.info(f"  Train scaffolds: {len(set(scaffolds[i] for i in train_indices))}")
        logger.info(f"  Test scaffolds: {len(set(scaffolds[i] for i in test_indices))}")
        
        return train_indices, test_indices
    
    def _generate_scaffolds(self,
                           smiles: List[str],
                           include_chirality: bool = False) -> List[str]:
        """
        Generate Bemis-Murcko scaffolds from SMILES strings.
        
        Args:
            smiles: List of SMILES strings
            include_chirality: Whether to preserve stereochemistry
            
        Returns:
            List of scaffold SMILES strings
        """
        scaffolds = []
        
        for i, smi in enumerate(smiles):
            try:
                if smi is None or smi == '' or pd.isna(smi):
                    # Assign unique scaffold to invalid SMILES
                    scaffolds.append(f"invalid_scaffold_{i}")
                    continue
                
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(
                        scaffold,
                        isomericSmiles=include_chirality
                    )
                    scaffolds.append(scaffold_smiles)
                else:
                    scaffolds.append(f"invalid_scaffold_{i}")
            except Exception as e:
                logger.warning(f"Error generating scaffold for SMILES {i}: {e}")
                scaffolds.append(f"invalid_scaffold_{i}")
        
        return scaffolds
    
    def _allocate_scaffolds_to_splits(self,
                                     sorted_scaffolds: List[Tuple[str, List[int]]],
                                     total_size: int,
                                     test_size: float) -> Tuple[List[int], List[int]]:
        """
        Allocate scaffolds to train and test splits.
        
        Uses a greedy algorithm to balance split sizes while maintaining
        scaffold disjointness.
        """
        np.random.seed(self.random_state)
        
        target_test_size = int(total_size * test_size)
        
        train_indices = []
        test_indices = []
        
        # Shuffle scaffolds to randomize allocation
        scaffolds_copy = sorted_scaffolds.copy()
        np.random.shuffle(scaffolds_copy)
        
        # Allocate scaffolds
        for scaffold, indices in scaffolds_copy:
            if len(test_indices) < target_test_size:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)
        
        return train_indices, test_indices
    
    def _validate_scaffold_split(self,
                                scaffolds: List[str],
                                train_indices: List[int],
                                test_indices: List[int]) -> None:
        """
        Validate that no scaffold appears in both train and test sets.
        
        Requirements: 17.2
        """
        train_scaffolds = set(scaffolds[i] for i in train_indices)
        test_scaffolds = set(scaffolds[i] for i in test_indices)
        
        overlap = train_scaffolds.intersection(test_scaffolds)
        
        if len(overlap) > 0:
            logger.error(f"Scaffold overlap detected: {len(overlap)} scaffolds in both splits!")
            logger.error(f"Overlapping scaffolds: {list(overlap)[:5]}...")
            raise ValueError("Scaffold split validation failed: overlapping scaffolds detected")
        else:
            logger.info("Scaffold split validated: no overlap between train and test")
    
    def _fallback_random_split(self,
                              n_samples: int,
                              test_size: float) -> Tuple[List[int], List[int]]:
        """
        Fallback to random splitting when scaffold splitting is not possible.
        """
        indices = list(range(n_samples))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.random_state
        )
        return train_indices, test_indices
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary with R², RMSE, and MAE
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
