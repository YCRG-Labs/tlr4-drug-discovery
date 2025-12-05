#!/usr/bin/env python3
"""
Fast baseline model training for the revised paper.

This trains the baseline ensemble model (RF, ElasticNet, Ridge, Bayesian)
which is sufficient for demonstrating the methodology improvements.

Usage:
    python scripts/train_baseline_fast.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_features(df):
    """Calculate molecular descriptors for the dataset."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    
    logger.info("Calculating molecular descriptors...")
    
    features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            if mol is None:
                continue
            
            # Calculate descriptors
            feat = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'MolMR': Descriptors.MolMR(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'HeavyAtomCount': Lipinski.HeavyAtomCount(mol),
            }
            
            features.append(feat)
            valid_indices.append(idx)
            
        except Exception as e:
            logger.warning(f"Failed to calculate features for compound {idx}: {e}")
            continue
    
    features_df = pd.DataFrame(features, index=valid_indices)
    logger.info(f"✓ Calculated {len(features_df.columns)} features for {len(features_df)} compounds")
    
    return features_df


def train_baseline_ensemble(X, y):
    """Train baseline ensemble model."""
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    logger.info("Training baseline ensemble...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    
    elasticnet = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42,
        max_iter=5000
    )
    
    ridge = Ridge(
        alpha=1.0,
        random_state=42
    )
    
    bayesian = BayesianRidge()
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('rf', rf),
        ('elasticnet', elasticnet),
        ('ridge', ridge),
        ('bayesian', bayesian)
    ])
    
    # Train
    logger.info("  Training Random Forest...")
    rf.fit(X_train_scaled, y_train)
    
    logger.info("  Training ElasticNet...")
    elasticnet.fit(X_train_scaled, y_train)
    
    logger.info("  Training Ridge...")
    ridge.fit(X_train_scaled, y_train)
    
    logger.info("  Training Bayesian Ridge...")
    bayesian.fit(X_train_scaled, y_train)
    
    logger.info("  Training Ensemble...")
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = ensemble.predict(X_train_scaled)
    y_pred_test = ensemble.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    logger.info("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        ensemble, X_train_scaled, y_train,
        cv=5, scoring='r2', n_jobs=-1
    )
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1]
    }
    
    logger.info(f"✓ Training complete")
    logger.info(f"  Train R²: {train_r2:.3f}")
    logger.info(f"  Test R²: {test_r2:.3f}")
    logger.info(f"  Test RMSE: {test_rmse:.3f} kcal/mol")
    logger.info(f"  Test MAE: {test_mae:.3f} kcal/mol")
    logger.info(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return ensemble, scaler, results


def main():
    logger.info("=" * 80)
    logger.info("FAST BASELINE MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info("")
    
    # Load dataset
    logger.info("Loading expanded dataset...")
    dataset_path = Path("binding-data/expanded_dataset.csv")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Please run: python scripts/collect_expanded_dataset.py")
        return
    
    df = pd.read_csv(dataset_path)
    logger.info(f"✓ Loaded {len(df)} compounds")
    logger.info("")
    
    # Calculate features
    features_df = calculate_features(df)
    
    # Merge with targets
    df_merged = df.loc[features_df.index].copy()
    df_merged = pd.concat([df_merged, features_df], axis=1)
    
    X = features_df.values
    y = df_merged['binding_affinity_kcal_mol'].values
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target range: {y.min():.2f} to {y.max():.2f} kcal/mol")
    logger.info("")
    
    # Train model
    model, scaler, results = train_baseline_ensemble(X, y)
    
    # Save model
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "baseline_ensemble.pkl"
    scaler_path = models_dir / "baseline_scaler.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("")
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info(f"✓ Scaler saved to: {scaler_path}")
    
    # Save results
    output_dir = Path("paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "baseline_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {results_file}")
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Dataset size: {len(df)} compounds")
    logger.info(f"Training set: {results['n_train']} compounds")
    logger.info(f"Test set: {results['n_test']} compounds")
    logger.info(f"Features: {results['n_features']}")
    logger.info("")
    logger.info(f"Test R²: {results['test_r2']:.3f}")
    logger.info(f"Test RMSE: {results['test_rmse']:.3f} kcal/mol")
    logger.info(f"Test MAE: {results['test_mae']:.3f} kcal/mol")
    logger.info(f"CV R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON WITH OLD PAPER")
    logger.info("=" * 80)
    logger.info("Old paper (49 compounds):")
    logger.info("  CV R²: 0.74 ± 0.10")
    logger.info("  Test R²: 0.79")
    logger.info("")
    logger.info(f"New results ({len(df)} compounds):")
    logger.info(f"  CV R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    logger.info(f"  Test R²: {results['test_r2']:.3f}")
    logger.info("")
    logger.info("✓ You now have results for your revised paper!")
    logger.info("")
    logger.info(f"Completed at: {datetime.now()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
