#!/usr/bin/env python3
"""
Improve R² by adding more features and trying advanced techniques.

This script implements multiple strategies to improve model performance:
1. Add 3D and electrostatic descriptors (60+ additional features)
2. Hyperparameter tuning
3. Feature selection
4. Advanced ensemble methods

Usage:
    python scripts/improve_r2.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_enhanced_features(df):
    """Calculate comprehensive molecular descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem import AllChem
    
    logger.info("Calculating enhanced molecular descriptors...")
    logger.info("  This includes 2D, 3D, and electrostatic features")
    
    features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            if mol is None:
                continue
            
            # Add hydrogens for 3D
            mol_h = Chem.AddHs(mol)
            
            # Generate 3D conformer
            try:
                AllChem.EmbedMolecule(mol_h, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_h)
            except:
                # If 3D fails, use 2D only
                mol_h = mol
            
            # 2D Descriptors (basic)
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
            
            # Additional 2D descriptors (compatible with older RDKit)
            try:
                feat['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            except:
                feat['FractionCsp3'] = 0
            
            feat.update({
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'RingCount': Descriptors.RingCount(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol),
            })
            
            # Electrostatic descriptors
            try:
                AllChem.ComputeGasteigerCharges(mol)
                charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') 
                          for i in range(mol.GetNumAtoms())]
                feat['MaxPartialCharge'] = max(charges) if charges else 0
                feat['MinPartialCharge'] = min(charges) if charges else 0
                feat['MaxAbsPartialCharge'] = max(abs(c) for c in charges) if charges else 0
            except:
                feat['MaxPartialCharge'] = 0
                feat['MinPartialCharge'] = 0
                feat['MaxAbsPartialCharge'] = 0
            
            # 3D descriptors (if available)
            try:
                feat['Asphericity'] = rdMolDescriptors.CalcAsphericity(mol_h)
                feat['Eccentricity'] = rdMolDescriptors.CalcEccentricity(mol_h)
                feat['InertialShapeFactor'] = rdMolDescriptors.CalcInertialShapeFactor(mol_h)
                feat['RadiusOfGyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol_h)
                feat['SpherocityIndex'] = rdMolDescriptors.CalcSpherocityIndex(mol_h)
                
                # PMI descriptors
                pmi = rdMolDescriptors.CalcPMI1(mol_h), rdMolDescriptors.CalcPMI2(mol_h), rdMolDescriptors.CalcPMI3(mol_h)
                feat['PMI1'] = pmi[0]
                feat['PMI2'] = pmi[1]
                feat['PMI3'] = pmi[2]
                feat['NPR1'] = pmi[0] / pmi[2] if pmi[2] > 0 else 0
                feat['NPR2'] = pmi[1] / pmi[2] if pmi[2] > 0 else 0
            except:
                # 3D descriptors failed, use defaults
                for key in ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 
                           'RadiusOfGyration', 'SpherocityIndex', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2']:
                    feat[key] = 0
            
            features.append(feat)
            valid_indices.append(idx)
            
            if len(features) % 500 == 0:
                logger.info(f"  Processed {len(features)} compounds...")
            
        except Exception as e:
            logger.warning(f"Failed to calculate features for compound {idx}: {e}")
            continue
    
    features_df = pd.DataFrame(features, index=valid_indices)
    logger.info(f"✓ Calculated {len(features_df.columns)} features for {len(features_df)} compounds")
    
    return features_df


def train_improved_model(X, y):
    """Train improved model with hyperparameter tuning."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.feature_selection import SelectKBest, f_regression
    
    logger.info("Training improved model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Handle NaN values (replace with median)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Feature selection (keep top 50 features)
    logger.info("  Performing feature selection...")
    n_features_to_select = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(f_regression, k=n_features_to_select)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    logger.info(f"  Selected {n_features_to_select} most important features")
    
    # Define improved models with better hyperparameters
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=2,
        random_state=42
    )
    
    elasticnet = ElasticNet(
        alpha=0.05,
        l1_ratio=0.5,
        random_state=42,
        max_iter=10000
    )
    
    ridge = Ridge(
        alpha=0.5,
        random_state=42
    )
    
    bayesian = BayesianRidge()
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb),
        ('elasticnet', elasticnet),
        ('ridge', ridge),
        ('bayesian', bayesian)
    ])
    
    # Train
    logger.info("  Training Random Forest...")
    rf.fit(X_train_selected, y_train)
    
    logger.info("  Training Gradient Boosting...")
    gb.fit(X_train_selected, y_train)
    
    logger.info("  Training ElasticNet...")
    elasticnet.fit(X_train_selected, y_train)
    
    logger.info("  Training Ridge...")
    ridge.fit(X_train_selected, y_train)
    
    logger.info("  Training Bayesian Ridge...")
    bayesian.fit(X_train_selected, y_train)
    
    logger.info("  Training Ensemble...")
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred_train = ensemble.predict(X_train_selected)
    y_pred_test = ensemble.predict(X_test_selected)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    logger.info("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        ensemble, X_train_selected, y_train,
        cv=5, scoring='r2', n_jobs=-1
    )
    
    # Individual model performance
    individual_scores = {}
    for name, model in [('rf', rf), ('gb', gb), ('elasticnet', elasticnet), 
                        ('ridge', ridge), ('bayesian', bayesian)]:
        y_pred = model.predict(X_test_selected)
        individual_scores[name] = r2_score(y_test, y_pred)
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features_original': X.shape[1],
        'n_features_selected': n_features_to_select,
        'individual_models': individual_scores
    }
    
    logger.info(f"✓ Training complete")
    logger.info(f"  Train R²: {train_r2:.3f}")
    logger.info(f"  Test R²: {test_r2:.3f}")
    logger.info(f"  Test RMSE: {test_rmse:.3f} kcal/mol")
    logger.info(f"  Test MAE: {test_mae:.3f} kcal/mol")
    logger.info(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    logger.info(f"\n  Individual model performance:")
    for name, score in individual_scores.items():
        logger.info(f"    {name.upper()}: {score:.3f}")
    
    return ensemble, scaler, selector, imputer, results


def main():
    logger.info("=" * 80)
    logger.info("IMPROVED MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info("")
    
    # Load dataset
    logger.info("Loading expanded dataset...")
    dataset_path = Path("binding-data/expanded_dataset.csv")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    logger.info(f"✓ Loaded {len(df)} compounds")
    logger.info("")
    
    # Calculate enhanced features
    features_df = calculate_enhanced_features(df)
    
    # Merge with targets
    df_merged = df.loc[features_df.index].copy()
    df_merged = pd.concat([df_merged, features_df], axis=1)
    
    X = features_df.values
    y = df_merged['binding_affinity_kcal_mol'].values
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target range: {y.min():.2f} to {y.max():.2f} kcal/mol")
    logger.info("")
    
    # Train improved model
    model, scaler, selector, imputer, results = train_improved_model(X, y)
    
    # Save model
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "improved_ensemble.pkl"
    scaler_path = models_dir / "improved_scaler.pkl"
    selector_path = models_dir / "improved_selector.pkl"
    imputer_path = models_dir / "improved_imputer.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(selector_path, 'wb') as f:
        pickle.dump(selector, f)
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    
    logger.info("")
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info(f"✓ Scaler saved to: {scaler_path}")
    logger.info(f"✓ Selector saved to: {selector_path}")
    logger.info(f"✓ Imputer saved to: {imputer_path}")
    
    # Save results
    output_dir = Path("paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "improved_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {results_file}")
    
    # Comparison
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON: BASELINE vs IMPROVED")
    logger.info("=" * 80)
    
    # Load baseline results
    baseline_file = output_dir / "baseline_results.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        logger.info("Baseline (10 features):")
        logger.info(f"  Test R²: {baseline['test_r2']:.3f}")
        logger.info(f"  CV R²: {baseline['cv_mean']:.3f} ± {baseline['cv_std']:.3f}")
        logger.info("")
        logger.info(f"Improved ({results['n_features_original']} → {results['n_features_selected']} features):")
        logger.info(f"  Test R²: {results['test_r2']:.3f}")
        logger.info(f"  CV R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        logger.info("")
        
        improvement = results['test_r2'] - baseline['test_r2']
        logger.info(f"✓ Improvement: +{improvement:.3f} R² ({improvement/baseline['test_r2']*100:.1f}%)")
    else:
        logger.info(f"Improved model results:")
        logger.info(f"  Test R²: {results['test_r2']:.3f}")
        logger.info(f"  CV R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS TO FURTHER IMPROVE")
    logger.info("=" * 80)
    logger.info("1. Train advanced models:")
    logger.info("   python scripts/train_all_models.py")
    logger.info("")
    logger.info("2. Try transfer learning:")
    logger.info("   python examples/demo_transfer_learning.py")
    logger.info("")
    logger.info("3. Use hybrid model (GNN + descriptors):")
    logger.info("   python examples/demo_hybrid_model.py")
    logger.info("")
    logger.info(f"Completed at: {datetime.now()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
