#!/usr/bin/env python3
"""
Optimize Gradient Boosting model since it showed best performance (R² = 0.575).

This script tunes GB hyperparameters to push R² even higher.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("GRADIENT BOOSTING OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("Goal: Push R² from 0.575 to 0.60+")
    logger.info("")
    
    # Load improved features
    logger.info("Loading improved features...")
    features_path = Path("models/trained/improved_imputer.pkl")
    
    if not features_path.exists():
        logger.error("Run scripts/improve_r2.py first!")
        return
    
    # Load dataset
    df = pd.read_csv("binding-data/expanded_dataset.csv")
    logger.info(f"✓ Loaded {len(df)} compounds")
    
    # Load preprocessing objects
    with open("models/trained/improved_imputer.pkl", 'rb') as f:
        imputer = pickle.load(f)
    with open("models/trained/improved_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Calculate features (reuse from improve_r2.py logic)
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    
    logger.info("Calculating features...")
    features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            if mol is None:
                continue
            
            mol_h = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol_h, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_h)
            except:
                mol_h = mol
            
            feat = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'MolMR': Descriptors.MolMR(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            }
            
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
            
            try:
                feat['Asphericity'] = rdMolDescriptors.CalcAsphericity(mol_h)
                feat['Eccentricity'] = rdMolDescriptors.CalcEccentricity(mol_h)
                feat['InertialShapeFactor'] = rdMolDescriptors.CalcInertialShapeFactor(mol_h)
                feat['RadiusOfGyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol_h)
                feat['SpherocityIndex'] = rdMolDescriptors.CalcSpherocityIndex(mol_h)
                
                pmi = rdMolDescriptors.CalcPMI1(mol_h), rdMolDescriptors.CalcPMI2(mol_h), rdMolDescriptors.CalcPMI3(mol_h)
                feat['PMI1'] = pmi[0]
                feat['PMI2'] = pmi[1]
                feat['PMI3'] = pmi[2]
                feat['NPR1'] = pmi[0] / pmi[2] if pmi[2] > 0 else 0
                feat['NPR2'] = pmi[1] / pmi[2] if pmi[2] > 0 else 0
            except:
                for key in ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 
                           'RadiusOfGyration', 'SpherocityIndex', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2']:
                    feat[key] = 0
            
            features.append(feat)
            valid_indices.append(idx)
            
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(features, index=valid_indices)
    df_merged = df.loc[features_df.index].copy()
    
    X = features_df.values
    y = df_merged['binding_affinity_kcal_mol'].values
    
    logger.info(f"✓ Features ready: {X.shape}")
    logger.info("")
    
    # Train optimized GB
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    logger.info("Optimizing Gradient Boosting hyperparameters...")
    logger.info("Testing different configurations...")
    
    # Quick grid search
    param_grid = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5, 6],
        'min_samples_split': [2, 4],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_gb = grid_search.best_estimator_
    logger.info(f"✓ Best parameters: {grid_search.best_params_}")
    
    # Evaluate
    y_pred_train = best_gb.predict(X_train_scaled)
    y_pred_test = best_gb.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    cv_scores = cross_val_score(best_gb, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZED GRADIENT BOOSTING RESULTS")
    logger.info("=" * 80)
    logger.info(f"Train R²: {train_r2:.3f}")
    logger.info(f"Test R²: {test_r2:.3f}")
    logger.info(f"Test RMSE: {test_rmse:.3f} kcal/mol")
    logger.info(f"Test MAE: {test_mae:.3f} kcal/mol")
    logger.info(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    logger.info("")
    
    # Save
    model_path = Path("models/trained/optimized_gb.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_gb, f)
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    with open("paper_results/optimized_gb_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info("")
    
    # Comparison
    logger.info("=" * 80)
    logger.info("PROGRESSION")
    logger.info("=" * 80)
    logger.info("Baseline (10 features):        R² = 0.344")
    logger.info("Improved (33 features):        R² = 0.469 (+36%)")
    logger.info("GB default (33 features):      R² = 0.575 (+67%)")
    logger.info(f"GB optimized (33 features):    R² = {test_r2:.3f} (+{(test_r2-0.344)/0.344*100:.0f}%)")
    logger.info("")
    
    if test_r2 >= 0.60:
        logger.info("🎉 SUCCESS: R² >= 0.60! This is publication-ready!")
    elif test_r2 >= 0.55:
        logger.info("✓ GOOD: R² >= 0.55! Strong results for TLR4.")
    else:
        logger.info("✓ SOLID: R² = {:.3f} is respectable for complex targets.".format(test_r2))
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
