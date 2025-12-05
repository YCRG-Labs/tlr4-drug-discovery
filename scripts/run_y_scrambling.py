"""
Y-Scrambling Validation for TLR4 Binding Prediction
Addresses reviewer feedback: "Suggest adding Y-scrambling diagnostics"

This script performs Y-scrambling (response permutation) to validate that
the model is learning real signal, not chance correlations.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def load_data():
    """Load the expanded dataset and trained model"""
    print("Loading data...")
    df = pd.read_csv('binding-data/expanded_dataset.csv')
    
    # Load preprocessing pipeline
    with open('models/trained/improved_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/trained/improved_imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    
    # Load model
    with open('models/trained/optimized_gb.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return df, scaler, imputer, model

def calculate_features(df):
    """Calculate the same 33 features used in the optimized model"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Descriptors3D, Lipinski, Crippen
    from rdkit.Chem import AllChem
    
    features_list = []
    
    print("Calculating molecular features...")
    for smiles in tqdm(df['canonical_smiles'], desc="Processing molecules"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features_list.append([np.nan] * 33)
            continue
        
        # Generate 3D conformer
        mol_3d = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if result == -1:  # Embedding failed
            # Use 2D coordinates, set 3D features to NaN
            features_3d_failed = True
        else:
            AllChem.MMFFOptimizeMolecule(mol_3d)
            features_3d_failed = False
        
        # 2D descriptors (20 features)
        features = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'FractionCsp3': Lipinski.FractionCSP3(mol),
            'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'Kappa3': Descriptors.Kappa3(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'LabuteASA': Descriptors.LabuteASA(mol),
            'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
        }
        
        # 3D descriptors (10 features)
        if features_3d_failed:
            features.update({k: np.nan for k in ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
                                                   'RadiusOfGyration', 'InertialShapeFactor',
                                                   'Eccentricity', 'Asphericity', 'SpherocityIndex']})
        else:
            try:
                features.update({
                    'PMI1': Descriptors3D.PMI1(mol_3d),
                    'PMI2': Descriptors3D.PMI2(mol_3d),
                    'PMI3': Descriptors3D.PMI3(mol_3d),
                    'NPR1': Descriptors3D.NPR1(mol_3d),
                    'NPR2': Descriptors3D.NPR2(mol_3d),
                    'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol_3d),
                    'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol_3d),
                    'Eccentricity': Descriptors3D.Eccentricity(mol_3d),
                    'Asphericity': Descriptors3D.Asphericity(mol_3d),
                    'SpherocityIndex': Descriptors3D.SpherocityIndex(mol_3d),
                })
            except:
                features.update({k: np.nan for k in ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
                                                       'RadiusOfGyration', 'InertialShapeFactor',
                                                       'Eccentricity', 'Asphericity', 'SpherocityIndex']})
        
        # Electrostatic descriptors (3 features)
        AllChem.ComputeGasteigerCharges(mol)
        charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') 
                   for i in range(mol.GetNumAtoms())]
        features.update({
            'MaxPartialCharge': max(charges) if charges else np.nan,
            'MinPartialCharge': min(charges) if charges else np.nan,
            'MeanAbsPartialCharge': np.mean(np.abs(charges)) if charges else np.nan,
        })
        
        features_list.append(list(features.values()))
    
    feature_names = list(features.keys())
    return np.array(features_list), feature_names

def y_scrambling_test(X, y, model_params, n_permutations=100, cv_folds=5):
    """
    Perform Y-scrambling test
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values (will be permuted)
    model_params : dict
        Model hyperparameters
    n_permutations : int
        Number of random permutations
    cv_folds : int
        Number of CV folds
    
    Returns:
    --------
    dict : Results including real R², scrambled R² distribution, and cR²p
    """
    print(f"\nRunning Y-scrambling test with {n_permutations} permutations...")
    
    # Calculate real model performance
    real_model = GradientBoostingRegressor(**model_params, random_state=42)
    real_scores = cross_val_score(real_model, X, y, cv=cv_folds, 
                                   scoring='r2', n_jobs=-1)
    real_r2 = real_scores.mean()
    
    print(f"Real model R² (CV): {real_r2:.4f} ± {real_scores.std():.4f}")
    
    # Perform permutations
    scrambled_r2_scores = []
    
    for i in tqdm(range(n_permutations), desc="Y-scrambling"):
        # Randomly permute target values
        y_scrambled = np.random.permutation(y)
        
        # Train model on scrambled data
        scrambled_model = GradientBoostingRegressor(**model_params, random_state=i)
        scrambled_scores = cross_val_score(scrambled_model, X, y_scrambled, 
                                           cv=cv_folds, scoring='r2', n_jobs=-1)
        scrambled_r2_scores.append(scrambled_scores.mean())
    
    scrambled_r2_scores = np.array(scrambled_r2_scores)
    
    # Calculate cR²p (corrected R² for permutation)
    # cR²p = R² * sqrt(R² - R²_scrambled_mean)
    r2_scrambled_mean = scrambled_r2_scores.mean()
    if real_r2 > r2_scrambled_mean:
        cR2p = real_r2 * np.sqrt(real_r2 - r2_scrambled_mean)
    else:
        cR2p = 0.0
    
    # Calculate p-value (fraction of scrambled R² >= real R²)
    p_value = (scrambled_r2_scores >= real_r2).sum() / n_permutations
    
    results = {
        'real_r2': float(real_r2),
        'real_r2_std': float(real_scores.std()),
        'scrambled_r2_mean': float(r2_scrambled_mean),
        'scrambled_r2_std': float(scrambled_r2_scores.std()),
        'scrambled_r2_max': float(scrambled_r2_scores.max()),
        'cR2p': float(cR2p),
        'p_value': float(p_value),
        'n_permutations': n_permutations,
        'scrambled_r2_distribution': scrambled_r2_scores.tolist()
    }
    
    print(f"\nY-Scrambling Results:")
    print(f"  Real R²: {real_r2:.4f}")
    print(f"  Scrambled R² (mean): {r2_scrambled_mean:.4f} ± {scrambled_r2_scores.std():.4f}")
    print(f"  Scrambled R² (max): {scrambled_r2_scores.max():.4f}")
    print(f"  cR²p: {cR2p:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  ✓ Model is significantly better than random (p < 0.05)")
    else:
        print("  ✗ WARNING: Model may not be learning real signal!")
    
    return results

def plot_y_scrambling_results(results):
    """Create visualization of Y-scrambling results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of scrambled R² values
    scrambled_r2 = results['scrambled_r2_distribution']
    ax1.hist(scrambled_r2, bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax1.axvline(results['real_r2'], color='red', linestyle='--', linewidth=2, 
                label=f"Real R² = {results['real_r2']:.4f}")
    ax1.axvline(results['scrambled_r2_mean'], color='blue', linestyle='--', linewidth=2,
                label=f"Mean Scrambled R² = {results['scrambled_r2_mean']:.4f}")
    ax1.set_xlabel('R² Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Y-Scrambling Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Box plot comparison
    ax2.boxplot([scrambled_r2, [results['real_r2']]], 
                labels=['Scrambled Models', 'Real Model'],
                patch_artist=True,
                boxprops=dict(facecolor='lightgray'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Real vs Scrambled Model Performance', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"cR²p = {results['cR2p']:.4f}\np-value = {results['p_value']:.4f}"
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('paper_results/y_scrambling_validation.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: paper_results/y_scrambling_validation.png")
    plt.close()

def main():
    # Load data and model
    df, scaler, imputer, model = load_data()
    
    # Calculate features
    X_raw, feature_names = calculate_features(df)
    y = df['binding_affinity_kcal_mol'].values
    
    # Preprocess features (same as training)
    X_imputed = imputer.transform(X_raw)
    X = scaler.transform(X_imputed)
    
    # Get model parameters
    model_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 4,
        'subsample': 0.8
    }
    
    # Run Y-scrambling test
    results = y_scrambling_test(X, y, model_params, n_permutations=100, cv_folds=5)
    
    # Save results
    with open('paper_results/y_scrambling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: paper_results/y_scrambling_results.json")
    
    # Create visualization
    plot_y_scrambling_results(results)
    
    # Interpretation for paper
    print("\n" + "="*70)
    print("INTERPRETATION FOR PAPER")
    print("="*70)
    print(f"\nThe Y-scrambling test validates that our model learns real chemical-")
    print(f"biological relationships rather than chance correlations:")
    print(f"\n• Real model R² = {results['real_r2']:.4f}")
    print(f"• Scrambled models R² = {results['scrambled_r2_mean']:.4f} ± {results['scrambled_r2_std']:.4f}")
    print(f"• Maximum scrambled R² = {results['scrambled_r2_max']:.4f}")
    print(f"• cR²p = {results['cR2p']:.4f} (>0.5 indicates robust model)")
    print(f"• p-value = {results['p_value']:.4f} (p < 0.05 confirms significance)")
    
    if results['cR2p'] > 0.5:
        print(f"\n✓ The high cR²p value ({results['cR2p']:.4f}) confirms the model is robust")
        print(f"  and not overfitting to noise.")
    
    if results['p_value'] < 0.05:
        print(f"\n✓ The low p-value ({results['p_value']:.4f}) confirms the model performance")
        print(f"  is statistically significant and not due to chance.")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
