"""
Applicability Domain Analysis for TLR4 Binding Prediction
Addresses reviewer feedback: "Suggest adding applicability domain estimation"

This script calculates the applicability domain to identify when predictions
are reliable vs when compounds are outside the training space.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
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
    from tqdm import tqdm
    
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

def calculate_leverage(X_train, X_test):
    """
    Calculate leverage (hat) values for applicability domain
    
    Leverage h_i = x_i^T (X^T X)^-1 x_i
    
    High leverage indicates compound is far from training space
    """
    print("\nCalculating leverage values...")
    
    # Calculate hat matrix diagonal (leverage values)
    # For training set
    XtX_inv = np.linalg.pinv(X_train.T @ X_train)
    h_train = np.sum((X_train @ XtX_inv) * X_train, axis=1)
    
    # For test set
    h_test = np.sum((X_test @ XtX_inv) * X_test, axis=1)
    
    # Warning threshold: h* = 3p/n (p = features, n = samples)
    n_samples, n_features = X_train.shape
    h_star = 3 * n_features / n_samples
    
    print(f"  Training set leverage: {h_train.mean():.4f} ± {h_train.std():.4f}")
    print(f"  Test set leverage: {h_test.mean():.4f} ± {h_test.std():.4f}")
    print(f"  Warning threshold h*: {h_star:.4f}")
    
    return h_train, h_test, h_star

def analyze_applicability_domain(X_train, X_test, y_train, y_test, 
                                  y_pred_test, h_train, h_test, h_star):
    """Analyze prediction reliability based on applicability domain"""
    
    # Classify compounds
    in_domain_mask = h_test <= h_star
    out_domain_mask = h_test > h_star
    
    n_in_domain = in_domain_mask.sum()
    n_out_domain = out_domain_mask.sum()
    
    print(f"\nApplicability Domain Analysis:")
    print(f"  Compounds in domain: {n_in_domain} ({100*n_in_domain/len(h_test):.1f}%)")
    print(f"  Compounds out of domain: {n_out_domain} ({100*n_out_domain/len(h_test):.1f}%)")
    
    # Calculate metrics for in-domain vs out-of-domain
    if n_in_domain > 0:
        r2_in = r2_score(y_test[in_domain_mask], y_pred_test[in_domain_mask])
        rmse_in = np.sqrt(mean_squared_error(y_test[in_domain_mask], y_pred_test[in_domain_mask]))
        mae_in = mean_absolute_error(y_test[in_domain_mask], y_pred_test[in_domain_mask])
        print(f"\n  In-domain performance:")
        print(f"    R² = {r2_in:.4f}")
        print(f"    RMSE = {rmse_in:.4f} kcal/mol")
        print(f"    MAE = {mae_in:.4f} kcal/mol")
    else:
        r2_in, rmse_in, mae_in = None, None, None
    
    if n_out_domain > 0:
        r2_out = r2_score(y_test[out_domain_mask], y_pred_test[out_domain_mask])
        rmse_out = np.sqrt(mean_squared_error(y_test[out_domain_mask], y_pred_test[out_domain_mask]))
        mae_out = mean_absolute_error(y_test[out_domain_mask], y_pred_test[out_domain_mask])
        print(f"\n  Out-of-domain performance:")
        print(f"    R² = {r2_out:.4f}")
        print(f"    RMSE = {rmse_out:.4f} kcal/mol")
        print(f"    MAE = {mae_out:.4f} kcal/mol")
    else:
        r2_out, rmse_out, mae_out = None, None, None
    
    results = {
        'n_train': len(h_train),
        'n_test': len(h_test),
        'n_in_domain': int(n_in_domain),
        'n_out_domain': int(n_out_domain),
        'pct_in_domain': float(100 * n_in_domain / len(h_test)),
        'h_star_threshold': float(h_star),
        'h_train_mean': float(h_train.mean()),
        'h_train_std': float(h_train.std()),
        'h_test_mean': float(h_test.mean()),
        'h_test_std': float(h_test.std()),
        'in_domain_r2': float(r2_in) if r2_in is not None else None,
        'in_domain_rmse': float(rmse_in) if rmse_in is not None else None,
        'in_domain_mae': float(mae_in) if mae_in is not None else None,
        'out_domain_r2': float(r2_out) if r2_out is not None else None,
        'out_domain_rmse': float(rmse_out) if rmse_out is not None else None,
        'out_domain_mae': float(mae_out) if mae_out is not None else None,
    }
    
    return results, in_domain_mask, out_domain_mask

def plot_applicability_domain(h_train, h_test, h_star, y_test, y_pred_test, 
                               in_domain_mask, out_domain_mask, results):
    """Create visualizations of applicability domain analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Leverage distribution
    ax = axes[0, 0]
    ax.hist(h_train, bins=50, alpha=0.6, label='Training set', color='blue', edgecolor='black')
    ax.hist(h_test, bins=50, alpha=0.6, label='Test set', color='orange', edgecolor='black')
    ax.axvline(h_star, color='red', linestyle='--', linewidth=2, label=f'h* = {h_star:.4f}')
    ax.set_xlabel('Leverage (h)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Leverage Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 2. Predicted vs Actual (colored by domain)
    ax = axes[0, 1]
    ax.scatter(y_test[in_domain_mask], y_pred_test[in_domain_mask], 
               alpha=0.6, s=30, label=f'In domain (n={in_domain_mask.sum()})', color='green')
    ax.scatter(y_test[out_domain_mask], y_pred_test[out_domain_mask], 
               alpha=0.6, s=30, label=f'Out of domain (n={out_domain_mask.sum()})', color='red')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Binding Affinity (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted Binding Affinity (kcal/mol)', fontsize=12)
    ax.set_title('Predictions by Applicability Domain', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 3. Prediction error vs leverage
    ax = axes[1, 0]
    errors = np.abs(y_test - y_pred_test)
    ax.scatter(h_test[in_domain_mask], errors[in_domain_mask], 
               alpha=0.6, s=30, label='In domain', color='green')
    ax.scatter(h_test[out_domain_mask], errors[out_domain_mask], 
               alpha=0.6, s=30, label='Out of domain', color='red')
    ax.axvline(h_star, color='red', linestyle='--', linewidth=2, label=f'h* = {h_star:.4f}')
    ax.set_xlabel('Leverage (h)', fontsize=12)
    ax.set_ylabel('Absolute Error (kcal/mol)', fontsize=12)
    ax.set_title('Prediction Error vs Leverage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 4. Performance comparison
    ax = axes[1, 1]
    categories = ['In Domain', 'Out of Domain']
    r2_values = [results['in_domain_r2'], results['out_domain_r2']]
    rmse_values = [results['in_domain_rmse'], results['out_domain_rmse']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, r2_values, width, label='R²', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, rmse_values, width, label='RMSE', color='coral', alpha=0.8)
    
    ax.set_ylabel('R² Score', fontsize=12, color='steelblue')
    ax2.set_ylabel('RMSE (kcal/mol)', fontsize=12, color='coral')
    ax.set_xlabel('Applicability Domain', fontsize=12)
    ax.set_title('Performance by Domain', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('paper_results/applicability_domain_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: paper_results/applicability_domain_analysis.png")
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
    
    # Split into train/test (same as original split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} compounds")
    print(f"Test set: {len(X_test)} compounds")
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    
    # Calculate leverage
    h_train, h_test, h_star = calculate_leverage(X_train, X_test)
    
    # Analyze applicability domain
    results, in_domain_mask, out_domain_mask = analyze_applicability_domain(
        X_train, X_test, y_train, y_test, y_pred_test, h_train, h_test, h_star
    )
    
    # Save results
    with open('paper_results/applicability_domain_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: paper_results/applicability_domain_results.json")
    
    # Create visualizations
    plot_applicability_domain(h_train, h_test, h_star, y_test, y_pred_test,
                               in_domain_mask, out_domain_mask, results)
    
    # Interpretation for paper
    print("\n" + "="*70)
    print("INTERPRETATION FOR PAPER")
    print("="*70)
    print(f"\nThe applicability domain analysis ensures predictions are reliable:")
    print(f"\n• {results['pct_in_domain']:.1f}% of test compounds are within the domain")
    print(f"• In-domain R² = {results['in_domain_r2']:.4f} (reliable predictions)")
    
    if results['out_domain_r2'] is not None:
        print(f"• Out-of-domain R² = {results['out_domain_r2']:.4f} (less reliable)")
        
        if results['in_domain_r2'] > results['out_domain_r2']:
            print(f"\n✓ In-domain predictions are more accurate, as expected")
            print(f"  This validates the applicability domain approach.")
    
    print(f"\n• Leverage threshold h* = {results['h_star_threshold']:.4f}")
    print(f"• Compounds with h > h* should be flagged for manual review")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
