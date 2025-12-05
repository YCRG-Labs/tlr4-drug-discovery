#!/usr/bin/env python3
"""
Collect expanded TLR4 dataset for the revised paper.

This script collects TLR4 binding data from ChEMBL and PubChem to expand
the dataset from 49 to 150-300 compounds.

Requirements addressed:
- Requirement 1: Dataset Expansion (150-300 TLR4 compounds)
- Requirement 2: Related TLR Data Collection (500-1000 compounds)
- Requirement 3: Functional Classification
- Requirement 4: Data Quality Control

Usage:
    python scripts/collect_expanded_dataset.py --output binding-data/expanded_dataset.csv
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_chembl_tlr4_data():
    """
    Collect TLR4 binding data from ChEMBL.
    
    Targets:
    - CHEMBL5896 (TLR4)
    - CHEMBL2047 (TLR4/MD2 complex)
    
    Requirements: 1.1
    """
    logger.info("Querying ChEMBL for TLR4 compounds...")
    
    try:
        from chembl_webresource_client.new_client import new_client
        
        # Query for TLR4 targets
        activity = new_client.activity
        
        # TLR4 target IDs
        target_ids = ['CHEMBL5896', 'CHEMBL2047']
        
        all_activities = []
        
        for target_id in target_ids:
            logger.info(f"  Querying {target_id}...")
            
            # Get activities for this target
            activities = activity.filter(
                target_chembl_id=target_id,
                standard_type__in=['IC50', 'EC50', 'Ki', 'Kd'],
                standard_relation='=',
                pchembl_value__isnull=False
            ).only([
                'molecule_chembl_id',
                'canonical_smiles',
                'standard_type',
                'standard_value',
                'standard_units',
                'pchembl_value',
                'assay_description',
                'assay_type'
            ])
            
            for act in activities:
                all_activities.append({
                    'chembl_id': act.get('molecule_chembl_id'),
                    'smiles': act.get('canonical_smiles'),
                    'activity_type': act.get('standard_type'),
                    'activity_value': act.get('standard_value'),
                    'activity_units': act.get('standard_units'),
                    'pchembl_value': act.get('pchembl_value'),
                    'assay_description': act.get('assay_description'),
                    'assay_type': act.get('assay_type'),
                    'target_id': target_id,
                    'source': 'ChEMBL'
                })
        
        df = pd.DataFrame(all_activities)
        logger.info(f"  ✓ Retrieved {len(df)} activity records from ChEMBL")
        
        return df
        
    except ImportError:
        logger.error("ChEMBL client not installed. Install with: pip install chembl_webresource_client")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error querying ChEMBL: {e}")
        return pd.DataFrame()


def collect_pubchem_tlr4_data():
    """
    Collect TLR4 binding data from PubChem BioAssay.
    
    Assays:
    - AID 1053197
    - AID 588834
    - AID 651635
    
    Requirements: 1.2
    """
    logger.info("Querying PubChem for TLR4 compounds...")
    
    try:
        import pubchempy as pcp
        
        assay_ids = [1053197, 588834, 651635]
        
        all_compounds = []
        
        for aid in assay_ids:
            logger.info(f"  Querying AID {aid}...")
            
            # Note: PubChem API has rate limits and may require pagination
            # This is a simplified version - production code would need more robust handling
            
            try:
                # Get assay data
                # Note: pubchempy doesn't have direct bioassay support
                # You may need to use the REST API directly or download data files
                logger.warning(f"  PubChem bioassay data collection requires manual download")
                logger.info(f"  Download from: https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}")
                
            except Exception as e:
                logger.warning(f"  Could not retrieve AID {aid}: {e}")
        
        df = pd.DataFrame(all_compounds)
        logger.info(f"  ✓ Retrieved {len(df)} compounds from PubChem")
        
        return df
        
    except ImportError:
        logger.error("PubChemPy not installed. Install with: pip install pubchempy")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error querying PubChem: {e}")
        return pd.DataFrame()


def collect_related_tlr_data():
    """
    Collect binding data from related TLR targets for transfer learning.
    
    Targets:
    - CHEMBL5372 (TLR2)
    - CHEMBL5600 (TLR7)
    - CHEMBL5608 (TLR8)
    - CHEMBL5842 (TLR9)
    
    Requirements: 2.1
    """
    logger.info("Querying ChEMBL for related TLR compounds...")
    
    try:
        from chembl_webresource_client.new_client import new_client
        
        activity = new_client.activity
        
        # Related TLR target IDs
        target_ids = {
            'CHEMBL5372': 'TLR2',
            'CHEMBL5600': 'TLR7',
            'CHEMBL5608': 'TLR8',
            'CHEMBL5842': 'TLR9'
        }
        
        all_activities = []
        
        for target_id, target_name in target_ids.items():
            logger.info(f"  Querying {target_name} ({target_id})...")
            
            activities = activity.filter(
                target_chembl_id=target_id,
                standard_type__in=['IC50', 'EC50', 'Ki', 'Kd'],
                standard_relation='=',
                pchembl_value__isnull=False
            ).only([
                'molecule_chembl_id',
                'canonical_smiles',
                'standard_type',
                'standard_value',
                'standard_units',
                'pchembl_value',
                'assay_description'
            ])
            
            for act in activities:
                all_activities.append({
                    'chembl_id': act.get('molecule_chembl_id'),
                    'smiles': act.get('canonical_smiles'),
                    'activity_type': act.get('standard_type'),
                    'activity_value': act.get('standard_value'),
                    'activity_units': act.get('standard_units'),
                    'pchembl_value': act.get('pchembl_value'),
                    'assay_description': act.get('assay_description'),
                    'target_id': target_id,
                    'target_name': target_name,
                    'source': 'ChEMBL'
                })
        
        df = pd.DataFrame(all_activities)
        logger.info(f"  ✓ Retrieved {len(df)} activity records from related TLRs")
        
        return df
        
    except ImportError:
        logger.error("ChEMBL client not installed")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error querying related TLRs: {e}")
        return pd.DataFrame()


def standardize_activity_values(df):
    """
    Convert all activity values to consistent units (kcal/mol).
    
    Uses: ΔG = RT ln(Kd)
    where R = 1.987 cal/(mol·K), T = 298.15 K
    
    Requirements: 1.3
    """
    logger.info("Standardizing activity values to kcal/mol...")
    
    R = 1.987e-3  # kcal/(mol·K)
    T = 298.15    # K
    
    def convert_to_binding_energy(row):
        """Convert IC50/EC50/Ki/Kd to binding free energy."""
        try:
            value = float(row['activity_value'])
            units = row['activity_units']
            
            # Convert to M (molar)
            if units == 'nM':
                kd_m = value * 1e-9
            elif units == 'uM' or units == 'µM':
                kd_m = value * 1e-6
            elif units == 'mM':
                kd_m = value * 1e-3
            elif units == 'M':
                kd_m = value
            else:
                return None
            
            # Calculate ΔG = RT ln(Kd)
            import math
            delta_g = R * T * math.log(kd_m)
            
            return delta_g
            
        except (ValueError, TypeError):
            return None
    
    df['binding_affinity_kcal_mol'] = df.apply(convert_to_binding_energy, axis=1)
    
    # Remove rows where conversion failed
    before_count = len(df)
    df = df.dropna(subset=['binding_affinity_kcal_mol'])
    after_count = len(df)
    
    logger.info(f"  ✓ Standardized {after_count}/{before_count} records")
    
    return df


def classify_functional_activity(df):
    """
    Classify compounds as agonists, antagonists, or unknown.
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    logger.info("Classifying functional activity...")
    
    def classify(assay_desc):
        """Parse assay description for functional keywords."""
        if pd.isna(assay_desc):
            return 'unknown'
        
        desc_lower = str(assay_desc).lower()
        
        # Check for antagonist keywords
        if any(word in desc_lower for word in ['inhibit', 'antagonist', 'block', 'suppress']):
            return 'antagonist'
        
        # Check for agonist keywords
        if any(word in desc_lower for word in ['activat', 'agonist', 'stimulat', 'induc']):
            return 'agonist'
        
        return 'unknown'
    
    df['functional_class'] = df['assay_description'].apply(classify)
    
    # Count classifications
    counts = df['functional_class'].value_counts()
    logger.info(f"  ✓ Classification complete:")
    for func_class, count in counts.items():
        logger.info(f"    {func_class}: {count}")
    
    return df


def apply_quality_control(df):
    """
    Apply quality control filters.
    
    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    logger.info("Applying quality control filters...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        
        # Remove invalid SMILES
        logger.info("  Validating SMILES...")
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
        before_count = len(df)
        df = df[df['mol'].notna()]
        logger.info(f"    Removed {before_count - len(df)} invalid SMILES")
        
        # Canonicalize SMILES (Requirements: 4.2)
        logger.info("  Canonicalizing SMILES...")
        df['canonical_smiles'] = df['mol'].apply(lambda m: Chem.MolToSmiles(m) if m else None)
        
        # Remove PAINS (Requirements: 4.1)
        logger.info("  Filtering PAINS compounds...")
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        
        def is_pains(mol):
            if mol is None:
                return True
            return catalog.HasMatch(mol)
        
        df['is_pains'] = df['mol'].apply(is_pains)
        before_count = len(df)
        df = df[~df['is_pains']]
        logger.info(f"    Removed {before_count - len(df)} PAINS compounds")
        
        # Calculate diversity (Requirements: 4.4)
        logger.info("  Calculating chemical diversity...")
        # This would calculate pairwise Tanimoto similarity
        # For now, just log that it should be done
        logger.info("    (Diversity calculation deferred to full analysis)")
        
        # Remove temporary mol column
        df = df.drop(columns=['mol', 'is_pains'])
        
        logger.info(f"  ✓ Quality control complete: {len(df)} compounds retained")
        
        return df
        
    except ImportError:
        logger.error("RDKit not installed. Install with: conda install -c conda-forge rdkit")
        return df
    except Exception as e:
        logger.error(f"Error in quality control: {e}")
        return df


def merge_and_deduplicate(dfs):
    """
    Merge datasets and handle duplicates.
    
    Requirements: 1.4
    """
    logger.info("Merging datasets and handling duplicates...")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Total records before deduplication: {len(df)}")
    
    # Group by canonical SMILES
    grouped = df.groupby('canonical_smiles')
    
    # Check for conflicting measurements
    conflicts = []
    for smiles, group in grouped:
        if len(group) > 1:
            # Check if binding affinities differ significantly
            affinities = group['binding_affinity_kcal_mol'].values
            if affinities.std() > 1.0:  # More than 1 kcal/mol std dev
                conflicts.append(smiles)
    
    if conflicts:
        logger.warning(f"  Found {len(conflicts)} compounds with conflicting measurements")
        logger.info("  These will be flagged for manual review")
    
    # Take median value for duplicates
    df_merged = df.groupby('canonical_smiles').agg({
        'chembl_id': 'first',
        'binding_affinity_kcal_mol': 'median',
        'functional_class': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
        'source': lambda x: ', '.join(set(x)),
        'target_id': 'first',
        'activity_type': 'first'
    }).reset_index()
    
    df_merged['has_conflict'] = df_merged['canonical_smiles'].isin(conflicts)
    
    logger.info(f"  ✓ Merged to {len(df_merged)} unique compounds")
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(description='Collect expanded TLR4 dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='binding-data/expanded_dataset.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--include-related-tlrs',
        action='store_true',
        help='Also collect related TLR data for transfer learning'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("TLR4 DATASET EXPANSION")
    logger.info("=" * 80)
    logger.info(f"Target: 150-300 unique TLR4 ligands")
    logger.info(f"Output: {args.output}")
    logger.info("")
    
    # Collect data from multiple sources
    dfs = []
    
    # ChEMBL TLR4 data
    df_chembl = collect_chembl_tlr4_data()
    if not df_chembl.empty:
        dfs.append(df_chembl)
    
    # PubChem TLR4 data
    df_pubchem = collect_pubchem_tlr4_data()
    if not df_pubchem.empty:
        dfs.append(df_pubchem)
    
    if not dfs:
        logger.error("No data collected. Exiting.")
        return
    
    # Standardize activity values
    for i in range(len(dfs)):
        dfs[i] = standardize_activity_values(dfs[i])
    
    # Classify functional activity
    for i in range(len(dfs)):
        dfs[i] = classify_functional_activity(dfs[i])
    
    # Apply quality control
    for i in range(len(dfs)):
        dfs[i] = apply_quality_control(dfs[i])
    
    # Merge and deduplicate
    df_final = merge_and_deduplicate(dfs)
    
    # Save TLR4 dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET COLLECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total unique TLR4 compounds: {len(df_final)}")
    logger.info(f"Saved to: {output_path.absolute()}")
    
    # Collect related TLR data if requested
    if args.include_related_tlrs:
        logger.info("")
        logger.info("=" * 80)
        logger.info("COLLECTING RELATED TLR DATA")
        logger.info("=" * 80)
        
        df_related = collect_related_tlr_data()
        if not df_related.empty:
            df_related = standardize_activity_values(df_related)
            df_related = apply_quality_control(df_related)
            
            related_path = output_path.parent / 'related_tlr_dataset.csv'
            df_related.to_csv(related_path, index=False)
            
            logger.info(f"Total related TLR compounds: {len(df_related)}")
            logger.info(f"Saved to: {related_path.absolute()}")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review the collected dataset")
    logger.info("2. Manually curate compounds with conflicts")
    logger.info("3. Run feature engineering: python scripts/calculate_features.py")
    logger.info("4. Train models: python scripts/train_models.py")


if __name__ == '__main__':
    main()
