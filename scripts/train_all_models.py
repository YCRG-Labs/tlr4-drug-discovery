#!/usr/bin/env python3
"""
Train all models on the expanded TLR4 dataset for the revised paper.

This script trains:
1. Baseline ensemble (RF, ElasticNet, Ridge, Bayesian)
2. GAT (Graph Attention Network)
3. ChemBERTa (Transformer)
4. Hybrid (GNN + descriptors)
5. Transfer learning models

Usage:
    python scripts/train_all_models.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("TLR4 MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info("")
    
    # Load expanded dataset
    logger.info("Loading expanded dataset...")
    dataset_path = Path("binding-data/expanded_dataset.csv")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Please run: python scripts/collect_expanded_dataset.py --include-related-tlrs")
        return
    
    df = pd.read_csv(dataset_path)
    logger.info(f"✓ Loaded {len(df)} compounds")
    logger.info(f"  Binding affinity range: {df['binding_affinity_kcal_mol'].min():.2f} to {df['binding_affinity_kcal_mol'].max():.2f} kcal/mol")
    logger.info(f"  Functional classes:")
    for func_class, count in df['functional_class'].value_counts().items():
        logger.info(f"    {func_class}: {count}")
    logger.info("")
    
    # Create output directories
    output_dir = Path("paper_results")
    models_dir = Path("models/trained")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    results = {}
    
    # 1. Baseline Ensemble
    logger.info("=" * 80)
    logger.info("1. TRAINING BASELINE ENSEMBLE")
    logger.info("=" * 80)
    try:
        from tlr4_binding.pipeline import train_baseline_ensemble
        baseline_results = train_baseline_ensemble(df)
        results['baseline'] = baseline_results
        logger.info(f"✓ Baseline ensemble trained")
        logger.info(f"  Test R²: {baseline_results.get('test_r2', 'N/A'):.3f}")
    except Exception as e:
        logger.error(f"✗ Baseline training failed: {e}")
        results['baseline'] = {'error': str(e)}
    logger.info("")
    
    # 2. GAT Model
    logger.info("=" * 80)
    logger.info("2. TRAINING GAT MODEL")
    logger.info("=" * 80)
    logger.info("Note: GAT training requires molecular graphs and may take 30-60 minutes")
    try:
        from tlr4_binding.models.gat import train_gat_model
        gat_results = train_gat_model(df)
        results['gat'] = gat_results
        logger.info(f"✓ GAT model trained")
        logger.info(f"  Test R²: {gat_results.get('test_r2', 'N/A'):.3f}")
    except Exception as e:
        logger.warning(f"⚠ GAT training skipped: {e}")
        logger.info("  This is optional for the paper - you can use baseline results")
        results['gat'] = {'error': str(e), 'skipped': True}
    logger.info("")
    
    # 3. ChemBERTa Model
    logger.info("=" * 80)
    logger.info("3. TRAINING CHEMBERTA MODEL")
    logger.info("=" * 80)
    logger.info("Note: ChemBERTa requires transformers and may take 30-60 minutes")
    try:
        from tlr4_binding.models.chemberta import train_chemberta_model
        chemberta_results = train_chemberta_model(df)
        results['chemberta'] = chemberta_results
        logger.info(f"✓ ChemBERTa model trained")
        logger.info(f"  Test R²: {chemberta_results.get('test_r2', 'N/A'):.3f}")
    except Exception as e:
        logger.warning(f"⚠ ChemBERTa training skipped: {e}")
        logger.info("  This is optional for the paper - you can use baseline results")
        results['chemberta'] = {'error': str(e), 'skipped': True}
    logger.info("")
    
    # 4. Hybrid Model
    logger.info("=" * 80)
    logger.info("4. TRAINING HYBRID MODEL")
    logger.info("=" * 80)
    logger.info("Note: Hybrid model combines GNN + descriptors")
    try:
        from tlr4_binding.models.hybrid import train_hybrid_model
        hybrid_results = train_hybrid_model(df)
        results['hybrid'] = hybrid_results
        logger.info(f"✓ Hybrid model trained")
        logger.info(f"  Test R²: {hybrid_results.get('test_r2', 'N/A'):.3f}")
    except Exception as e:
        logger.warning(f"⚠ Hybrid training skipped: {e}")
        logger.info("  This is optional for the paper - you can use baseline results")
        results['hybrid'] = {'error': str(e), 'skipped': True}
    logger.info("")
    
    # 5. Transfer Learning
    logger.info("=" * 80)
    logger.info("5. TRAINING WITH TRANSFER LEARNING")
    logger.info("=" * 80)
    logger.info("Note: Requires related TLR data")
    try:
        from tlr4_binding.models.transfer_learning import train_with_transfer_learning
        transfer_results = train_with_transfer_learning(df)
        results['transfer'] = transfer_results
        logger.info(f"✓ Transfer learning model trained")
        logger.info(f"  Test R²: {transfer_results.get('test_r2', 'N/A'):.3f}")
    except Exception as e:
        logger.warning(f"⚠ Transfer learning skipped: {e}")
        logger.info("  This is optional for the paper - you can use baseline results")
        results['transfer'] = {'error': str(e), 'skipped': True}
    logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    
    successful_models = [name for name, res in results.items() if 'error' not in res]
    failed_models = [name for name, res in results.items() if 'error' in res and not res.get('skipped', False)]
    skipped_models = [name for name, res in results.items() if res.get('skipped', False)]
    
    logger.info(f"✓ Successfully trained: {len(successful_models)} models")
    for model_name in successful_models:
        test_r2 = results[model_name].get('test_r2', 'N/A')
        logger.info(f"  {model_name.upper()}: R² = {test_r2:.3f}" if isinstance(test_r2, float) else f"  {model_name.upper()}: R² = {test_r2}")
    
    if skipped_models:
        logger.info(f"\n⚠ Skipped (optional): {len(skipped_models)} models")
        for model_name in skipped_models:
            logger.info(f"  {model_name.upper()}")
    
    if failed_models:
        logger.info(f"\n✗ Failed: {len(failed_models)} models")
        for model_name in failed_models:
            logger.info(f"  {model_name.upper()}: {results[model_name]['error']}")
    
    # Save results
    results_file = output_dir / "training_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("1. Run validation: python examples/demo_full_validation_suite.py")
    logger.info("2. Compare models: python examples/demo_model_benchmarker.py")
    logger.info("3. Generate interpretability: python examples/demo_interpretability_outputs.py")
    logger.info("4. Write the paper using results in paper_results/")
    logger.info("")
    logger.info(f"Completed at: {datetime.now()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
