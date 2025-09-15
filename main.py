#!/usr/bin/env python3
"""
Main entry point for TLR4 Binding Affinity Prediction System.

This script provides a command-line interface for running the complete
TLR4 binding prediction pipeline.
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tlr4_binding.config.settings import get_config
from tlr4_binding.utils.logging_config import setup_logging
from tlr4_binding.molecular_analysis import MolecularFeatureExtractor
from tlr4_binding.data_processing import DataPreprocessor
from tlr4_binding.ml_components import MLModelTrainer

logger = logging.getLogger(__name__)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="TLR4 Binding Affinity Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --pipeline complete --pdbqt-dir binding-data/raw/pdbqt --binding-csv binding-data/processed/processed_logs.csv
  
  # Extract features only
  python main.py --pipeline features --pdbqt-dir binding-data/raw/pdbqt
  
  # Train models only
  python main.py --pipeline train --features-csv data/processed/features.csv --binding-csv binding-data/processed/processed_logs.csv
  
  # Make predictions
  python main.py --pipeline predict --model-path models/trained/best_model.joblib --pdbqt-file compound.pdbqt
        """
    )
    
    # Pipeline options
    parser.add_argument('--pipeline', 
                       choices=['complete', 'features', 'train', 'predict'],
                       required=True,
                       help='Pipeline stage to run')
    
    # Data paths
    parser.add_argument('--pdbqt-dir', 
                       type=str,
                       help='Directory containing PDBQT files')
    parser.add_argument('--pdbqt-file',
                       type=str,
                       help='Single PDBQT file for prediction')
    parser.add_argument('--binding-csv',
                       type=str,
                       help='Path to binding affinity CSV file')
    parser.add_argument('--features-csv',
                       type=str,
                       help='Path to features CSV file')
    
    # Model options
    parser.add_argument('--model-path',
                       type=str,
                       help='Path to trained model file')
    parser.add_argument('--output-dir',
                       type=str,
                       default='results',
                       help='Output directory for results')
    
    # Configuration options
    parser.add_argument('--config',
                       type=str,
                       help='Path to configuration file')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Load configuration
    config = get_config()
    if args.config:
        config = config.load_config(args.config)
    
    logger.info("Starting TLR4 Binding Affinity Prediction System")
    logger.info(f"Pipeline stage: {args.pipeline}")
    
    try:
        if args.pipeline == 'complete':
            run_complete_pipeline(args, config)
        elif args.pipeline == 'features':
            run_feature_extraction(args, config)
        elif args.pipeline == 'train':
            run_model_training(args, config)
        elif args.pipeline == 'predict':
            run_prediction(args, config)
        else:
            logger.error(f"Unknown pipeline stage: {args.pipeline}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully")


def run_complete_pipeline(args, config):
    """Run the complete prediction pipeline."""
    logger.info("Running complete pipeline")
    
    # Step 1: Extract molecular features
    logger.info("Step 1: Extracting molecular features")
    feature_extractor = MolecularFeatureExtractor()
    
    if not args.pdbqt_dir:
        raise ValueError("--pdbqt-dir is required for complete pipeline")
    
    features_df = feature_extractor.batch_extract(args.pdbqt_dir)
    logger.info(f"Extracted features for {len(features_df)} compounds")
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data")
    preprocessor = DataPreprocessor()
    
    if not args.binding_csv:
        raise ValueError("--binding-csv is required for complete pipeline")
    
    integrated_df = preprocessor.preprocess_pipeline(features_df, args.binding_csv)
    logger.info(f"Integrated dataset with {len(integrated_df)} records")
    
    # Step 3: Train models
    logger.info("Step 3: Training models")
    trainer = MLModelTrainer()
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # Drop non-numeric columns for ML training
    columns_to_drop = [config.data.affinity_column]
    
    # Add string/object columns that should not be used for training
    string_columns = ['compound_name', 'pdbqt_file', 'smiles', 'inchi', 'ligand', 'matched_compound']
    for col in string_columns:
        if col in integrated_df.columns:
            columns_to_drop.append(col)
    
    X = integrated_df.drop(columns=columns_to_drop)
    y = integrated_df[config.data.affinity_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.data.random_state
    )
    
    # Train models
    trained_models = trainer.train_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    logger.info(f"Best model: {best_model_name}")
    
    # Step 4: Generate results
    logger.info("Step 4: Generating results")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame(evaluation_results[best_model_name]['predictions'])
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    logger.info(f"Results saved to {output_dir}")


def run_feature_extraction(args, config):
    """Run feature extraction only."""
    logger.info("Running feature extraction")
    
    if not args.pdbqt_dir:
        raise ValueError("--pdbqt-dir is required for feature extraction")
    
    feature_extractor = MolecularFeatureExtractor()
    features_df = feature_extractor.batch_extract(args.pdbqt_dir)
    
    # Save features
    output_path = args.output_dir + '/features.csv'
    features_df.to_csv(output_path, index=False)
    logger.info(f"Features saved to {output_path}")


def run_model_training(args, config):
    """Run model training only."""
    logger.info("Running model training")
    
    if not args.features_csv or not args.binding_csv:
        raise ValueError("--features-csv and --binding-csv are required for training")
    
    # Load data
    features_df = pd.read_csv(args.features_csv)
    preprocessor = DataPreprocessor()
    integrated_df = preprocessor.preprocess_pipeline(features_df, args.binding_csv)
    
    # Train models
    trainer = MLModelTrainer()
    
    # Drop non-numeric columns for ML training
    columns_to_drop = [config.data.affinity_column]
    
    # Add string/object columns that should not be used for training
    string_columns = ['compound_name', 'pdbqt_file', 'smiles', 'inchi', 'ligand', 'matched_compound']
    for col in string_columns:
        if col in integrated_df.columns:
            columns_to_drop.append(col)
    
    X = integrated_df.drop(columns=columns_to_drop)
    y = integrated_df[config.data.affinity_column]
    
    trained_models = trainer.train_models(X, y)
    logger.info(f"Trained {len(trained_models)} models")


def run_prediction(args, config):
    """Run prediction only."""
    logger.info("Running prediction")
    
    if not args.model_path or not args.pdbqt_file:
        raise ValueError("--model-path and --pdbqt-file are required for prediction")
    
    # Load model and make prediction
    from tlr4_binding.ml_components import BindingPredictor
    
    predictor = BindingPredictor(args.model_path)
    
    # Extract features for single compound
    feature_extractor = MolecularFeatureExtractor()
    features = feature_extractor.extract_features(args.pdbqt_file)
    
    # Make prediction
    result = predictor.predict_single(features)
    
    print(f"Predicted binding affinity: {result.predicted_affinity:.3f} kcal/mol")
    print(f"Confidence interval: [{result.confidence_interval_lower:.3f}, {result.confidence_interval_upper:.3f}]")


if __name__ == "__main__":
    main()
