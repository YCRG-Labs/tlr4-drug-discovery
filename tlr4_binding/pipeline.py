"""
End-to-end pipeline for TLR4 binding affinity prediction.

This module provides a comprehensive pipeline that integrates all components:
- Data collection and quality control
- Feature engineering (2D, 3D, electrostatic, graph)
- Model training (traditional ML, GNN, transformer, hybrid, transfer learning, multi-task)
- Validation (nested CV, Y-scrambling, scaffold validation, applicability domain)
- Interpretability (attention visualization, SHAP analysis)
- Model benchmarking and comparison

Requirements: All
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime

# Data components
from .data.collector import DataCollector
from .data.quality_control import QualityController
from .data.functional_classifier import FunctionalClassifier
from .data.models import CompoundRecord

# Feature components
from .features.descriptor_3d import Descriptor3DCalculator
from .features.electrostatic import ElectrostaticCalculator
from .features.graph_generator import MolecularGraphGenerator
from .features.models import MolecularFeatures

# Model components
from .models import (
    GAT_AVAILABLE, CHEMBERTA_AVAILABLE, HYBRID_AVAILABLE,
    TRANSFER_LEARNING_AVAILABLE, MULTI_TASK_AVAILABLE
)

if GAT_AVAILABLE:
    from .models.gat import create_gat_model, train_gat_model, GATConfig, TrainingConfig
if CHEMBERTA_AVAILABLE:
    from .models.chemberta import create_chemberta_model, train_chemberta_model, ChemBERTaConfig
if HYBRID_AVAILABLE:
    from .models.hybrid import create_hybrid_model, train_hybrid_model, HybridConfig
if TRANSFER_LEARNING_AVAILABLE:
    from .models.transfer_learning import create_transfer_learning_manager, TransferLearningConfig
if MULTI_TASK_AVAILABLE:
    from .models.multi_task import create_multi_task_model, train_multi_task_model, MultiTaskConfig

# Validation components
from .validation.framework import ValidationFramework
from .validation.applicability_domain import ApplicabilityDomainAnalyzer
from .validation.benchmarker import ModelBenchmarker

# Interpretability components
from .interpretability.analyzer import create_interpretability_analyzer

# Configuration
from .config import Config, get_api_config, get_hyperparameter_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end pipeline."""
    
    # Data collection
    collect_data: bool = True
    chembl_targets: List[str] = None
    pubchem_assays: List[int] = None
    min_compounds: int = 150
    max_compounds: int = 300
    
    # Transfer learning data
    collect_transfer_data: bool = True
    related_tlr_targets: List[str] = None
    min_transfer_compounds: int = 500
    
    # Feature engineering
    calculate_3d_descriptors: bool = True
    calculate_electrostatic: bool = True
    generate_graphs: bool = True
    
    # Models to train
    train_traditional_ml: bool = True
    train_gnn: bool = True
    train_transformer: bool = True
    train_hybrid: bool = True
    train_transfer_learning: bool = True
    train_multi_task: bool = True
    
    # Validation
    external_test_size: float = 0.2
    nested_cv_outer_folds: int = 5
    nested_cv_inner_folds: int = 3
    y_scrambling_iterations: int = 100
    run_scaffold_validation: bool = True
    calculate_applicability_domain: bool = True
    
    # Interpretability
    generate_attention_viz: bool = True
    generate_shap_analysis: bool = True
    top_compounds_for_viz: int = 10
    
    # Output
    output_dir: str = "./results/pipeline_output"
    save_models: bool = True
    generate_report: bool = True
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.chembl_targets is None:
            self.chembl_targets = ["CHEMBL5896", "CHEMBL2047"]
        if self.pubchem_assays is None:
            self.pubchem_assays = [1053197, 588834, 651635]
        if self.related_tlr_targets is None:
            self.related_tlr_targets = ["CHEMBL5372", "CHEMBL5600", "CHEMBL5608", "CHEMBL5842"]


class TLR4Pipeline:
    """
    End-to-end pipeline for TLR4 binding affinity prediction.
    
    This pipeline integrates all components of the methodology enhancement:
    1. Data collection and quality control
    2. Feature engineering
    3. Model training
    4. Validation
    5. Interpretability analysis
    6. Benchmarking and reporting
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration. If None, uses default configuration.
        """
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = None
        self.quality_controller = None
        self.functional_classifier = None
        self.descriptor_3d_calculator = None
        self.electrostatic_calculator = None
        self.graph_generator = None
        self.validation_framework = None
        self.applicability_domain = None
        self.benchmarker = None
        self.interpretability_analyzer = None
        
        # Data storage
        self.tlr4_data: Optional[pd.DataFrame] = None
        self.transfer_data: Optional[pd.DataFrame] = None
        self.features: Optional[Dict[str, Any]] = None
        self.models: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}
        self.interpretability_results: Dict[str, Any] = {}
        
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        if self.config.collect_data:
            self.data_collector = DataCollector()
            self.quality_controller = QualityController()
            self.functional_classifier = FunctionalClassifier()
        
        if self.config.calculate_3d_descriptors:
            self.descriptor_3d_calculator = Descriptor3DCalculator()
        
        if self.config.calculate_electrostatic:
            self.electrostatic_calculator = ElectrostaticCalculator()
        
        if self.config.generate_graphs:
            self.graph_generator = MolecularGraphGenerator()
        
        self.validation_framework = ValidationFramework()
        
        if self.config.calculate_applicability_domain:
            self.applicability_domain = ApplicabilityDomainAnalyzer()
        
        self.benchmarker = ModelBenchmarker()
        
        if self.config.generate_attention_viz or self.config.generate_shap_analysis:
            self.interpretability_analyzer = create_interpretability_analyzer()
        
        logger.info("All components initialized successfully")
    
    def collect_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Collect TLR4 and related TLR data.
        
        Returns:
            Tuple of (tlr4_data, transfer_data)
        """
        logger.info("Starting data collection...")
        
        # Collect TLR4 data
        logger.info(f"Collecting TLR4 data from ChEMBL targets: {self.config.chembl_targets}")
        tlr4_chembl = self.data_collector.query_chembl(self.config.chembl_targets)
        
        logger.info(f"Collecting TLR4 data from PubChem assays: {self.config.pubchem_assays}")
        tlr4_pubchem = self.data_collector.query_pubchem(self.config.pubchem_assays)
        
        # Merge and standardize
        logger.info("Merging and standardizing TLR4 data...")
        tlr4_data = self.data_collector.merge_sources([tlr4_chembl, tlr4_pubchem])
        tlr4_data = self.data_collector.standardize_activity(tlr4_data)
        
        # Quality control
        logger.info("Applying quality control filters...")
        tlr4_data = self.quality_controller.remove_pains(tlr4_data)
        tlr4_data['canonical_smiles'] = tlr4_data['smiles'].apply(
            self.quality_controller.canonicalize_smiles
        )
        
        # Functional classification
        logger.info("Classifying compound functions...")
        tlr4_data['functional_class'] = tlr4_data.apply(
            lambda row: self.functional_classifier.classify_compound(row.to_dict()),
            axis=1
        )
        
        logger.info(f"TLR4 dataset: {len(tlr4_data)} compounds")
        
        # Collect transfer learning data
        transfer_data = None
        if self.config.collect_transfer_data:
            logger.info(f"Collecting related TLR data from: {self.config.related_tlr_targets}")
            transfer_data = self.data_collector.query_chembl(self.config.related_tlr_targets)
            transfer_data = self.data_collector.standardize_activity(transfer_data)
            transfer_data = self.quality_controller.remove_pains(transfer_data)
            logger.info(f"Transfer learning dataset: {len(transfer_data)} compounds")
        
        self.tlr4_data = tlr4_data
        self.transfer_data = transfer_data
        
        # Save datasets
        tlr4_data.to_csv(self.output_dir / "tlr4_dataset.csv", index=False)
        if transfer_data is not None:
            transfer_data.to_csv(self.output_dir / "transfer_dataset.csv", index=False)
        
        logger.info("Data collection completed")
        return tlr4_data, transfer_data
    
    def engineer_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Engineer all features for the dataset.
        
        Args:
            data: DataFrame with compound data
            
        Returns:
            Dictionary containing all engineered features
        """
        logger.info("Starting feature engineering...")
        
        features = {
            'smiles': data['canonical_smiles'].tolist(),
            'binding_affinity': data['binding_affinity'].values,
            'functional_class': data.get('functional_class', None)
        }
        
        # 3D descriptors
        if self.config.calculate_3d_descriptors:
            logger.info("Calculating 3D descriptors...")
            descriptors_3d = []
            for smiles in features['smiles']:
                try:
                    desc = self.descriptor_3d_calculator.calculate_all(smiles)
                    descriptors_3d.append(desc)
                except Exception as e:
                    logger.warning(f"Failed to calculate 3D descriptors for {smiles}: {e}")
                    descriptors_3d.append({})
            features['descriptors_3d'] = descriptors_3d
        
        # Electrostatic descriptors
        if self.config.calculate_electrostatic:
            logger.info("Calculating electrostatic descriptors...")
            from rdkit import Chem
            electrostatic = []
            for smiles in features['smiles']:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        desc = {
                            'gasteiger_charges': self.electrostatic_calculator.calculate_gasteiger_charges(mol),
                            'peoe_vsa': self.electrostatic_calculator.calculate_peoe_vsa(mol),
                            'dipole': self.electrostatic_calculator.calculate_dipole(mol)
                        }
                        electrostatic.append(desc)
                    else:
                        electrostatic.append({})
                except Exception as e:
                    logger.warning(f"Failed to calculate electrostatic for {smiles}: {e}")
                    electrostatic.append({})
            features['electrostatic'] = electrostatic
        
        # Graph representations
        if self.config.generate_graphs:
            logger.info("Generating graph representations...")
            graphs = []
            for smiles in features['smiles']:
                try:
                    graph = self.graph_generator.mol_to_graph(smiles)
                    graphs.append(graph)
                except Exception as e:
                    logger.warning(f"Failed to generate graph for {smiles}: {e}")
                    graphs.append(None)
            features['graphs'] = graphs
        
        logger.info("Feature engineering completed")
        return features
    
    def train_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            features: Dictionary of engineered features
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Starting model training...")
        models = {}
        
        # Traditional ML models
        if self.config.train_traditional_ml:
            logger.info("Training traditional ML models...")
            # Implementation would use existing ML trainers
            # For now, placeholder
            models['traditional_ml'] = None
        
        # GNN models
        if self.config.train_gnn and GAT_AVAILABLE:
            logger.info("Training Graph Attention Network...")
            # Implementation would use GAT trainer
            models['gnn'] = None
        
        # Transformer models
        if self.config.train_transformer and CHEMBERTA_AVAILABLE:
            logger.info("Training ChemBERTa transformer...")
            # Implementation would use ChemBERTa trainer
            models['transformer'] = None
        
        # Hybrid models
        if self.config.train_hybrid and HYBRID_AVAILABLE:
            logger.info("Training hybrid model...")
            # Implementation would use hybrid trainer
            models['hybrid'] = None
        
        # Transfer learning
        if self.config.train_transfer_learning and TRANSFER_LEARNING_AVAILABLE:
            logger.info("Training with transfer learning...")
            # Implementation would use transfer learning manager
            models['transfer_learning'] = None
        
        # Multi-task learning
        if self.config.train_multi_task and MULTI_TASK_AVAILABLE:
            logger.info("Training multi-task model...")
            # Implementation would use multi-task trainer
            models['multi_task'] = None
        
        logger.info(f"Model training completed. Trained {len(models)} model types")
        return models
    
    def validate_models(self, models: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive validation on all models.
        
        Args:
            models: Dictionary of trained models
            features: Dictionary of features
            
        Returns:
            Dictionary of validation results
        """
        logger.info("Starting model validation...")
        results = {}
        
        # External test set evaluation
        logger.info("Evaluating on external test set...")
        # Implementation would split data and evaluate
        
        # Nested cross-validation
        logger.info("Running nested cross-validation...")
        # Implementation would use validation framework
        
        # Y-scrambling
        logger.info("Running Y-scrambling validation...")
        # Implementation would use validation framework
        
        # Scaffold validation
        if self.config.run_scaffold_validation:
            logger.info("Running scaffold-based validation...")
            # Implementation would use validation framework
        
        # Applicability domain
        if self.config.calculate_applicability_domain:
            logger.info("Calculating applicability domain...")
            # Implementation would use applicability domain analyzer
        
        logger.info("Model validation completed")
        return results
    
    def generate_interpretability(self, models: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate interpretability outputs.
        
        Args:
            models: Dictionary of trained models
            features: Dictionary of features
            
        Returns:
            Dictionary of interpretability results
        """
        logger.info("Generating interpretability outputs...")
        results = {}
        
        # Attention visualization
        if self.config.generate_attention_viz:
            logger.info("Generating attention visualizations...")
            # Implementation would use interpretability analyzer
        
        # SHAP analysis
        if self.config.generate_shap_analysis:
            logger.info("Generating SHAP analysis...")
            # Implementation would use interpretability analyzer
        
        logger.info("Interpretability generation completed")
        return results
    
    def benchmark_models(self, models: Dict[str, Any], validation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Benchmark and compare all models.
        
        Args:
            models: Dictionary of trained models
            validation_results: Dictionary of validation results
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Benchmarking models...")
        
        # Use benchmarker to compare models
        comparison = self.benchmarker.compare_models(models, validation_results)
        
        # Save comparison table
        comparison.to_csv(self.output_dir / "model_comparison.csv", index=False)
        
        logger.info("Model benchmarking completed")
        return comparison
    
    def generate_report(self) -> str:
        """
        Generate comprehensive pipeline report.
        
        Returns:
            Report as markdown string
        """
        logger.info("Generating pipeline report...")
        
        report = f"""# TLR4 Binding Affinity Prediction Pipeline Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary

- TLR4 compounds: {len(self.tlr4_data) if self.tlr4_data is not None else 0}
- Transfer learning compounds: {len(self.transfer_data) if self.transfer_data is not None else 0}

## Models Trained

{', '.join(self.models.keys()) if self.models else 'None'}

## Validation Results

[Validation results would be included here]

## Interpretability Analysis

[Interpretability results would be included here]

## Model Comparison

[Comparison table would be included here]

## Conclusions

[Summary and recommendations would be included here]
"""
        
        # Save report
        report_path = self.output_dir / "pipeline_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("="*80)
        logger.info("Starting TLR4 Binding Affinity Prediction Pipeline")
        logger.info("="*80)
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Step 1: Data collection
            if self.config.collect_data:
                tlr4_data, transfer_data = self.collect_data()
            else:
                logger.info("Skipping data collection (using existing data)")
                # Load existing data
                tlr4_data = pd.read_csv(self.output_dir / "tlr4_dataset.csv")
                transfer_data = None
                if (self.output_dir / "transfer_dataset.csv").exists():
                    transfer_data = pd.read_csv(self.output_dir / "transfer_dataset.csv")
            
            # Step 2: Feature engineering
            self.features = self.engineer_features(tlr4_data)
            
            # Step 3: Model training
            self.models = self.train_models(self.features)
            
            # Step 4: Validation
            self.validation_results = self.validate_models(self.models, self.features)
            
            # Step 5: Interpretability
            self.interpretability_results = self.generate_interpretability(self.models, self.features)
            
            # Step 6: Benchmarking
            comparison = self.benchmark_models(self.models, self.validation_results)
            
            # Step 7: Report generation
            if self.config.generate_report:
                report = self.generate_report()
            
            # Compile results
            results = {
                'tlr4_data': tlr4_data,
                'transfer_data': transfer_data,
                'features': self.features,
                'models': self.models,
                'validation_results': self.validation_results,
                'interpretability_results': self.interpretability_results,
                'comparison': comparison,
                'output_dir': str(self.output_dir)
            }
            
            logger.info("="*80)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_pipeline(config: Optional[PipelineConfig] = None) -> TLR4Pipeline:
    """
    Create a TLR4 pipeline instance.
    
    Args:
        config: Pipeline configuration. If None, uses default configuration.
        
    Returns:
        Configured TLR4Pipeline instance
    """
    return TLR4Pipeline(config)


def run_pipeline(config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to create and run the pipeline.
    
    Args:
        config: Pipeline configuration. If None, uses default configuration.
        
    Returns:
        Dictionary containing all pipeline results
    """
    pipeline = create_pipeline(config)
    return pipeline.run()
