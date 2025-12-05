# TLR4 Binding Affinity Prediction Pipeline

## Overview

The TLR4 Pipeline provides an end-to-end solution for predicting binding affinities of small molecules to the TLR4 receptor. It integrates all components of the methodology enhancement into a single, cohesive workflow.

## Features

### Data Collection & Quality Control
- ChEMBL and PubChem data collection
- Activity standardization (ΔG conversion)
- PAINS filtering and SMILES canonicalization
- Functional classification (agonist/antagonist)
- Related TLR data collection for transfer learning

### Feature Engineering
- 2D molecular descriptors
- 3D conformational descriptors (PMI, shape, WHIM)
- Electrostatic properties (Gasteiger charges, PEOE-VSA, dipole)
- Graph representations for GNN models

### Model Training
- Traditional ML ensemble (Random Forest, XGBoost, SVR)
- Graph Attention Network (GAT)
- ChemBERTa transformer
- Hybrid architecture (GNN + descriptors)
- Transfer learning (pre-trained on related TLRs)
- Multi-task learning (affinity + function)

### Validation
- External test set (stratified by affinity)
- Nested cross-validation (5x3 default)
- Y-scrambling validation (100+ iterations)
- Scaffold-based validation
- Applicability domain analysis

### Interpretability
- Attention weight visualization (GNN models)
- SHAP feature importance (traditional models)
- Molecular structure overlays

### Benchmarking
- Systematic model comparison
- Statistical significance testing
- Performance metrics (R², RMSE, MAE)
- Comprehensive reporting

## Quick Start

### Basic Usage

```python
from tlr4_binding.pipeline import run_pipeline, PipelineConfig

# Use default configuration
results = run_pipeline()

# Or customize configuration
config = PipelineConfig(
    output_dir="./my_results",
    train_gnn=True,
    train_transformer=True,
    nested_cv_outer_folds=5
)
results = run_pipeline(config)
```

### Command Line

```bash
# Quick demo mode (minimal validation)
python run_pipeline.py --quick

# Full validation mode (all models and validation)
python run_pipeline.py --full

# Custom configuration
python run_pipeline.py --config my_config.json

# Generate configuration template
python run_pipeline.py --save-template
```

## Configuration

### PipelineConfig Parameters

#### Data Collection
- `collect_data` (bool): Collect data from ChEMBL/PubChem
- `chembl_targets` (List[str]): ChEMBL target IDs for TLR4
- `pubchem_assays` (List[int]): PubChem assay IDs
- `min_compounds` (int): Minimum compounds required (default: 150)
- `max_compounds` (int): Maximum compounds to collect (default: 300)
- `collect_transfer_data` (bool): Collect related TLR data
- `related_tlr_targets` (List[str]): ChEMBL IDs for related TLRs
- `min_transfer_compounds` (int): Minimum transfer learning compounds (default: 500)

#### Feature Engineering
- `calculate_3d_descriptors` (bool): Calculate 3D conformational descriptors
- `calculate_electrostatic` (bool): Calculate electrostatic properties
- `generate_graphs` (bool): Generate graph representations

#### Model Training
- `train_traditional_ml` (bool): Train traditional ML ensemble
- `train_gnn` (bool): Train Graph Attention Network
- `train_transformer` (bool): Train ChemBERTa transformer
- `train_hybrid` (bool): Train hybrid model
- `train_transfer_learning` (bool): Use transfer learning
- `train_multi_task` (bool): Train multi-task model

#### Validation
- `external_test_size` (float): Test set fraction (default: 0.2)
- `nested_cv_outer_folds` (int): Outer CV folds (default: 5)
- `nested_cv_inner_folds` (int): Inner CV folds (default: 3)
- `y_scrambling_iterations` (int): Y-scrambling iterations (default: 100)
- `run_scaffold_validation` (bool): Run scaffold-based validation
- `calculate_applicability_domain` (bool): Calculate applicability domain

#### Interpretability
- `generate_attention_viz` (bool): Generate attention visualizations
- `generate_shap_analysis` (bool): Generate SHAP analysis
- `top_compounds_for_viz` (int): Number of compounds to visualize (default: 10)

#### Output
- `output_dir` (str): Output directory path
- `save_models` (bool): Save trained models
- `generate_report` (bool): Generate comprehensive report

### Example Configuration File

```json
{
  "collect_data": true,
  "chembl_targets": ["CHEMBL5896", "CHEMBL2047"],
  "pubchem_assays": [1053197, 588834, 651635],
  "calculate_3d_descriptors": true,
  "calculate_electrostatic": true,
  "generate_graphs": true,
  "train_traditional_ml": true,
  "train_gnn": true,
  "train_transformer": true,
  "train_hybrid": true,
  "train_transfer_learning": true,
  "train_multi_task": true,
  "nested_cv_outer_folds": 5,
  "nested_cv_inner_folds": 3,
  "y_scrambling_iterations": 100,
  "run_scaffold_validation": true,
  "calculate_applicability_domain": true,
  "generate_attention_viz": true,
  "generate_shap_analysis": true,
  "output_dir": "./results/full_pipeline",
  "save_models": true,
  "generate_report": true
}
```

## Pipeline Workflow

```
1. Data Collection
   ├── Query ChEMBL for TLR4 targets
   ├── Query PubChem for bioassays
   ├── Merge and standardize activities
   ├── Quality control (PAINS, canonicalization)
   └── Functional classification

2. Feature Engineering
   ├── 2D molecular descriptors
   ├── 3D conformational descriptors
   ├── Electrostatic properties
   └── Graph representations

3. Model Training
   ├── Traditional ML ensemble
   ├── Graph Attention Network
   ├── ChemBERTa transformer
   ├── Hybrid model
   ├── Transfer learning
   └── Multi-task learning

4. Validation
   ├── External test set evaluation
   ├── Nested cross-validation
   ├── Y-scrambling validation
   ├── Scaffold-based validation
   └── Applicability domain analysis

5. Interpretability
   ├── Attention weight extraction
   ├── Attention visualization
   ├── SHAP analysis
   └── Feature importance plots

6. Benchmarking
   ├── Model comparison
   ├── Statistical significance testing
   └── Performance metrics

7. Reporting
   ├── Dataset summary
   ├── Model performance tables
   ├── Validation results
   ├── Interpretability outputs
   └── Recommendations
```

## Output Structure

```
output_dir/
├── tlr4_dataset.csv              # Curated TLR4 dataset
├── transfer_dataset.csv          # Related TLR dataset
├── model_comparison.csv          # Model performance comparison
├── pipeline_report.md            # Comprehensive report
├── models/                       # Saved model files
│   ├── traditional_ml/
│   ├── gnn/
│   ├── transformer/
│   ├── hybrid/
│   └── multi_task/
├── validation/                   # Validation results
│   ├── nested_cv_results.json
│   ├── y_scrambling_results.json
│   ├── scaffold_validation.json
│   └── applicability_domain.json
└── interpretability/             # Interpretability outputs
    ├── attention_visualizations/
    └── shap_analysis/
```

## Examples

See `examples/demo_end_to_end_pipeline.py` for comprehensive demonstrations:

1. **Minimal Pipeline**: Quick demo with synthetic data
2. **Feature Engineering Focus**: Emphasis on descriptor calculation
3. **Validation Focus**: Comprehensive validation suite
4. **Full Pipeline**: All components enabled
5. **Custom Configuration**: Tailored for specific research questions

## Requirements

### Core Dependencies
- Python >= 3.8
- RDKit >= 2022.03
- PyTorch >= 1.12
- PyTorch Geometric >= 2.0
- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.21

### Optional Dependencies
- transformers >= 4.20 (for ChemBERTa)
- mordred (for additional descriptors)
- SHAP >= 0.41 (for interpretability)
- matplotlib >= 3.5 (for visualization)

## Performance Considerations

### Computational Requirements

- **Data Collection**: ~5-10 minutes (API dependent)
- **Feature Engineering**: ~1-2 seconds per compound
- **Traditional ML Training**: ~5-10 minutes
- **GNN Training**: ~30-60 minutes (GPU recommended)
- **Transformer Training**: ~1-2 hours (GPU required)
- **Validation**: ~2-4x training time (nested CV)

### Optimization Tips

1. **Quick Testing**: Disable expensive components
   ```python
   config = PipelineConfig(
       calculate_3d_descriptors=False,
       train_transformer=False,
       nested_cv_outer_folds=3,
       y_scrambling_iterations=50
   )
   ```

2. **GPU Acceleration**: Enable for deep learning models
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

3. **Parallel Processing**: Use multiple cores for feature calculation
   ```python
   from joblib import Parallel, delayed
   # Parallel feature calculation
   ```

4. **Caching**: Save intermediate results
   ```python
   config = PipelineConfig(
       collect_data=False,  # Use cached data
       save_models=True     # Save trained models
   )
   ```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU Memory**: Reduce batch size or use CPU
   ```python
   config.train_transformer = False  # Skip GPU-intensive models
   ```

3. **API Timeouts**: Increase timeout or use cached data
   ```python
   config.collect_data = False
   ```

4. **Missing Data**: Check ChEMBL/PubChem availability
   ```python
   # Verify targets exist before running
   ```

## Citation

If you use this pipeline in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please contact:
- Brandon Yee: b.yee@ycrg-labs.org
- Maximilian Rutowski

## Acknowledgments

This pipeline implements the methodology enhancements addressing peer review feedback for the TLR4 binding affinity prediction system.
