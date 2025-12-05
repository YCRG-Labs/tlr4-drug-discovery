#!/usr/bin/env python3
"""
Main entry point for TLR4 binding affinity prediction pipeline.

This script provides a command-line interface to run the complete
end-to-end pipeline with customizable configuration.

Usage:
    python run_pipeline.py --config config.json
    python run_pipeline.py --quick  # Quick demo mode
    python run_pipeline.py --full   # Full validation mode
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tlr4_binding.pipeline import TLR4Pipeline, PipelineConfig, run_pipeline


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from file or use defaults.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        PipelineConfig instance
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return PipelineConfig(**config_dict)
    else:
        return PipelineConfig()


def create_quick_config() -> PipelineConfig:
    """
    Create a quick demo configuration with minimal validation.
    
    Returns:
        PipelineConfig for quick demo
    """
    return PipelineConfig(
        collect_data=False,  # Use existing data
        collect_transfer_data=False,
        calculate_3d_descriptors=True,
        calculate_electrostatic=True,
        generate_graphs=True,
        train_traditional_ml=True,
        train_gnn=False,  # Skip for quick demo
        train_transformer=False,
        train_hybrid=False,
        train_transfer_learning=False,
        train_multi_task=False,
        nested_cv_outer_folds=3,  # Reduced for speed
        nested_cv_inner_folds=2,
        y_scrambling_iterations=50,  # Reduced for speed
        run_scaffold_validation=False,
        calculate_applicability_domain=True,
        generate_attention_viz=False,
        generate_shap_analysis=True,
        top_compounds_for_viz=5,
        output_dir="./results/quick_demo",
        generate_report=True
    )


def create_full_config() -> PipelineConfig:
    """
    Create a full validation configuration.
    
    Returns:
        PipelineConfig for full validation
    """
    return PipelineConfig(
        collect_data=True,
        collect_transfer_data=True,
        calculate_3d_descriptors=True,
        calculate_electrostatic=True,
        generate_graphs=True,
        train_traditional_ml=True,
        train_gnn=True,
        train_transformer=True,
        train_hybrid=True,
        train_transfer_learning=True,
        train_multi_task=True,
        external_test_size=0.2,
        nested_cv_outer_folds=5,
        nested_cv_inner_folds=3,
        y_scrambling_iterations=100,
        run_scaffold_validation=True,
        calculate_applicability_domain=True,
        generate_attention_viz=True,
        generate_shap_analysis=True,
        top_compounds_for_viz=10,
        output_dir="./results/full_validation",
        save_models=True,
        generate_report=True
    )


def save_config_template(output_path: str = "pipeline_config_template.json"):
    """
    Save a template configuration file.
    
    Args:
        output_path: Path to save the template
    """
    config = PipelineConfig()
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration template saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TLR4 Binding Affinity Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with custom configuration
  python run_pipeline.py --config my_config.json
  
  # Quick demo mode (minimal validation)
  python run_pipeline.py --quick
  
  # Full validation mode (all models and validation)
  python run_pipeline.py --full
  
  # Generate configuration template
  python run_pipeline.py --save-template
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick demo mode (minimal validation)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run in full validation mode (all models and validation)'
    )
    
    parser.add_argument(
        '--save-template',
        action='store_true',
        help='Save a configuration template file and exit'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Handle save template
    if args.save_template:
        save_config_template()
        return 0
    
    # Determine configuration
    if args.quick:
        print("Running in QUICK DEMO mode")
        config = create_quick_config()
    elif args.full:
        print("Running in FULL VALIDATION mode")
        config = create_full_config()
    elif args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("Using DEFAULT configuration")
        config = PipelineConfig()
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Print configuration summary
    print("\n" + "="*80)
    print("Pipeline Configuration Summary")
    print("="*80)
    print(f"Output directory: {config.output_dir}")
    print(f"Data collection: {config.collect_data}")
    print(f"Transfer learning data: {config.collect_transfer_data}")
    print(f"Models to train:")
    print(f"  - Traditional ML: {config.train_traditional_ml}")
    print(f"  - GNN: {config.train_gnn}")
    print(f"  - Transformer: {config.train_transformer}")
    print(f"  - Hybrid: {config.train_hybrid}")
    print(f"  - Transfer Learning: {config.train_transfer_learning}")
    print(f"  - Multi-Task: {config.train_multi_task}")
    print(f"Validation:")
    print(f"  - Nested CV: {config.nested_cv_outer_folds}x{config.nested_cv_inner_folds}")
    print(f"  - Y-scrambling: {config.y_scrambling_iterations} iterations")
    print(f"  - Scaffold validation: {config.run_scaffold_validation}")
    print(f"  - Applicability domain: {config.calculate_applicability_domain}")
    print("="*80 + "\n")
    
    # Run pipeline
    try:
        results = run_pipeline(config)
        
        print("\n" + "="*80)
        print("Pipeline completed successfully!")
        print("="*80)
        print(f"\nResults saved to: {results['output_dir']}")
        print("\nGenerated files:")
        print(f"  - tlr4_dataset.csv")
        print(f"  - model_comparison.csv")
        print(f"  - pipeline_report.md")
        
        return 0
        
    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
