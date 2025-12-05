"""
Interpretability output generator for TLR4 binding prediction models.

This module provides comprehensive interpretability output generation including:
- Attention weight visualization for GNN models
- SHAP analysis and feature importance plots
- Molecular structure overlays with importance highlights
- Batch processing for multiple compounds
- Report generation

Requirements: 18.1, 18.3, 18.4
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from .analyzer import InterpretabilityAnalyzer, create_interpretability_analyzer

logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability output generation."""
    
    # Attention visualization
    generate_attention_viz: bool = True
    attention_top_k: int = 10  # Top K compounds to visualize
    attention_colormap: str = "RdYlGn_r"  # Red (high) to Green (low)
    
    # SHAP analysis
    generate_shap_analysis: bool = True
    shap_top_features: int = 20  # Top features to show
    shap_sample_size: Optional[int] = None  # None = use all samples
    
    # Feature importance
    generate_feature_importance: bool = True
    feature_importance_top_k: int = 30
    
    # Molecular visualization
    mol_viz_size: Tuple[int, int] = (400, 400)
    mol_viz_highlight_atoms: bool = True
    mol_viz_highlight_bonds: bool = True
    
    # Output
    output_dir: str = "./results/interpretability"
    save_individual_plots: bool = True
    save_summary_plots: bool = True
    generate_report: bool = True
    
    # Plot styling
    figure_dpi: int = 300
    figure_format: str = "png"  # or "pdf", "svg"


@dataclass
class InterpretabilityOutputs:
    """Container for interpretability outputs."""
    
    # Attention outputs
    attention_visualizations: Optional[Dict[str, Any]] = None
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    
    # SHAP outputs
    shap_values: Optional[np.ndarray] = None
    shap_feature_importance: Optional[pd.DataFrame] = None
    shap_summary_plot_path: Optional[str] = None
    
    # Feature importance outputs
    feature_importance: Optional[pd.DataFrame] = None
    feature_importance_plot_path: Optional[str] = None
    
    # Report
    report_path: Optional[str] = None
    
    # Metadata
    n_compounds_analyzed: int = 0
    n_features_analyzed: int = 0
    timestamp: str = None
    
    def __post_init__(self):
        """Set timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class InterpretabilityOutputGenerator:
    """
    Generator for comprehensive interpretability outputs.
    
    This class orchestrates the generation of all interpretability outputs
    including attention visualizations, SHAP analysis, and feature importance.
    """
    
    def __init__(self, config: Optional[InterpretabilityConfig] = None):
        """
        Initialize the output generator.
        
        Args:
            config: Configuration for output generation
        """
        self.config = config or InterpretabilityConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.attention_dir = self.output_dir / "attention_visualizations"
        self.shap_dir = self.output_dir / "shap_analysis"
        self.feature_dir = self.output_dir / "feature_importance"
        
        if self.config.save_individual_plots:
            self.attention_dir.mkdir(exist_ok=True)
            self.shap_dir.mkdir(exist_ok=True)
            self.feature_dir.mkdir(exist_ok=True)
        
        # Initialize analyzer
        self.analyzer = create_interpretability_analyzer()
        
        # Results storage
        self.outputs = InterpretabilityOutputs()
        
        logger.info(f"Interpretability output generator initialized: {self.output_dir}")
    
    def generate_attention_visualizations(
        self,
        model: Any,
        graph_data_list: List[Any],
        smiles_list: List[str],
        predictions: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate attention weight visualizations for top compounds.
        
        Args:
            model: Trained GNN model with attention mechanism
            graph_data_list: List of graph data objects
            smiles_list: List of SMILES strings
            predictions: Optional model predictions
            true_values: Optional true binding affinities
            
        Returns:
            Dictionary of attention visualization results
        """
        logger.info("="*80)
        logger.info("Generating Attention Visualizations")
        logger.info("="*80)
        
        if not self.config.generate_attention_viz:
            logger.info("Attention visualization disabled")
            return {}
        
        # Select top K compounds
        if predictions is not None and true_values is not None:
            # Select compounds with best predictions (lowest error)
            errors = np.abs(predictions - true_values)
            top_indices = np.argsort(errors)[:self.config.attention_top_k]
            logger.info(f"Selecting top {self.config.attention_top_k} compounds by prediction accuracy")
        else:
            # Select first K compounds
            top_indices = list(range(min(self.config.attention_top_k, len(smiles_list))))
            logger.info(f"Selecting first {len(top_indices)} compounds")
        
        attention_results = {}
        
        for idx in top_indices:
            smiles = smiles_list[idx]
            graph_data = graph_data_list[idx]
            
            logger.info(f"\nProcessing compound {idx}: {smiles}")
            
            try:
                # Extract attention weights
                attention_weights = self.analyzer.extract_attention(model, graph_data)
                
                # Visualize attention on molecular structure
                if self.config.save_individual_plots:
                    output_path = self.attention_dir / f"attention_compound_{idx}.{self.config.figure_format}"
                    self.analyzer.visualize_attention(
                        smiles=smiles,
                        attention=attention_weights,
                        save_path=str(output_path),
                        size=self.config.mol_viz_size,
                        colormap=self.config.attention_colormap
                    )
                    logger.info(f"  Saved visualization: {output_path}")
                
                # Store results
                attention_results[smiles] = {
                    'attention_weights': attention_weights,
                    'prediction': predictions[idx] if predictions is not None else None,
                    'true_value': true_values[idx] if true_values is not None else None,
                    'error': errors[idx] if predictions is not None and true_values is not None else None
                }
                
            except Exception as e:
                logger.error(f"  Failed to process compound {idx}: {e}")
        
        logger.info(f"\nGenerated {len(attention_results)} attention visualizations")
        
        self.outputs.attention_visualizations = attention_results
        self.outputs.n_compounds_analyzed = len(attention_results)
        
        return attention_results
    
    def generate_shap_analysis(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate SHAP analysis and feature importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Optional feature names
            sample_indices: Optional indices of samples to analyze
            
        Returns:
            Tuple of (shap_values, feature_importance_df)
        """
        logger.info("\n" + "="*80)
        logger.info("Generating SHAP Analysis")
        logger.info("="*80)
        
        if not self.config.generate_shap_analysis:
            logger.info("SHAP analysis disabled")
            return None, None
        
        # Sample data if needed
        if self.config.shap_sample_size and len(X) > self.config.shap_sample_size:
            if sample_indices is None:
                sample_indices = np.random.choice(
                    len(X), 
                    self.config.shap_sample_size, 
                    replace=False
                )
            X_sample = X[sample_indices]
            logger.info(f"Using {len(X_sample)} samples for SHAP analysis")
        else:
            X_sample = X
            logger.info(f"Using all {len(X_sample)} samples for SHAP analysis")
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        try:
            shap_values = self.analyzer.calculate_shap(model, X_sample)
            logger.info(f"SHAP values shape: {shap_values.shape}")
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return None, None
        
        # Calculate feature importance
        logger.info("Calculating feature importance from SHAP values...")
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {min(10, len(importance_df))} most important features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Generate SHAP summary plot
        if self.config.save_summary_plots:
            logger.info("\nGenerating SHAP summary plot...")
            summary_path = self.shap_dir / f"shap_summary.{self.config.figure_format}"
            
            try:
                self.analyzer.plot_feature_importance(
                    shap_values=shap_values,
                    feature_names=feature_names,
                    save_path=str(summary_path),
                    top_k=self.config.shap_top_features
                )
                logger.info(f"SHAP summary plot saved: {summary_path}")
                self.outputs.shap_summary_plot_path = str(summary_path)
            except Exception as e:
                logger.error(f"Failed to generate SHAP summary plot: {e}")
        
        # Save feature importance
        importance_path = self.shap_dir / "shap_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved: {importance_path}")
        
        self.outputs.shap_values = shap_values
        self.outputs.shap_feature_importance = importance_df
        self.outputs.n_features_analyzed = len(feature_names)
        
        return shap_values, importance_df
    
    def generate_feature_importance_plot(
        self,
        model: Any,
        feature_names: List[str],
        method: str = "permutation"
    ) -> pd.DataFrame:
        """
        Generate feature importance plot using model-specific methods.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            method: Method to use ("permutation", "tree", "coefficients")
            
        Returns:
            DataFrame with feature importance
        """
        logger.info("\n" + "="*80)
        logger.info("Generating Feature Importance Plot")
        logger.info("="*80)
        logger.info(f"Method: {method}")
        
        if not self.config.generate_feature_importance:
            logger.info("Feature importance generation disabled")
            return None
        
        # Extract feature importance based on model type
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
                logger.info("Using tree-based feature importance")
            elif hasattr(model, 'coef_'):
                # Linear models
                importance = np.abs(model.coef_)
                logger.info("Using coefficient-based feature importance")
            else:
                logger.warning("Model does not have built-in feature importance")
                return None
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot top features
            if self.config.save_summary_plots:
                logger.info(f"Plotting top {self.config.feature_importance_top_k} features...")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                top_features = importance_df.head(self.config.feature_importance_top_k)
                ax.barh(range(len(top_features)), top_features['importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top {self.config.feature_importance_top_k} Feature Importance')
                ax.invert_yaxis()
                
                plt.tight_layout()
                
                plot_path = self.feature_dir / f"feature_importance.{self.config.figure_format}"
                plt.savefig(plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Feature importance plot saved: {plot_path}")
                self.outputs.feature_importance_plot_path = str(plot_path)
            
            # Save CSV
            csv_path = self.feature_dir / "feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)
            logger.info(f"Feature importance saved: {csv_path}")
            
            self.outputs.feature_importance = importance_df
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Failed to generate feature importance: {e}")
            return None
    
    def generate_report(self) -> str:
        """
        Generate comprehensive interpretability report.
        
        Returns:
            Report as markdown string
        """
        logger.info("\n" + "="*80)
        logger.info("Generating Interpretability Report")
        logger.info("="*80)
        
        report = f"""# TLR4 Binding Prediction - Interpretability Analysis Report

Generated: {self.outputs.timestamp}

## Summary

- Compounds analyzed: {self.outputs.n_compounds_analyzed}
- Features analyzed: {self.outputs.n_features_analyzed}

## Attention Visualizations

"""
        
        if self.outputs.attention_visualizations:
            report += f"Generated attention visualizations for {len(self.outputs.attention_visualizations)} compounds.\n\n"
            report += "### Top Compounds by Prediction Accuracy\n\n"
            
            for smiles, result in list(self.outputs.attention_visualizations.items())[:5]:
                report += f"**{smiles}**\n"
                if result['prediction'] is not None:
                    report += f"- Predicted: {result['prediction']:.2f} kcal/mol\n"
                if result['true_value'] is not None:
                    report += f"- True: {result['true_value']:.2f} kcal/mol\n"
                if result['error'] is not None:
                    report += f"- Error: {result['error']:.2f} kcal/mol\n"
                report += "\n"
        else:
            report += "No attention visualizations generated.\n\n"
        
        report += "## SHAP Analysis\n\n"
        
        if self.outputs.shap_feature_importance is not None:
            report += f"### Top {min(20, len(self.outputs.shap_feature_importance))} Most Important Features\n\n"
            report += "| Rank | Feature | Importance |\n"
            report += "|------|---------|------------|\n"
            
            for idx, (_, row) in enumerate(self.outputs.shap_feature_importance.head(20).iterrows(), 1):
                report += f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n"
            
            report += "\n"
            
            if self.outputs.shap_summary_plot_path:
                report += f"SHAP summary plot: `{self.outputs.shap_summary_plot_path}`\n\n"
        else:
            report += "No SHAP analysis performed.\n\n"
        
        report += "## Feature Importance\n\n"
        
        if self.outputs.feature_importance is not None:
            report += f"### Top {min(15, len(self.outputs.feature_importance))} Features\n\n"
            report += "| Rank | Feature | Importance |\n"
            report += "|------|---------|------------|\n"
            
            for idx, (_, row) in enumerate(self.outputs.feature_importance.head(15).iterrows(), 1):
                report += f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n"
            
            report += "\n"
            
            if self.outputs.feature_importance_plot_path:
                report += f"Feature importance plot: `{self.outputs.feature_importance_plot_path}`\n\n"
        else:
            report += "No feature importance analysis performed.\n\n"
        
        report += "## Interpretation Guidelines\n\n"
        report += """
### Attention Weights

- Higher attention weights (red) indicate atoms that the model focuses on for prediction
- These atoms are likely important for TLR4 binding
- Compare attention patterns across agonists vs antagonists

### SHAP Values

- SHAP values represent the contribution of each feature to the prediction
- Positive SHAP values increase predicted binding affinity (stronger binding)
- Negative SHAP values decrease predicted binding affinity (weaker binding)
- Features with high absolute SHAP values are most important

### Feature Importance

- Feature importance shows which molecular descriptors are most predictive
- High importance features should be prioritized in drug design
- Compare importance across different model types for robustness

## Recommendations

1. Focus on molecular features with high SHAP importance
2. Examine attention patterns for structure-activity relationships
3. Validate important features with experimental data
4. Use interpretability to guide compound optimization

## Output Files

"""
        
        report += f"- Attention visualizations: `{self.attention_dir}/`\n"
        report += f"- SHAP analysis: `{self.shap_dir}/`\n"
        report += f"- Feature importance: `{self.feature_dir}/`\n"
        
        # Save report
        if self.config.generate_report:
            report_path = self.output_dir / "interpretability_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved: {report_path}")
            self.outputs.report_path = str(report_path)
        
        return report
    
    def generate_all_outputs(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        smiles_list: Optional[List[str]] = None,
        graph_data_list: Optional[List[Any]] = None,
        predictions: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None
    ) -> InterpretabilityOutputs:
        """
        Generate all interpretability outputs.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            smiles_list: Optional SMILES for attention visualization
            graph_data_list: Optional graph data for attention visualization
            predictions: Optional model predictions
            true_values: Optional true values
            
        Returns:
            InterpretabilityOutputs object
        """
        logger.info("="*80)
        logger.info("Generating All Interpretability Outputs")
        logger.info("="*80)
        
        # 1. Attention visualizations (if GNN model)
        if smiles_list and graph_data_list and self.config.generate_attention_viz:
            self.generate_attention_visualizations(
                model=model,
                graph_data_list=graph_data_list,
                smiles_list=smiles_list,
                predictions=predictions,
                true_values=true_values
            )
        
        # 2. SHAP analysis
        if self.config.generate_shap_analysis:
            self.generate_shap_analysis(
                model=model,
                X=X,
                feature_names=feature_names
            )
        
        # 3. Feature importance
        if self.config.generate_feature_importance:
            self.generate_feature_importance_plot(
                model=model,
                feature_names=feature_names
            )
        
        # 4. Generate report
        if self.config.generate_report:
            self.generate_report()
        
        logger.info("\n" + "="*80)
        logger.info("All Interpretability Outputs Generated")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        
        return self.outputs


def create_output_generator(config: Optional[InterpretabilityConfig] = None) -> InterpretabilityOutputGenerator:
    """
    Create an interpretability output generator.
    
    Args:
        config: Configuration for output generation
        
    Returns:
        InterpretabilityOutputGenerator instance
    """
    return InterpretabilityOutputGenerator(config)
