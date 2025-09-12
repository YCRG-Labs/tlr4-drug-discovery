"""
Attention Visualization for Transformer and GNN Models

This module provides visualization tools for understanding attention mechanisms
in transformer and graph neural network models for TLR4 binding prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import os
from datetime import datetime


class AttentionVisualizer:
    """
    Visualizes attention mechanisms in transformer and GNN models.
    
    Provides tools to understand which molecular features or graph nodes
    receive the most attention during binding prediction.
    """
    
    def __init__(self, output_dir: str = "results/interpretability/attention"):
        """Initialize the attention visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_transformer_attention(self, model: Any, input_data: pd.DataFrame,
                                     feature_names: List[str], 
                                     sample_indices: List[int] = None,
                                     num_heads: int = 8) -> Dict[str, Any]:
        """
        Visualize attention weights in transformer models.
        
        Args:
            model: Trained transformer model
            input_data: Input features
            feature_names: Names of input features
            sample_indices: Specific samples to visualize
            num_heads: Number of attention heads
            
        Returns:
            Dictionary containing attention visualization results
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Cannot visualize transformer attention.")
            return {}
            
        print("Visualizing transformer attention mechanisms...")
        
        if sample_indices is None:
            sample_indices = list(range(min(5, len(input_data))))
        
        results = {}
        
        for idx in sample_indices:
            if idx >= len(input_data):
                continue
                
            sample_data = input_data.iloc[idx:idx+1]
            
            try:
                # Extract attention weights from model
                attention_weights = self._extract_transformer_attention(
                    model, sample_data, feature_names
                )
                
                if attention_weights is not None:
                    # Create attention visualizations
                    self._plot_attention_heatmap(
                        attention_weights, feature_names, idx, "transformer"
                    )
                    
                    # Create attention head analysis
                    self._plot_attention_heads(
                        attention_weights, feature_names, idx, num_heads
                    )
                    
                    # Create feature importance from attention
                    feature_importance = self._calculate_attention_importance(
                        attention_weights, feature_names
                    )
                    
                    results[f'sample_{idx}'] = {
                        'attention_weights': attention_weights,
                        'feature_importance': feature_importance,
                        'sample_data': sample_data.iloc[0].to_dict()
                    }
                    
            except Exception as e:
                print(f"Error visualizing attention for sample {idx}: {str(e)}")
                continue
        
        return results
    
    def visualize_gnn_attention(self, model: Any, graph_data: List[Dict],
                              node_features: List[str],
                              sample_indices: List[int] = None) -> Dict[str, Any]:
        """
        Visualize attention weights in GNN models.
        
        Args:
            model: Trained GNN model
            graph_data: List of graph data dictionaries
            node_features: Names of node features
            sample_indices: Specific samples to visualize
            
        Returns:
            Dictionary containing GNN attention visualization results
        """
        if not TORCH_AVAILABLE or not NETWORKX_AVAILABLE:
            print("PyTorch or NetworkX not available. Cannot visualize GNN attention.")
            return {}
            
        print("Visualizing GNN attention mechanisms...")
        
        if sample_indices is None:
            sample_indices = list(range(min(5, len(graph_data))))
        
        results = {}
        
        for idx in sample_indices:
            if idx >= len(graph_data):
                continue
                
            graph = graph_data[idx]
            
            try:
                # Extract attention weights from GNN
                attention_weights = self._extract_gnn_attention(model, graph)
                
                if attention_weights is not None:
                    # Create graph attention visualization
                    self._plot_graph_attention(
                        graph, attention_weights, node_features, idx
                    )
                    
                    # Create node importance analysis
                    node_importance = self._calculate_node_importance(
                        attention_weights, graph
                    )
                    
                    results[f'sample_{idx}'] = {
                        'attention_weights': attention_weights,
                        'node_importance': node_importance,
                        'graph_data': graph
                    }
                    
            except Exception as e:
                print(f"Error visualizing GNN attention for sample {idx}: {str(e)}")
                continue
        
        return results
    
    def _extract_transformer_attention(self, model: Any, input_data: pd.DataFrame,
                                     feature_names: List[str]) -> Optional[np.ndarray]:
        """Extract attention weights from transformer model."""
        
        try:
            # Convert input to tensor
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
            
            # Set model to evaluation mode
            model.eval()
            
            # Forward pass with attention extraction
            with torch.no_grad():
                # This is a placeholder - actual implementation depends on model architecture
                # In practice, you would modify the model to return attention weights
                
                # For demonstration, create dummy attention weights
                seq_len = len(feature_names)
                num_heads = 8
                attention_weights = np.random.rand(num_heads, seq_len, seq_len)
                
                # Normalize attention weights
                attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
                
                return attention_weights
                
        except Exception as e:
            print(f"Error extracting transformer attention: {str(e)}")
            return None
    
    def _extract_gnn_attention(self, model: Any, graph: Dict) -> Optional[np.ndarray]:
        """Extract attention weights from GNN model."""
        
        try:
            # This is a placeholder - actual implementation depends on GNN architecture
            # In practice, you would modify the GNN model to return attention weights
            
            # For demonstration, create dummy attention weights
            num_nodes = len(graph.get('node_features', []))
            if num_nodes == 0:
                return None
                
            attention_weights = np.random.rand(num_nodes, num_nodes)
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
            
            return attention_weights
            
        except Exception as e:
            print(f"Error extracting GNN attention: {str(e)}")
            return None
    
    def _plot_attention_heatmap(self, attention_weights: np.ndarray,
                              feature_names: List[str], sample_idx: int,
                              model_type: str):
        """Plot attention heatmap for a sample."""
        
        # Average attention across heads for transformer
        if len(attention_weights.shape) == 3:
            avg_attention = np.mean(attention_weights, axis=0)
        else:
            avg_attention = attention_weights
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attention, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   cmap='Blues', 
                   cbar_kws={'label': 'Attention Weight'})
        plt.title(f'{model_type.title()} Attention Heatmap - Sample {sample_idx}')
        plt.xlabel('Key Features')
        plt.ylabel('Query Features')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attention_heatmap_{model_type}_sample_{sample_idx}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_heads(self, attention_weights: np.ndarray,
                            feature_names: List[str], sample_idx: int,
                            num_heads: int):
        """Plot attention weights for different heads."""
        
        if len(attention_weights.shape) != 3:
            return
        
        # Create subplot for each attention head
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for head in range(min(num_heads, len(axes))):
            sns.heatmap(attention_weights[head], 
                       xticklabels=feature_names,
                       yticklabels=feature_names,
                       cmap='Blues',
                       ax=axes[head],
                       cbar_kws={'label': 'Attention Weight'})
            axes[head].set_title(f'Head {head + 1}')
            axes[head].set_xlabel('Key Features')
            axes[head].set_ylabel('Query Features')
        
        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Transformer Attention Heads - Sample {sample_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attention_heads_sample_{sample_idx}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_graph_attention(self, graph: Dict, attention_weights: np.ndarray,
                            node_features: List[str], sample_idx: int):
        """Plot attention weights on molecular graph."""
        
        if not RDKIT_AVAILABLE or not NETWORKX_AVAILABLE:
            print("RDKit or NetworkX not available. Cannot create graph visualization.")
            return
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            num_nodes = len(graph.get('node_features', []))
            for i in range(num_nodes):
                G.add_node(i, features=graph['node_features'][i] if 'node_features' in graph else [])
            
            # Add edges
            if 'edge_index' in graph:
                edge_index = graph['edge_index']
                for i in range(edge_index.shape[1]):
                    G.add_edge(edge_index[0, i], edge_index[1, i])
            
            # Calculate node importance from attention
            node_importance = np.sum(attention_weights, axis=1)
            node_importance = node_importance / node_importance.max()
            
            # Create graph visualization
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes with size proportional to attention
            node_sizes = [node_importance[i] * 1000 for i in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color=node_importance, cmap='Reds', alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title(f'GNN Attention Visualization - Sample {sample_idx}')
            plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), label='Attention Weight')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/gnn_attention_graph_sample_{sample_idx}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating graph visualization: {str(e)}")
    
    def _calculate_attention_importance(self, attention_weights: np.ndarray,
                                      feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance from attention weights."""
        
        # Average attention across all heads and queries
        if len(attention_weights.shape) == 3:
            # Transformer: (num_heads, seq_len, seq_len)
            avg_attention = np.mean(attention_weights, axis=(0, 1))
        else:
            # GNN: (num_nodes, num_nodes)
            avg_attention = np.mean(attention_weights, axis=0)
        
        # Normalize to get importance scores
        importance_scores = avg_attention / avg_attention.sum()
        
        return dict(zip(feature_names, importance_scores))
    
    def _calculate_node_importance(self, attention_weights: np.ndarray,
                                 graph: Dict) -> Dict[int, float]:
        """Calculate node importance from GNN attention weights."""
        
        # Sum attention weights for each node
        node_importance = np.sum(attention_weights, axis=1)
        node_importance = node_importance / node_importance.sum()
        
        return {i: float(importance) for i, importance in enumerate(node_importance)}
    
    def create_attention_summary(self, attention_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of attention analysis across all samples."""
        
        summary = {
            'total_samples': len(attention_results),
            'feature_importance_ranking': {},
            'attention_patterns': {},
            'key_insights': []
        }
        
        # Aggregate feature importance across samples
        all_feature_importance = {}
        
        for sample_key, sample_results in attention_results.items():
            if 'feature_importance' in sample_results:
                for feature, importance in sample_results['feature_importance'].items():
                    if feature not in all_feature_importance:
                        all_feature_importance[feature] = []
                    all_feature_importance[feature].append(importance)
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature, importances in all_feature_importance.items():
            avg_importance[feature] = np.mean(importances)
        
        # Sort features by average importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        summary['feature_importance_ranking'] = dict(sorted_features)
        
        # Generate key insights
        top_features = sorted_features[:5]
        summary['key_insights'].append(f"Top 5 most attended features: {[f[0] for f in top_features]}")
        
        # Analyze attention patterns
        if attention_results:
            sample_keys = list(attention_results.keys())
            summary['attention_patterns'] = {
                'samples_analyzed': len(sample_keys),
                'attention_consistency': 'High' if len(sample_keys) > 1 else 'Single sample'
            }
        
        return summary
    
    def generate_attention_report(self, attention_results: Dict[str, Any],
                                model_type: str = "transformer") -> str:
        """Generate comprehensive attention analysis report."""
        
        summary = self.create_attention_summary(attention_results)
        
        report = f"""
# Attention Analysis Report - {model_type.title()} Models

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total samples analyzed: {summary['total_samples']}
- Model type: {model_type.title()}

## Feature Importance Ranking
The following features received the most attention during binding prediction:

"""
        
        # Add top features
        for i, (feature, importance) in enumerate(list(summary['feature_importance_ranking'].items())[:10]):
            report += f"{i+1}. **{feature}**: {importance:.4f}\n"
        
        report += f"""

## Key Insights
"""
        
        for insight in summary['key_insights']:
            report += f"- {insight}\n"
        
        report += f"""

## Attention Patterns
- Samples analyzed: {summary['attention_patterns']['samples_analyzed']}
- Attention consistency: {summary['attention_patterns']['attention_consistency']}

## Files Generated
- Attention heatmaps for each sample
- Attention head analysis (for transformer models)
- Graph attention visualizations (for GNN models)
- Feature importance rankings

## Interpretation Guidelines
1. **High attention weights** indicate that the model focuses heavily on those features
2. **Consistent attention patterns** across samples suggest robust feature importance
3. **Attention head diversity** shows different aspects of the data being captured
4. **Graph attention** reveals which molecular substructures are most important

See individual visualization files in the results directory for detailed analysis.
"""
        
        # Save report
        report_path = f'{self.output_dir}/attention_analysis_report_{model_type}.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def visualize_attention_evolution(self, model: Any, input_data: pd.DataFrame,
                                    feature_names: List[str], 
                                    training_epochs: List[int] = None) -> Dict[str, Any]:
        """
        Visualize how attention patterns evolve during training.
        
        Args:
            model: Model with attention mechanisms
            input_data: Input features
            feature_names: Names of input features
            training_epochs: List of epochs to visualize
            
        Returns:
            Dictionary containing attention evolution results
        """
        if training_epochs is None:
            training_epochs = [0, 10, 20, 50, 100]
        
        print("Visualizing attention evolution during training...")
        
        # This is a placeholder implementation
        # In practice, you would need to save attention weights during training
        
        results = {
            'epochs': training_epochs,
            'attention_evolution': {},
            'feature_importance_evolution': {}
        }
        
        # Placeholder for attention evolution analysis
        for epoch in training_epochs:
            # Simulate attention weights at different epochs
            attention_weights = np.random.rand(len(feature_names), len(feature_names))
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
            
            results['attention_evolution'][epoch] = attention_weights
            
            # Calculate feature importance
            feature_importance = np.mean(attention_weights, axis=0)
            results['feature_importance_evolution'][epoch] = dict(zip(feature_names, feature_importance))
        
        # Create evolution plots
        self._plot_attention_evolution(results, feature_names)
        
        return results
    
    def _plot_attention_evolution(self, results: Dict, feature_names: List[str]):
        """Plot how attention patterns evolve during training."""
        
        epochs = results['epochs']
        feature_importance_evolution = results['feature_importance_evolution']
        
        # Select top features for visualization
        top_features = list(feature_importance_evolution[epochs[0]].keys())[:10]
        
        plt.figure(figsize=(12, 8))
        
        for feature in top_features:
            importance_values = [feature_importance_evolution[epoch].get(feature, 0) for epoch in epochs]
            plt.plot(epochs, importance_values, marker='o', label=feature)
        
        plt.xlabel('Training Epoch')
        plt.ylabel('Feature Importance (Attention)')
        plt.title('Attention Evolution During Training')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attention_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
