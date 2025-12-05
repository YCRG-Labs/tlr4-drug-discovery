"""
Demo: Applicability Domain Analysis

This script demonstrates how to use the ApplicabilityDomainAnalyzer to:
1. Define the applicability domain based on training data
2. Calculate leverage and Mahalanobis distance for test compounds
3. Determine domain membership and confidence scores
4. Calculate structural similarity to training set

Requirements: 16.1, 16.2, 16.3, 16.4
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import logging

from tlr4_binding.validation import ApplicabilityDomainAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples=200, n_features=50, random_state=42):
    """Generate synthetic molecular descriptor data."""
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    
    # Generate regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=30,
        noise=0.5,
        random_state=random_state
    )
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def demo_basic_usage():
    """Demonstrate basic applicability domain analysis."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Basic Applicability Domain Analysis")
    logger.info("="*80)
    
    # Generate sample data
    X_train, X_test, y_train, y_test = generate_sample_data()
    
    # Initialize analyzer
    analyzer = ApplicabilityDomainAnalyzer(threshold_multiplier=3.0)
    
    # Fit on training data
    logger.info("\nFitting applicability domain on training data...")
    analyzer.fit(X_train)
    
    # Calculate leverage for test compounds
    logger.info("\nCalculating leverage values for test compounds...")
    leverages = analyzer.calculate_leverage(X_test)
    
    logger.info(f"Leverage statistics:")
    logger.info(f"  Mean: {np.mean(leverages):.4f}")
    logger.info(f"  Std: {np.std(leverages):.4f}")
    logger.info(f"  Min: {np.min(leverages):.4f}")
    logger.info(f"  Max: {np.max(leverages):.4f}")
    logger.info(f"  Threshold h*: {analyzer.leverage_threshold:.4f}")
    
    # Calculate Mahalanobis distance
    logger.info("\nCalculating Mahalanobis distances...")
    distances = analyzer.calculate_mahalanobis_distance(X_test)
    
    logger.info(f"Mahalanobis distance statistics:")
    logger.info(f"  Mean: {np.mean(distances):.4f}")
    logger.info(f"  Std: {np.std(distances):.4f}")
    logger.info(f"  Min: {np.min(distances):.4f}")
    logger.info(f"  Max: {np.max(distances):.4f}")
    
    # Determine domain membership
    logger.info("\nDetermining domain membership...")
    in_domain = analyzer.is_in_domain(X_test)
    
    n_in = np.sum(in_domain)
    n_out = len(in_domain) - n_in
    
    logger.info(f"Domain membership:")
    logger.info(f"  In domain: {n_in} ({100*n_in/len(in_domain):.1f}%)")
    logger.info(f"  Out of domain: {n_out} ({100*n_out/len(in_domain):.1f}%)")
    
    # Calculate confidence scores
    logger.info("\nCalculating confidence scores...")
    confidence = analyzer.get_confidence(X_test, method='leverage')
    
    logger.info(f"Confidence statistics:")
    logger.info(f"  Mean: {np.mean(confidence):.4f}")
    logger.info(f"  Std: {np.std(confidence):.4f}")
    logger.info(f"  Min: {np.min(confidence):.4f}")
    logger.info(f"  Max: {np.max(confidence):.4f}")
    
    # Show examples of in-domain and out-of-domain compounds
    logger.info("\nExample compounds:")
    for i in range(min(5, len(X_test))):
        status = "IN" if in_domain[i] else "OUT"
        logger.info(
            f"  Compound {i}: {status} domain "
            f"(leverage={leverages[i]:.4f}, confidence={confidence[i]:.4f})"
        )


def demo_with_outliers():
    """Demonstrate applicability domain with intentional outliers."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Applicability Domain with Outliers")
    logger.info("="*80)
    
    # Generate sample data
    X_train, X_test, y_train, y_test = generate_sample_data()
    
    # Add some outliers to test set
    n_outliers = 5
    logger.info(f"\nAdding {n_outliers} outliers to test set...")
    
    # Create outliers by scaling some test samples
    outlier_indices = np.random.choice(len(X_test), n_outliers, replace=False)
    X_test_with_outliers = X_test.copy()
    X_test_with_outliers[outlier_indices] *= 5.0  # Scale by 5x
    
    # Initialize and fit analyzer
    analyzer = ApplicabilityDomainAnalyzer(threshold_multiplier=3.0)
    analyzer.fit(X_train)
    
    # Analyze test set with outliers
    leverages = analyzer.calculate_leverage(X_test_with_outliers)
    in_domain = analyzer.is_in_domain(X_test_with_outliers)
    confidence = analyzer.get_confidence(X_test_with_outliers)
    
    # Check if outliers are detected
    logger.info("\nOutlier detection results:")
    for idx in outlier_indices:
        logger.info(
            f"  Outlier {idx}: "
            f"leverage={leverages[idx]:.4f}, "
            f"in_domain={in_domain[idx]}, "
            f"confidence={confidence[idx]:.4f}"
        )
    
    # Compare with normal compounds
    normal_indices = [i for i in range(len(X_test)) if i not in outlier_indices]
    logger.info("\nNormal compound statistics:")
    logger.info(f"  Mean leverage: {np.mean(leverages[normal_indices]):.4f}")
    logger.info(f"  Mean confidence: {np.mean(confidence[normal_indices]):.4f}")
    logger.info(f"  In domain: {np.sum(in_domain[normal_indices])}/{len(normal_indices)}")
    
    logger.info("\nOutlier statistics:")
    logger.info(f"  Mean leverage: {np.mean(leverages[outlier_indices]):.4f}")
    logger.info(f"  Mean confidence: {np.mean(confidence[outlier_indices]):.4f}")
    logger.info(f"  In domain: {np.sum(in_domain[outlier_indices])}/{len(outlier_indices)}")


def demo_domain_statistics():
    """Demonstrate comprehensive domain statistics."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Comprehensive Domain Statistics")
    logger.info("="*80)
    
    # Generate sample data
    X_train, X_test, y_train, y_test = generate_sample_data()
    
    # Initialize and fit analyzer
    analyzer = ApplicabilityDomainAnalyzer(threshold_multiplier=3.0)
    analyzer.fit(X_train)
    
    # Get comprehensive statistics
    logger.info("\nCalculating comprehensive domain statistics...")
    stats = analyzer.get_domain_statistics(X_test)
    
    logger.info("\nDomain Statistics:")
    logger.info(f"  Total samples: {stats['n_samples']}")
    logger.info(f"  In domain: {stats['n_in_domain']}")
    logger.info(f"  Out of domain: {stats['n_out_domain']}")
    logger.info(f"  Leverage threshold: {stats['leverage_threshold']:.4f}")
    logger.info(f"  Leverage (mean ± std): {stats['leverage_mean']:.4f} ± {stats['leverage_std']:.4f}")
    logger.info(f"  Leverage (max): {stats['leverage_max']:.4f}")
    logger.info(f"  Distance (mean ± std): {stats['distance_mean']:.4f} ± {stats['distance_std']:.4f}")
    logger.info(f"  Distance (max): {stats['distance_max']:.4f}")
    logger.info(f"  Confidence (mean ± std): {stats['confidence_mean']:.4f} ± {stats['confidence_std']:.4f}")
    logger.info(f"  Confidence (min): {stats['confidence_min']:.4f}")


def demo_with_smiles():
    """Demonstrate applicability domain with SMILES similarity."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Applicability Domain with SMILES Similarity")
    logger.info("="*80)
    
    # Sample SMILES strings (TLR4-related compounds)
    train_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen-like
        "CC(=O)Oc1ccccc1C(O)=O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)NCC(COc1ccccc1)O",  # Propranolol-like
        "c1ccc2c(c1)ccc3c2cccc3",  # Anthracene
    ]
    
    test_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Same as training (high similarity)
        "CCCCc1ccc(cc1)C(C)C(O)=O",  # Similar to training
        "c1ccccc1",  # Benzene (simple, different)
        "CCCCCCCCCCCCCCCCCC",  # Long alkane (very different)
    ]
    
    # Generate corresponding descriptor data
    np.random.seed(42)
    X_train = np.random.randn(len(train_smiles), 50)
    X_test = np.random.randn(len(test_smiles), 50)
    
    # Initialize and fit analyzer with SMILES
    logger.info("\nFitting applicability domain with SMILES...")
    analyzer = ApplicabilityDomainAnalyzer(threshold_multiplier=3.0)
    analyzer.fit(X_train, train_smiles=train_smiles)
    
    # Calculate domain metrics
    leverages = analyzer.calculate_leverage(X_test)
    in_domain = analyzer.is_in_domain(X_test)
    confidence = analyzer.get_confidence(X_test)
    
    # Calculate similarity to training set
    logger.info("\nCalculating Tanimoto similarity to training set...")
    similarities = analyzer.calculate_similarity(test_smiles)
    
    # Display results
    logger.info("\nTest compound analysis:")
    for i, smi in enumerate(test_smiles):
        logger.info(f"\nCompound {i}: {smi}")
        logger.info(f"  Leverage: {leverages[i]:.4f}")
        logger.info(f"  In domain: {in_domain[i]}")
        logger.info(f"  Confidence: {confidence[i]:.4f}")
        logger.info(f"  Max similarity: {similarities[i]:.4f}")


def demo_confidence_methods():
    """Compare different confidence calculation methods."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: Confidence Calculation Methods")
    logger.info("="*80)
    
    # Generate sample data
    X_train, X_test, y_train, y_test = generate_sample_data()
    
    # Initialize and fit analyzer
    analyzer = ApplicabilityDomainAnalyzer(threshold_multiplier=3.0)
    analyzer.fit(X_train)
    
    # Calculate confidence using different methods
    logger.info("\nCalculating confidence using different methods...")
    
    confidence_leverage = analyzer.get_confidence(X_test, method='leverage')
    confidence_distance = analyzer.get_confidence(X_test, method='distance')
    
    logger.info("\nConfidence comparison:")
    logger.info(f"Leverage method:")
    logger.info(f"  Mean: {np.mean(confidence_leverage):.4f}")
    logger.info(f"  Std: {np.std(confidence_leverage):.4f}")
    logger.info(f"  Range: [{np.min(confidence_leverage):.4f}, {np.max(confidence_leverage):.4f}]")
    
    logger.info(f"\nDistance method:")
    logger.info(f"  Mean: {np.mean(confidence_distance):.4f}")
    logger.info(f"  Std: {np.std(confidence_distance):.4f}")
    logger.info(f"  Range: [{np.min(confidence_distance):.4f}, {np.max(confidence_distance):.4f}]")
    
    # Show correlation between methods
    correlation = np.corrcoef(confidence_leverage, confidence_distance)[0, 1]
    logger.info(f"\nCorrelation between methods: {correlation:.4f}")
    
    # Show examples
    logger.info("\nExample compounds (first 5):")
    for i in range(min(5, len(X_test))):
        logger.info(
            f"  Compound {i}: "
            f"leverage_conf={confidence_leverage[i]:.4f}, "
            f"distance_conf={confidence_distance[i]:.4f}"
        )


def main():
    """Run all demos."""
    logger.info("Starting Applicability Domain Analysis Demos")
    logger.info("=" * 80)
    
    try:
        demo_basic_usage()
        demo_with_outliers()
        demo_domain_statistics()
        demo_with_smiles()
        demo_confidence_methods()
        
        logger.info("\n" + "="*80)
        logger.info("All demos completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error running demos: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
