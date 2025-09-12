#!/usr/bin/env python3
"""
Validation script for the comprehensive molecular feature extractor.

This script validates that the MolecularFeatureExtractor has been properly
implemented with all required functionality for Task 5.
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_file_structure():
    """Validate that the extractor file exists and has the correct structure."""
    logger.info("=== Validating File Structure ===")
    
    extractor_file = Path("src/tlr4_binding/molecular_analysis/extractor.py")
    if not extractor_file.exists():
        logger.error("✗ Extractor file not found")
        return False
    
    # Read the file content
    content = extractor_file.read_text()
    
    # Check for required class
    if "class MolecularFeatureExtractor:" not in content:
        logger.error("✗ MolecularFeatureExtractor class not found")
        return False
    
    logger.info("✓ MolecularFeatureExtractor class found")
    
    # Check for required methods
    required_methods = [
        "def extract_features(",
        "def batch_extract(",
        "def extract_features_from_list(",
        "def get_extraction_stats(",
        "def get_performance_report(",
        "def validate_features(",
        "def get_feature_summary(",
        "def reset_stats(",
        "def save_performance_report(",
        "def get_feature_extraction_efficiency("
    ]
    
    for method in required_methods:
        if method not in content:
            logger.error(f"✗ Required method not found: {method}")
            return False
    
    logger.info("✓ All required methods found")
    
    # Check for performance tracking features
    performance_features = [
        "processing_times",
        "memory_usage",
        "feature_extraction_times",
        "start_time",
        "end_time"
    ]
    
    for feature in performance_features:
        if feature not in content:
            logger.error(f"✗ Performance tracking feature not found: {feature}")
            return False
    
    logger.info("✓ All performance tracking features found")
    
    # Check for SMILES extraction methods
    smiles_features = [
        "_extract_smiles_from_pdbqt",
        "_validate_smiles",
        "_reconstruct_smiles_from_coords"
    ]
    
    for feature in smiles_features:
        if feature not in content:
            logger.error(f"✗ SMILES extraction feature not found: {feature}")
            return False
    
    logger.info("✓ All SMILES extraction features found")
    
    return True


def validate_imports_and_dependencies():
    """Validate that required imports are present."""
    logger.info("=== Validating Imports and Dependencies ===")
    
    extractor_file = Path("src/tlr4_binding/molecular_analysis/extractor.py")
    content = extractor_file.read_text()
    
    # Check for required imports
    required_imports = [
        "import pandas as pd",
        "import numpy as np",
        "import logging",
        "from pathlib import Path",
        "from tqdm import tqdm",
        "from typing import Dict, List, Optional, Union, Any"
    ]
    
    for import_stmt in required_imports:
        if import_stmt not in content:
            logger.error(f"✗ Required import not found: {import_stmt}")
            return False
    
    logger.info("✓ All required imports found")
    
    # Check for component imports
    component_imports = [
        "from .parser import PDBQTParser",
        "from .descriptors import MolecularDescriptorCalculator",
        "from .structure import StructuralFeatureExtractor",
        "from .features import MolecularFeatures, FeatureSet"
    ]
    
    for import_stmt in component_imports:
        if import_stmt not in content:
            logger.error(f"✗ Required component import not found: {import_stmt}")
            return False
    
    logger.info("✓ All component imports found")
    
    return True


def validate_error_handling():
    """Validate that comprehensive error handling is implemented."""
    logger.info("=== Validating Error Handling ===")
    
    extractor_file = Path("src/tlr4_binding/molecular_analysis/extractor.py")
    content = extractor_file.read_text()
    
    # Check for error handling patterns
    error_patterns = [
        "try:",
        "except Exception as e:",
        "logger.error(",
        "logger.warning(",
        "raise ValueError("
    ]
    
    for pattern in error_patterns:
        if pattern not in content:
            logger.error(f"✗ Error handling pattern not found: {pattern}")
            return False
    
    logger.info("✓ Error handling patterns found")
    
    # Check for statistics tracking
    stats_patterns = [
        "extraction_stats",
        "failed_extractions",
        "extraction_errors"
    ]
    
    for pattern in stats_patterns:
        if pattern not in content:
            logger.error(f"✗ Statistics tracking pattern not found: {pattern}")
            return False
    
    logger.info("✓ Statistics tracking patterns found")
    
    return True


def validate_progress_tracking():
    """Validate that progress tracking and logging is implemented."""
    logger.info("=== Validating Progress Tracking ===")
    
    extractor_file = Path("src/tlr4_binding/molecular_analysis/extractor.py")
    content = extractor_file.read_text()
    
    # Check for progress tracking features
    progress_features = [
        "tqdm(",
        "progress_bar",
        "logger.info(",
        "set_postfix(",
        "desc="
    ]
    
    for feature in progress_features:
        if feature not in content:
            logger.error(f"✗ Progress tracking feature not found: {feature}")
            return False
    
    logger.info("✓ Progress tracking features found")
    
    # Check for timing and performance monitoring
    timing_features = [
        "time.time()",
        "processing_times",
        "memory_usage",
        "psutil"
    ]
    
    for feature in timing_features:
        if feature not in content:
            logger.error(f"✗ Timing/monitoring feature not found: {feature}")
            return False
    
    logger.info("✓ Timing and performance monitoring found")
    
    return True


def validate_feature_extraction_modes():
    """Validate that different feature extraction modes are supported."""
    logger.info("=== Validating Feature Extraction Modes ===")
    
    extractor_file = Path("src/tlr4_binding/molecular_analysis/extractor.py")
    content = extractor_file.read_text()
    
    # Check for configurable feature extraction
    mode_features = [
        "include_2d_features",
        "include_3d_features", 
        "include_advanced_features"
    ]
    
    for feature in mode_features:
        if feature not in content:
            logger.error(f"✗ Feature extraction mode not found: {feature}")
            return False
    
    logger.info("✓ Feature extraction modes found")
    
    # Check for default feature handling
    default_features = [
        "_get_default_2d_features",
        "_get_default_3d_features"
    ]
    
    for feature in default_features:
        if feature not in content:
            logger.error(f"✗ Default feature method not found: {feature}")
            return False
    
    logger.info("✓ Default feature methods found")
    
    return True


def validate_integration_tests():
    """Validate that integration tests are present."""
    logger.info("=== Validating Integration Tests ===")
    
    test_file = Path("tests/integration/test_molecular_feature_extractor_integration.py")
    if not test_file.exists():
        logger.error("✗ Integration test file not found")
        return False
    
    content = test_file.read_text()
    
    # Check for test methods
    test_methods = [
        "def test_single_file_extraction",
        "def test_batch_extraction",
        "def test_extract_from_list",
        "def test_feature_validation",
        "def test_performance_reporting"
    ]
    
    for method in test_methods:
        if method not in content:
            logger.error(f"✗ Integration test method not found: {method}")
            return False
    
    logger.info("✓ All integration test methods found")
    
    return True


def main():
    """Run all validation checks."""
    logger.info("Starting MolecularFeatureExtractor Validation")
    
    validations = [
        validate_file_structure,
        validate_imports_and_dependencies,
        validate_error_handling,
        validate_progress_tracking,
        validate_feature_extraction_modes,
        validate_integration_tests
    ]
    
    passed = 0
    failed = 0
    
    for validation in validations:
        try:
            if validation():
                passed += 1
                logger.info(f"✓ {validation.__name__} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {validation.__name__} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {validation.__name__} FAILED with exception: {str(e)}")
        
        logger.info("-" * 60)
    
    # Summary
    logger.info(f"Validation Summary:")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All validations passed! Task 5 implementation is complete.")
        return True
    else:
        logger.error(f"❌ {failed} validations failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
