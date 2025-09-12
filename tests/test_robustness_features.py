#!/usr/bin/env python3
"""
Comprehensive test script for error handling and robustness features.

This script validates all error handling, recovery mechanisms, and robustness
features implemented in the TLR4 binding prediction pipeline.
"""

import sys
import os
import logging
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_error_handling_utilities():
    """Test error handling utilities."""
    logger.info("Testing error handling utilities...")
    
    try:
        from src.tlr4_binding.utils.error_handling import (
            RobustnessManager, CheckpointManager, CircuitBreaker,
            robust_execution, safe_execution, graceful_degradation,
            PipelineError, DataQualityError, ModelTrainingError, FeatureExtractionError
        )
        
        # Test custom exceptions
        try:
            raise PipelineError("Test pipeline error", "TEST_ERROR", {'context': 'test'})
        except PipelineError as e:
            assert e.error_code == "TEST_ERROR"
            assert 'context' in e.context
        
        # Test robustness manager
        robustness_manager = RobustnessManager()
        test_error = ValueError("Test error")
        robustness_manager.log_error(test_error, {'context': 'test'})
        assert len(robustness_manager.error_log) == 1
        
        # Test circuit breaker
        cb = CircuitBreaker(failure_threshold=2, timeout=1)
        def failing_function():
            raise ValueError("Always fails")
        
        # First two failures should pass through
        try:
            cb.call(failing_function)
        except ValueError:
            pass
        
        try:
            cb.call(failing_function)
        except ValueError:
            pass
        
        # Third call should be blocked (circuit open)
        try:
            cb.call(failing_function)
            assert False, "Circuit breaker should have blocked the call"
        except Exception as e:
            assert "Circuit breaker is OPEN" in str(e)
        
        logger.info("✓ Error handling utilities tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling utilities tests failed: {e}")
        return False


def test_checkpoint_manager():
    """Test checkpoint manager functionality."""
    logger.info("Testing checkpoint manager...")
    
    try:
        from src.tlr4_binding.utils.error_handling import CheckpointManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(str(temp_dir))
            
            # Test save and load
            checkpoint_id = "test_checkpoint"
            test_data = {
                'model_data': {'weights': [1, 2, 3]},
                'metadata': {'epoch': 10, 'loss': 0.5}
            }
            
            success = checkpoint_manager.save_checkpoint(checkpoint_id, test_data)
            assert success
            
            loaded_data = checkpoint_manager.load_checkpoint(checkpoint_id)
            assert loaded_data is not None
            assert loaded_data['model_data'] == test_data['model_data']
            
            # Test list checkpoints
            checkpoints = checkpoint_manager.list_checkpoints()
            assert checkpoint_id in checkpoints
            
            # Test delete checkpoint
            success = checkpoint_manager.delete_checkpoint(checkpoint_id)
            assert success
            assert checkpoint_manager.load_checkpoint(checkpoint_id) is None
        
        logger.info("✓ Checkpoint manager tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Checkpoint manager tests failed: {e}")
        return False


def test_data_quality_validation():
    """Test data quality validation."""
    logger.info("Testing data quality validation...")
    
    try:
        from src.tlr4_binding.utils.data_quality import (
            DataQualityValidator, DataAnomalyDetector, validate_molecular_data
        )
        
        # Create test data
        good_data = pd.DataFrame({
            'molecular_weight': [100, 150, 200, 180, 120],
            'logp': [1.0, 2.0, 1.5, 0.5, 2.5],
            'tpsa': [50, 60, 55, 45, 65],
            'compound_name': ['A', 'B', 'C', 'D', 'E']
        })
        
        bad_data = pd.DataFrame({
            'molecular_weight': [100, 150, -50, 180, 120],  # Negative value
            'logp': [1.0, 2.0, 1.5, None, 2.5],  # Missing value
            'tpsa': [50, 60, 55, 45, 65],
            'compound_name': ['A', 'B', None, 'D', 'E']  # Missing name
        })
        
        # Test validator
        validator = DataQualityValidator()
        
        # Test good data
        results = validator.validate_dataset(good_data, "good_data")
        assert results['validation_passed']
        assert results['quality_score'] > 0.8
        
        # Test bad data
        results = validator.validate_dataset(bad_data, "bad_data")
        # Bad data should have lower quality score than perfect data
        assert results['quality_score'] < 1.0  # Should be lower than perfect
        # The validation should detect some quality issues
        # Check if any validation components found issues
        has_issues = (
            len(results.get('structure_issues', [])) > 0 or
            len(results.get('missing_value_issues', [])) > 0 or
            len(results.get('data_type_issues', [])) > 0 or
            len(results.get('range_issues', [])) > 0 or
            len(results.get('outlier_issues', [])) > 0 or
            len(results.get('correlation_issues', [])) > 0
        )
        assert has_issues, "Bad data should trigger some validation issues"
        
        # Test anomaly detector
        detector = DataAnomalyDetector()
        
        # Create baseline data
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100)
        })
        
        detector.fit_baseline(baseline_data)
        
        # Create test data with anomalies
        test_data = baseline_data.copy()
        test_data.loc[0, 'feature_1'] = 10  # Anomaly
        
        results = detector.detect_anomalies(test_data)
        assert results['total_anomalies'] > 0
        
        # Test convenience function
        results = validate_molecular_data(good_data)
        assert 'validation_passed' in results
        
        logger.info("✓ Data quality validation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data quality validation tests failed: {e}")
        return False


def test_robust_execution_decorators():
    """Test robust execution decorators."""
    logger.info("Testing robust execution decorators...")
    
    try:
        from src.tlr4_binding.utils.error_handling import robust_execution, graceful_degradation
        
        # Test robust execution with success
        @robust_execution(max_retries=3, delay=0.01)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
        
        # Test robust execution with retry
        call_count = 0
        
        @robust_execution(max_retries=3, delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = failing_then_success()
        assert result == "success"
        assert call_count == 3
        
        # Test graceful degradation
        @graceful_degradation(fallback_value="degraded")
        def failing_function():
            raise ValueError("Function fails")
        
        result = failing_function()
        assert result == "degraded"
        
        logger.info("✓ Robust execution decorators tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Robust execution decorators tests failed: {e}")
        return False


def test_robustness_configuration():
    """Test robustness configuration management."""
    logger.info("Testing robustness configuration...")
    
    try:
        from src.tlr4_binding.utils.robustness_config import (
            RobustnessConfig, RobustnessConfigManager,
            get_development_config, get_production_config,
            get_research_config, get_minimal_config,
            validate_config, setup_default_configs
        )
        
        # Test configuration creation
        config = RobustnessConfig()
        assert config.max_retry_attempts == 3
        assert config.enable_checkpointing == True
        
        # Test environment-specific configs
        dev_config = get_development_config()
        assert dev_config.max_retry_attempts == 2
        assert dev_config.enable_checkpointing == False
        
        prod_config = get_production_config()
        assert prod_config.max_retry_attempts == 5
        assert prod_config.enable_checkpointing == True
        
        # Test configuration validation
        issues = validate_config(config)
        assert len(issues) == 0  # Default config should be valid
        
        # Test invalid configuration
        invalid_config = RobustnessConfig(max_retry_attempts=-1)
        issues = validate_config(invalid_config)
        assert len(issues) > 0
        assert any("max_retry_attempts must be non-negative" in issue for issue in issues)
        
        # Test configuration manager
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = RobustnessConfigManager(str(temp_dir))
            
            # Save and load config
            success = config_manager.save_config(config, "test_config")
            assert success
            
            loaded_config = config_manager.load_config("test_config")
            assert loaded_config is not None
            assert loaded_config.max_retry_attempts == config.max_retry_attempts
            
            # List configs
            configs = config_manager.list_configs()
            assert "test_config" in configs
            
            # Delete config
            success = config_manager.delete_config("test_config")
            assert success
        
        logger.info("✓ Robustness configuration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Robustness configuration tests failed: {e}")
        return False


def test_molecular_extractor_robustness():
    """Test molecular feature extractor robustness (if dependencies available)."""
    logger.info("Testing molecular feature extractor robustness...")
    
    try:
        from src.tlr4_binding.molecular_analysis.extractor import MolecularFeatureExtractor
        
        # Test initialization with robustness features
        extractor = MolecularFeatureExtractor(
            enable_checkpointing=True,
            checkpoint_interval=50,
            robustness_config={'max_retry_attempts': 3}
        )
        
        assert extractor.enable_checkpointing == True
        assert extractor.checkpoint_interval == 50
        assert extractor.robustness_manager is not None
        assert extractor.checkpoint_manager is not None
        
        # Test statistics tracking
        assert 'checkpoint_saves' in extractor.extraction_stats
        assert 'checkpoint_loads' in extractor.extraction_stats
        assert 'recovery_attempts' in extractor.extraction_stats
        
        logger.info("✓ Molecular feature extractor robustness tests passed")
        return True
        
    except ImportError as e:
        logger.warning(f"Molecular feature extractor tests skipped due to missing dependencies: {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Molecular feature extractor robustness tests failed: {e}")
        return False


def test_pipeline_orchestrator_robustness():
    """Test pipeline orchestrator robustness (if dependencies available)."""
    logger.info("Testing pipeline orchestrator robustness...")
    
    try:
        from src.tlr4_binding.ml_components.pipeline_orchestrator import (
            ResearchPipelineOrchestrator, ExperimentConfig
        )
        
        # Create test configuration
        config = ExperimentConfig(
            experiment_name="test_robustness",
            description="Test robustness features",
            tags={"test": "true"},
            data_path="test_data",
            output_path="test_output",
            enable_mlflow=False,  # Disable MLflow for testing
            enable_optuna=False   # Disable Optuna for testing
        )
        
        # Test initialization with robustness features
        orchestrator = ResearchPipelineOrchestrator(config)
        
        assert orchestrator.robustness_manager is not None
        assert orchestrator.checkpoint_manager is not None
        assert orchestrator.data_validator is not None
        assert orchestrator.anomaly_detector is not None
        assert 'model_training' in orchestrator.circuit_breakers
        assert 'hyperparameter_optimization' in orchestrator.circuit_breakers
        
        logger.info("✓ Pipeline orchestrator robustness tests passed")
        return True
        
    except ImportError as e:
        logger.warning(f"Pipeline orchestrator tests skipped due to missing dependencies: {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline orchestrator robustness tests failed: {e}")
        return False


def run_comprehensive_robustness_tests():
    """Run all robustness tests."""
    logger.info("Starting comprehensive robustness tests...")
    
    tests = [
        ("Error Handling Utilities", test_error_handling_utilities),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Data Quality Validation", test_data_quality_validation),
        ("Robust Execution Decorators", test_robust_execution_decorators),
        ("Robustness Configuration", test_robustness_configuration),
        ("Molecular Extractor Robustness", test_molecular_extractor_robustness),
        ("Pipeline Orchestrator Robustness", test_pipeline_orchestrator_robustness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"ROBUSTNESS TESTS SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL ROBUSTNESS TESTS PASSED!")
        return True
    else:
        logger.error("❌ SOME ROBUSTNESS TESTS FAILED!")
        return False


def main():
    """Main function."""
    logger.info("TLR4 Binding Prediction - Robustness Features Test Suite")
    logger.info("=" * 60)
    
    success = run_comprehensive_robustness_tests()
    
    if success:
        logger.info("\n✅ All robustness features are working correctly!")
        logger.info("The pipeline is ready for production use with comprehensive error handling.")
        return 0
    else:
        logger.error("\n❌ Some robustness features need attention.")
        logger.error("Please review the failed tests and fix the issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
