#!/usr/bin/env python3
"""
Check readiness for paper revision.

This script checks:
1. What infrastructure is implemented
2. What data is available
3. What results have been generated
4. What's still needed for the paper

Usage:
    python scripts/check_paper_readiness.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_mark(condition: bool) -> str:
    """Return checkmark or X based on condition."""
    return "✓" if condition else "✗"


def check_infrastructure() -> List[Tuple[str, bool, str]]:
    """Check what infrastructure is implemented."""
    checks = []
    
    # Data collection
    try:
        from tlr4_binding.data import collector
        checks.append(("Data collection module", True, "tlr4_binding/data/"))
    except ImportError:
        checks.append(("Data collection module", False, "Need to implement"))
    
    # Feature engineering
    try:
        from tlr4_binding.features import descriptors_3d
        checks.append(("3D descriptor calculator", True, "tlr4_binding/features/"))
    except ImportError:
        checks.append(("3D descriptor calculator", False, "Need to implement"))
    
    # Models
    try:
        from tlr4_binding.models import gat
        checks.append(("GAT model", True, "tlr4_binding/models/gat.py"))
    except ImportError:
        checks.append(("GAT model", False, "Need to implement"))
    
    try:
        from tlr4_binding.models import chemberta
        checks.append(("ChemBERTa model", True, "tlr4_binding/models/chemberta.py"))
    except ImportError:
        checks.append(("ChemBERTa model", False, "Need to implement"))
    
    try:
        from tlr4_binding.models import hybrid
        checks.append(("Hybrid model", True, "tlr4_binding/models/hybrid.py"))
    except ImportError:
        checks.append(("Hybrid model", False, "Need to implement"))
    
    try:
        from tlr4_binding.models import transfer_learning
        checks.append(("Transfer learning", True, "tlr4_binding/models/transfer_learning.py"))
    except ImportError:
        checks.append(("Transfer learning", False, "Need to implement"))
    
    try:
        from tlr4_binding.models import multi_task
        checks.append(("Multi-task model", True, "tlr4_binding/models/multi_task.py"))
    except ImportError:
        checks.append(("Multi-task model", False, "Need to implement"))
    
    # Validation
    try:
        from tlr4_binding.validation import framework
        checks.append(("Validation framework", True, "tlr4_binding/validation/framework.py"))
    except ImportError:
        checks.append(("Validation framework", False, "Need to implement"))
    
    try:
        from tlr4_binding.validation import applicability_domain
        checks.append(("Applicability domain", True, "tlr4_binding/validation/applicability_domain.py"))
    except ImportError:
        checks.append(("Applicability domain", False, "Need to implement"))
    
    # Interpretability
    try:
        from tlr4_binding.interpretability import analyzer
        checks.append(("Interpretability tools", True, "tlr4_binding/interpretability/"))
    except ImportError:
        checks.append(("Interpretability tools", False, "Need to implement"))
    
    return checks


def check_data() -> List[Tuple[str, bool, str]]:
    """Check what data is available."""
    checks = []
    
    # Original dataset
    original_data = Path("binding-data")
    if original_data.exists():
        csv_files = list(original_data.glob("*.csv"))
        checks.append((f"Original dataset ({len(csv_files)} files)", True, str(original_data)))
    else:
        checks.append(("Original dataset", False, "binding-data/ not found"))
    
    # Expanded dataset
    expanded_data = Path("binding-data/expanded_dataset.csv")
    checks.append(("Expanded TLR4 dataset", expanded_data.exists(), str(expanded_data)))
    
    # Related TLR data
    related_data = Path("binding-data/related_tlr_dataset.csv")
    checks.append(("Related TLR dataset", related_data.exists(), str(related_data)))
    
    # Features
    features = Path("binding-data/features.csv")
    checks.append(("Calculated features", features.exists(), str(features)))
    
    return checks


def check_models() -> List[Tuple[str, bool, str]]:
    """Check what trained models exist."""
    checks = []
    
    models_dir = Path("models/trained")
    if not models_dir.exists():
        checks.append(("Trained models directory", False, "models/trained/ not found"))
        return checks
    
    # Check for specific model files
    model_types = [
        "ensemble",
        "gat",
        "chemberta",
        "hybrid",
        "transfer"
    ]
    
    for model_type in model_types:
        model_files = list(models_dir.glob(f"{model_type}*.pt")) + list(models_dir.glob(f"{model_type}*.pkl"))
        checks.append((f"{model_type.upper()} model", len(model_files) > 0, f"Found {len(model_files)} files"))
    
    return checks


def check_results() -> List[Tuple[str, bool, str]]:
    """Check what results have been generated."""
    checks = []
    
    results_dir = Path("paper_results")
    if not results_dir.exists():
        checks.append(("Results directory", False, "paper_results/ not found"))
        return checks
    
    # Check for specific result types
    result_types = [
        ("Validation results", "validation"),
        ("Model comparison", "comparison"),
        ("Interpretability outputs", "interpretability"),
        ("Figures", "figures"),
        ("Tables", "tables")
    ]
    
    for name, subdir in result_types:
        path = results_dir / subdir
        if path.exists():
            file_count = len(list(path.glob("*")))
            checks.append((name, True, f"{file_count} files"))
        else:
            checks.append((name, False, f"{subdir}/ not found"))
    
    return checks


def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check required dependencies."""
    checks = []
    
    dependencies = [
        ("RDKit", "rdkit"),
        ("PyTorch", "torch"),
        ("PyTorch Geometric", "torch_geometric"),
        ("Transformers", "transformers"),
        ("ChEMBL client", "chembl_webresource_client"),
        ("PubChemPy", "pubchempy"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    for name, module in dependencies:
        try:
            __import__(module)
            checks.append((name, True, "Installed"))
        except ImportError:
            checks.append((name, False, "Not installed"))
    
    return checks


def print_section(title: str, checks: List[Tuple[str, bool, str]]):
    """Print a section of checks."""
    print(f"\n{title}")
    print("=" * 80)
    
    for name, status, detail in checks:
        mark = check_mark(status)
        print(f"  {mark} {name:40s} {detail}")
    
    # Summary
    passed = sum(1 for _, status, _ in checks if status)
    total = len(checks)
    print(f"\n  Summary: {passed}/{total} checks passed")


def main():
    print("=" * 80)
    print("TLR4 PAPER READINESS CHECK")
    print("=" * 80)
    
    # Check dependencies
    dep_checks = check_dependencies()
    print_section("DEPENDENCIES", dep_checks)
    
    # Check infrastructure
    infra_checks = check_infrastructure()
    print_section("INFRASTRUCTURE", infra_checks)
    
    # Check data
    data_checks = check_data()
    print_section("DATA", data_checks)
    
    # Check models
    model_checks = check_models()
    print_section("TRAINED MODELS", model_checks)
    
    # Check results
    result_checks = check_results()
    print_section("RESULTS", result_checks)
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    all_checks = dep_checks + infra_checks + data_checks + model_checks + result_checks
    total_passed = sum(1 for _, status, _ in all_checks if status)
    total_checks = len(all_checks)
    percentage = (total_passed / total_checks) * 100
    
    print(f"\nOverall completion: {total_passed}/{total_checks} ({percentage:.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Check critical items
    has_expanded_data = any(name == "Expanded TLR4 dataset" and status for name, status, _ in data_checks)
    has_trained_models = any(status for _, status, _ in model_checks)
    has_results = any(status for _, status, _ in result_checks)
    
    if not has_expanded_data:
        print("\n🔴 CRITICAL: Expand dataset from 49 to 150-300 compounds")
        print("   Action: python scripts/collect_expanded_dataset.py --include-related-tlrs")
    
    if not has_trained_models:
        print("\n🟡 IMPORTANT: Train models on expanded dataset")
        print("   Action: Run training scripts in examples/")
    
    if not has_results:
        print("\n🟡 IMPORTANT: Generate validation results")
        print("   Action: python examples/demo_full_validation_suite.py")
    
    if has_expanded_data and has_trained_models and has_results:
        print("\n🟢 READY: You have everything needed to write the revised paper!")
        print("   Next: Review results in paper_results/ and start writing")
    else:
        print("\n⏳ IN PROGRESS: Complete the actions above to be paper-ready")
    
    print("\n" + "=" * 80)
    print("For detailed guidance, see: PAPER_EXECUTION_GUIDE.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
