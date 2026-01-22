#!/usr/bin/env python3
"""
Installation Test Script
Verifies that all dependencies are installed and the pipeline can run.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages are importable."""
    print("Testing package imports...")

    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
    ]

    failed = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            failed.append(package_name)

    return len(failed) == 0, failed


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")

    data_dir = Path("data")
    required_files = ["winequality-red.csv", "winequality-white.csv"]

    failed = []
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            failed.append(filename)

    return len(failed) == 0, failed


def test_modules():
    """Test that custom modules are importable."""
    print("\nTesting custom modules...")

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    modules = [
        "wine_quality_ml.data_loader",
        "wine_quality_ml.eda",
        "wine_quality_ml.models",
        "wine_quality_ml.trainer",
        "wine_quality_ml.visualizer",
    ]

    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name} - IMPORT ERROR: {e}")
            failed.append(module_name)

    return len(failed) == 0, failed


def test_quick_load():
    """Test quick data loading."""
    print("\nTesting data loading...")

    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from wine_quality_ml.data_loader import WineDataLoader

        loader = WineDataLoader("data")
        df = loader.load_data("red")

        print(f"  ✓ Successfully loaded {len(df)} red wine samples")
        print(f"  ✓ Dataset has {len(df.columns)} columns")
        return True, None

    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False, str(e)


def main():
    """Run all tests."""
    print("=" * 70)
    print("WINE QUALITY ML PIPELINE - INSTALLATION TEST")
    print("=" * 70)

    results = []

    # Test 1: Package imports
    success, failed = test_imports()
    results.append(("Package Imports", success, failed))

    # Test 2: Data files
    success, failed = test_data_files()
    results.append(("Data Files", success, failed))

    # Test 3: Custom modules
    success, failed = test_modules()
    results.append(("Custom Modules", success, failed))

    # Test 4: Quick load
    success, failed = test_quick_load()
    results.append(("Data Loading", success, failed))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, success, failed in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20} {status}")
        if not success and failed:
            print(f"  Failed items: {failed}")
        all_passed = all_passed and success

    print("=" * 70)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou can now run the pipeline:")
        print("  python main.py --quick-test")
        print("  python main.py")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("\nPlease install missing dependencies:")
        print("  uv pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
