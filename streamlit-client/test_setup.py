#!/usr/bin/env python3
"""
Test Setup Script for SciDER Streamlit Interface

This script validates that all dependencies and configurations are correct.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing imports...")

    try:
        import streamlit as st

        print("  ✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"  ❌ Streamlit import failed: {e}")
        print("     Fix: pip install streamlit")
        return False

    try:
        from scider.workflows.full_workflow_with_ideation import FullWorkflowWithIdeation

        print("  ✅ SciDER workflow imported successfully")
    except ImportError as e:
        print(f"  ❌ SciDER import failed: {e}")
        print("     Fix: Ensure parent directory is set up correctly")
        return False

    try:
        from scider.core.brain import Brain
        from scider.core.llms import ModelRegistry

        print("  ✅ SciDER core modules imported successfully")
    except ImportError as e:
        print(f"  ❌ SciDER core import failed: {e}")
        return False

    return True


def test_env_file():
    """Test that .env file exists and has required keys."""
    print("\n🔍 Testing environment configuration...")

    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        print(f"  ⚠️  .env file not found at {env_path}")
        print("     Recommendation: Copy .env.template to .env and add your API keys")
        return False

    # Read .env and check for API keys
    env_content = env_path.read_text()

    required_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    found_keys = []

    for key in required_keys:
        if key in env_content:
            # Check if it's not just a placeholder
            for line in env_content.split("\n"):
                if line.startswith(key) and "..." not in line and "your_" not in line.lower():
                    found_keys.append(key)
                    break

    if len(found_keys) == len(required_keys):
        print(f"  ✅ All required API keys found")
        return True
    else:
        missing = set(required_keys) - set(found_keys)
        print(f"  ⚠️  Missing or incomplete API keys: {missing}")
        print(f"     Found: {found_keys}")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\n🔍 Testing directory structure...")

    parent_dir = Path(__file__).parent.parent
    required_dirs = ["scider", "scider/workflows", "scider/agents", "scider/tools"]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = parent_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name} exists")
        else:
            print(f"  ❌ {dir_name} not found")
            all_exist = False

    return all_exist


def test_streamlit_files():
    """Test that streamlit client files exist."""
    print("\n🔍 Testing Streamlit client files...")

    client_dir = Path(__file__).parent
    required_files = [
        "app.py",
        "app_enhanced.py",
        "display_components.py",
        "workflow_monitor.py",
        "requirements.txt",
        "README.md",
    ]

    all_exist = True
    for file_name in required_files:
        file_path = client_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name} exists")
        else:
            print(f"  ❌ {file_name} not found")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("SciDER Streamlit Interface - Setup Validation")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_env_file),
        ("Directory Structure Test", test_directories),
        ("Streamlit Files Test", test_streamlit_files),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")

    print("\n" + f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! You're ready to run the Streamlit interface.")
        print("\nRun the interface with:")
        print("  ./run.sh          (Linux/Mac)")
        print("  run.bat           (Windows)")
        print("  streamlit run app_enhanced.py  (Direct)")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the interface.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
