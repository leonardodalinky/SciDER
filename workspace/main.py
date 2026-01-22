#!/usr/bin/env python3
"""
Wine Quality Machine Learning Pipeline
Main execution script for training and evaluating ML models on wine quality data.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wine_quality_ml.data_loader import WineDataLoader
from wine_quality_ml.eda import WineEDA
from wine_quality_ml.models import WineQualityModels
from wine_quality_ml.trainer import ModelTrainer
from wine_quality_ml.visualizer import ResultsVisualizer


def verify_outputs(check_optional=True):
    """
    Verify that all expected output files were created successfully.

    Args:
        check_optional: If False, skip optional files like learning curves
    """
    import os

    # Required files that should always be present
    required_files = {
        "results": [
            "model_comparison.csv",
            "detailed_results.json",
            "model_comparison.png",
            "confusion_matrices.png",
            "roc_curves.png",
            "training_time_comparison.png",
        ],
        "models": [],
    }

    # Optional files (EDA outputs, learning curves, etc.)
    optional_files = {
        "results": [
            "quality_distribution.png",
            "correlation_matrix.png",
            "feature_distributions.png",
            "quality_vs_features.png",
        ]
    }

    all_verified = True

    for directory, files in required_files.items():
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"✗ Directory '{directory}/' does not exist!")
            all_verified = False
            continue

        print(f"\nVerifying {directory}/ directory:")

        # Check required files
        for filename in files:
            file_path = dir_path / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"  ✓ {filename} ({file_size:,} bytes)")
            else:
                print(f"  ✗ {filename} - NOT FOUND")
                all_verified = False

        # Check optional files if requested
        if check_optional and directory in optional_files:
            for filename in optional_files[directory]:
                file_path = dir_path / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"  ✓ {filename} ({file_size:,} bytes)")

        # List all model files dynamically
        if directory == "models":
            model_files = list(dir_path.glob("*.joblib"))
            if model_files:
                for model_file in model_files:
                    file_size = model_file.stat().st_size
                    print(f"  ✓ {model_file.name} ({file_size:,} bytes)")
            else:
                print(f"  ✗ No model files found!")
                all_verified = False

        # List any additional files (like learning curves)
        if directory == "results":
            additional_files = list(dir_path.glob("learning_curve_*.png"))
            for additional_file in additional_files:
                file_size = additional_file.stat().st_size
                print(f"  ✓ {additional_file.name} ({file_size:,} bytes)")

    if all_verified:
        print("\n✓ All expected output files verified successfully!")
    else:
        print("\n✗ Warning: Some required output files are missing. See details above.")

    return all_verified


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Wine Quality ML Pipeline")
    parser.add_argument(
        "--wine-type",
        choices=["red", "white", "both"],
        default="both",
        help="Type of wine to analyze (default: both)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--skip-eda", action="store_true", help="Skip exploratory data analysis")
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick test with limited models"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WINE QUALITY MACHINE LEARNING PIPELINE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Wine Type: {args.wine_type}")
    print(f"  - Test Size: {args.test_size}")
    print(f"  - CV Folds: {args.cv_folds}")
    print(f"  - Random Seed: {args.random_seed}")
    print(f"  - Quick Test Mode: {args.quick_test}")
    print("=" * 80 + "\n")

    # Step 1: Load Data
    print("Step 1: Loading Data...")
    data_loader = WineDataLoader(data_dir="data")
    df = data_loader.load_data(wine_type=args.wine_type)

    data_info = data_loader.get_data_info(df)
    print(f"Dataset loaded: {data_info['shape'][0]} samples, {data_info['shape'][1]} features")
    print(f"Quality distribution: {data_info['quality_distribution']}")
    if "wine_type_distribution" in data_info:
        print(f"Wine type distribution: {data_info['wine_type_distribution']}")
    print()

    # Step 2: Exploratory Data Analysis
    if not args.skip_eda:
        print("Step 2: Performing Exploratory Data Analysis...")
        eda = WineEDA(output_dir="results")
        eda_report = eda.generate_full_report(df)
        print(f"EDA complete. Generated {len(eda_report['plots_saved'])} plots.\n")
    else:
        print("Step 2: Skipping EDA (--skip-eda flag set)\n")

    # Step 3: Preprocess Data
    print("Step 3: Preprocessing Data...")
    X_train, X_test, y_train, y_test, feature_names, target_name = data_loader.preprocess_data(
        df, test_size=args.test_size, random_state=args.random_seed
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {len(feature_names)}")
    print(f"Target: {target_name} (binary classification: good quality >= 6)\n")

    # Step 4: Initialize Models
    print("Step 4: Initializing Models...")
    model_factory = WineQualityModels(random_state=args.random_seed)
    models = model_factory.get_all_models()

    # For quick test, only use a subset of models
    if args.quick_test:
        quick_models = ["Logistic Regression", "Random Forest", "XGBoost"]
        models = {k: v for k, v in models.items() if k in quick_models}
        print(f"Quick test mode: Using {len(models)} models: {list(models.keys())}")
    else:
        print(f"Using {len(models)} models: {list(models.keys())}")
    print()

    # Step 5: Train and Evaluate Models
    print("Step 5: Training and Evaluating Models...")
    trainer = ModelTrainer(output_dir="results")
    comparison_df = trainer.train_and_evaluate_all(
        models, X_train, X_test, y_train, y_test, perform_cv=True, cv_folds=args.cv_folds
    )

    # Step 6: Get Best Model
    print("Step 6: Identifying Best Model...")
    best_model_name, best_model, best_metrics = trainer.get_best_model(metric="accuracy")
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    if best_metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print()

    # Print detailed classification report for best model
    trainer.print_classification_report(best_model, X_test, y_test, best_model_name)

    # Step 7: Visualize Results
    print("Step 7: Generating Visualizations...")
    visualizer = ResultsVisualizer(output_dir="results")

    # Model comparison
    visualizer.plot_model_comparison(comparison_df)
    print("  - Model comparison plot saved")

    # Confusion matrices
    visualizer.plot_confusion_matrices(trainer.results)
    print("  - Confusion matrices saved")

    # ROC curves
    trained_models = {name: result["model"] for name, result in trainer.results.items()}
    visualizer.plot_roc_curves(trained_models, X_test, y_test)
    print("  - ROC curves saved")

    # Training time comparison
    visualizer.plot_training_time_comparison(comparison_df)
    print("  - Training time comparison saved")

    # Feature importance for best model
    if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
        visualizer.plot_feature_importance(best_model, best_model_name, list(feature_names))
        print(f"  - Feature importance for {best_model_name} saved")

    # Learning curve for best model
    if not args.quick_test:
        print(f"  - Generating learning curve for {best_model_name}...")
        import numpy as np

        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        visualizer.plot_learning_curves(
            models[best_model_name], best_model_name, X_full, y_full, cv=args.cv_folds
        )
        print(f"  - Learning curve saved")

    print()

    # Step 8: Save Best Model
    print("Step 8: Saving Best Model...")
    model_factory.save_model(best_model, best_model_name, output_dir="models")
    print(f"Best model ({best_model_name}) saved to models/\n")

    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 80)

    # Step 9: Verify all outputs were created
    print("\nStep 9: Verifying Output Files...")
    verify_outputs()

    print("\nResults saved in:")
    print("  - results/: Visualizations and evaluation metrics")
    print("  - models/: Trained model files")
    print("=" * 80)


if __name__ == "__main__":
    main()
