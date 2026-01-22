"""
Model Training and Evaluation Module
Handles training, evaluation, and comparison of ML models.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    """Train and evaluate machine learning models."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize trainer.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}

    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray, model_name: str):
        """
        Train a single model.

        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model

        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"{model_name} training completed in {training_time:.2f} seconds")

        return model, training_time

    def evaluate_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Get probability predictions if available
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except AttributeError:
            y_pred_proba = None
            roc_auc = None

        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1_score": f1_score(y_test, y_pred, average="binary"),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        return metrics

    def cross_validate_model(
        self, model, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model instance
            X: Features
            y: Labels
            cv: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        return {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    def train_and_evaluate_all(
        self,
        models_dict: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        perform_cv: bool = True,
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Train and evaluate all models.

        Args:
            models_dict: Dictionary of model name to model instance
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            perform_cv: Whether to perform cross-validation
            cv_folds: Number of cross-validation folds

        Returns:
            DataFrame with comparison of all models
        """
        results = []

        for model_name, model in models_dict.items():
            # Train model
            trained_model, training_time = self.train_model(model, X_train, y_train, model_name)

            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test, model_name)
            metrics["training_time"] = training_time

            # Cross-validation
            if perform_cv:
                print(f"Performing {cv_folds}-fold cross-validation for {model_name}...")
                cv_results = self.cross_validate_model(
                    model,
                    np.vstack([X_train, X_test]),
                    np.concatenate([y_train, y_test]),
                    cv=cv_folds,
                )
                metrics.update(cv_results)

            results.append(metrics)
            self.results[model_name] = {"model": trained_model, "metrics": metrics}

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)

        # Select columns for display (excluding confusion matrix and cv_scores for clarity)
        display_cols = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        if perform_cv:
            display_cols.extend(["cv_mean", "cv_std"])
        display_cols.append("training_time")

        comparison_df_display = comparison_df[display_cols].copy()
        comparison_df_display = comparison_df_display.sort_values("accuracy", ascending=False)

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(comparison_df_display.to_string(index=False))
        print("=" * 80 + "\n")

        # Save results
        self._save_results(comparison_df)

        return comparison_df

    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, Any, Dict]:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, model, metrics)
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]["metrics"][metric])

        best_result = self.results[best_model_name]
        return best_model_name, best_result["model"], best_result["metrics"]

    def _save_results(self, comparison_df: pd.DataFrame) -> None:
        """
        Save results to files with error handling and verification.

        Args:
            comparison_df: DataFrame with model comparison results
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(exist_ok=True, parents=True)

            # Save as CSV
            csv_path = self.output_dir / "model_comparison.csv"
            comparison_df.to_csv(csv_path, index=False)

            # Verify CSV was created
            if csv_path.exists() and csv_path.stat().st_size > 0:
                print(f"✓ Results saved to {csv_path} ({csv_path.stat().st_size} bytes)")
            else:
                print(f"✗ Warning: Failed to save or verify {csv_path}")

            # Save detailed results as JSON
            json_results = {}
            for model_name, result in self.results.items():
                json_results[model_name] = result["metrics"]

            json_path = self.output_dir / "detailed_results.json"
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2)

            # Verify JSON was created
            if json_path.exists() and json_path.stat().st_size > 0:
                print(f"✓ Detailed results saved to {json_path} ({json_path.stat().st_size} bytes)")
            else:
                print(f"✗ Warning: Failed to save or verify {json_path}")

        except IOError as e:
            print(f"✗ Error saving results: {e}")
            raise
        except Exception as e:
            print(f"✗ Unexpected error saving results: {e}")
            raise

    def print_classification_report(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> None:
        """
        Print detailed classification report.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        """
        y_pred = model.predict(X_test)
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION REPORT - {model_name}")
        print("=" * 80)
        print(classification_report(y_test, y_pred, target_names=["Low Quality", "High Quality"]))
        print("=" * 80 + "\n")
