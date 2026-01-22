"""
Results Visualization Module
Creates visualizations for model performance and analysis.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


class ResultsVisualizer:
    """Visualize model results and performance."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    def plot_model_comparison(self, comparison_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot comparison of model performances with error handling.

        Args:
            comparison_df: DataFrame with model comparison results
            save: Whether to save the plot
        """
        try:
            metrics = ["accuracy", "precision", "recall", "f1_score"]

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            for idx, metric in enumerate(metrics):
                data = comparison_df[["model_name", metric]].sort_values(metric, ascending=False)

                axes[idx].barh(
                    data["model_name"], data[metric], color="steelblue", edgecolor="black"
                )
                axes[idx].set_xlabel(
                    metric.replace("_", " ").title(), fontsize=12, fontweight="bold"
                )
                axes[idx].set_ylabel("Model", fontsize=12, fontweight="bold")
                axes[idx].set_title(
                    f'{metric.replace("_", " ").title()} Comparison', fontsize=13, fontweight="bold"
                )
                axes[idx].grid(axis="x", alpha=0.3)

                # Add value labels
                for i, v in enumerate(data[metric]):
                    axes[idx].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

            plt.tight_layout()

            if save:
                # Ensure directory exists
                self.output_dir.mkdir(exist_ok=True, parents=True)
                plot_path = self.output_dir / "model_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Verify file was saved
                if plot_path.exists() and plot_path.stat().st_size > 0:
                    return True
                else:
                    print(f"✗ Warning: Failed to save {plot_path}")
                    return False
            else:
                plt.show()
                return True

        except Exception as e:
            print(f"✗ Error creating model comparison plot: {e}")
            plt.close()
            raise

    def plot_confusion_matrices(
        self, results_dict: Dict[str, Dict[str, Any]], save: bool = True
    ) -> None:
        """
        Plot confusion matrices for all models.

        Args:
            results_dict: Dictionary of model results
            save: Whether to save the plot
        """
        n_models = len(results_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, result) in enumerate(results_dict.items()):
            cm = np.array(result["metrics"]["confusion_matrix"])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar=False,
                square=True,
                xticklabels=["Low Quality", "High Quality"],
                yticklabels=["Low Quality", "High Quality"],
            )
            axes[idx].set_title(f"{model_name}", fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("True Label", fontsize=10)
            axes[idx].set_xlabel("Predicted Label", fontsize=10)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_roc_curves(
        self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, save: bool = True
    ) -> None:
        """
        Plot ROC curves for all models.

        Args:
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 8))

        for model_name, model in models_dict.items():
            try:
                # Get probability predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                # Plot
                plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

            except AttributeError:
                print(f"Skipping ROC curve for {model_name} (no probability predictions)")

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        plt.title("ROC Curves Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        if save:
            plt.savefig(self.output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_learning_curves(
        self, model, model_name: str, X: np.ndarray, y: np.ndarray, cv: int = 5, save: bool = True
    ) -> None:
        """
        Plot learning curves for a model.

        Args:
            model: Model instance
            model_name: Name of the model
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            save: Whether to save the plot
        """
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))

        plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker="o")
        plt.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue"
        )

        plt.plot(train_sizes, val_mean, label="Cross-validation score", color="red", marker="s")
        plt.fill_between(
            train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="red"
        )

        plt.xlabel("Training Set Size", fontsize=12, fontweight="bold")
        plt.ylabel("Accuracy Score", fontsize=12, fontweight="bold")
        plt.title(f"Learning Curves - {model_name}", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)

        if save:
            filename = f'learning_curve_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(
        self, model, model_name: str, feature_names: List[str], top_n: int = 10, save: bool = True
    ) -> None:
        """
        Plot feature importance for tree-based models.

        Args:
            model: Trained model
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to display
            save: Whether to save the plot
        """
        try:
            # Get feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                print(f"Model {model_name} does not support feature importance")
                return

            # Create DataFrame
            feature_imp = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(top_n)
            )

            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(
                feature_imp["feature"],
                feature_imp["importance"],
                color="steelblue",
                edgecolor="black",
            )
            plt.xlabel("Importance", fontsize=12, fontweight="bold")
            plt.ylabel("Feature", fontsize=12, fontweight="bold")
            plt.title(
                f"Top {top_n} Feature Importances - {model_name}", fontsize=14, fontweight="bold"
            )
            plt.gca().invert_yaxis()
            plt.grid(axis="x", alpha=0.3)

            if save:
                filename = f'feature_importance_{model_name.replace(" ", "_").lower()}.png'
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting feature importance for {model_name}: {e}")

    def plot_training_time_comparison(self, comparison_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot training time comparison.

        Args:
            comparison_df: DataFrame with model comparison results
            save: Whether to save the plot
        """
        if "training_time" not in comparison_df.columns:
            print("Training time information not available")
            return

        data = comparison_df[["model_name", "training_time"]].sort_values("training_time")

        plt.figure(figsize=(10, 6))
        plt.barh(data["model_name"], data["training_time"], color="coral", edgecolor="black")
        plt.xlabel("Training Time (seconds)", fontsize=12, fontweight="bold")
        plt.ylabel("Model", fontsize=12, fontweight="bold")
        plt.title("Training Time Comparison", fontsize=14, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, v in enumerate(data["training_time"]):
            plt.text(v + 0.1, i, f"{v:.2f}s", va="center", fontsize=10)

        if save:
            plt.savefig(
                self.output_dir / "training_time_comparison.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()
