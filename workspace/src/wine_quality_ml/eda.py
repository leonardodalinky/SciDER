"""
Exploratory Data Analysis Module
Provides visualization and statistical analysis of wine quality data.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class WineEDA:
    """Perform exploratory data analysis on wine quality data."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize EDA module.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with summary statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe()
        return summary

    def plot_quality_distribution(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot quality score distribution.

        Args:
            df: Input DataFrame
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))

        if "wine_type" in df.columns:
            for wine_type in df["wine_type"].unique():
                subset = df[df["wine_type"] == wine_type]
                plt.hist(subset["quality"], bins=range(3, 11), alpha=0.6, label=wine_type)
            plt.legend()
        else:
            plt.hist(
                df["quality"], bins=range(3, 11), alpha=0.7, color="skyblue", edgecolor="black"
            )

        plt.xlabel("Quality Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Wine Quality Distribution", fontsize=14, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)

        if save:
            plt.savefig(self.output_dir / "quality_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            df: Input DataFrame
            save: Whether to save the plot
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_feature_distributions(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot distribution of all features.

        Args:
            df: Input DataFrame
            save: Whether to save the plot
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(
                df[col].dropna(), bins=30, alpha=0.7, color="steelblue", edgecolor="black"
            )
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel("Frequency", fontsize=10)
            axes[idx].set_title(f"Distribution of {col}", fontsize=10, fontweight="bold")
            axes[idx].grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_quality_vs_features(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot quality score against key features.

        Args:
            df: Input DataFrame
            save: Whether to save the plot
        """
        # Select key features (most correlated with quality)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "quality"]

        # Calculate correlation with quality
        correlations = df[numeric_cols + ["quality"]].corr()["quality"].drop("quality")
        top_features = correlations.abs().sort_values(ascending=False).head(6).index.tolist()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            axes[idx].scatter(df[feature], df["quality"], alpha=0.3, s=10)
            axes[idx].set_xlabel(feature, fontsize=10)
            axes[idx].set_ylabel("Quality", fontsize=10)
            axes[idx].set_title(
                f"Quality vs {feature}\n(corr: {correlations[feature]:.3f})",
                fontsize=10,
                fontweight="bold",
            )
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "quality_vs_features.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def generate_full_report(self, df: pd.DataFrame) -> dict:
        """
        Generate comprehensive EDA report with all visualizations.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with analysis results
        """
        print("Generating EDA report...")

        # Summary statistics
        summary_stats = self.generate_summary_statistics(df)

        # Generate all plots
        self.plot_quality_distribution(df)
        self.plot_correlation_matrix(df)
        self.plot_feature_distributions(df)
        self.plot_quality_vs_features(df)

        report = {
            "summary_statistics": summary_stats,
            "plots_saved": [
                "quality_distribution.png",
                "correlation_matrix.png",
                "feature_distributions.png",
                "quality_vs_features.png",
            ],
        }

        print(f"EDA report complete. Plots saved to {self.output_dir}")
        return report
