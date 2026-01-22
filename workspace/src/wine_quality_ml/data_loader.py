"""
Data Loading and Preprocessing Module
Handles loading wine quality datasets and preprocessing operations.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class WineDataLoader:
    """Load and preprocess wine quality datasets."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the wine quality CSV files
        """
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()

    def load_data(self, wine_type: str = "both") -> pd.DataFrame:
        """
        Load wine quality data.

        Args:
            wine_type: Type of wine to load ('red', 'white', or 'both')

        Returns:
            DataFrame with wine quality data
        """
        if wine_type == "red":
            df = pd.read_csv(self.data_dir / "winequality-red.csv", sep=";", encoding="utf-8")
            df["wine_type"] = "red"
        elif wine_type == "white":
            df = pd.read_csv(self.data_dir / "winequality-white.csv", sep=";", encoding="utf-8")
            df["wine_type"] = "white"
        else:  # both
            red_df = pd.read_csv(self.data_dir / "winequality-red.csv", sep=";", encoding="utf-8")
            red_df["wine_type"] = "red"

            white_df = pd.read_csv(
                self.data_dir / "winequality-white.csv", sep=";", encoding="utf-8"
            )
            white_df["wine_type"] = "white"

            df = pd.concat([red_df, white_df], ignore_index=True)

        return df

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str = "quality",
        classification: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Preprocess data for machine learning.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            classification: If True, convert quality to binary classification
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names, target_name)
        """
        # Handle missing values (if any)
        df = df.dropna()

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert wine_type to numeric if present
        if "wine_type" in X.columns:
            X["wine_type"] = (X["wine_type"] == "red").astype(int)

        feature_names = X.columns

        # For classification, create binary labels (good wine: quality >= 6)
        if classification:
            y = (y >= 6).astype(int)
            target_name = "quality_binary"
        else:
            target_name = target_col

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if classification else None,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return (
            X_train_scaled,
            X_test_scaled,
            y_train.values,
            y_test.values,
            feature_names,
            target_name,
        )

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get information about the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with dataset information
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "quality_distribution": (
                df["quality"].value_counts().sort_index().to_dict()
                if "quality" in df.columns
                else None
            ),
        }

        if "wine_type" in df.columns:
            info["wine_type_distribution"] = df["wine_type"].value_counts().to_dict()

        return info
