"""
Machine Learning Models Module
Implements multiple ML algorithms for wine quality prediction.
"""

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class WineQualityModels:
    """Collection of ML models for wine quality prediction."""

    def __init__(self, random_state: int = 42):
        """
        Initialize models with default hyperparameters.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize all models with default parameters."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000, n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "SVM": SVC(
                kernel="rbf", C=1.0, gamma="scale", random_state=self.random_state, probability=True
            ),
        }

    def get_model(self, model_name: str):
        """
        Get a specific model by name.

        Args:
            model_name: Name of the model

        Returns:
            Model instance

        Raises:
            ValueError: If model name is not found
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            )
        return self.models[model_name]

    def get_all_models(self) -> Dict[str, Any]:
        """
        Get all available models.

        Returns:
            Dictionary of model name to model instance
        """
        return self.models.copy()

    def save_model(self, model, model_name: str, output_dir: str = "models") -> None:
        """
        Save a trained model to disk with error handling and verification.

        Args:
            model: Trained model instance
            model_name: Name to save the model as
            output_dir: Directory to save the model
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

            filename = output_path / f"{model_name.replace(' ', '_').lower()}.joblib"
            joblib.dump(model, filename)

            # Verify model was saved
            if filename.exists() and filename.stat().st_size > 0:
                print(f"✓ Model saved: {filename} ({filename.stat().st_size} bytes)")
            else:
                print(f"✗ Warning: Failed to save or verify {filename}")

        except IOError as e:
            print(f"✗ Error saving model: {e}")
            raise
        except Exception as e:
            print(f"✗ Unexpected error saving model: {e}")
            raise

    def load_model(self, model_name: str, model_dir: str = "models"):
        """
        Load a trained model from disk.

        Args:
            model_name: Name of the model to load
            model_dir: Directory containing the model

        Returns:
            Loaded model instance
        """
        model_path = Path(model_dir)
        filename = model_path / f"{model_name.replace(' ', '_').lower()}.joblib"

        if not filename.exists():
            raise FileNotFoundError(f"Model file not found: {filename}")

        return joblib.load(filename)

    @staticmethod
    def get_model_params(model) -> dict:
        """
        Get model hyperparameters.

        Args:
            model: Model instance

        Returns:
            Dictionary of hyperparameters
        """
        return model.get_params()


class ModelOptimizer:
    """Optimize model hyperparameters using grid search or random search."""

    def __init__(self, random_state: int = 42):
        """
        Initialize optimizer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def get_param_grid(self, model_name: str) -> dict:
        """
        Get parameter grid for hyperparameter tuning.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of parameter grid
        """
        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 5, 10],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9],
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto", 0.001, 0.01],
                "kernel": ["rbf", "poly"],
            },
            "Logistic Regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            },
            "Decision Tree": {
                "max_depth": [5, 10, 15, 20],
                "min_samples_split": [5, 10, 20],
                "min_samples_leaf": [2, 5, 10],
            },
        }

        return param_grids.get(model_name, {})
