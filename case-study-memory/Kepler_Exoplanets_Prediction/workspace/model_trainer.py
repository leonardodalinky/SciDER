"""Model training and evaluation for exoplanet classification."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42

# Models to compare
MODELS = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": None,  # Lazy import
    "LightGBM": None,  # Lazy import
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

CV_FOLDS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _get_models():
    """Lazy load XGBoost and LightGBM."""
    models = dict(MODELS)
    if models["XGBoost"] is None:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(random_state=RANDOM_STATE)
        except ImportError:
            models.pop("XGBoost", None)
    if models["LightGBM"] is None:
        try:
            from lightgbm import LGBMClassifier

            models["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
        except ImportError:
            models.pop("LightGBM", None)
    return {k: v for k, v in models.items() if v is not None}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train each model with 5-fold CV, compute metrics.
    Returns dict of model_name -> {cv_metrics, test_metrics, model, feature_importance}.
    """
    results = {}
    models = _get_models()

    for name, model in models.items():
        cv = cross_validate(
            model, X_train, y_train, cv=CV_FOLDS, scoring=list(SCORING.values()), n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        cv_metrics = {
            "accuracy": cv["test_accuracy"].mean(),
            "precision": cv["test_precision"].mean(),
            "recall": cv["test_recall"].mean(),
            "f1": cv["test_f1"].mean(),
            "roc_auc": cv["test_roc_auc"].mean(),
        }
        # Handle NaN (e.g. from single-class fold)
        for k in cv_metrics:
            if np.isnan(cv_metrics[k]):
                cv_metrics[k] = 0.0

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
        }

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = np.abs(model.coef_).ravel()

        results[name] = {
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "model": model,
            "feature_importance": fi,
        }

    return results
