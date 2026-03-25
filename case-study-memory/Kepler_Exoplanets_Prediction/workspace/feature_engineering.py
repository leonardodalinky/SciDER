"""Physics-inspired feature engineering and selection for Kepler KOI data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TOP_K_FEATURES = 20


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide with zero handling."""
    return np.where(b != 0, a / b, 0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-inspired features. Returns df with new columns."""
    df = df.copy()

    if "koi_period" in df.columns:
        df["period_squared"] = df["koi_period"] ** 2

    if "koi_model_snr" in df.columns and "koi_depth" in df.columns:
        df["snr_depth_product"] = df["koi_model_snr"] * df["koi_depth"]

    if "koi_depth" in df.columns and "koi_srad" in df.columns:
        df["relative_depth"] = _safe_divide(df["koi_depth"], df["koi_srad"])

    if "koi_slogg" in df.columns and "koi_srad" in df.columns:
        df["stellar_density_proxy"] = _safe_divide(df["koi_slogg"], df["koi_srad"])

    if "koi_period_err1" in df.columns and "koi_period_err2" in df.columns:
        df["period_uncertainty"] = np.abs(df["koi_period_err1"] - df["koi_period_err2"])

    return df


def select_top_features(
    X_train: pd.DataFrame, y_train: pd.Series, k: int = TOP_K_FEATURES
) -> list[str]:
    """Use Random Forest feature importance to select top k features."""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    return importance.nlargest(k).index.tolist()


def build_feature_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Build preprocessing pipeline: add features, select top k, scale.
    Returns (feature_names, scaler, selected_columns).
    """
    X = add_engineered_features(X_train)
    selected = select_top_features(X, y_train, k=TOP_K_FEATURES)
    X_selected = X[selected]
    scaler = StandardScaler()
    scaler.fit(X_selected)
    return selected, scaler
