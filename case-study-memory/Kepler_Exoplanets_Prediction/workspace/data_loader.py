"""Data loading and preprocessing for Kepler KOI exoplanet detection."""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Identifier columns to drop (non-numeric, not useful for classification)
ID_COLUMNS = [
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_tce_plnt_num",
    "koi_tce_delivname",
    "ra",
    "dec",
]


def load_and_preprocess(data_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load cumulative.csv, filter CONFIRMED/FALSE POSITIVE, preprocess, split.
    Returns (X_train, y_train, X_test, y_test).
    """
    df = pd.read_csv(data_path)

    # Filter target classes
    valid = df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    df = df[valid].copy()

    # Encode labels: CONFIRMED=1, FALSE POSITIVE=0
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
    df = df.drop(columns=["koi_disposition"])

    # Drop identifier columns (keep only if present)
    to_drop = [c for c in ID_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].copy()

    # Handle missing values: median imputation, then 0 for any remaining
    df = df.fillna(df.median())
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # SMOTE on training data only
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test
