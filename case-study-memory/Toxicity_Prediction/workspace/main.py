#!/usr/bin/env python3
"""Run the full mutagenicity classification pipeline."""

from pathlib import Path

from preprocess_data import save_preprocessed
from train_models import run_and_save
from utils import WORKSPACE

DATA_DIR = WORKSPACE / "data"
RESULTS_DIR = WORKSPACE / "results"


def main():
    data_path = WORKSPACE / "Mutagenicity_N6512 2.csv"
    if not data_path.exists():
        data_path = WORKSPACE / "Mutagenicity_N6512.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected Mutagenicity_N6512.csv in {WORKSPACE}"
        )

    print("Step 1: Preprocessing...")
    X, y, names = save_preprocessed(DATA_DIR)
    print(f"  Loaded {len(y)} molecules, {X.shape[1]} features")

    print("Step 2: Training models...")
    metrics = run_and_save(RESULTS_DIR)
    print("\nMetrics:")
    for name, m in metrics.items():
        print(f"  {name}: F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
