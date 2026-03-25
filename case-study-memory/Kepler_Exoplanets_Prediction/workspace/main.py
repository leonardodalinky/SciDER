#!/usr/bin/env python3
"""Run the Kepler KOI exoplanet detection experiment."""

from pathlib import Path

from experiment_runner import run_experiment

WORKSPACE = Path(__file__).parent
DATA_PATH = WORKSPACE / "cumulative.csv"
RESULTS_DIR = WORKSPACE / "results"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")
    report = run_experiment(DATA_PATH, RESULTS_DIR)
    print(f"\nBest model: {report['best_model']}")
    print(f"CV F1: {report['best_cv_metrics']['f1']:.4f}")
    print(f"Test F1: {report['best_test_metrics']['f1']:.4f}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
