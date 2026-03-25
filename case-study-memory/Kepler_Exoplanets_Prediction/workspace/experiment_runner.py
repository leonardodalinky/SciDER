"""Orchestrates the full exoplanet detection experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data_loader import load_and_preprocess
from feature_engineering import add_engineered_features, build_feature_pipeline
from model_trainer import train_and_evaluate


def run_experiment(data_path: Path, output_dir: Path) -> dict:
    """
    Run full pipeline: load -> engineer -> select -> scale -> train -> report.
    Returns results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    X_train, y_train, X_test, y_test = load_and_preprocess(data_path)

    # Add engineered features
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    # Select top 20 features and fit scaler
    selected_cols, scaler = build_feature_pipeline(X_train_fe, y_train)
    X_train_selected = X_train_fe[selected_cols]
    X_test_selected = X_test_fe[selected_cols]
    X_train_scaled = scaler.transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Train and evaluate models
    results = train_and_evaluate(
        X_train_scaled,
        y_train.values,
        X_test_scaled,
        y_test.values,
    )

    # Select best by mean F1
    best_name = max(results.keys(), key=lambda k: results[k]["cv_metrics"]["f1"])
    best_cv = results[best_name]["cv_metrics"]
    best_test = results[best_name]["test_metrics"]

    # Feature importance for best model
    fi = results[best_name]["feature_importance"]
    fi_dict = dict(zip(selected_cols, fi.tolist())) if fi is not None else {}

    # Build report
    report = {
        "model_comparison": {
            name: {
                "cv_f1": r["cv_metrics"]["f1"],
                "cv_accuracy": r["cv_metrics"]["accuracy"],
                "cv_precision": r["cv_metrics"]["precision"],
                "cv_recall": r["cv_metrics"]["recall"],
                "cv_roc_auc": r["cv_metrics"]["roc_auc"],
            }
            for name, r in results.items()
        },
        "best_model": best_name,
        "best_cv_metrics": best_cv,
        "best_test_metrics": best_test,
        "feature_importance": dict(sorted(fi_dict.items(), key=lambda x: -x[1])),
        "selected_features": selected_cols,
    }

    # Save report.json
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save report.txt
    lines = [
        "=" * 60,
        "KEPLER KOI EXOPLANET DETECTION - EXPERIMENT REPORT",
        "=" * 60,
        "",
        "MODEL COMPARISON (5-fold CV):",
        "-" * 40,
    ]
    for name, m in report["model_comparison"].items():
        lines.append(f"  {name}:")
        lines.append(
            f"    F1={m['cv_f1']:.4f}  Acc={m['cv_accuracy']:.4f}  Prec={m['cv_precision']:.4f}  Rec={m['cv_recall']:.4f}  AUC={m['cv_roc_auc']:.4f}"
        )
    lines.extend(
        [
            "",
            "BEST MODEL: " + report["best_model"],
            "",
            "BEST CV METRICS:",
            f"  F1:      {best_cv['f1']:.4f}",
            f"  Accuracy: {best_cv['accuracy']:.4f}",
            f"  Precision: {best_cv['precision']:.4f}",
            f"  Recall:   {best_cv['recall']:.4f}",
            f"  ROC-AUC: {best_cv['roc_auc']:.4f}",
            "",
            "BEST TEST METRICS:",
            f"  F1:      {best_test['f1']:.4f}",
            f"  Accuracy: {best_test['accuracy']:.4f}",
            f"  Precision: {best_test['precision']:.4f}",
            f"  Recall:   {best_test['recall']:.4f}",
            f"  ROC-AUC: {best_test['roc_auc']:.4f}",
            "",
            "TOP FEATURES (by importance):",
        ]
    )
    for feat, imp in list(report["feature_importance"].items())[:10]:
        lines.append(f"  {feat}: {imp:.4f}")
    lines.append("")
    with open(output_dir / "report.txt", "w") as f:
        f.write("\n".join(lines))

    return report
