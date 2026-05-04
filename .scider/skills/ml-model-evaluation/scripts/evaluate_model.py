#!/usr/bin/env python3
"""
evaluate_model.py — ML Model Evaluation Script for SciDER

Usage:
    python evaluate_model.py --predictions pred.csv --labels labels.csv \
        --task classification [--output report.md] [--features features.csv] \
        [--prob_col prob] [--label_col label] [--pred_col prediction]

Arguments:
    --predictions   CSV file containing model predictions (required)
    --labels        CSV file containing ground-truth labels (required)
    --task          Task type: classification | regression | clustering | ranking
    --output        Output markdown report path (default: evaluation_report.md)
    --features      CSV of original features (used for clustering silhouette score)
    --prob_col      Column name for predicted probabilities (classification only)
    --label_col     Column name for true labels (default: 'label')
    --pred_col      Column name for predictions (default: 'prediction')
    --k             k for Precision@k / NDCG@k (ranking only, default: 10)
"""

import argparse
import sys
import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, description: str) -> pd.DataFrame:
    """Load a CSV file with informative error messages."""
    if not os.path.exists(path):
        print(f"ERROR: {description} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: Could not read {description} file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    if df.empty:
        print(f"ERROR: {description} file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return df


def get_column(df: pd.DataFrame, col: str, df_name: str) -> pd.Series:
    """Extract a column with a helpful error if it's missing."""
    if col not in df.columns:
        available = ", ".join(df.columns.tolist())
        print(
            f"ERROR: Column '{col}' not found in {df_name}.\n"
            f"       Available columns: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    return df[col]


def to_numeric_array(series: pd.Series, name: str) -> np.ndarray:
    """Convert Series to float array with a clear error if conversion fails."""
    try:
        arr = pd.to_numeric(series, errors="raise").values
    except (ValueError, TypeError):
        bad = series[~series.apply(lambda x: str(x).replace(".", "", 1).lstrip("-").isdigit())]
        print(
            f"ERROR: Column '{name}' contains non-numeric values: "
            f"{bad.head(5).tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return arr.astype(float)


def check_lengths_match(a, b, name_a, name_b):
    if len(a) != len(b):
        print(
            f"ERROR: Length mismatch — {name_a} has {len(a)} rows, "
            f"{name_b} has {len(b)} rows.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def evaluate_classification(y_true, y_pred, y_prob=None):
    """Return dict of classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
        matthews_corrcoef,
        confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    results["n_classes"] = len(classes)
    results["classes"] = classes.tolist()

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    results["class_counts"] = dict(zip(unique.tolist(), counts.tolist()))
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    results["imbalance_ratio"] = imbalance_ratio
    results["imbalance_warning"] = imbalance_ratio > 3

    for avg in ["macro", "micro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        results[f"precision_{avg}"] = p
        results[f"recall_{avg}"] = r
        results[f"f1_{avg}"] = f1

    results["mcc"] = matthews_corrcoef(y_true, y_pred)
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob, dtype=float)
            if len(classes) == 2:
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
                results["auc_roc"] = roc_auc_score(y_true, y_prob)
                results["auc_pr"] = average_precision_score(y_true, y_prob)
            else:
                if y_prob.ndim == 2 and y_prob.shape[1] == len(classes):
                    results["auc_roc_macro_ovr"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    )
                    results["auc_pr_macro"] = average_precision_score(
                        y_true, y_prob, average="macro"
                    )
                else:
                    results["auc_roc"] = "SKIPPED: probability array shape mismatch for multi-class"
        except Exception as e:
            results["auc_roc"] = f"SKIPPED: {e}"
            results["auc_pr"] = f"SKIPPED: {e}"

    return results


def format_classification_report(results: dict, y_true, y_pred) -> str:
    from sklearn.metrics import classification_report

    lines = []
    lines.append("## Classification Metrics\n")

    if results.get("imbalance_warning"):
        ratio = results["imbalance_ratio"]
        lines.append(
            f"> **Imbalance Warning**: Class imbalance ratio is {ratio:.1f}:1. "
            "Do NOT rely on accuracy alone. Use F1-macro, AUC-PR, or MCC as primary metrics.\n"
        )

    lines.append("### Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {results['accuracy']:.4f} |")
    lines.append(f"| Balanced Accuracy | {results['balanced_accuracy']:.4f} |")
    lines.append(f"| F1 (macro) | {results['f1_macro']:.4f} |")
    lines.append(f"| F1 (weighted) | {results['f1_weighted']:.4f} |")
    lines.append(f"| Precision (macro) | {results['precision_macro']:.4f} |")
    lines.append(f"| Recall (macro) | {results['recall_macro']:.4f} |")
    lines.append(f"| MCC | {results['mcc']:.4f} |")

    if "auc_roc" in results:
        val = results["auc_roc"]
        lines.append(f"| AUC-ROC | {val if isinstance(val, str) else f'{val:.4f}'} |")
    if "auc_pr" in results:
        val = results["auc_pr"]
        lines.append(f"| AUC-PR | {val if isinstance(val, str) else f'{val:.4f}'} |")
    if "auc_roc_macro_ovr" in results:
        lines.append(f"| AUC-ROC (macro OvR) | {results['auc_roc_macro_ovr']:.4f} |")
    if "auc_pr_macro" in results:
        lines.append(f"| AUC-PR (macro) | {results['auc_pr_macro']:.4f} |")

    lines.append("")
    lines.append("### Class Distribution\n")
    lines.append("| Class | Count |")
    lines.append("|-------|-------|")
    for cls, cnt in results["class_counts"].items():
        lines.append(f"| {cls} | {cnt} |")
    lines.append(f"\nImbalance ratio: {results['imbalance_ratio']:.2f}:1\n")

    lines.append("### Per-Class Report\n")
    lines.append("```")
    lines.append(classification_report(y_true, y_pred, zero_division=0))
    lines.append("```\n")

    lines.append("### Confusion Matrix\n")
    lines.append("```")
    cm = np.array(results["confusion_matrix"])
    lines.append(str(cm))
    lines.append("```\n")

    lines.append("### Interpretation\n")
    mcc = results["mcc"]
    if mcc > 0.7:
        lines.append(f"- MCC = {mcc:.3f}: Strong predictive performance across all classes.")
    elif mcc > 0.4:
        lines.append(f"- MCC = {mcc:.3f}: Moderate performance. Review per-class recall for weak classes.")
    else:
        lines.append(f"- MCC = {mcc:.3f}: Poor performance. Compare against majority-class baseline.")

    ba = results["balanced_accuracy"]
    n_classes = results["n_classes"]
    random_ba = 1.0 / n_classes
    lines.append(
        f"- Balanced accuracy = {ba:.3f} vs. random baseline = {random_ba:.3f} "
        f"({'above' if ba > random_ba else 'at or below'} random chance)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results = {}
    results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    results["mae"] = float(mean_absolute_error(y_true, y_pred))
    results["r2"] = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    if nonzero.sum() > 0:
        results["mape"] = float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        )
    else:
        results["mape"] = None

    # Huber loss (delta = 1.35 * MAD)
    residuals = y_true - y_pred
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    delta = 1.35 * mad if mad > 0 else 1.0
    huber = np.where(
        np.abs(residuals) <= delta,
        0.5 * residuals**2,
        delta * (np.abs(residuals) - 0.5 * delta),
    )
    results["huber_loss"] = float(np.mean(huber))
    results["huber_delta"] = delta

    # Descriptive stats
    results["y_true_mean"] = float(y_true.mean())
    results["y_true_std"] = float(y_true.std())
    results["n_samples"] = len(y_true)

    return results


def format_regression_report(results: dict) -> str:
    lines = []
    lines.append("## Regression Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| RMSE | {results['rmse']:.4f} |")
    lines.append(f"| MAE | {results['mae']:.4f} |")
    lines.append(f"| R² | {results['r2']:.4f} |")
    if results["mape"] is not None:
        lines.append(f"| MAPE (%) | {results['mape']:.2f}% |")
    else:
        lines.append("| MAPE | N/A (zero targets present) |")
    lines.append(f"| Huber Loss (δ={results['huber_delta']:.3f}) | {results['huber_loss']:.4f} |")
    lines.append("")

    lines.append("### Target Statistics\n")
    lines.append(f"- n = {results['n_samples']}")
    lines.append(f"- Mean: {results['y_true_mean']:.4f}, Std: {results['y_true_std']:.4f}")
    lines.append(
        f"- RMSE / Std = {results['rmse'] / (results['y_true_std'] + 1e-10):.3f} "
        "(normalized RMSE; < 0.5 is reasonable)\n"
    )

    lines.append("### Interpretation\n")
    r2 = results["r2"]
    if r2 > 0.9:
        lines.append(f"- R² = {r2:.3f}: Excellent fit.")
    elif r2 > 0.7:
        lines.append(f"- R² = {r2:.3f}: Good fit.")
    elif r2 > 0.4:
        lines.append(f"- R² = {r2:.3f}: Moderate fit. Consider feature engineering or model complexity.")
    elif r2 > 0:
        lines.append(f"- R² = {r2:.3f}: Weak fit. Model is only marginally better than predicting the mean.")
    else:
        lines.append(f"- R² = {r2:.3f}: Model performs worse than predicting the mean. Investigate data and pipeline.")

    lines.append(
        f"- RMSE = {results['rmse']:.4f}: Errors penalized quadratically. "
        "Compare to MAE to gauge outlier influence."
    )
    mae, rmse = results["mae"], results["rmse"]
    if rmse > 1.5 * mae:
        lines.append(
            "- RMSE >> MAE: Large outliers are present and inflating RMSE. "
            "Consider reporting MAE as primary metric or investigating outliers."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def evaluate_clustering(labels_pred, labels_true=None, X_features=None):
    results = {}
    labels_pred = np.asarray(labels_pred)
    n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    results["n_clusters_predicted"] = n_clusters

    if n_clusters < 2:
        results["silhouette"] = None
        results["davies_bouldin"] = None
        results["note"] = "Cannot compute internal metrics with fewer than 2 clusters."
    elif X_features is not None:
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        mask = labels_pred != -1
        if mask.sum() >= n_clusters * 2:
            results["silhouette"] = float(
                silhouette_score(X_features[mask], labels_pred[mask])
            )
            results["davies_bouldin"] = float(
                davies_bouldin_score(X_features[mask], labels_pred[mask])
            )
        else:
            results["silhouette"] = None
            results["davies_bouldin"] = None
            results["note"] = "Too few non-noise samples for internal metrics."
    else:
        results["silhouette"] = None
        results["davies_bouldin"] = None
        results["note"] = "Features not provided — internal metrics (silhouette, DB) skipped."

    if labels_true is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        labels_true = np.asarray(labels_true)
        results["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
        results["nmi"] = float(
            normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")
        )
        results["n_true_clusters"] = len(set(labels_true))
    else:
        results["ari"] = None
        results["nmi"] = None

    return results


def format_clustering_report(results: dict) -> str:
    lines = []
    lines.append("## Clustering Metrics\n")
    lines.append(f"- Number of clusters (predicted): {results['n_clusters_predicted']}\n")

    lines.append("### Internal Metrics (no ground truth required)\n")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")
    sil = results.get("silhouette")
    db = results.get("davies_bouldin")
    lines.append(
        f"| Silhouette Score | {sil:.4f if sil is not None else 'N/A'} | "
        "Range [-1, 1]; higher is better; > 0.5 = strong structure |"
    )
    lines.append(
        f"| Davies-Bouldin Index | {db:.4f if db is not None else 'N/A'} | "
        "Lower is better; < 1.0 = good separation |"
    )
    if "note" in results:
        lines.append(f"\n_Note: {results['note']}_\n")

    if results.get("ari") is not None:
        lines.append("\n### External Metrics (require ground truth)\n")
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|--------|-------|----------------|")
        lines.append(
            f"| ARI | {results['ari']:.4f} | "
            "Range [-1, 1]; 1 = perfect; 0 = random; robust to chance |"
        )
        lines.append(
            f"| NMI | {results['nmi']:.4f} | "
            "Range [0, 1]; 1 = perfect alignment with true labels |"
        )
        lines.append(f"| True clusters | {results['n_true_clusters']} |  |")

    lines.append("\n### Interpretation\n")
    if sil is not None:
        if sil > 0.7:
            lines.append(f"- Silhouette = {sil:.3f}: Strong cluster structure.")
        elif sil > 0.5:
            lines.append(f"- Silhouette = {sil:.3f}: Reasonable cluster structure.")
        elif sil > 0.25:
            lines.append(f"- Silhouette = {sil:.3f}: Weak structure. Consider different k or algorithm.")
        else:
            lines.append(f"- Silhouette = {sil:.3f}: Very weak structure. Clusters may not be meaningful.")

    if results.get("ari") is not None:
        ari = results["ari"]
        if ari > 0.8:
            lines.append(f"- ARI = {ari:.3f}: Clustering closely matches ground truth labels.")
        elif ari > 0.5:
            lines.append(f"- ARI = {ari:.3f}: Partial alignment with ground truth.")
        else:
            lines.append(
                f"- ARI = {ari:.3f}: Poor alignment with ground truth. "
                "Review cluster count and algorithm choice."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def evaluate_ranking(df_pred: pd.DataFrame, k: int):
    """
    Expects df_pred to have columns: query_id, item_id, score (predicted rank score),
    and relevance (ground-truth relevance grade).
    """
    required = {"query_id", "item_id", "score", "relevance"}
    missing = required - set(df_pred.columns)
    if missing:
        print(
            f"ERROR: Ranking evaluation requires columns: {required}. "
            f"Missing: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = {}
    ndcg_scores, ap_scores, rr_scores, pk_scores = [], [], [], []

    for query_id, group in df_pred.groupby("query_id"):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        relevance = group["relevance"].values

        # NDCG@k
        def dcg(rels, n):
            rels = rels[:n]
            return sum(r / np.log2(i + 2) for i, r in enumerate(rels))

        ideal = sorted(relevance, reverse=True)
        idcg = dcg(ideal, k)
        ndcg_scores.append(dcg(relevance, k) / (idcg + 1e-10) if idcg > 0 else 0.0)

        # Average Precision
        binary_rel = (relevance > 0).astype(int)
        hits, ap = 0, 0.0
        for i, r in enumerate(binary_rel, 1):
            if r:
                hits += 1
                ap += hits / i
        n_relevant = binary_rel.sum()
        ap_scores.append(ap / max(n_relevant, 1))

        # Reciprocal Rank
        rr = 0.0
        for i, r in enumerate(binary_rel, 1):
            if r:
                rr = 1.0 / i
                break
        rr_scores.append(rr)

        # Precision@k
        pk_scores.append(binary_rel[:k].sum() / k)

    results["ndcg_at_k"] = float(np.mean(ndcg_scores))
    results["map"] = float(np.mean(ap_scores))
    results["mrr"] = float(np.mean(rr_scores))
    results["precision_at_k"] = float(np.mean(pk_scores))
    results["k"] = k
    results["n_queries"] = len(ndcg_scores)
    return results


def format_ranking_report(results: dict) -> str:
    k = results["k"]
    lines = []
    lines.append("## Ranking / Retrieval Metrics\n")
    lines.append(f"- Evaluated at k = {k}, over {results['n_queries']} queries\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| NDCG@{k} | {results['ndcg_at_k']:.4f} |")
    lines.append(f"| MAP | {results['map']:.4f} |")
    lines.append(f"| MRR | {results['mrr']:.4f} |")
    lines.append(f"| Precision@{k} | {results['precision_at_k']:.4f} |")
    lines.append("")

    lines.append("### Interpretation\n")
    ndcg = results["ndcg_at_k"]
    if ndcg > 0.8:
        lines.append(f"- NDCG@{k} = {ndcg:.3f}: Excellent ranking quality.")
    elif ndcg > 0.6:
        lines.append(f"- NDCG@{k} = {ndcg:.3f}: Good ranking performance.")
    elif ndcg > 0.4:
        lines.append(f"- NDCG@{k} = {ndcg:.3f}: Moderate performance. Review feature quality and model.")
    else:
        lines.append(f"- NDCG@{k} = {ndcg:.3f}: Weak performance. Compare against BM25 or random baseline.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(task: str, metric_section: str, args) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = textwrap.dedent(f"""\
        # Model Evaluation Report

        **Task**: {task}
        **Generated**: {ts}
        **Predictions file**: `{args.predictions}`
        **Labels file**: `{args.labels}`

        ---

    """)
    footer = textwrap.dedent("""
        ---

        ## Checklist

        - [ ] Primary metric justified for task type
        - [ ] Confidence intervals / std across folds reported (if applicable)
        - [ ] Compared against trivial baseline (random, majority, mean predictor)
        - [ ] Imbalance ratio reviewed (classification)
        - [ ] No train/test contamination confirmed
        - [ ] All secondary metrics inspected, not just primary

        _Generated by SciDER `evaluate_model.py`_
    """)
    return header + metric_section + footer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SciDER ML evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--predictions", required=True,
                        help="CSV file with model predictions")
    parser.add_argument("--labels", required=True,
                        help="CSV file with ground-truth labels")
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression", "clustering", "ranking"],
        help="Task type",
    )
    parser.add_argument("--output", default="evaluation_report.md",
                        help="Output markdown report (default: evaluation_report.md)")
    parser.add_argument("--features",
                        help="CSV file with original features (clustering silhouette)")
    parser.add_argument("--prob_col", default=None,
                        help="Column name for predicted probabilities (classification)")
    parser.add_argument("--label_col", default="label",
                        help="Column name for true labels (default: label)")
    parser.add_argument("--pred_col", default="prediction",
                        help="Column name for predictions (default: prediction)")
    parser.add_argument("--k", type=int, default=10,
                        help="k for Precision@k / NDCG@k (ranking, default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()

    df_pred = load_csv(args.predictions, "predictions")
    df_labels = load_csv(args.labels, "labels")

    task = args.task
    print(f"[evaluate_model] Task: {task}")
    print(f"[evaluate_model] Predictions: {args.predictions} ({len(df_pred)} rows)")
    print(f"[evaluate_model] Labels:      {args.labels} ({len(df_labels)} rows)")

    # ---------------------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------------------
    if task == "classification":
        y_pred_raw = get_column(df_pred, args.pred_col, "predictions")
        y_true_raw = get_column(df_labels, args.label_col, "labels")
        check_lengths_match(y_pred_raw, y_true_raw, "predictions", "labels")

        y_pred = y_pred_raw.values
        y_true = y_true_raw.values

        y_prob = None
        if args.prob_col:
            prob_series = get_column(df_pred, args.prob_col, "predictions")
            try:
                y_prob = prob_series.values.astype(float)
                print(f"[evaluate_model] Using probability column: '{args.prob_col}'")
            except Exception as e:
                print(f"WARNING: Could not parse probability column '{args.prob_col}': {e}. "
                      "Skipping AUC metrics.", file=sys.stderr)

        results = evaluate_classification(y_true, y_pred, y_prob)
        metric_section = format_classification_report(results, y_true, y_pred)

    # ---------------------------------------------------------------------------
    # Regression
    # ---------------------------------------------------------------------------
    elif task == "regression":
        y_pred_raw = get_column(df_pred, args.pred_col, "predictions")
        y_true_raw = get_column(df_labels, args.label_col, "labels")
        check_lengths_match(y_pred_raw, y_true_raw, "predictions", "labels")

        y_pred = to_numeric_array(y_pred_raw, args.pred_col)
        y_true = to_numeric_array(y_true_raw, args.label_col)

        results = evaluate_regression(y_true, y_pred)
        metric_section = format_regression_report(results)

    # ---------------------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------------------
    elif task == "clustering":
        labels_pred_raw = get_column(df_pred, args.pred_col, "predictions")
        labels_pred = to_numeric_array(labels_pred_raw, args.pred_col).astype(int)

        labels_true = None
        if args.label_col in df_labels.columns:
            labels_true_raw = get_column(df_labels, args.label_col, "labels")
            check_lengths_match(labels_pred, labels_true_raw, "predictions", "labels")
            labels_true = to_numeric_array(labels_true_raw, args.label_col).astype(int)
            print("[evaluate_model] True labels found — computing external metrics (ARI, NMI).")
        else:
            print(
                f"[evaluate_model] Column '{args.label_col}' not found in labels file. "
                "External metrics (ARI, NMI) will be skipped.",
                file=sys.stderr,
            )

        X_features = None
        if args.features:
            df_feat = load_csv(args.features, "features")
            try:
                X_features = df_feat.select_dtypes(include=[np.number]).values
                if X_features.shape[0] != len(labels_pred):
                    print(
                        f"WARNING: features file has {X_features.shape[0]} rows but "
                        f"predictions have {len(labels_pred)}. Skipping silhouette.",
                        file=sys.stderr,
                    )
                    X_features = None
                else:
                    print(f"[evaluate_model] Features loaded: {X_features.shape}")
            except Exception as e:
                print(f"WARNING: Could not load features for silhouette: {e}", file=sys.stderr)

        results = evaluate_clustering(labels_pred, labels_true, X_features)
        metric_section = format_clustering_report(results)

    # ---------------------------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------------------------
    elif task == "ranking":
        # Merge predictions and labels on a common key; expect both to have
        # query_id, item_id columns; pred has 'score', labels has 'relevance'
        required_pred = {"query_id", "item_id", "score"}
        required_labels = {"query_id", "item_id", "relevance"}

        missing_pred = required_pred - set(df_pred.columns)
        missing_labels = required_labels - set(df_labels.columns)
        if missing_pred:
            print(
                f"ERROR: predictions CSV for ranking must have columns: {required_pred}. "
                f"Missing: {missing_pred}",
                file=sys.stderr,
            )
            sys.exit(1)
        if missing_labels:
            print(
                f"ERROR: labels CSV for ranking must have columns: {required_labels}. "
                f"Missing: {missing_labels}",
                file=sys.stderr,
            )
            sys.exit(1)

        df_merged = df_pred.merge(df_labels[["query_id", "item_id", "relevance"]],
                                  on=["query_id", "item_id"], how="inner")
        if df_merged.empty:
            print(
                "ERROR: No matching (query_id, item_id) pairs found after merging "
                "predictions and labels.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[evaluate_model] Merged {len(df_merged)} prediction-label pairs for ranking.")

        results = evaluate_ranking(df_merged, k=args.k)
        metric_section = format_ranking_report(results)

    else:
        print(f"ERROR: Unknown task '{task}'.", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Write report
    # ---------------------------------------------------------------------------
    report = build_report(task, metric_section, args)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[evaluate_model] Report written to: {args.output}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("METRIC SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        if key in ("confusion_matrix", "classes", "class_counts",
                   "imbalance_warning", "note"):
            continue
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4f}")
        elif isinstance(val, (int, str, bool)):
            print(f"  {key:30s}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
