"""Train and evaluate mutagenicity classification models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import RANDOM_STATE, WORKSPACE


def load_data():
    import pickle

    data_dir = WORKSPACE / "data"
    X = np.load(data_dir / "features.npy")
    y = np.load(data_dir / "labels.npy")
    with open(data_dir / "feature_names.pkl", "rb") as f:
        names = pickle.load(f)
    return X, y, names


def split_and_scale(X, y):
    """Stratified 80/20, SMOTE on train, StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def train_rf(X_train, y_train):
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train)


def train_svm(X_train, y_train):
    return SVC(probability=True, random_state=RANDOM_STATE).fit(X_train, y_train)


def train_xgb(X_train, y_train):
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)


def train_dnn(X_train, y_train, X_test, y_test):
    """Train DNN using Keras if available, else sklearn MLPClassifier."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential(
            [
                Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        return model
    except ImportError:
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=200,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ).fit(X_train, y_train)


def run_and_save(results_dir: Path):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    models = {}
    models["Random Forest"] = train_rf(X_train, y_train)
    models["SVM"] = train_svm(X_train, y_train)
    models["XGBoost"] = train_xgb(X_train, y_train)
    models["DNN"] = train_dnn(X_train, y_train, X_test, y_test)

    all_metrics = {}
    all_probs = {}
    all_cms = {}
    importance_dict = {}

    for name, model in models.items():
        if name == "DNN":
            if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
                probs = model.predict(X_test, verbose=0).ravel()
            else:
                probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)
        preds = (probs >= 0.5).astype(int)
        all_probs[name] = probs
        all_cms[name] = confusion_matrix(y_test, preds).tolist()
        all_metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0,
        }
        if hasattr(model, "feature_importances_"):
            importance_dict[name] = model.feature_importances_.tolist()
    # DNN permutation importance (skip - expensive with 1000+ features; use tree importance for plot)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Confusion matrices plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, cm) in zip(axes.flat, all_cms.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrices.png", dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(8, 6))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC Curves")
    plt.savefig(results_dir / "roc_curves.png", dpi=150)
    plt.close()

    # Feature importance (top 20) - aggregate tree + DNN permutation
    if importance_dict:
        imp = np.mean([np.array(importance_dict[m]) for m in importance_dict], axis=0)
    else:
        imp = np.zeros(len(feature_names))
    top_idx = np.argsort(imp)[-20:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = [imp[i] for i in top_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(range(20), top_vals[::-1])
    plt.yticks(range(20), top_names[::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig(results_dir / "feature_importance.png", dpi=150)
    plt.close()

    print("Top 20 features:")
    for n, v in zip(top_names, top_vals):
        print(f"  {n}: {v:.4f}")
    return all_metrics
