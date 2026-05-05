---
name: ml-model-evaluation
description: Model evaluation framework — cross-validation strategies, metric selection by task type, baseline requirements, ablation design, and train/val/test discipline. Use for all ML experiments before reporting results.
allowed_agents: [experiment, experiment_coding, native_coding]
preload_for: [experiment]
---

# ML Model Evaluation

## Overview

Rigorous model evaluation is the foundation of credible ML research. This skill covers the complete evaluation pipeline: how to split data correctly, which metrics to report for which tasks, what baselines are required, how to structure ablation studies, and how to verify that results are statistically meaningful. Apply this skill before writing any results section or claiming model performance.

## When to Use This Skill

Use this skill when:
- Selecting cross-validation strategy for a new experiment
- Choosing which metrics to compute and report for a task
- Setting up baseline comparisons before training models
- Designing ablation studies to identify component contributions
- Assessing whether train/val/test splits are clean and leak-free
- Tuning hyperparameters with nested cross-validation
- Reporting confidence intervals and checking statistical significance
- Completing the results section of a paper or technical report

---

## Hard Rules (follow for every experiment)

1. NEVER report accuracy alone for imbalanced datasets — always pair with F1, AUC-PR, or MCC
2. ALWAYS compare against at least one trivial baseline before claiming model performance
3. ALWAYS use cross-validation (or a proper held-out test set) — never evaluate on training data
4. ALWAYS report confidence intervals or standard deviation across folds
5. For ablation studies: change ONE component at a time and report the delta

---

## Cross-Validation Taxonomy

Choose the CV strategy based on your data structure. Wrong CV choice is one of the most common sources of inflated reported performance.

### Standard K-Fold

Use when: data is IID, classes are balanced, no group structure.

```python
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

results = cross_validate(
    model, X, y,
    cv=kf,
    scoring=['accuracy', 'f1_macro', 'roc_auc'],
    return_train_score=True
)

print(f"Val accuracy:  {results['test_accuracy'].mean():.3f} ± {results['test_accuracy'].std():.3f}")
print(f"Train accuracy: {results['train_accuracy'].mean():.3f} ± {results['train_accuracy'].std():.3f}")
# Large train-val gap → overfitting
```

### Stratified K-Fold

Use when: classification task with class imbalance. Preserves class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Pass y to cross_validate so stratification is applied
results = cross_validate(model, X, y, cv=skf, scoring='f1_macro')
```

### Group K-Fold

Use when: samples are not independent — they come from groups (e.g., multiple measurements per patient, sentences from the same document). Leaking groups across folds gives optimistically biased estimates.

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
# groups: array of group labels, same length as X
results = cross_validate(model, X, y, cv=gkf.split(X, y, groups=groups),
                         scoring='f1_macro')
```

### Time-Series Split

Use when: data has temporal ordering. Future data must never appear in training folds.

**Expanding window** (default `TimeSeriesSplit`) — training set grows with each fold:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)
    # evaluate on X_val
```

**Sliding window** — fixed-size training window (manual):

```python
window_size = 1000
step_size = 200

fold_results = []
for start in range(0, len(X) - window_size - step_size, step_size):
    train_idx = slice(start, start + window_size)
    val_idx = slice(start + window_size, start + window_size + step_size)
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
    fold_results.append(score)

print(f"Sliding-window val: {np.mean(fold_results):.3f} ± {np.std(fold_results):.3f}")
```

### Nested Cross-Validation (Hyperparameter Tuning + Evaluation)

Use nested CV when you tune hyperparameters AND want an unbiased generalization estimate. The outer loop estimates generalization; the inner loop selects hyperparameters.

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
clf = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='f1_macro')

# Outer loop gives unbiased estimate
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='f1_macro')
print(f"Nested CV F1: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

**Rule**: never use the same CV split for both tuning and evaluation — this is a common source of optimistic bias.

---

## Metric Selection Guide by Task Type

### Classification

| Metric | When to Use | When NOT to Use |
|--------|-------------|-----------------|
| Accuracy | Balanced classes only | Imbalanced datasets |
| Balanced Accuracy | Imbalanced, multi-class | — |
| Precision / Recall | When false positives or false negatives have asymmetric costs | Don't use alone |
| F1 (macro) | Multi-class, treat all classes equally | When class sizes matter |
| F1 (weighted) | Multi-class, weight by support | Obscures minority class performance |
| AUC-ROC | Binary, probabilistic output | Severely imbalanced (prefer AUC-PR) |
| AUC-PR | Imbalanced binary problems | Not useful for balanced data |
| MCC | Best single metric for imbalanced binary | Less intuitive to report |

```python
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import numpy as np

def classification_metrics(y_true, y_pred, y_prob=None, pos_label=1):
    """Compute full classification metric suite."""
    results = {}

    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    results['precision_per_class'] = p
    results['recall_per_class'] = r
    results['f1_per_class'] = f1

    for avg in ['macro', 'micro', 'weighted']:
        _, _, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        results[f'f1_{avg}'] = f

    results['mcc'] = matthews_corrcoef(y_true, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    if y_prob is not None:
        if y_prob.ndim == 2:
            # Multi-class: use OvR
            results['auc_roc'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
            results['auc_pr'] = average_precision_score(
                y_true, y_prob, average='macro'
            )
        else:
            results['auc_roc'] = roc_auc_score(y_true, y_prob)
            results['auc_pr'] = average_precision_score(y_true, y_prob)

    # Imbalance check
    class_counts = np.bincount(y_true)
    imbalance_ratio = class_counts.max() / class_counts.min()
    results['imbalance_ratio'] = imbalance_ratio
    if imbalance_ratio > 3:
        results['imbalance_warning'] = (
            f"Class imbalance detected (ratio {imbalance_ratio:.1f}:1). "
            "Do NOT rely on accuracy. Use F1-macro, AUC-PR, or MCC."
        )

    return results
```

### Regression

| Metric | Formula | When to Use |
|--------|---------|-------------|
| RMSE | √(mean((ŷ-y)²)) | Default; penalizes large errors; interpretable in target units |
| MAE | mean(\|ŷ-y\|) | When outliers are expected and should not dominate |
| R² | 1 - SS_res/SS_tot | Explained variance; use as secondary metric |
| MAPE | mean(\|ŷ-y\|/\|y\|)×100 | When relative error matters; AVOID when y≈0 |
| Huber Loss | Hybrid MAE/MSE | Robust to outliers; use δ≈1.35×MAD |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def regression_metrics(y_true, y_pred, huber_delta=1.35):
    """Compute full regression metric suite."""
    results = {}

    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['r2'] = r2_score(y_true, y_pred)

    # MAPE — guard against zero targets
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        results['mape'] = np.mean(
            np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
        ) * 100
    else:
        results['mape'] = float('nan')

    # Huber loss
    residuals = y_true - y_pred
    huber = np.where(
        np.abs(residuals) <= huber_delta,
        0.5 * residuals**2,
        huber_delta * (np.abs(residuals) - 0.5 * huber_delta)
    )
    results['huber_loss'] = np.mean(huber)

    return results
```

### Clustering

Distinguish **external** metrics (require true labels) from **internal** metrics (only use features):

```python
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

def clustering_metrics(X, labels_pred, labels_true=None):
    results = {}

    # Internal metrics (no ground truth needed)
    if len(set(labels_pred)) > 1:  # silhouette requires ≥2 clusters
        results['silhouette'] = silhouette_score(X, labels_pred)
        results['davies_bouldin'] = davies_bouldin_score(X, labels_pred)
        # Lower DB → better; Higher silhouette → better

    # External metrics (require ground truth)
    if labels_true is not None:
        results['ari'] = adjusted_rand_score(labels_true, labels_pred)
        results['nmi'] = normalized_mutual_info_score(
            labels_true, labels_pred, average_method='arithmetic'
        )
        # ARI=1, NMI=1 → perfect; ARI=0 → random

    return results
```

### Ranking / Retrieval

```python
import numpy as np

def ndcg_at_k(relevance_scores, k):
    """relevance_scores: list of relevance for ranked items (desc order)."""
    def dcg(scores):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores[:k]))
    ideal = sorted(relevance_scores, reverse=True)
    return dcg(relevance_scores) / (dcg(ideal) + 1e-10)

def mean_average_precision(relevant_sets, ranked_lists):
    """AP averaged over queries."""
    aps = []
    for relevant, ranked in zip(relevant_sets, ranked_lists):
        hits, ap = 0, 0.0
        for i, item in enumerate(ranked, 1):
            if item in relevant:
                hits += 1
                ap += hits / i
        aps.append(ap / max(len(relevant), 1))
    return np.mean(aps)

def mean_reciprocal_rank(relevant_sets, ranked_lists):
    rrs = []
    for relevant, ranked in zip(relevant_sets, ranked_lists):
        for i, item in enumerate(ranked, 1):
            if item in relevant:
                rrs.append(1 / i)
                break
        else:
            rrs.append(0.0)
    return np.mean(rrs)
```

### Generation / Seq2Seq

```python
# Install: pip install evaluate bert_score
import evaluate

# BLEU
bleu = evaluate.load("bleu")
result = bleu.compute(predictions=["the cat sat"], references=[["the cat sat on the mat"]])
print(f"BLEU: {result['bleu']:.4f}")

# ROUGE
rouge = evaluate.load("rouge")
result = rouge.compute(predictions=["the cat sat"], references=["the cat sat on the mat"])
# rouge1, rouge2, rougeL keys

# BERTScore (semantic similarity — preferred over BLEU for meaning)
bertscore = evaluate.load("bertscore")
result = bertscore.compute(
    predictions=["the cat sat"],
    references=["the cat sat on the mat"],
    lang="en"
)
print(f"BERTScore F1: {result['f1'][0]:.4f}")
```

---

## Baseline Requirements

Every paper must include baselines. Report your model's metrics alongside all of the following that apply:

| Baseline | When Required | Code |
|----------|--------------|------|
| Random | Always | `DummyClassifier(strategy='uniform')` |
| Majority class | Classification | `DummyClassifier(strategy='most_frequent')` |
| Mean predictor | Regression | `DummyRegressor(strategy='mean')` |
| Linear model | Always | `LogisticRegression()` / `Ridge()` |
| Published SOTA | If benchmarking | Cite paper + reproduce or use reported numbers |

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

baselines = {
    'random':         DummyClassifier(strategy='uniform', random_state=42),
    'majority':       DummyClassifier(strategy='most_frequent'),
    'logistic_reg':   LogisticRegression(max_iter=1000, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in baselines.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"{name:20s}: {scores.mean():.3f} ± {scores.std():.3f}")

# Your model
scores = cross_val_score(your_model, X_train, y_train, cv=cv, scoring='f1_macro')
print(f"{'your_model':20s}: {scores.mean():.3f} ± {scores.std():.3f}")
```

A model that does not beat the majority-class or linear baseline is not publishable — investigate before proceeding.

---

## Train / Val / Test Discipline

### Split Ratios

| Dataset Size | Train | Val | Test |
|---|---|---|---|
| < 1K samples | 60% | 20% | 20% (or use CV) |
| 1K–100K | 70% | 15% | 15% |
| > 100K | 80–90% | 5–10% | 5–10% |

```python
from sklearn.model_selection import train_test_split

# Always stratify for classification
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    # 0.176 ≈ 15% of original dataset
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### Data Leakage Prevention

Leakage causes inflated metrics that vanish at deployment. Check each of the following:

1. **Fit preprocessing only on training data**:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# CORRECT — scaler fitted only on train
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)
pipe.score(X_val, y_val)

# WRONG — leaks test distribution into scaler
scaler = StandardScaler().fit(X)  # DO NOT DO THIS
```

2. **Never use test set for any model selection decision** — not for threshold tuning, feature selection, or early stopping.
3. **Group leakage**: ensure the same entity (patient, user, document) does not appear in both train and test.
4. **Temporal leakage**: ensure all test timestamps are strictly after all training timestamps.

### When to Touch the Test Set

Touch the test set **exactly once**: after all hyperparameter decisions are finalized using the validation set or CV. If you look at test results and change anything, your test set has become a validation set — find or hold out new data for a true final evaluation.

---

## Ablation Study Design Protocol

Ablations identify which components actually matter. A weak ablation section is a common reviewer complaint.

**Protocol**:
1. Define your full system: list all components (e.g., data augmentation, pre-training, attention mechanism, loss function, post-processing).
2. Train the full system first — this is your ceiling.
3. Remove **one component at a time** and retrain from scratch.
4. Report the metric delta (full − ablated) for each component.
5. Run the same CV setup for all ablations so comparisons are valid.
6. If components interact, run a 2×2 factorial for the interacting pair.

```python
import pandas as pd

components = ['data_augmentation', 'pretrained_encoder', 'attention', 'auxiliary_loss']
results = {}

# Full system
results['full_system'] = run_experiment(use_all_components=True)

# Ablate one at a time
for component in components:
    config = {c: True for c in components}
    config[component] = False
    results[f'w/o_{component}'] = run_experiment(**config)

# Report deltas
df = pd.DataFrame(results, index=['f1_macro', 'std']).T
df['delta_from_full'] = df['f1_macro'] - df.loc['full_system', 'f1_macro']
print(df.sort_values('delta_from_full'))
```

---

## Overfitting Signals

### Train / Val Gap Analysis

```python
import matplotlib.pyplot as plt

train_scores, val_scores = [], []
for train_idx, val_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    train_scores.append(metric(y[train_idx], model.predict(X[train_idx])))
    val_scores.append(metric(y[val_idx],   model.predict(X[val_idx])))

gap = np.mean(train_scores) - np.mean(val_scores)
print(f"Train: {np.mean(train_scores):.3f}, Val: {np.mean(val_scores):.3f}, Gap: {gap:.3f}")
if gap > 0.05:
    print("WARNING: Large train/val gap — likely overfitting.")
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.fill_between(train_sizes,
                 train_scores.mean(1) - train_scores.std(1),
                 train_scores.mean(1) + train_scores.std(1), alpha=0.2)
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.fill_between(train_sizes,
                 val_scores.mean(1) - val_scores.std(1),
                 val_scores.mean(1) + val_scores.std(1), alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('F1 (macro)')
plt.legend()
plt.title('Learning Curve')
plt.tight_layout()
plt.savefig('learning_curve.pdf', dpi=300, bbox_inches='tight')
```

### Regularization Checklist

When overfitting is detected, try in order:
- [ ] Reduce model complexity (fewer layers, smaller max_depth)
- [ ] Add L2 regularization (increase `alpha` / decrease `C`)
- [ ] Add dropout (neural nets)
- [ ] Increase training data (check learning curve plateau)
- [ ] Add data augmentation
- [ ] Use early stopping

---

## Hyperparameter Search

| Method | When to Use | Library |
|--------|-------------|---------|
| Grid Search | ≤ 3 params, small search space | `GridSearchCV` |
| Random Search | > 3 params, limited budget | `RandomizedSearchCV` |
| Bayesian (TPE) | Large budget, expensive model | `optuna` |

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial):
    params = {
        'n_estimators':   trial.suggest_int('n_estimators', 50, 500),
        'max_depth':      trial.suggest_int('max_depth', 2, 8),
        'learning_rate':  trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample':      trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

print("Best params:", study.best_params)
print("Best val F1:", study.best_value)
```

---

## Statistical Significance

### Bootstrap Confidence Intervals for Metrics

```python
import numpy as np
from sklearn.utils import resample

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap CI for any scalar metric."""
    scores = []
    for _ in range(n_bootstrap):
        idx = resample(np.arange(len(y_true)), replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return np.mean(scores), lower, upper

from sklearn.metrics import f1_score
mean_f1, lo, hi = bootstrap_metric(
    y_test, y_pred,
    lambda yt, yp: f1_score(yt, yp, average='macro')
)
print(f"F1 = {mean_f1:.3f} [{lo:.3f}, {hi:.3f}]")
```

### Paired t-Test for Comparing Two Models

Use when you have matched predictions from two models on the same CV folds:

```python
from scipy import stats

model_a_scores = np.array([0.81, 0.83, 0.79, 0.82, 0.80])  # per-fold scores
model_b_scores = np.array([0.76, 0.78, 0.74, 0.77, 0.75])

t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
diff = model_a_scores - model_b_scores
mean_diff = diff.mean()
ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))

print(f"Mean difference: {mean_diff:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"t({len(diff)-1}) = {t_stat:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    print("Models differ significantly (p < 0.05)")
else:
    print("No significant difference detected — difference may be noise")
```

For multiple model comparisons, use Bonferroni correction or Wilcoxon signed-rank test as a nonparametric alternative.

---

## Reporting Checklist

A complete results section must include all of the following:

### Metrics
- [ ] Primary metric selected and justified for the task type
- [ ] Secondary metrics reported (do not report just one number)
- [ ] Imbalance ratio stated if dataset is imbalanced
- [ ] Confidence intervals or std across folds reported

### Baselines
- [ ] Trivial baseline (random, majority, mean) reported
- [ ] Linear baseline reported
- [ ] Prior work baseline cited and reproduced on the same split

### Experimental Setup
- [ ] CV strategy stated (K-fold, stratified, grouped, time-series)
- [ ] Number of folds / train-val-test proportions stated
- [ ] Stratification strategy described
- [ ] How hyperparameters were selected (which method, which folds used)

### Ablations
- [ ] Each ablation removes exactly one component
- [ ] Delta contribution reported per component
- [ ] All ablations run under the same CV setup

### Reproducibility
- [ ] Random seeds reported
- [ ] Software versions stated
- [ ] Whether test set was used once only

### Statistical Validity
- [ ] Significance test reported when comparing models
- [ ] Bootstrap or fold-level CI reported for primary metric
- [ ] No train/val/test contamination (explicitly state split procedure)

---

## Resources

### Scripts Directory

- **evaluate_model.py**: Command-line evaluation script for classification, regression, clustering, and ranking. Run as:
  `python evaluate_model.py --predictions pred.csv --labels labels.csv --task classification --output report.md`

---

## Common Pitfalls

1. **Reporting a single number**: Always pair with CI or fold std.
2. **Accuracy on imbalanced data**: A 95%-majority-class dataset gives 95% accuracy with a trivial classifier.
3. **Evaluating on training data**: Always use held-out fold or test set.
4. **Tuning on test data**: Any decision made after looking at test results makes test = validation.
5. **Ignoring group structure**: Patient-level splits look much worse than sample-level splits — that is correct, not a bug.
6. **Running ablations with different random seeds**: Seed variation can exceed component contribution — control seeds.
7. **Claiming significance without a significance test**: A 0.2 F1 improvement in one fold is not "significant".
