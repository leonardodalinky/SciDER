---
name: statistics-advanced
description: Advanced statistical methods — Bayesian inference with PyMC/MCMC, mixed-effects models, resampling (bootstrap, permutation), spatial statistics, and missing data handling. Use when standard frequentist tests are insufficient or you need full uncertainty quantification.
allowed_agents: [data, experiment, ideation]
---

# Advanced Statistics

## Overview

This skill extends the basic `statistical-analysis` skill with advanced methods for complex data structures: hierarchical/multilevel data, Bayesian inference, resampling-based inference, spatial data, and missing data. Use the basic skill first for standard hypothesis tests.

## When to Use This Skill

- Data has hierarchical or nested structure (students in schools, repeated measures per subject)
- You want a full posterior distribution, not just a point estimate and p-value
- Small sample sizes where asymptotic normality doesn't hold
- Data has spatial autocorrelation
- Substantial missing data that cannot be ignored

---

## 1. Bayesian Inference with PyMC

### When to Use Bayesian Analysis
- Small n (< 30 per group) where priors help regularize
- You have genuine prior knowledge about parameter ranges
- You need full uncertainty quantification (not just confidence intervals)
- Hierarchical / multilevel models are needed

```python
import pymc as pm
import arviz as az
import numpy as np

# Example: Bayesian t-test
np.random.seed(42)
control = np.random.normal(10, 2, 20)
treatment = np.random.normal(12, 2.5, 20)

with pm.Model() as model:
    # Priors (weakly informative)
    mu_ctrl = pm.Normal("mu_ctrl", mu=10, sigma=5)
    mu_treat = pm.Normal("mu_treat", mu=10, sigma=5)
    sigma_ctrl = pm.HalfNormal("sigma_ctrl", sigma=3)
    sigma_treat = pm.HalfNormal("sigma_treat", sigma=3)

    # Effect size (Cohen's d)
    diff = pm.Deterministic("difference", mu_treat - mu_ctrl)
    pooled_sigma = pm.Deterministic("pooled_sigma",
                                    pm.math.sqrt((sigma_ctrl**2 + sigma_treat**2) / 2))
    effect_size = pm.Deterministic("effect_size", diff / pooled_sigma)

    # Likelihood
    obs_ctrl = pm.Normal("obs_ctrl", mu=mu_ctrl, sigma=sigma_ctrl, observed=control)
    obs_treat = pm.Normal("obs_treat", mu=mu_treat, sigma=sigma_treat, observed=treatment)

    # Sample
    trace = pm.sample(2000, chains=4, target_accept=0.9, random_seed=42)

# Diagnostics
summary = az.summary(trace, var_names=["difference", "effect_size"])
print(summary)

# Key diagnostics to check:
# r_hat < 1.01 → chains converged
# ess_bulk > 400 → enough effective samples
az.plot_trace(trace, var_names=["difference"])

# Posterior predictive check
with model:
    ppc = pm.sample_posterior_predictive(trace)
```

### Credible Intervals vs. Confidence Intervals
- **Credible interval (Bayesian)**: "There is 95% probability the parameter lies in this range" — direct probability statement about the parameter
- **Confidence interval (frequentist)**: "If we repeated this experiment many times, 95% of such intervals would contain the true parameter" — NOT a probability statement about this specific interval

### Prior Predictive Checks
```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=42)
# Visualize: does the prior predictive cover realistic data ranges?
az.plot_ppc(prior_pred, group="prior")
```

---

## 2. Mixed-Effects (Multilevel) Models

### When to Use
- Repeated measures per subject (multiple observations per patient, student, etc.)
- Nested data (students in classes in schools)
- Longitudinal data with individual trajectories

```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Simulate: 30 subjects, 5 repeated measurements each
np.random.seed(42)
n_subjects = 30
n_obs = 5
subjects = np.repeat(range(n_subjects), n_obs)
time = np.tile(range(n_obs), n_subjects)
# Subject-level random intercepts
random_intercepts = np.random.normal(0, 2, n_subjects)[subjects]
y = 5 + 0.5 * time + random_intercepts + np.random.normal(0, 1, n_subjects * n_obs)

df = pd.DataFrame({"y": y, "time": time, "subject": subjects})

# Mixed-effects model: fixed effect of time + random intercept per subject
model = smf.mixedlm("y ~ time", df, groups=df["subject"])
result = model.fit()
print(result.summary())
# Fixed effects: population-level slope/intercept
# Random effects variance: between-subject variability

# Random slopes too (each subject has own time trajectory)
model_slopes = smf.mixedlm("y ~ time", df, groups=df["subject"],
                             re_formula="~time")
result_slopes = model_slopes.fit()
```

### Hausman Test (FE vs RE choice for panel data)
```python
# Use linearmodels for panel data Hausman test
from linearmodels.panel import PanelOLS, RandomEffects

df_panel = df.set_index(["subject", "time"])

fe = PanelOLS.from_formula("y ~ time + EntityEffects", df_panel).fit()
re = RandomEffects.from_formula("y ~ time", df_panel).fit()

# Hausman: if significant → use FE; if not → RE is more efficient
from linearmodels.panel.results import compare
# print(compare([fe, re]))
```

---

## 3. Resampling Methods

### Bootstrap Confidence Intervals
```python
from scipy.stats import bootstrap
import numpy as np

# Bootstrap CI for the median (or any statistic)
data = np.random.exponential(scale=2, size=50)

result = bootstrap(
    (data,),
    statistic=np.median,
    n_resamples=9999,
    confidence_level=0.95,
    random_state=42,
    method="BCa",        # bias-corrected and accelerated (best)
)
print(f"Median: {np.median(data):.3f}")
print(f"95% Bootstrap CI: ({result.confidence_interval.low:.3f}, {result.confidence_interval.high:.3f})")

# Manual bootstrap for more control
def bootstrap_ci(data, statistic, n_boot=9999, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boot_stats = [statistic(rng.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lo, hi

lo, hi = bootstrap_ci(data, np.mean)
print(f"Bootstrap mean CI: ({lo:.3f}, {hi:.3f})")
```

### Permutation Tests
```python
# Permutation test for two-group mean difference
def permutation_test(group1, group2, n_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    null_diffs = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        null_diffs.append(shuffled[:n1].mean() - shuffled[n1:].mean())

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

group_a = np.random.normal(5, 2, 30)
group_b = np.random.normal(6, 2, 30)
diff, p = permutation_test(group_a, group_b)
print(f"Observed difference: {diff:.3f}, p-value (permutation): {p:.4f}")
```

---

## 4. Dimensionality Reduction (Statistical Perspective)

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(200, 20)  # 200 samples, 20 features

# PCA
pca = PCA()
pca.fit(X)

# Scree plot to choose n_components
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumvar)+1), cumvar, "bo-")
plt.axhline(0.90, color="r", ls="--", label="90% variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")

# Loadings interpretation
n_comp = 3
loadings = pca.components_[:n_comp]  # shape: (n_comp, n_features)
# loadings[i, j] = contribution of feature j to component i

# UMAP vs t-SNE guidelines:
# - UMAP: faster, preserves global structure better, use for exploring clusters
# - t-SNE: better at separating local clusters, perplexity controls neighborhood size
# - Neither preserves distances — do NOT interpret absolute distances in the embedding
```

---

## 5. Missing Data

```python
import pandas as pd
import numpy as np

# Diagnose missingness mechanism
def analyze_missing(df):
    missing_pct = df.isnull().mean() * 100
    print("Missing data by column:")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))

    # MCAR test: if data are MCAR, missingness is unrelated to observed values
    # Simple check: compare means of complete vs incomplete cases on other variables
    for col in df.columns:
        if df[col].isnull().any():
            missing_mask = df[col].isnull()
            for other_col in df.select_dtypes(include=[np.number]).columns:
                if other_col != col:
                    m1 = df[other_col][missing_mask].mean()
                    m2 = df[other_col][~missing_mask].mean()
                    if abs(m1 - m2) / (df[other_col].std() + 1e-8) > 0.3:
                        print(f"  ⚠️ {col} missingness correlated with {other_col} (suggests MAR not MCAR)")

# Multiple imputation with sklearn
from sklearn.impute import IterativeImputer

# IterativeImputer models each feature with missing values as a function of others
# (equivalent to mice in R)
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(df.select_dtypes(include=[np.number]))

# Run analysis on M=5 imputed datasets and pool results (Rubin's rules)
M = 5
estimates = []
for i in range(M):
    imputer_m = IterativeImputer(max_iter=10, random_state=i)
    X_m = imputer_m.fit_transform(df.select_dtypes(include=[np.number]))
    # Run your model on X_m, get estimate + variance
    # estimates.append((estimate, variance))

# Pool: Q_bar = mean of estimates; W = mean within-variance; B = between-variance
# Total variance = W + (1 + 1/M)*B
```

**Decision guide**:
| Mechanism | What it means | Appropriate action |
|---|---|---|
| **MCAR** (missing completely at random) | Probability of missing is unrelated to any data | Complete case analysis is valid (but loses power) |
| **MAR** (missing at random) | Probability of missing depends on observed data only | Multiple imputation or FIML |
| **MNAR** (missing not at random) | Probability of missing depends on the missing value itself | Sensitivity analysis; selection models |

---

## 6. Spatial Statistics

```python
# Moran's I: global spatial autocorrelation
# Requires: esda, libpysal
from esda.moran import Moran
from libpysal.weights import Queen
import geopandas as gpd

# Load spatial data
# gdf = gpd.read_file("spatial_data.shp")
# w = Queen.from_dataframe(gdf)
# w.transform = "r"  # row-standardized
# mi = Moran(gdf["variable"], w)
# print(f"Moran's I = {mi.I:.3f}, p-value = {mi.p_sim:.4f}")
# If significant: data is spatially clustered (I > 0) or dispersed (I < 0)

# When spatial autocorrelation in residuals is detected:
# 1. Add spatial lag: y_i = ρ Wy_i + Xβ + ε (Spatial Lag Model)
# 2. Add spatial error: y = Xβ + u; u = λWu + ε (Spatial Error Model)
# Use spreg (PySAL) for these models
```
