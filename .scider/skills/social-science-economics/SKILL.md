---
name: social-science-economics
description: Social science and economics methods — survey analysis (Cronbach's alpha, factor analysis), econometrics (OLS diagnostics, robust SEs), panel data, behavioral experiments, administrative data linkage, and program evaluation (ITT vs LATE). Use when analyzing social, behavioral, or economic data.
allowed_agents: [data, experiment, ideation]
---

# Social Science and Economics

## Overview

This skill covers quantitative social science and economics methods: from survey instrument validation and regression diagnostics to causal program evaluation and administrative data linkage. For causal identification strategies (IV, DiD, RDD), also see the `causal-inference` skill.

## When to Use This Skill

- Analyzing survey data (Likert scales, factor analysis, reliability)
- Running econometric regressions with proper diagnostic tests
- Working with panel/longitudinal data
- Evaluating policy interventions or field experiments
- Merging administrative datasets with fuzzy matching

---

## 1. Survey Data Analysis

### Likert Scale Handling

```python
import pandas as pd
import numpy as np
import pingouin as pg

# Likert scales are ORDINAL — treat carefully
# 5-point scale: 1=Strongly Disagree, 5=Strongly Agree

# ❌ Wrong: treating Likert as continuous without justification
# ✅ Better: report median and IQR; use non-parametric tests

likert_data = pd.DataFrame({
    "Q1": [3, 4, 5, 2, 4, 3, 5, 4, 3, 2],
    "Q2": [4, 4, 5, 3, 5, 4, 4, 5, 3, 3],
    "Q3": [2, 3, 4, 2, 3, 3, 4, 3, 2, 2],
    "Q4": [3, 4, 4, 2, 4, 3, 5, 4, 3, 2],
})

# Summary statistics
print(likert_data.describe())
print("\nMedians:")
print(likert_data.median())

# Compare two groups: Mann-Whitney U (non-parametric)
from scipy.stats import mannwhitneyu
group_A = likert_data["Q1"][:5]
group_B = likert_data["Q1"][5:]
stat, p = mannwhitneyu(group_A, group_B, alternative="two-sided")
print(f"Mann-Whitney: U={stat}, p={p:.4f}")
```

### Internal Consistency: Cronbach's Alpha

```python
# Cronbach's alpha: measures how consistently items measure the same construct
# α ≥ 0.90: excellent, 0.80-0.89: good, 0.70-0.79: acceptable, < 0.70: questionable

alpha_result = pg.cronbach_alpha(data=likert_data)
print(f"Cronbach's α = {alpha_result[0]:.3f} (95% CI: {alpha_result[1]})")

# Item-total correlations: identify items that don't fit
for col in likert_data.columns:
    rest = likert_data.drop(columns=[col])
    r = likert_data[col].corr(rest.sum(axis=1))
    print(f"{col}: item-total r = {r:.3f}  {'⚠️ low' if r < 0.3 else '✅'}")
```

### Exploratory Factor Analysis (EFA)

```python
from factor_analyzer import FactorAnalyzer
import numpy as np

# First: check if factor analysis is appropriate
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

chi2, p = calculate_bartlett_sphericity(likert_data)
kmo_all, kmo_model = calculate_kmo(likert_data)
print(f"Bartlett's test: χ²={chi2:.2f}, p={p:.4f} (should be < 0.05)")
print(f"KMO measure: {kmo_model:.3f} (should be > 0.60)")

# Determine number of factors: scree plot + parallel analysis
fa_scree = FactorAnalyzer(n_factors=len(likert_data.columns), rotation=None)
fa_scree.fit(likert_data)
ev, v = fa_scree.get_eigenvalues()
# Retain factors with eigenvalue > 1 (Kaiser rule)
n_factors = sum(ev > 1)
print(f"Factors with eigenvalue > 1: {n_factors}")

# Fit with rotation
fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin")  # oblimin for correlated factors
fa.fit(likert_data)
loadings = pd.DataFrame(fa.loadings_, index=likert_data.columns,
                         columns=[f"F{i+1}" for i in range(n_factors)])
print("\nFactor loadings (|loading| > 0.4 = significant):")
print(loadings.round(3))
```

---

## 2. Econometrics: OLS Diagnostics

```python
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as diag
import pandas as pd
import numpy as np

# Generate example data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "y": 2 + 0.5*np.random.randn(n) + 3*np.random.randn(n),
    "x1": np.random.randn(n),
    "x2": np.random.randn(n),
})
df["y"] = 2 + 1.5*df["x1"] + 0.8*df["x2"] + np.random.randn(n)

# OLS regression
model = smf.ols("y ~ x1 + x2", data=df)
result = model.fit()

# ── Check OLS Assumptions ──────────────────────────────────────

# 1. Heteroscedasticity: White test (H₀: homoscedastic)
white_stat, white_p, white_f, white_fp = diag.het_white(result.resid, result.model.exog)
print(f"White test for heteroscedasticity: F={white_f:.2f}, p={white_fp:.4f}")
if white_fp < 0.05:
    print("  → Use heteroscedasticity-robust SEs: result.get_robustcov_results(cov_type='HC3')")

# Fit with robust SEs (always safer in applied econometrics)
result_robust = model.fit(cov_type="HC3")  # HC3: recommended for small samples
print(result_robust.summary())

# 2. Autocorrelation: Durbin-Watson test (2 = no autocorrelation)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(result.resid)
print(f"Durbin-Watson: {dw:.2f}  (2.0 = no autocorrelation; < 1.5 or > 2.5 = concern)")
if dw < 1.5 or dw > 2.5:
    print("  → Use Newey-West HAC SEs: result.get_robustcov_results(cov_type='HAC', maxlags=4)")

# 3. Multicollinearity: Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = result.model.exog
vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
for name, vif in zip(result.model.exog_names, vifs):
    print(f"VIF({name}): {vif:.2f}  {'⚠️ HIGH' if vif > 10 else '✅'}")
# VIF > 10 → serious multicollinearity; VIF 5-10 → moderate concern

# 4. Normality of residuals (less critical for large n, CLT applies)
from scipy import stats
_, normality_p = stats.shapiro(result.resid[:50])  # Shapiro on subset
print(f"Shapiro-Wilk normality of residuals: p={normality_p:.4f}")
```

---

## 3. Panel Data

```python
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
import pandas as pd
import numpy as np

# Multi-period panel data setup
np.random.seed(42)
n_units, n_periods = 50, 10
unit_ids = np.repeat(range(n_units), n_periods)
time_ids = np.tile(range(n_periods), n_units)

df_panel = pd.DataFrame({
    "unit": unit_ids,
    "time": time_ids,
    "y": np.random.randn(n_units * n_periods),
    "x": np.random.randn(n_units * n_periods),
    "treatment": (np.random.rand(n_units * n_periods) > 0.6).astype(int),
})

# Set multi-index (entity, time) required by linearmodels
df_indexed = df_panel.set_index(["unit", "time"])

# Fixed Effects (within estimator) — controls for all time-invariant unit heterogeneity
fe = PanelOLS.from_formula("y ~ x + treatment + EntityEffects", df_indexed)
fe_result = fe.fit(cov_type="clustered", cluster_entity=True)  # cluster SEs by entity
print(fe_result.summary)

# Random Effects (GLS estimator) — assumes RE uncorrelated with regressors
re = RandomEffects.from_formula("y ~ x + treatment", df_indexed)
re_result = re.fit()

# Two-way FE: entity + time fixed effects
twfe = PanelOLS.from_formula("y ~ x + treatment + EntityEffects + TimeEffects", df_indexed)
twfe_result = twfe.fit(cov_type="clustered", cluster_entity=True)

# Hausman test: FE vs RE (significant p → use FE)
# Compare FE and RE coefficients: large differences → endogeneity → use FE
```

---

## 4. Behavioral Experiments

### Common Issues and Fixes

**Attention checks**:
```python
# Include 1-2 obvious questions in your survey
# Example: "For quality control, please select 'Strongly Agree' for this item"
attention_check_col = "attention_1"
valid_respondents = df[df[attention_check_col] == 5]  # only keep those who passed
n_excluded = len(df) - len(valid_respondents)
print(f"Excluded {n_excluded} ({n_excluded/len(df)*100:.1f}%) respondents for failing attention check")
```

**Demand effects**: Participants guess the study hypothesis and respond accordingly.
- Mitigation: Cover story (don't reveal hypothesis), filler items, behavioral measure instead of self-report
- Check: compare across framing conditions where demand should differ

**Incentive compatibility** (economic experiments):
- Real stakes: use actual payoffs, not hypothetical
- Proper scoring rules for belief elicitation: Brier score, log score
- BDM mechanism for willingness-to-pay: randomly drawn threshold compared to stated WTP

---

## 5. Administrative Data and Record Linkage

```python
import recordlinkage
import pandas as pd

# Exact linkage (when shared unique ID exists — use pd.merge)
merged = pd.merge(df_admin, df_survey, on="national_id", how="inner")

# Probabilistic linkage (no shared ID — match on name, DOB, address)
# Using recordlinkage library
df_a = pd.DataFrame({"name": ["John Smith", "Jane Doe"], "dob": ["1980-01-15", "1975-06-20"]})
df_b = pd.DataFrame({"name": ["Jon Smith", "Jane Doe"],  "dob": ["1980-01-15", "1975-06-21"]})

indexer = recordlinkage.Index()
indexer.block("dob")  # block on DOB to reduce candidates
candidate_links = indexer.index(df_a, df_b)

# Compare fields
compare = recordlinkage.Compare()
compare.string("name", "name", method="jarowinkler", label="name_similarity")
compare.exact("dob", "dob", label="dob_match")
features = compare.compute(candidate_links, df_a, df_b)

# Score and threshold
features["score"] = features.sum(axis=1)
matches = features[features["score"] >= 1.5]  # adjust threshold based on data quality
print(f"Matched {len(matches)} records")

# Disclosure risk: k-anonymity
def check_k_anonymity(df, quasi_identifiers, k=5):
    """Check if any group is smaller than k (disclosure risk)."""
    groups = df.groupby(quasi_identifiers).size()
    risky = groups[groups < k]
    if len(risky) > 0:
        print(f"⚠️ {len(risky)} groups with < {k} records — consider suppression or generalization")
    return groups
```

---

## 6. Program Evaluation

```python
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# Intent-to-Treat (ITT): effect of assignment to treatment, regardless of compliance
# Estimates: population-average effect of offering the program
model_itt = smf.ols("outcome ~ assigned_treatment + controls", data=df).fit(cov_type="HC3")
att_itt = model_itt.params["assigned_treatment"]
print(f"ITT estimate: {att_itt:.3f}")

# Local Average Treatment Effect (LATE / CACE): effect on compliers
# Requires: assigned_treatment as instrument for actual_treatment
# LATE = ITT / compliance rate (= first-stage estimate)
compliance_rate = df.groupby("assigned_treatment")["actual_treatment"].mean().diff().iloc[-1]
late = att_itt / compliance_rate
print(f"LATE estimate: {late:.3f}  (compliance rate: {compliance_rate:.2%})")

# OR use 2SLS for LATE with controls
from linearmodels.iv import IV2SLS
result_late = IV2SLS.from_formula(
    "outcome ~ 1 + controls + [actual_treatment ~ assigned_treatment]",
    data=df
).fit(cov_type="robust")

# Attrition bias check
attrition = smf.ols("attrited ~ assigned_treatment + baseline_controls", data=df).fit()
print(f"Differential attrition p-value: {attrition.pvalues['assigned_treatment']:.4f}")
if attrition.pvalues["assigned_treatment"] < 0.05:
    print("⚠️ Differential attrition — consider bounds analysis or Lee bounds")

# Multiple testing correction (multiple outcomes)
from statsmodels.stats.multitest import multipletests

p_values = [0.03, 0.07, 0.02, 0.15, 0.04]
reject_bonferroni, p_bonferroni, _, _ = multipletests(p_values, method="bonferroni")
reject_bh, p_bh, _, _ = multipletests(p_values, method="fdr_bh")
for p, p_b, p_fdr, r_b, r_fdr in zip(p_values, p_bonferroni, p_bh, reject_bonferroni, reject_bh):
    print(f"p={p:.3f}  |  Bonferroni p={p_b:.3f} ({'+' if r_b else '-'})  |  BH p={p_fdr:.3f} ({'+' if r_fdr else '-'})")
```

**Reporting checklist for program evaluations**:
- [ ] Pre-specified primary outcome and statistical test (pre-registration)
- [ ] Balance table: compare baseline covariates across treatment/control
- [ ] ITT and (if applicable) LATE estimates with SEs and CIs
- [ ] Attrition rates and differential attrition test
- [ ] Multiple testing correction for secondary outcomes
- [ ] Subgroup analyses were pre-specified (not data-mined)
