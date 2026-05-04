---
name: causal-inference
description: Causal inference methods — DAG-based causal thinking, distinguishing observational from experimental data, IV, DiD, RDD, propensity score matching, and sensitivity analysis. Use when making causal claims from data.
allowed_agents: [ideation, experiment]
---

# Causal Inference

## Overview

Causal inference provides methods for estimating causal effects rather than merely correlational associations. Use this skill whenever you need to claim that X *causes* Y, not just that X and Y are correlated.

## When to Use This Skill

- When your research question is causal ("Does X increase Y?")
- When you have observational data and want to make causal claims
- When designing a study and choosing between experimental and observational approaches
- When a reviewer asks about confounding, endogeneity, or causal identification

## When NOT to Use This Skill

- When your goal is purely predictive (correlations are sufficient for prediction)
- When you have a true RCT with perfect compliance (standard t-test is enough)

---

## Hard Rules

1. **Never use causal language (causes, increases, reduces) without causal identification** — observational correlation alone is insufficient
2. **Draw the DAG before choosing a method** — the causal graph determines what you can and cannot control for
3. **Conditioning on a collider opens a backdoor path** — it can create spurious associations
4. **Always run a placebo test** — if your method claims an effect where no effect should exist, the method is flawed

---

## 1. Causal DAGs (Directed Acyclic Graphs)

Drawing the causal graph is the first step before any analysis.

**Node types**:
- **Treatment (T)**: the variable you want to study
- **Outcome (Y)**: the variable you want to affect
- **Confounder (C)**: common cause of T and Y — must control for
- **Mediator (M)**: on the causal path T → M → Y — do NOT control for if you want total effect
- **Collider (K)**: caused by both T and Y (or their descendants) — do NOT condition on

```python
# Simple DAG visualization with networkx
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from([
    ("Education", "Income"),      # T → Y (causal path)
    ("Family_SES", "Education"),  # C → T (confounder)
    ("Family_SES", "Income"),     # C → Y (confounder)
])
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=2000, arrows=True, arrowsize=20)
```

**Backdoor criterion**: A set of variables Z satisfies the backdoor criterion if:
1. Z blocks all backdoor paths from T to Y
2. Z does not contain any descendant of T

If you can find such Z, controlling for Z gives you the causal effect.

---

## 2. Experimental Design (Gold Standard)

**Randomized Controlled Trial (RCT)**:
- Randomly assign treatment → eliminates all confounding by design
- Check balance at baseline: run t-tests on pre-treatment covariates

```python
import pandas as pd
from scipy import stats

# Check covariate balance between treatment groups
for covariate in ["age", "income", "education"]:
    t_stat, p_val = stats.ttest_ind(
        df[df.treated == 1][covariate],
        df[df.treated == 0][covariate]
    )
    print(f"{covariate}: p={p_val:.3f} {'⚠️ IMBALANCED' if p_val < 0.05 else '✅'}")
```

**Encouragement design**: Randomize access/encouragement (instrument), not actual treatment → use IV to estimate LATE.

---

## 3. Instrumental Variables (IV)

**When to use**: Treatment is endogenous (correlated with unobservables), but you have a valid instrument.

**Validity conditions for instrument Z**:
1. **Relevance**: Z is correlated with T (test: first-stage F-statistic > 10)
2. **Exclusion restriction**: Z affects Y ONLY through T (must be argued theoretically)
3. **Independence**: Z is as-good-as-random (no confounders of Z-Y)

```python
from linearmodels.iv import IV2SLS

# Two-Stage Least Squares
res = IV2SLS.from_formula(
    "outcome ~ 1 + controls + [treatment ~ instrument]",
    data=df
).fit(cov_type="robust")
print(res.summary)

# Check first-stage relevance (F-stat > 10 rule of thumb)
from linearmodels.iv.model import _OLS
first_stage = _OLS.from_formula("treatment ~ 1 + controls + instrument", data=df).fit()
print(f"First-stage F-stat: {first_stage.f_statistic.stat:.2f}")
```

---

## 4. Difference-in-Differences (DiD)

**When to use**: Panel data with treatment affecting some units at some point in time.

**Key assumption (parallel trends)**: In the absence of treatment, treated and control groups would have followed parallel trends. Test by checking pre-treatment trends.

```python
import statsmodels.formula.api as smf

# Two-period DiD
# treated: dummy for treatment group; post: dummy for post-treatment period
result = smf.ols(
    "outcome ~ treated + post + treated:post + controls",
    data=df
).fit(cov_type="HC3")
did_estimate = result.params["treated:post"]
print(f"DiD estimate: {did_estimate:.3f}")
print(result.summary())

# Parallel trends test: run DiD only on pre-treatment data
# The interaction should be near zero
pre_data = df[df.post == 0]
# ... (use period dummies instead of post)
```

**Two-way fixed effects** (multiple periods):
```python
from linearmodels.panel import PanelOLS
result = PanelOLS.from_formula(
    "outcome ~ treatment + EntityEffects + TimeEffects",
    data=df.set_index(["unit_id", "time_id"])
).fit(cov_type="clustered", cluster_entity=True)
```

---

## 5. Regression Discontinuity Design (RDD)

**When to use**: Treatment is assigned based on crossing a threshold of a continuous "running variable".

```python
import numpy as np
import statsmodels.formula.api as smf

cutoff = 50  # assignment threshold
df["above_cutoff"] = (df.running_var >= cutoff).astype(int)
df["centered"] = df.running_var - cutoff

# Sharp RDD: linear fit on each side of cutoff
result = smf.ols(
    "outcome ~ centered * above_cutoff",
    data=df[np.abs(df.centered) <= 10]  # bandwidth = 10
).fit(cov_type="HC3")
rdd_estimate = result.params["above_cutoff"]
print(f"RDD estimate at cutoff: {rdd_estimate:.3f}")

# Validity checks:
# 1. Continuity of density at cutoff (McCrary test — rddensity package)
# 2. Continuity of pre-determined covariates at cutoff (run RDD with covariate as outcome)
```

---

## 6. Propensity Score Matching (PSM)

**When to use**: Observational data where selection into treatment depends on observed covariates.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Step 1: Estimate propensity score
X = df[["age", "income", "education", "prior_outcome"]]
y = df["treated"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_scaled, y)
df["propensity_score"] = ps_model.predict_proba(X_scaled)[:, 1]

# Step 2: Check overlap (common support)
import matplotlib.pyplot as plt
for t in [0, 1]:
    plt.hist(df[df.treated == t]["propensity_score"],
             alpha=0.5, label=f"Treated={t}", bins=30)
plt.xlabel("Propensity Score")
plt.legend()
plt.title("Common Support Check")

# Step 3: 1:1 nearest-neighbor matching
from scipy.spatial.distance import cdist
treated = df[df.treated == 1]
control = df[df.treated == 0]
distances = cdist(
    treated[["propensity_score"]].values,
    control[["propensity_score"]].values
)
matched_control_idx = distances.argmin(axis=1)
matched = pd.concat([
    treated.reset_index(drop=True),
    control.iloc[matched_control_idx].reset_index(drop=True).add_suffix("_ctrl")
], axis=1)

# Step 4: Check covariate balance after matching
att_estimate = (matched["outcome"] - matched["outcome_ctrl"]).mean()
print(f"ATT estimate: {att_estimate:.3f}")
```

For robust matching, use `causalml` or `dowhy`:
```python
# With dowhy (recommended for production analysis)
import dowhy
model = dowhy.CausalModel(
    data=df,
    treatment="treated",
    outcome="outcome",
    common_causes=["age", "income", "education"],
)
identified = model.identify_effect()
estimate = model.estimate_effect(identified, method_name="backdoor.propensity_score_matching")
refute = model.refute_estimate(identified, estimate, method_name="placebo_treatment_refuter")
```

---

## 7. Sensitivity Analysis

After estimating a causal effect, test its robustness:

**Placebo tests**:
- Replace treatment with random assignment → estimate should be ~0
- Apply treatment to pre-treatment period → estimate should be ~0
- Apply to a group that shouldn't be affected → estimate should be ~0

**Rosenbaum bounds**: How strong would unmeasured confounding need to be to explain away the effect?

**E-value**: Minimum strength of association an unmeasured confounder would need with both treatment and outcome to fully explain the observed association.

```python
# E-value formula for relative risks (VanderWeele & Ding, 2017)
def e_value(rr):
    """E-value for a relative risk estimate."""
    return rr + np.sqrt(rr * (rr - 1))

rr = 2.5  # observed relative risk
print(f"E-value: {e_value(rr):.2f}")
# Interpretation: an unmeasured confounder would need to have a risk ratio ≥ E-value
# with both treatment and outcome to explain away the result
```

---

## Libraries Quick Reference

| Library | Best for |
|---------|----------|
| `dowhy` | End-to-end causal analysis with refutation tests |
| `econml` | Heterogeneous treatment effects (CATE), double ML |
| `causalml` | Uplift modeling, propensity-based methods |
| `linearmodels` | IV (2SLS, LIML), panel data (FE, RE, GMM) |
| `statsmodels` | DiD, OLS with robust SEs, basic regression |
