---
name: biology-ecology
description: Experimental and ecological biology — experimental design with controls/replicates, biology-specific statistical tests, diversity indices, cell biology assays (IC50, ELISA, flow cytometry), imaging analysis, and survival analysis. Use when working with biological experimental data.
allowed_agents: [data, experiment]
---

# Biology and Ecology

## Overview

This skill covers experimental biology and ecology workflows: from designing valid experiments with appropriate controls and replicates, to analyzing assay data, microscopy images, ecological surveys, and survival data. For genomics and sequencing data, also see the `bioinformatics-analysis` skill.

## When to Use This Skill

- Analyzing in vitro or in vivo experimental data (cell lines, animal studies)
- Processing ecology survey data (species counts, diversity indices)
- Fitting dose-response curves (IC50, EC50)
- Analyzing flow cytometry, ELISA, or imaging data
- Survival analysis (Kaplan-Meier, Cox regression)
- Choosing the right statistical test for biological data

---

## 1. Experimental Design for Biology

### Controls
| Control type | Purpose | Example |
|---|---|---|
| **Negative control** | Establish background / baseline | Vehicle (DMSO, PBS), untreated cells |
| **Positive control** | Confirm assay works | Known active compound, reference drug |
| **Vehicle control** | Separate drug effect from solvent effect | DMSO at matched concentration |
| **Isotype control** (flow) | Estimate non-specific antibody binding | Matched isotype antibody |

**Rule**: Always include both negative AND positive controls in every experiment run.

### Replicates
- **Biological replicates**: independent samples (different cell passages, different animals, different patients) → what you use for statistics
- **Technical replicates**: same sample measured multiple times → estimate measurement variability only, do NOT treat as independent samples

**Minimum n**: Aim for n ≥ 3 biological replicates per condition. For animal studies, use power analysis (see `experiment-design` skill) targeting power ≥ 0.80.

### Randomization and Blinding
```python
import random

# Randomize treatment assignment
wells = list(range(96))
random.seed(42)
random.shuffle(wells)
treatment_wells = wells[:24]  # randomly assigned
control_wells = wells[24:48]

# Document blinding: encode sample IDs before measurement
sample_map = {f"Sample_{i}": f"Unknown_{i:03d}" for i in range(1, 25)}
# Decode only after all measurements are complete
```

---

## 2. Biology-Specific Statistical Tests

```python
import scipy.stats as stats
import pingouin as pg
import numpy as np
import pandas as pd

# Two groups, continuous, normal distribution → Welch's t-test (unequal variance assumed)
control = np.array([2.1, 2.4, 2.2, 2.5, 2.3])
treatment = np.array([3.5, 3.8, 3.2, 3.9, 3.6])
t_stat, p_val = stats.ttest_ind(control, treatment, equal_var=False)
print(f"Welch's t-test: t={t_stat:.2f}, p={p_val:.4f}")

# Effect size (Cohen's d)
result = pg.ttest(treatment, control)
print(result[["T", "p-val", "cohen-d", "power"]])

# Non-parametric alternative: Mann-Whitney U
u_stat, p_mw = stats.mannwhitneyu(control, treatment, alternative="two-sided")
print(f"Mann-Whitney U: p={p_mw:.4f}")

# Multiple groups → one-way ANOVA + Tukey HSD post-hoc
groups = {"Control": [2.1, 2.4, 2.2], "Drug_A": [3.5, 3.8, 3.2], "Drug_B": [4.1, 4.3, 4.0]}
df_long = pd.DataFrame([{"group": g, "value": v} for g, vals in groups.items() for v in vals])
aov = pg.anova(dv="value", between="group", data=df_long)
posthoc = pg.pairwise_tukey(dv="value", between="group", data=df_long)
print(aov)
print(posthoc[["A", "B", "diff", "p-tukey"]])

# Non-parametric: Kruskal-Wallis + Dunn's test
kw_stat, kw_p = stats.kruskal(*[v for v in groups.values()])
print(f"Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f}")
dunn = pg.pairwise_tests(dv="value", between="group", data=df_long, parametric=False)

# Paired comparison: paired t-test
before = np.array([5.2, 4.8, 5.5, 4.9])
after = np.array([4.1, 3.9, 4.3, 3.8])
t_paired, p_paired = stats.ttest_rel(before, after)
print(f"Paired t-test: t={t_paired:.2f}, p={p_paired:.4f}")
```

**Test selection quick guide**:
| Data type | 2 independent groups | 2 paired | ≥3 independent | ≥3 paired |
|---|---|---|---|---|
| Normal | Welch's t-test | Paired t-test | ANOVA + Tukey | Repeated ANOVA |
| Non-normal / small n | Mann-Whitney U | Wilcoxon signed-rank | Kruskal-Wallis + Dunn | Friedman |

---

## 3. Dose-Response Analysis (IC50 / EC50)

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def four_pl(x, bottom, top, ec50, hill):
    """Four-parameter logistic (4PL) dose-response curve."""
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)

# Concentration in nM, response as % inhibition
conc = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])  # nM
response = np.array([2, 5, 15, 50, 85, 95, 98])         # % inhibition

# Fit the curve
try:
    popt, pcov = curve_fit(
        four_pl, conc, response,
        p0=[0, 100, 1.0, 1.0],         # initial guesses
        bounds=([0, 50, 0, 0.1], [20, 110, 1e6, 10]),  # realistic bounds
        maxfev=10000,
    )
    bottom, top, ic50, hill = popt
    perr = np.sqrt(np.diag(pcov))

    print(f"IC50 = {ic50:.3f} nM (95% CI: {ic50-2*perr[2]:.3f} – {ic50+2*perr[2]:.3f})")
    print(f"Hill slope = {hill:.2f}")
    print(f"Bottom = {bottom:.1f}%, Top = {top:.1f}%")
except RuntimeError:
    print("Curve fitting failed — check data range covers full sigmoidal curve")

# Plot
x_fit = np.logspace(np.log10(conc.min()), np.log10(conc.max()), 200)
plt.semilogx(conc, response, "o", label="Data")
plt.semilogx(x_fit, four_pl(x_fit, *popt), "-", label=f"4PL fit (IC50={ic50:.2f} nM)")
plt.axvline(ic50, ls="--", color="gray", alpha=0.5)
plt.xlabel("Concentration (nM)")
plt.ylabel("% Inhibition")
plt.legend()
```

---

## 4. ELISA Quantification

```python
# Standard curve fitting and interpolation
std_conc = np.array([0, 0.5, 1, 2, 5, 10, 20])  # ng/mL
std_od = np.array([0.05, 0.12, 0.21, 0.38, 0.82, 1.45, 2.1])  # OD450

# Fit 4PL to standard curve
popt_std, _ = curve_fit(four_pl, std_conc[1:], std_od[1:],
                         p0=[0, 2.5, 5, 1.5], maxfev=10000)

# Interpolate unknown samples
unknown_od = np.array([0.45, 0.78, 1.12])

def interpolate_from_curve(od_values, popt):
    """Inverse 4PL: solve for concentration given OD."""
    from scipy.optimize import brentq
    concentrations = []
    for od in od_values:
        try:
            c = brentq(lambda x: four_pl(x, *popt) - od, 0.01, 1000)
            concentrations.append(c)
        except ValueError:
            concentrations.append(np.nan)
    return np.array(concentrations)

conc_unknown = interpolate_from_curve(unknown_od, popt_std)
print(f"Concentrations: {conc_unknown} ng/mL")
```

---

## 5. Ecology: Diversity Indices

```python
import numpy as np
from scipy.stats import entropy as scipy_entropy

# Community composition (species counts)
community_A = np.array([50, 30, 10, 5, 3, 2])
community_B = np.array([15, 14, 13, 12, 11, 10])

def shannon_index(counts):
    props = counts / counts.sum()
    return scipy_entropy(props, base=np.e)

def simpson_index(counts):
    n = counts.sum()
    return 1 - np.sum(counts * (counts - 1)) / (n * (n - 1))

def chao1(counts):
    """Chao1 species richness estimator."""
    f1 = (counts == 1).sum()  # singletons
    f2 = (counts == 2).sum()  # doubletons
    return len(counts) + (f1**2) / (2 * max(f2, 1))

print(f"Community A — Shannon H': {shannon_index(community_A):.3f}, Simpson D: {simpson_index(community_A):.3f}")
print(f"Community B — Shannon H': {shannon_index(community_B):.3f}, Simpson D: {simpson_index(community_B):.3f}")

# Beta diversity: Bray-Curtis dissimilarity
def bray_curtis(a, b):
    return np.sum(np.abs(a - b)) / np.sum(a + b)

bc = bray_curtis(community_A, community_B)
print(f"Bray-Curtis dissimilarity: {bc:.3f} (0=identical, 1=completely different)")
```

---

## 6. Survival Analysis

```python
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd

# Kaplan-Meier survival curves
df = pd.DataFrame({
    "time": [5, 12, 20, 28, 35, 8, 15, 22, 30, 40],
    "event": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],       # 1=event, 0=censored
    "group": ["ctrl"]*5 + ["treat"]*5,
})

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7, 5))
for group in ["ctrl", "treat"]:
    mask = df["group"] == group
    kmf.fit(df[mask]["time"], df[mask]["event"], label=group)
    kmf.plot_survival_function(ax=ax, ci_show=True)

# Log-rank test between groups
ctrl = df[df.group == "ctrl"]
treat = df[df.group == "treat"]
lr = logrank_test(ctrl["time"], treat["time"], ctrl["event"], treat["event"])
print(f"Log-rank p-value: {lr.p_value:.4f}")

# Cox proportional hazards
df["treated"] = (df["group"] == "treat").astype(int)
cph = CoxPHFitter()
cph.fit(df[["time", "event", "treated"]], duration_col="time", event_col="event")
cph.print_summary()
# HR < 1 means treatment reduces hazard (protective)
# HR > 1 means treatment increases hazard
```

**Assumptions to check**:
- **Proportional hazards assumption**: use Schoenfeld residuals (`cph.check_assumptions(df)`)
- **Censoring**: must be non-informative (censored patients don't systematically differ from non-censored)
