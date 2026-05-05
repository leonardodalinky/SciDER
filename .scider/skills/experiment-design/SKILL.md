---
name: experiment-design
description: Controlled experiment design — hypothesis formation, statistical power and sample size estimation, confound control, ablation structure, baseline selection, and reproducibility requirements. Use when designing research experiments to ensure valid, reproducible results.
allowed_agents: [ideation, experiment]
preload_for: [ideation]
---

# Experiment Design

## Overview

Good experimental design determines whether your results are interpretable, valid, and reproducible. This skill covers the process from hypothesis formation through post-experiment reporting, with actionable checklists and code examples.

## When to Use This Skill

- Before designing any experiment: choose baselines, define metrics, plan sample sizes
- When writing a research idea: include a concrete experimental protocol
- When results look unexpected: check whether experimental design is the cause
- When a reviewer asks about confounds, baselines, or statistical power

---

## Hard Rules

1. **Define your hypothesis BEFORE collecting data** — post-hoc hypothesis selection inflates false positive rates
2. **Always include at least one trivial baseline** — random, majority class, or simple heuristic
3. **Set random seeds everywhere** — numpy, random, torch, tensorflow, sklearn
4. **Use held-out test data only once** — never tune on it
5. **Report N (sample size) and how it was determined** — underpowered studies mislead

---

## 1. Hypothesis Formation

A well-formed hypothesis has three parts:

| Part | Example |
|------|---------|
| **Null hypothesis (H₀)** | The proposed method performs no better than the baseline |
| **Alternative hypothesis (H₁)** | The proposed method achieves higher F1 than the baseline |
| **Expected effect size** | Δ = 0.03 F1 (3 percentage points) |

**Directionality**:
- **One-tailed**: you predict direction of effect (method > baseline) → more statistical power, but only justified if direction is theoretically grounded
- **Two-tailed**: you test for any difference → safer for exploratory work

---

## 2. Statistical Power and Sample Size

Power analysis determines how many samples you need to detect an effect of a given size.

**Key parameters**:
- α (Type I error rate): probability of false positive — typically 0.05
- β (Type II error rate): probability of false negative — typically 0.20
- Power = 1 - β = 0.80 (minimum), 0.90 (preferred)
- Effect size: domain-specific (Cohen's d for continuous, Cohen's h for proportions)

```python
from statsmodels.stats.power import TTestIndPower, NormalIndPower

# Two-sample t-test: minimum n per group
analysis = TTestIndPower()
n = analysis.solve_power(
    effect_size=0.5,   # Cohen's d: small=0.2, medium=0.5, large=0.8
    alpha=0.05,
    power=0.80,
    ratio=1.0,         # equal group sizes
)
print(f"Required n per group: {n:.0f}")

# For proportions (e.g., accuracy comparison)
from statsmodels.stats.proportion import proportion_effectsize
effect = proportion_effectsize(0.75, 0.80)  # baseline vs expected accuracy
n_prop = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.80)
print(f"Required n per condition: {n_prop:.0f}")
```

**Rule of thumb for ML experiments**: use at least 30 test examples per class for classification; for regression, aim for n ≥ 50 per parameter estimated.

---

## 3. Confound Identification and Control

**Confound**: a variable that affects both treatment and outcome, creating spurious correlations.

**Step-by-step confound analysis**:
1. List all variables that could affect your outcome
2. For each, ask: "Is this variable correlated with my treatment?"
3. If yes → it is a potential confound
4. Decide: hold constant, measure and control, or randomize

**Common confounds in ML research**:
- **Dataset size**: compare methods on same-size training data
- **Compute budget**: compare methods given same FLOPs or wall time
- **Random seed**: report mean ± std across multiple seeds
- **Hardware**: specify GPU/CPU for timing comparisons
- **Preprocessing choices**: same pipeline for all methods

**Control strategies**:
- **Hold constant**: run all conditions with the same hyperparameters
- **Block and randomize**: if you must vary, do so randomly across conditions
- **Measure and partial out**: include confound as a covariate in statistical model

---

## 4. Ablation Study Structure

Ablation studies isolate the contribution of each component.

**Protocol**:
1. List ALL system components (e.g., data augmentation, loss function, pretraining, regularization)
2. Define a "full system" (all components enabled)
3. Create one ablation per component: full system minus component X
4. Run each ablation on the SAME data, with the SAME random seeds
5. Report delta contribution: Δ(full − ablated)

```
| Configuration          | Metric | Δ vs Full |
|------------------------|--------|-----------|
| Full system            | 82.3   | —         |
| − Data augmentation    | 79.1   | −3.2      |
| − Pretraining          | 76.8   | −5.5      |
| − Regularization       | 81.7   | −0.6      |
| − All (baseline only)  | 71.2   | −11.1     |
```

**Common mistake**: changing multiple things between ablations — you cannot attribute the change to any one component.

---

## 5. Baseline Selection Criteria

A valid baseline must be:
1. **Reproducible**: from published paper or fully specified implementation
2. **Comparable**: evaluated on the same data split and preprocessing
3. **Appropriate**: same task, same evaluation protocol

**Baseline hierarchy**:
- **Trivial baseline**: random guess, majority class, mean prediction, constant output
- **Simple heuristic**: TF-IDF + logistic regression for NLP, linear regression for tabular
- **Prior state of the art**: best published result on same benchmark

Always include at least the trivial baseline. If your method cannot beat a trivial baseline, revisit the problem formulation.

---

## 6. Reproducibility Requirements

### Random Seeds (set ALL of these)
```python
import random, os, numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except ImportError:
    pass
```

### Environment Logging
```bash
# Save exact package versions
pip freeze > requirements.txt
# OR with uv:
uv pip freeze > requirements.txt

# Log hardware info
python -c "import torch; print(torch.cuda.get_device_name(0))"
nvidia-smi
python --version
```

### Data Integrity
```python
import hashlib, pathlib

def file_hash(path):
    h = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    print(f"SHA256({path}): {h}")
    return h

# Call this for every dataset file before training
file_hash("train.csv")
file_hash("test.csv")
```

---

## 7. Replication vs. Reproduction

| Term | Definition | What it verifies |
|------|-----------|-----------------|
| **Reproduction** | Same code, same data, same results | Code correctness |
| **Replication** | Different implementation, same protocol → same results | Scientific claim validity |

For AI research:
- Reproduction: re-run your own code → should match
- Replication: another team implements your method → the harder, more meaningful test

**Documentation for reproducibility**:
- Pin all random seeds
- Record all hyperparameters (save config file, not just code)
- Log training curves, not just final metrics
- Checkpoint models at multiple points

---

## Pre-Experiment Checklist

Before starting:
- [ ] Hypothesis is written down (H₀, H₁, expected effect size)
- [ ] Sample size / evaluation set size determined (power analysis or domain norm)
- [ ] All baselines identified and confirmed implementable
- [ ] Evaluation metric(s) chosen and justified
- [ ] Data splits created and fixed (no re-splitting after seeing results)
- [ ] Potential confounds listed and mitigation strategy defined
- [ ] Random seeds set and logged
- [ ] Ablation study components listed

After experiment:
- [ ] Results reported with confidence intervals or std across seeds
- [ ] All baselines evaluated on exact same test split
- [ ] Ablation table shows per-component delta
- [ ] Hyperparameter search budget documented
- [ ] Training/validation curves included
- [ ] Requirements file saved with pinned versions
