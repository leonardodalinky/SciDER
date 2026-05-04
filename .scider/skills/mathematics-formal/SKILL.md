---
name: mathematics-formal
description: Formal mathematics for scientific computing — symbolic computation (sympy), numerical linear algebra, optimization (convex/non-convex), information theory, and numerical precision issues. Use when working with mathematical derivations, proofs, or rigorous numerical analysis.
allowed_agents: [experiment, native_coding, ideation]
---

# Mathematics (Formal and Computational)

## Overview

This skill bridges formal mathematics and scientific computing, covering symbolic computation, numerical methods, optimization theory, and common numerical pitfalls. Use it when mathematical rigor or numerical correctness is central to your research.

## When to Use This Skill

- Deriving or verifying mathematical expressions symbolically
- Implementing numerically stable algorithms
- Choosing and applying optimization methods
- Working with probability distributions or information theory
- Checking gradient implementations or matrix computations

---

## 1. Symbolic Computation with SymPy

```python
import sympy as sp

# Define symbolic variables
x, y, t, n = sp.symbols("x y t n", real=True)
alpha, beta = sp.symbols("alpha beta", positive=True)

# Algebra
expr = (x + y)**3
print(sp.expand(expr))       # x³ + 3x²y + 3xy² + y³
print(sp.factor(x**2 - 1))  # (x-1)(x+1)

# Calculus
f = sp.exp(-alpha * x**2)
df = sp.diff(f, x)           # derivative
integral = sp.integrate(f, (x, -sp.oo, sp.oo))  # definite integral
print(f"∫ exp(-αx²) dx = {integral}")  # √(π/α)

# Taylor series
series = sp.series(sp.sin(x), x, 0, n=7)
print(series)  # x - x³/6 + x⁵/120 - ...

# Solve equations
solutions = sp.solve(x**2 + 2*x - 3, x)  # [1, -3]

# ODEs
y_func = sp.Function("y")
ode = sp.Eq(y_func(t).diff(t) + alpha * y_func(t), 0)
sol = sp.dsolve(ode, y_func(t))
print(sol)  # y(t) = C1 * exp(-αt)

# Linear algebra
A = sp.Matrix([[1, 2], [3, 4]])
print(A.det())         # -2
print(A.eigenvals())   # {3 - √5: 1, 3 + √5: 1}
print(A.inv())
```

---

## 2. Numerical Linear Algebra

```python
import numpy as np
from scipy import linalg

# ── Conditioning ──────────────────────────────────────────────
A = np.array([[1, 2], [2, 4.001]])  # nearly singular
cond = np.linalg.cond(A)
print(f"Condition number: {cond:.2e}")
# > 1e6 → ill-conditioned; results sensitive to input perturbations
# > 1/eps (≈ 4.5e15) → numerically singular

# ── Solving linear systems ─────────────────────────────────────
# NEVER: x = np.linalg.inv(A) @ b  (unstable, expensive)
# ALWAYS: x = np.linalg.solve(A, b)  (uses LU decomposition)
b = np.array([1.0, 2.0])
x = np.linalg.solve(A, b)

# Sparse systems (large n):
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
A_sparse = csr_matrix(A)
x_sparse = spsolve(A_sparse, b)

# ── SVD Decomposition ─────────────────────────────────────────
U, s, Vh = np.linalg.svd(A)
# Low-rank approximation (keep top k singular values)
k = 1
A_approx = (U[:, :k] * s[:k]) @ Vh[:k, :]

# Numerical rank (robust to noise)
rank = np.linalg.matrix_rank(A, tol=1e-10)

# Pseudo-inverse (for rank-deficient systems)
A_pinv = np.linalg.pinv(A)

# ── Eigendecomposition ────────────────────────────────────────
# For symmetric/Hermitian matrices (more stable):
eigenvalues, eigenvectors = np.linalg.eigh(A.T @ A)
# For general matrices:
eigenvalues_g, eigenvectors_g = np.linalg.eig(A)
```

---

## 3. Optimization

### Convex Optimization with CVXPY

```python
import cvxpy as cp
import numpy as np

# Recognize convex problems by checking:
# 1. Objective is convex (minimize) or concave (maximize)
# 2. Constraints are convex sets

# Example: Lasso regression
n, d = 100, 20
X = np.random.randn(n, d)
y = np.random.randn(n)
beta = cp.Variable(d)
lambda_reg = 0.1

objective = cp.Minimize(cp.sum_squares(X @ beta - y) + lambda_reg * cp.norm1(beta))
problem = cp.Problem(objective)
problem.solve()
print(f"Optimal beta: {beta.value}")
```

### Unconstrained Optimization with SciPy

```python
from scipy.optimize import minimize
import numpy as np

# Rosenbrock function (non-convex benchmark)
def rosenbrock(x):
    return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[:-1] += -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1]**2)
    return grad

result = minimize(
    rosenbrock,
    x0=np.zeros(5),
    jac=rosenbrock_grad,
    method="L-BFGS-B",    # best for smooth unconstrained
    options={"maxiter": 1000, "ftol": 1e-12},
)
print(f"Minimum: {result.fun:.6f} at x = {result.x}")
```

### Optimization Method Selection

| Scenario | Recommended method |
|----------|-------------------|
| Convex, structured problem | CVXPY (disciplined convex programming) |
| Smooth, unconstrained | L-BFGS-B (gradient-based) |
| Smooth, constrained | SLSQP (equality + inequality constraints) |
| Non-smooth or derivative-free | Nelder-Mead, differential evolution |
| Large-scale ML (neural nets) | Adam, AdamW (with `torch.optim`) |
| Bayesian hyperparameter opt | Optuna, BoTorch |

---

## 4. Probability Theory

```python
from scipy import stats
import numpy as np

# Common distributions
normal = stats.norm(loc=0, scale=1)
poisson = stats.poisson(mu=5)
beta_dist = stats.beta(a=2, b=5)

# Moments
print(f"Normal: mean={normal.mean()}, var={normal.var()}")
print(f"Poisson: mean={poisson.mean()}, var={poisson.var()}")

# PDF, CDF, survival function
x = np.linspace(-3, 3, 100)
pdf = normal.pdf(x)
cdf = normal.cdf(x)

# Sampling
samples = normal.rvs(size=1000, random_state=42)

# CLT: sample means converge to Normal(μ, σ²/n)
# Useful: with n≥30, sample mean is approximately normal regardless of distribution
n_samples = 1000
means = [stats.expon(scale=2).rvs(size=30).mean() for _ in range(n_samples)]
print(f"Mean of means: {np.mean(means):.3f}, std: {np.std(means):.3f}")
print(f"Expected (CLT): {2:.3f}, {2/np.sqrt(30):.3f}")
```

---

## 5. Information Theory

```python
import numpy as np
from scipy.stats import entropy

# Shannon entropy: H(X) = -∑ p(x) log p(x)
p = np.array([0.5, 0.3, 0.2])
H = entropy(p, base=2)  # in bits
print(f"Entropy: {H:.3f} bits (max for 3 outcomes = log₂(3) = {np.log2(3):.3f})")

# KL divergence: D_KL(P||Q) = ∑ P log(P/Q)
# NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
# Interpretation: extra bits needed to encode P using Q's code
p_dist = np.array([0.4, 0.4, 0.2])
q_dist = np.array([0.33, 0.33, 0.34])
kl_pq = entropy(p_dist, q_dist, base=2)
kl_qp = entropy(q_dist, p_dist, base=2)
print(f"KL(P||Q) = {kl_pq:.3f}, KL(Q||P) = {kl_qp:.3f}")

# Mutual information: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
# Joint distribution
p_xy = np.array([[0.2, 0.1], [0.15, 0.25], [0.1, 0.2]])
p_x = p_xy.sum(axis=1)
p_y = p_xy.sum(axis=0)
H_xy = entropy(p_xy.flatten(), base=2)
I_xy = entropy(p_x, base=2) + entropy(p_y, base=2) - H_xy
print(f"Mutual information I(X;Y) = {I_xy:.3f} bits")

# Log-sum-exp trick (avoid overflow in log probabilities)
def logsumexp(log_probs):
    max_lp = log_probs.max()
    return max_lp + np.log(np.exp(log_probs - max_lp).sum())

log_probs = np.array([-1000., -1001., -1002.])  # would overflow with exp directly
print(f"log∑exp = {logsumexp(log_probs):.3f}")  # stable computation
```

---

## 6. Numerical Precision

```python
import numpy as np

# Machine epsilon
eps = np.finfo(float).eps  # ≈ 2.2e-16
print(f"Machine epsilon: {eps:.2e}")

# Catastrophic cancellation: avoid a - b when a ≈ b
a, b = 1.0000001, 1.0
bad_result = a - b             # loses precision
good_result = np.float128(a) - np.float128(b)  # use higher precision if needed

# Use numerically stable formulas
# BAD:  sqrt(x+1) - 1  (cancellation when x is small)
# GOOD: x / (sqrt(x+1) + 1)  (algebraically equivalent, no cancellation)

def safe_log1p(x):
    """Numerically stable log(1+x) for small x."""
    return np.log1p(x)  # use numpy's built-in, not log(1+x)

def safe_expm1(x):
    """Numerically stable exp(x)-1 for small x."""
    return np.expm1(x)  # use numpy's built-in

# Kahan summation for summing many floats
def kahan_sum(values):
    total = 0.0
    compensation = 0.0
    for v in values:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total

# Numerically stable softmax
def stable_softmax(x):
    x_shifted = x - x.max()  # subtract max to prevent overflow
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()
```

**Key rules**:
1. Check condition number before solving linear systems — if > 1e6, add regularization
2. Use `np.linalg.solve()` not `inv(A) @ b`
3. Use `log1p(x)` and `expm1(x)` for small x
4. Apply the log-sum-exp trick when computing log-sums of probabilities
5. Never compare floats with `==` — use `np.isclose(a, b, rtol=1e-5)`
