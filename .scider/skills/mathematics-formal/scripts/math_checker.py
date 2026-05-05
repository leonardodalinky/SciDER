"""
Mathematical Verification Utilities for Research Code.

Provides gradient checks, matrix diagnostics, and numerical stability tests.

Usage as module:
    from math_checker import gradient_check, condition_number_check, check_positive_definite

Usage as script (demo):
    python math_checker.py
"""

import numpy as np
import sys


def gradient_check(func, x: np.ndarray, eps: float = 1e-5, rtol: float = 1e-3) -> dict:
    """Compare analytical gradient with finite-difference approximation.

    Args:
        func: Function that returns (scalar_value, gradient_array) or just scalar.
              If it returns a scalar, uses finite differences only.
        x: Input array at which to evaluate gradient.
        eps: Finite difference step size.
        rtol: Relative tolerance for passing the check.

    Returns:
        dict with 'passed', 'max_rel_error', 'analytic_grad', 'fd_grad'.
    """
    x = np.array(x, dtype=np.float64)

    # Compute analytical gradient
    result = func(x)
    if isinstance(result, tuple):
        val, analytic_grad = result
        analytic_grad = np.array(analytic_grad, dtype=np.float64)
    else:
        # Try torch autograd if available
        try:
            import torch
            x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float64)
            val_torch = func(x_torch)
            val_torch.backward()
            analytic_grad = x_torch.grad.numpy()
            val = float(val_torch.detach())
        except (ImportError, Exception):
            analytic_grad = None
            val = float(result)

    # Compute finite-difference gradient
    fd_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        r_plus = func(x_plus)
        r_minus = func(x_minus)
        v_plus = float(r_plus[0] if isinstance(r_plus, tuple) else r_plus)
        v_minus = float(r_minus[0] if isinstance(r_minus, tuple) else r_minus)
        fd_grad[i] = (v_plus - v_minus) / (2 * eps)

    if analytic_grad is None:
        return {
            "passed": None,
            "note": "No analytical gradient available; finite-difference computed",
            "fd_grad": fd_grad,
            "val": val,
        }

    # Relative error
    norm_analytic = np.linalg.norm(analytic_grad)
    norm_diff = np.linalg.norm(analytic_grad - fd_grad)
    rel_error = norm_diff / (norm_analytic + 1e-8)
    passed = rel_error < rtol

    return {
        "passed": passed,
        "max_rel_error": float(rel_error),
        "analytic_grad": analytic_grad,
        "fd_grad": fd_grad,
        "val": val,
        "interpretation": "✅ PASSED" if passed else f"❌ FAILED (rel_error={rel_error:.2e} > rtol={rtol})",
    }


def condition_number_check(matrix: np.ndarray) -> dict:
    """Check matrix conditioning and suggest remedies if ill-conditioned.

    Args:
        matrix: 2D numpy array.

    Returns:
        dict with 'condition_number', 'is_ill_conditioned', 'recommendations'.
    """
    A = np.array(matrix, dtype=np.float64)
    cond = np.linalg.cond(A)
    eps_machine = np.finfo(float).eps
    is_singular = cond > (1.0 / eps_machine)
    is_ill = cond > 1e6

    recommendations = []
    if is_ill:
        recommendations += [
            "Add Tikhonov regularization: solve (AᵀA + λI)x = Aᵀb",
            "Normalize/scale input features to similar magnitudes",
            "Use SVD-based pseudo-inverse for least-squares: np.linalg.lstsq(A, b)",
        ]
    if is_singular:
        recommendations.append("Matrix is numerically singular — use pseudo-inverse")

    return {
        "condition_number": float(cond),
        "log10_condition": float(np.log10(cond + 1)),
        "is_ill_conditioned": bool(is_ill),
        "is_numerically_singular": bool(is_singular),
        "recommendations": recommendations,
        "interpretation": (
            "✅ Well-conditioned" if not is_ill else
            ("❌ Numerically singular" if is_singular else "⚠️ Ill-conditioned")
        ),
    }


def matrix_rank_check(matrix: np.ndarray, tol: float = 1e-10) -> dict:
    """Check numerical rank vs declared rank.

    Args:
        matrix: 2D numpy array.
        tol: Tolerance for singular value threshold.

    Returns:
        dict with 'full_rank', 'numerical_rank', 'declared_rank', 'singular_values'.
    """
    A = np.array(matrix, dtype=np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    numerical_rank = int((s > tol * s[0]).sum())
    declared_rank = min(A.shape)

    return {
        "declared_rank": declared_rank,
        "numerical_rank": numerical_rank,
        "full_rank": numerical_rank == declared_rank,
        "singular_values": s.tolist(),
        "smallest_nonzero_sv": float(s[numerical_rank - 1]) if numerical_rank > 0 else 0.0,
        "interpretation": (
            "✅ Full rank" if numerical_rank == declared_rank
            else f"⚠️ Rank-deficient: {numerical_rank} < {declared_rank}"
        ),
    }


def check_positive_definite(matrix: np.ndarray) -> dict:
    """Test if a matrix is positive definite.

    Args:
        matrix: Square numpy array (should be symmetric).

    Returns:
        dict with 'is_positive_definite', 'min_eigenvalue', 'nearest_pd_suggestion'.
    """
    A = np.array(matrix, dtype=np.float64)
    A_sym = (A + A.T) / 2  # symmetrize

    try:
        np.linalg.cholesky(A_sym)
        is_pd = True
        eigenvalues = np.linalg.eigvalsh(A_sym)
        min_ev = float(eigenvalues.min())
    except np.linalg.LinAlgError:
        is_pd = False
        eigenvalues = np.linalg.eigvalsh(A_sym)
        min_ev = float(eigenvalues.min())

    suggestion = None
    if not is_pd:
        # Nearest positive definite matrix (Higham 1988)
        eps_reg = abs(min_ev) + 1e-10
        suggestion = f"Add {eps_reg:.2e} * I to make positive definite: A + {eps_reg:.2e}*np.eye(n)"

    return {
        "is_positive_definite": is_pd,
        "min_eigenvalue": min_ev,
        "max_eigenvalue": float(eigenvalues.max()),
        "all_eigenvalues": eigenvalues.tolist(),
        "nearest_pd_suggestion": suggestion,
        "interpretation": "✅ Positive definite" if is_pd else f"❌ Not PD (min eigenvalue = {min_ev:.2e})",
    }


def numerical_stability_test(func, x: np.ndarray, perturbation: float = 1e-8) -> dict:
    """Evaluate function sensitivity to small input perturbations.

    Args:
        func: Function from array to scalar.
        x: Nominal input point.
        perturbation: Magnitude of random perturbation.

    Returns:
        dict with sensitivity (condition number estimate) and pass/fail.
    """
    x = np.array(x, dtype=np.float64)
    f_x = float(func(x))

    sensitivities = []
    for _ in range(10):
        delta = perturbation * np.random.randn(len(x))
        f_perturbed = float(func(x + delta))
        rel_output_change = abs(f_perturbed - f_x) / (abs(f_x) + 1e-30)
        rel_input_change = np.linalg.norm(delta) / (np.linalg.norm(x) + 1e-30)
        if rel_input_change > 0:
            sensitivities.append(rel_output_change / rel_input_change)

    if not sensitivities:
        return {"error": "Could not compute sensitivity"}

    condition_est = float(np.mean(sensitivities))
    return {
        "condition_number_estimate": condition_est,
        "is_sensitive": condition_est > 1e6,
        "interpretation": (
            f"✅ Numerically stable (κ ≈ {condition_est:.2e})" if condition_est <= 1e6
            else f"⚠️ Sensitive to perturbations (κ ≈ {condition_est:.2e})"
        ),
    }


def print_result(name: str, result: dict):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 6:
            print(f"  {k}: [{', '.join(f'{x:.4g}' for x in v[:3])}, ..., {v[-1]:.4g}]")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    print("Math Checker — Demonstration\n")
    np.random.seed(42)

    # 1. Gradient check: quadratic function f(x) = ||x||²
    def quadratic(x):
        val = float(np.dot(x, x))
        grad = 2 * x
        return val, grad

    x0 = np.array([1.0, 2.0, -1.5, 0.5])
    print_result("Gradient Check (f(x) = ||x||², analytic grad = 2x)",
                 gradient_check(quadratic, x0))

    # 2. Condition number check: well-conditioned vs ill-conditioned
    A_good = np.array([[2.0, 1.0], [1.0, 3.0]])
    A_bad = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])

    print_result("Condition Number (well-conditioned 2×2)", condition_number_check(A_good))
    print_result("Condition Number (ill-conditioned 2×2)", condition_number_check(A_bad))

    # 3. Matrix rank check: rank-deficient matrix
    A_rankdef = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    print_result("Matrix Rank Check (rank-deficient 3×3)", matrix_rank_check(A_rankdef))

    # 4. Positive definite check
    A_pd = np.array([[4.0, 2.0], [2.0, 3.0]])       # positive definite
    A_not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])   # not positive definite
    print_result("Positive Definite Check (PD matrix)", check_positive_definite(A_pd))
    print_result("Positive Definite Check (non-PD matrix)", check_positive_definite(A_not_pd))

    # 5. Numerical stability
    def stable_func(x): return np.sum(np.log(1 + x**2))
    def unstable_func(x): return np.sum(np.exp(x) - np.exp(x - 1e-15) * np.exp(1e-15))

    print_result("Stability Test (log(1+x²), should be stable)",
                 numerical_stability_test(stable_func, np.array([1.0, 2.0, 3.0])))
