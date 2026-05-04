"""
Physics Simulation Data Inspector.

Checks numerical arrays from simulation output for physical plausibility:
- NaN / Inf detection
- Value range and statistical properties
- Energy conservation drift (if energy columns present)
- Autocorrelation (equilibration check for MD/MC data)

Usage:
    python physics_eda.py simulation_output.csv --output physics_report.md
    python physics_eda.py trajectory.npy --output report.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_file(path: str) -> pd.DataFrame:
    """Load simulation data from various formats into a DataFrame."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(p)
    elif ext in (".tsv", ".txt"):
        return pd.read_csv(p, sep="\t")
    elif ext == ".npy":
        arr = np.load(p)
        if arr.ndim == 1:
            return pd.DataFrame({"value": arr})
        return pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
    elif ext == ".npz":
        data = np.load(p)
        dfs = {}
        for key in data.files:
            arr = data[key]
            if arr.ndim == 1:
                dfs[key] = arr
            else:
                for i in range(arr.shape[1]):
                    dfs[f"{key}_{i}"] = arr[:, i]
        return pd.DataFrame(dfs)
    elif ext in (".h5", ".hdf5"):
        try:
            import h5py
            with h5py.File(p, "r") as f:
                data = {}
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.ndim <= 2:
                        arr = obj[()]
                        if arr.ndim == 1:
                            data[name] = arr
                        elif arr.ndim == 2 and arr.shape[1] <= 20:
                            for i in range(arr.shape[1]):
                                data[f"{name}_{i}"] = arr[:, i]
                f.visititems(collect)
            return pd.DataFrame(data)
        except ImportError:
            sys.exit("h5py not installed: uv pip install h5py")
    else:
        # Try CSV as fallback
        try:
            return pd.read_csv(p)
        except Exception:
            sys.exit(f"Unsupported format: {ext}")


def check_nan_inf(df: pd.DataFrame) -> dict:
    """Check for NaN and Inf values per column."""
    results = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col].dropna()).sum()
        results[col] = {"n_nan": int(n_nan), "n_inf": int(n_inf)}
    return results


def column_stats(df: pd.DataFrame) -> dict:
    """Compute basic statistics for each numeric column."""
    num_df = df.select_dtypes(include=[np.number])
    stats = {}
    for col in num_df.columns:
        vals = num_df[col].dropna().values
        if len(vals) == 0:
            continue
        stats[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "skewness": float(pd.Series(vals).skew()),
            "kurtosis": float(pd.Series(vals).kurtosis()),
        }
    return stats


def energy_conservation_check(df: pd.DataFrame) -> dict:
    """Check energy conservation if energy-related columns are present."""
    energy_keywords = ["energy", "etotal", "epot", "ekin", "e_total", "total_energy"]
    energy_cols = [c for c in df.columns
                   if any(kw in c.lower() for kw in energy_keywords)]

    results = {}
    for col in energy_cols:
        vals = df[col].dropna().values
        if len(vals) < 10:
            continue
        initial = vals[:10].mean()
        final = vals[-10:].mean()
        drift_pct = abs((final - initial) / (initial + 1e-30)) * 100
        std_pct = vals.std() / (abs(vals.mean()) + 1e-30) * 100
        results[col] = {
            "initial_avg": round(float(initial), 4),
            "final_avg": round(float(final), 4),
            "drift_pct": round(float(drift_pct), 2),
            "fluctuation_pct": round(float(std_pct), 2),
            "concern": drift_pct > 5.0,
        }
    return results


def autocorrelation_check(df: pd.DataFrame, max_lags: int = 100) -> dict:
    """Estimate autocorrelation time to check for equilibration."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    results = {}
    for col in list(num_cols)[:5]:  # check first 5 numeric columns
        vals = df[col].dropna().values
        if len(vals) < 50:
            continue
        mean = vals.mean()
        var = vals.var()
        if var < 1e-30:
            continue
        lags = min(max_lags, len(vals) // 4)
        acf = np.array([
            np.mean((vals[:len(vals)-k] - mean) * (vals[k:] - mean)) / var
            for k in range(1, lags + 1)
        ])
        # Integrated autocorrelation time
        tau = 1 + 2 * acf[acf > 0.05].sum()
        n_eff = len(vals) / (2 * tau) if tau > 0 else len(vals)
        results[col] = {
            "lag_1_acf": round(float(acf[0]), 3),
            "autocorr_time": round(float(tau), 1),
            "effective_samples": round(float(n_eff), 0),
            "possibly_unequilibrated": acf[0] > 0.9,
        }
    return results


def generate_report(path: str, df: pd.DataFrame, output: str) -> str:
    """Generate a markdown report from all checks."""
    nan_inf = check_nan_inf(df)
    stats = column_stats(df)
    energy = energy_conservation_check(df)
    autocorr = autocorrelation_check(df)

    lines = [
        f"# Physics Simulation Data Report: `{Path(path).name}`\n",
        f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n",
        "---\n",
        "## 1. NaN / Inf Check\n",
    ]

    has_issues = any(v["n_nan"] > 0 or v["n_inf"] > 0 for v in nan_inf.values())
    if has_issues:
        lines.append("| Column | NaN count | Inf count |")
        lines.append("|--------|-----------|-----------|")
        for col, vals in nan_inf.items():
            if vals["n_nan"] > 0 or vals["n_inf"] > 0:
                lines.append(f"| `{col}` | {vals['n_nan']} | {vals['n_inf']} |")
        lines.append("\n⚠️ **Action**: Investigate source of NaN/Inf — may indicate numerical instability.")
    else:
        lines.append("✅ No NaN or Inf values found.\n")

    lines += [
        "\n## 2. Column Statistics\n",
        "| Column | Mean | Std | Min | Max | Skewness |",
        "|--------|------|-----|-----|-----|----------|",
    ]
    for col, s in list(stats.items())[:20]:  # cap at 20 columns
        lines.append(
            f"| `{col}` | {s['mean']:.4g} | {s['std']:.4g} | {s['min']:.4g} | {s['max']:.4g} | {s['skewness']:.2f} |"
        )

    if energy:
        lines += ["\n## 3. Energy Conservation\n"]
        for col, e in energy.items():
            icon = "⚠️" if e["concern"] else "✅"
            lines.append(f"**{col}**: Initial avg = {e['initial_avg']:.4g}, Final avg = {e['final_avg']:.4g}")
            lines.append(f"- Drift: {e['drift_pct']}% | Fluctuation: {e['fluctuation_pct']}% {icon}")
            if e["concern"]:
                lines.append("  - > 5% drift suggests numerical instability or too-large timestep")
    else:
        lines.append("\n## 3. Energy Conservation\n")
        lines.append("No energy columns detected (looking for columns containing 'energy', 'etotal', etc.)\n")

    lines += ["\n## 4. Autocorrelation (Equilibration Check)\n"]
    if autocorr:
        lines += [
            "| Column | Lag-1 ACF | Autocorr time (τ) | Effective samples |",
            "|--------|-----------|-------------------|-------------------|",
        ]
        for col, a in autocorr.items():
            icon = " ⚠️ (possibly unequilibrated)" if a["possibly_unequilibrated"] else ""
            lines.append(
                f"| `{col}` | {a['lag_1_acf']} | {a['autocorr_time']} | {int(a['effective_samples'])} |{icon}"
            )
        lines.append("\n> **Lag-1 ACF > 0.9** suggests high serial correlation — run longer or discard burn-in.")
    else:
        lines.append("Not enough data for autocorrelation analysis (need ≥ 50 observations).\n")

    lines += [
        "\n## 5. Recommendations\n",
        "- Use `physics-simulation` skill for guidance on solver selection, MD analysis, and error propagation",
        "- If energy drift > 5%: reduce timestep, check force field parameters, or use symplectic integrator",
        "- If high autocorrelation: extend simulation, use replica exchange, or apply thinning",
    ]

    report = "\n".join(lines)
    Path(output).write_text(report)
    print(f"Report saved to: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Physics simulation data inspector")
    parser.add_argument("input_file", help="Path to simulation output file (CSV, NPY, NPZ, HDF5)")
    parser.add_argument("--output", default="physics_eda_report.md", help="Output report path")
    args = parser.parse_args()

    print(f"Loading {args.input_file}...")
    df = load_file(args.input_file)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    generate_report(args.input_file, df, args.output)


if __name__ == "__main__":
    main()
