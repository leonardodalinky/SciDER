"""
Time Series Profiler — stationarity tests, decomposition, and ACF/PACF analysis.

Usage:
    python ts_profiler.py data.csv --time_col date --value_col sales --output ts_report.md
    python ts_profiler.py data.csv --time_col timestamp --value_col temperature
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str, time_col: str, value_col: str) -> pd.Series:
    """Load a CSV/TSV file and return a time-indexed Series."""
    p = Path(path)
    sep = "\t" if p.suffix == ".tsv" else ","
    df = pd.read_csv(p, sep=sep, parse_dates=[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    if value_col not in df.columns:
        available = list(df.columns)
        sys.exit(f"Column '{value_col}' not found. Available: {available}")
    series = df[value_col].dropna()
    return series


def detect_frequency(series: pd.Series) -> str:
    """Infer frequency from datetime index."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return "unknown"
    try:
        freq = pd.infer_freq(series.index)
        return freq or "irregular"
    except Exception:
        return "irregular"


def stationarity_tests(series: pd.Series) -> dict:
    """Run ADF and KPSS stationarity tests."""
    results = {}
    try:
        from statsmodels.tsa.stattools import adfuller, kpss

        # ADF test: H0 = unit root (non-stationary). Reject H0 → stationary
        adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(series, autolag="AIC")
        results["adf"] = {
            "statistic": round(adf_stat, 4),
            "p_value": round(adf_p, 4),
            "lags_used": adf_lags,
            "critical_values": {k: round(v, 4) for k, v in adf_crit.items()},
            "stationary": adf_p < 0.05,
        }

        # KPSS test: H0 = stationary. Reject H0 → non-stationary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression="c")
        results["kpss"] = {
            "statistic": round(kpss_stat, 4),
            "p_value": round(kpss_p, 4),
            "lags_used": kpss_lags,
            "critical_values": {k: round(v, 4) for k, v in kpss_crit.items()},
            "stationary": kpss_p > 0.05,
        }
    except ImportError:
        results["error"] = "statsmodels not installed — run: uv pip install statsmodels"
    return results


def detect_seasonality(series: pd.Series) -> dict:
    """Detect dominant period using FFT."""
    n = len(series)
    if n < 10:
        return {"detected": False, "reason": "too few observations"}

    values = series.values - series.mean()
    fft = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(n)

    # Exclude DC component (freq=0)
    fft[0] = 0
    dominant_idx = np.argmax(fft)
    dominant_freq = freqs[dominant_idx]
    period = round(1 / dominant_freq) if dominant_freq > 0 else None

    return {
        "detected": dominant_freq > 0,
        "dominant_period": period,
        "dominant_frequency": round(float(dominant_freq), 4),
        "fft_strength": round(float(fft[dominant_idx] / fft.sum()), 3),
    }


def compute_autocorrelation(series: pd.Series, max_lags: int = 40) -> dict:
    """Compute ACF values and identify significant lags."""
    from statsmodels.tsa.stattools import acf

    try:
        acf_values, confint = acf(series, nlags=min(max_lags, len(series) // 2), alpha=0.05)
        ci_width = (confint[:, 1] - confint[:, 0]) / 2
        significant_lags = [
            int(i) for i, (acf_val, ci) in enumerate(zip(acf_values[1:], ci_width[1:]), 1)
            if abs(acf_val) > ci
        ]
        return {
            "significant_lags": significant_lags[:10],
            "lag_1_autocorr": round(float(acf_values[1]), 3) if len(acf_values) > 1 else None,
        }
    except Exception as e:
        return {"error": str(e)}


def generate_report(series: pd.Series, path: str, output: str) -> str:
    """Generate a markdown profiling report for the time series."""
    freq = detect_frequency(series)
    stat_tests = stationarity_tests(series)
    seasonality = detect_seasonality(series)
    autocorr = compute_autocorrelation(series)

    # Missing timestamps
    if isinstance(series.index, pd.DatetimeIndex) and freq not in ("irregular", "unknown"):
        try:
            expected = pd.date_range(series.index.min(), series.index.max(), freq=freq)
            missing_ts = len(expected) - len(series)
        except Exception:
            missing_ts = "unknown"
    else:
        missing_ts = "N/A (irregular frequency)"

    # Stationarity summary
    if "adf" in stat_tests and "kpss" in stat_tests:
        adf_stat = stat_tests["adf"]["stationary"]
        kpss_stat = stat_tests["kpss"]["stationary"]
        if adf_stat and kpss_stat:
            stationarity_conclusion = "✅ **Stationary** (both ADF and KPSS agree)"
        elif not adf_stat and not kpss_stat:
            stationarity_conclusion = "❌ **Non-stationary** — consider differencing or log-transform"
        else:
            stationarity_conclusion = "⚠️ **Inconclusive** — ADF and KPSS disagree; check for structural breaks"
    else:
        stationarity_conclusion = "⚠️ Could not run tests: " + stat_tests.get("error", "unknown error")

    lines = [
        f"# Time Series Profile: `{Path(path).name}`\n",
        "## Basic Statistics\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Observations | {len(series)} |",
        f"| Date range | {series.index.min()} → {series.index.max()} |",
        f"| Detected frequency | `{freq}` |",
        f"| Missing timestamps | {missing_ts} |",
        f"| Mean | {series.mean():.4f} |",
        f"| Std | {series.std():.4f} |",
        f"| Min | {series.min():.4f} |",
        f"| Max | {series.max():.4f} |",
        f"| Skewness | {series.skew():.3f} |",
        "",
        "## Stationarity Analysis\n",
        stationarity_conclusion,
        "",
    ]

    if "adf" in stat_tests:
        adf = stat_tests["adf"]
        lines += [
            "### ADF Test (H₀: unit root / non-stationary)",
            f"- Statistic: {adf['statistic']} | p-value: {adf['p_value']}",
            f"- {'Reject H₀ → **stationary**' if adf['stationary'] else 'Fail to reject H₀ → **non-stationary**'}",
        ]

    if "kpss" in stat_tests:
        kpss = stat_tests["kpss"]
        lines += [
            "\n### KPSS Test (H₀: stationary)",
            f"- Statistic: {kpss['statistic']} | p-value: {kpss['p_value']}",
            f"- {'Fail to reject H₀ → **stationary**' if kpss['stationary'] else 'Reject H₀ → **non-stationary**'}",
        ]

    lines += [
        "\n## Seasonality Detection (FFT)\n",
        f"- Dominant period: **{seasonality.get('dominant_period', 'none detected')}** time steps"
        if seasonality["detected"]
        else "- No dominant periodicity detected",
        f"- FFT strength: {seasonality.get('fft_strength', 'N/A')} (fraction of variance in dominant frequency)",
        "",
        "## Autocorrelation\n",
        f"- Lag-1 autocorrelation: {autocorr.get('lag_1_autocorr', 'N/A')}",
        f"- Significant lags (ACF > 95% CI): {autocorr.get('significant_lags', [])}",
        "",
        "## Recommendations\n",
    ]

    if "adf" in stat_tests and not stat_tests["adf"]["stationary"]:
        lines += [
            "- **Non-stationary data detected.** Try:",
            "  - First differencing: `series.diff().dropna()`",
            "  - Log transform: `np.log(series)` (if values > 0)",
            "  - Seasonal differencing: `series.diff(period)` (if periodic)",
        ]

    if seasonality["detected"] and seasonality.get("dominant_period"):
        p = seasonality["dominant_period"]
        lines += [
            f"- **Seasonality detected** (period ≈ {p}). Consider:",
            f"  - STL decomposition with `period={p}`",
            f"  - SARIMA with seasonal order s={p}",
            f"  - Temporal CV with gap of {p} time steps",
        ]

    lines += [
        "- **Cross-validation**: always use `TimeSeriesSplit` — never shuffle time series data",
        "- **Forecasting method selection**: see `time-series-analysis` skill for decision tree",
    ]

    report = "\n".join(lines)
    Path(output).write_text(report)
    print(f"Report saved to: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Profile a time series dataset")
    parser.add_argument("input_file", help="CSV/TSV file path")
    parser.add_argument("--time_col", default="time", help="Name of datetime column")
    parser.add_argument("--value_col", default="value", help="Name of value column")
    parser.add_argument("--output", default="ts_profile_report.md", help="Output report path")
    args = parser.parse_args()

    print(f"Loading {args.input_file}...")
    series = load_data(args.input_file, args.time_col, args.value_col)
    print(f"Loaded {len(series)} observations")

    generate_report(series, args.input_file, args.output)


if __name__ == "__main__":
    main()
