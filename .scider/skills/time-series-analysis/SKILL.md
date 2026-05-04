---
name: time-series-analysis
description: Time series data analysis — stationarity testing, decomposition, autocorrelation, forecasting method selection, temporal cross-validation, and evaluation metrics (MAPE, RMSE, MASE). Use when working with sequential temporal data.
allowed_agents: [data, experiment]
---

# Time Series Analysis

## Overview

Time series analysis covers the end-to-end workflow for sequential temporal data: data preparation, stationarity testing, decomposition, autocorrelation analysis, model selection, temporally-correct cross-validation, and evaluation. Apply this skill for any dataset where observations are ordered in time and the temporal structure is scientifically meaningful.

## When to Use This Skill

Use this skill when:
- The dataset has a datetime index and the ordering of observations matters
- You need to forecast future values of a variable
- You need to detect trends, seasonality, or change points
- You are performing anomaly detection on temporal data
- You need to evaluate a forecasting model with proper temporal splits
- You are fitting ARIMA, Prophet, LSTM, or any other time series model

---

## Data Preparation

### Datetime Index Setup

```python
import pandas as pd

# Load and parse dates
df = pd.read_csv("data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.set_index("timestamp")

# Verify index is datetime
print(df.index.dtype)         # should be datetime64[ns]
print(df.index.is_monotonic_increasing)  # must be True
```

### Frequency Detection and Setting

```python
# Infer frequency from data
inferred_freq = pd.infer_freq(df.index)
print(f"Inferred frequency: {inferred_freq}")
# Common codes: T (minute), H (hourly), D (daily), W (weekly), M (month-end), Q (quarter), A (annual)

# Explicitly set frequency (required by many statsmodels functions)
df = df.asfreq(inferred_freq)  # may introduce NaT rows for missing timestamps
print(df.index.freq)
```

### Handling Missing Timestamps

```python
# Check for gaps
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=inferred_freq)
missing_timestamps = full_index.difference(df.index)
print(f"Missing timestamps: {len(missing_timestamps)}")

# Reindex to fill in gaps, then interpolate
df = df.reindex(full_index)
n_missing = df["value"].isna().sum()
print(f"Missing values after reindex: {n_missing}")

# Interpolation options (choose based on expected pattern)
df["value"] = df["value"].interpolate(method="time")       # linear in time
# df["value"] = df["value"].interpolate(method="cubic")    # cubic spline
# df["value"] = df["value"].ffill()                        # forward fill (flat)
# df["value"] = df["value"].fillna(df["value"].rolling(7, min_periods=1).mean())  # rolling mean
```

### Resampling

```python
# Downsample: daily → weekly mean
df_weekly = df.resample("W").mean()

# Downsample: minute-level → hourly sum
df_hourly = df.resample("H").sum()

# Upsample: daily → hourly (creates NaN, then fill)
df_hourly = df.resample("H").interpolate(method="time")

# Multiple aggregations at once
df_monthly = df.resample("M").agg({
    "sales": "sum",
    "temperature": "mean",
    "visitors": "max"
})
```

---

## Stationarity Analysis

**Why it matters**: Most classical models (ARIMA) require stationarity. Non-stationary series have time-varying mean or variance, which violates model assumptions.

### ADF Test (Augmented Dickey-Fuller)

Null hypothesis: the series has a unit root (non-stationary). Reject H0 for stationarity.

```python
from statsmodels.tsa.stattools import adfuller

def test_stationarity_adf(series, name="Series", alpha=0.05):
    """Run ADF test and print interpretation."""
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value, n_lags, n_obs, critical_values, _ = result

    print(f"\nADF Test — {name}")
    print(f"  ADF Statistic:  {adf_stat:.4f}")
    print(f"  p-value:        {p_value:.4f}")
    print(f"  Lags used:      {n_lags}")
    print(f"  Observations:   {n_obs}")
    print(f"  Critical values:")
    for level, crit in critical_values.items():
        marker = " <-- ADF statistic is more extreme" if adf_stat < crit else ""
        print(f"    {level}: {crit:.4f}{marker}")

    if p_value < alpha:
        print(f"  CONCLUSION: Reject H0 (p={p_value:.4f} < {alpha}). Series is STATIONARY.")
    else:
        print(f"  CONCLUSION: Fail to reject H0 (p={p_value:.4f} >= {alpha}). Series is NON-STATIONARY.")

    return {"adf_stat": adf_stat, "p_value": p_value, "stationary": p_value < alpha}

result = test_stationarity_adf(df["value"], name="Sales")
```

### KPSS Test

Null hypothesis: the series is stationary (trend-stationary). Reject H0 for non-stationarity. Use alongside ADF — contradictory results suggest difference-stationary behavior.

```python
from statsmodels.tsa.stattools import kpss

def test_stationarity_kpss(series, name="Series", regression="c", alpha=0.05):
    """
    regression: 'c' (constant) or 'ct' (constant + trend)
    """
    kpss_stat, p_value, n_lags, critical_values = kpss(series.dropna(), regression=regression, nlags="auto")

    print(f"\nKPSS Test — {name}")
    print(f"  KPSS Statistic: {kpss_stat:.4f}")
    print(f"  p-value:        {p_value:.4f}  (Note: p is bounded [0.01, 0.10])")
    print(f"  Lags used:      {n_lags}")
    print(f"  Critical values: {critical_values}")

    if p_value < alpha:
        print(f"  CONCLUSION: Reject H0. Series is NON-STATIONARY (unit root present).")
    else:
        print(f"  CONCLUSION: Fail to reject H0. Series is STATIONARY.")

    return {"kpss_stat": kpss_stat, "p_value": p_value, "stationary": p_value >= alpha}

# Combining ADF and KPSS
# Both say stationary        → Strong evidence of stationarity
# Both say non-stationary    → Strong evidence; needs differencing
# ADF: stationary, KPSS: non-stationary → Trend-stationary (remove trend)
# ADF: non-stationary, KPSS: stationary → Difference-stationary (difference it)
```

### Making a Non-Stationary Series Stationary

```python
import numpy as np
import matplotlib.pyplot as plt

series = df["value"]

# 1. First-order differencing (removes linear trend)
diff1 = series.diff().dropna()

# 2. Log transform (stabilizes variance for exponential growth)
log_series = np.log(series.replace(0, np.nan))

# 3. Log + differencing (common for financial/economic data)
log_diff = log_series.diff().dropna()

# 4. Seasonal differencing (removes annual seasonality in monthly data)
# For monthly data with annual seasonality:
seasonal_diff = series.diff(12).dropna()

# 5. Double differencing (second-order non-stationarity)
diff2 = series.diff().diff().dropna()

# Test each until stationary — prefer the fewest differences needed
for name, s in [("Original", series), ("Diff(1)", diff1), ("Log", log_series), ("Log+Diff", log_diff)]:
    test_stationarity_adf(s, name=name)

# Plot to visually confirm
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
for ax, (name, s) in zip(axes, [("Original", series), ("Diff(1)", diff1),
                                  ("Log", log_series), ("Log+Diff", log_diff)]):
    ax.plot(s)
    ax.set_title(name)
    ax.set_xlabel("Time")
plt.tight_layout()
plt.savefig("stationarity_transforms.png", dpi=150)
```

---

## Decomposition

### When to Use Additive vs Multiplicative

- **Additive** (trend + seasonal + residual): seasonal fluctuations are constant in magnitude. Use when the series is roughly linear.
- **Multiplicative** (trend × seasonal × residual): seasonal fluctuations grow with the level. Use when the series grows exponentially. Can be applied as `log → additive decomposition → exp`.

### Classical Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# period: number of observations per seasonal cycle
# e.g., 12 for monthly data with annual seasonality, 7 for daily with weekly seasonality
result = seasonal_decompose(df["value"], model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
result.observed.plot(ax=axes[0], title="Observed")
result.trend.plot(ax=axes[1], title="Trend")
result.seasonal.plot(ax=axes[2], title="Seasonal")
result.resid.plot(ax=axes[3], title="Residual")
plt.tight_layout()
plt.savefig("decomposition_classical.png", dpi=150)
plt.show()

# Residual diagnostics
resid = result.resid.dropna()
print(f"Residual mean:   {resid.mean():.4f}  (should be ~0)")
print(f"Residual std:    {resid.std():.4f}")
```

### STL Decomposition (Robust, Recommended)

STL (Seasonal-Trend decomposition using LOESS) is more robust to outliers and handles non-constant seasonality.

```python
from statsmodels.tsa.seasonal import STL

stl = STL(
    df["value"],
    period=12,          # seasonal period
    seasonal=13,        # seasonal smoother window (odd, >= 7)
    trend=None,         # auto-selected based on period
    robust=True         # downweight outliers in LOESS
)
result = stl.fit()

fig = result.plot()
fig.set_size_inches(12, 10)
plt.tight_layout()
plt.savefig("decomposition_stl.png", dpi=150)

# Extract components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Seasonal strength: how much variance is explained by the seasonal component
var_resid = residual.var()
var_seasonal_plus_resid = (seasonal + residual).var()
seasonal_strength = max(0, 1 - var_resid / var_seasonal_plus_resid)
print(f"Seasonal strength: {seasonal_strength:.3f}  (>0.6 = strong seasonality)")
```

### Detecting Seasonality: ACF and FFT

```python
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.pyplot as plt

# ACF approach: peaks at multiples of the seasonal period
fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(df["value"].dropna(), lags=60, ax=ax)
ax.set_title("ACF — look for peaks at lag s, 2s, 3s...")
plt.savefig("acf_seasonality.png", dpi=150)

# FFT periodogram approach
series_values = df["value"].dropna().values
n = len(series_values)
fft_vals = np.abs(np.fft.rfft(series_values - series_values.mean())) ** 2
freqs = np.fft.rfftfreq(n)

# Find dominant frequency
dominant_freq_idx = np.argmax(fft_vals[1:]) + 1  # skip DC component
dominant_period = 1 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] > 0 else np.inf
print(f"Dominant period from FFT: {dominant_period:.1f} time steps")

plt.figure(figsize=(12, 4))
plt.plot(1/freqs[1:], fft_vals[1:])
plt.xlabel("Period (time steps)")
plt.ylabel("Power")
plt.title("FFT Periodogram")
plt.xlim(0, n // 2)
plt.savefig("fft_periodogram.png", dpi=150)
```

---

## Autocorrelation Analysis

Autocorrelation analysis reveals the temporal memory structure of the series and guides model order selection.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# ACF: lag at which autocorrelation cuts off → MA(q) order
# PACF: lag at which partial autocorrelation cuts off → AR(p) order

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df["value"].dropna(), lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title("ACF — indicates MA order (q): note where it drops inside confidence band")

plot_pacf(df["value"].dropna(), lags=40, ax=axes[1], alpha=0.05, method="ywmle")
axes[1].set_title("PACF — indicates AR order (p): note where it drops inside confidence band")

plt.tight_layout()
plt.savefig("acf_pacf.png", dpi=150)
plt.show()

# ACF / PACF interpretation guide:
# AR(p):   ACF decays exponentially/sinusoidally; PACF cuts off at lag p
# MA(q):   PACF decays; ACF cuts off at lag q
# ARMA:    Both decay gradually → use AIC/BIC to select (p, q)

# Ljung-Box test: are residuals white noise?
# Apply to residuals AFTER fitting a model
lb_result = acorr_ljungbox(df["value"].dropna(), lags=[10, 20, 30], return_df=True)
print(lb_result)
# If p-value > 0.05 at all lags → residuals are white noise → good model fit
# If p-value < 0.05 → model has not captured all autocorrelation structure
```

---

## Forecasting Method Selection

```
What are the properties of your series?
│
├─ UNIVARIATE + SIMPLE PATTERN + LINEAR TREND
│  └─ Holt-Winters (Exponential Smoothing)
│     Good for: well-behaved series with trend and/or seasonality
│     Library: statsmodels.tsa.holtwinters.ExponentialSmoothing
│
├─ UNIVARIATE + COMPLEX / MULTIPLE SEASONALITIES + BUSINESS DATA
│  └─ Prophet (Meta/Facebook)
│     Good for: daily business data, holidays, missing data
│     Library: prophet (pip install prophet)
│
├─ UNIVARIATE + STATIONARY OR DIFFERENCED
│  └─ ARIMA / SARIMA
│     Good for: classic time series, medium sample sizes
│     Library: statsmodels.tsa.statespace.sarimax.SARIMAX
│     Auto-selection: pmdarima.auto_arima
│
├─ MULTIVARIATE + SHORT-TO-MEDIUM HORIZON
│  └─ SARIMAX (with exogenous) or VAR (Vector Autoregression)
│     Good for: multiple correlated series, known external drivers
│     Library: statsmodels.tsa.statespace.sarimax.SARIMAX
│
├─ LARGE DATASET + DEEP LEARNING PREFERRED
│  ├─ LSTM (classic RNN approach)
│  │  Library: PyTorch / TensorFlow
│  └─ Temporal Fusion Transformer (TFT) — state-of-the-art
│     Library: pytorch-forecasting
│
└─ TABULAR ML APPROACH (many series, feature engineering preferred)
   └─ LightGBM / XGBoost with lag features
      Library: lightgbm, tsfresh (for automated feature extraction)
```

### Holt-Winters (Exponential Smoothing)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    df["value"],
    trend="add",           # "add", "mul", or None
    seasonal="add",        # "add", "mul", or None
    seasonal_periods=12,   # periods per seasonal cycle
    damped_trend=True      # recommended to prevent trend extrapolation
)
fitted = model.fit(optimized=True)
print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=12)
print(forecast)

# AIC for model comparison
print(f"AIC: {fitted.aic:.2f}")
```

### Prophet

```python
# pip install prophet
from prophet import Prophet
import pandas as pd

# Prophet requires columns 'ds' (datetime) and 'y' (value)
df_prophet = df[["value"]].reset_index()
df_prophet.columns = ["ds", "y"]

model = Prophet(
    changepoint_prior_scale=0.05,    # flexibility of trend (increase for more flexibility)
    seasonality_prior_scale=10.0,    # strength of seasonality
    holidays_prior_scale=10.0,
    seasonality_mode="additive",     # or "multiplicative"
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add custom seasonality (e.g., monthly)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

# Add holidays
from prophet.make_holidays import make_holidays_df
# model.add_country_holidays(country_name="US")

model.fit(df_prophet)

# Make future dataframe and predict
future = model.make_future_dataframe(periods=30, freq="D")
forecast = model.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

# Plot
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
```

### SARIMA with Auto-Selection

```python
# pip install pmdarima
import pmdarima as pm

# Auto-select (p, d, q)(P, D, Q, s) via AIC
auto_model = pm.auto_arima(
    df["value"],
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,              # auto-determine differencing
    D=1,                 # seasonal differencing
    seasonal=True,
    m=12,                # seasonal period
    information_criterion="aic",
    stepwise=True,       # faster than exhaustive search
    trace=True,
    error_action="ignore",
    suppress_warnings=True
)
print(auto_model.summary())
forecast, conf_int = auto_model.predict(n_periods=12, return_conf_int=True)
```

### LightGBM with Lag Features

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def create_lag_features(df, target_col, lags, rolling_windows):
    """Create lag and rolling features for tabular ML forecasting."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df[target_col].shift(1).rolling(window).std()
    # Calendar features
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    return df

df_features = create_lag_features(
    df, "value",
    lags=[1, 2, 3, 6, 12, 24],
    rolling_windows=[3, 6, 12]
)
df_features = df_features.dropna()

feature_cols = [c for c in df_features.columns if c != "value"]
X = df_features[feature_cols]
y = df_features["value"]

# Temporal split — NEVER shuffle
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(50, verbose=False)])

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

---

## Temporal Cross-Validation

**CRITICAL: Never shuffle time series data before splitting. This causes data leakage.**

### Concepts

- **Expanding window**: train on all data up to time t, test on [t, t+h]. Train set grows with each fold.
- **Sliding window**: train on a fixed window [t-w, t], test on [t, t+h]. Used when older data is less relevant.
- **Forecast horizon h**: how many steps ahead you are predicting. Must be consistent across all folds.

### sklearn TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

X = df_features[feature_cols].values
y = df_features["value"].values

# Expanding window (default)
tscv = TimeSeriesSplit(n_splits=5, gap=0)

print("TimeSeriesSplit fold sizes:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"  Fold {fold+1}: train={len(train_idx)}, test={len(test_idx)}, "
          f"train_end={train_idx[-1]}, test_start={test_idx[0]}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Fit and evaluate model here...

# With a gap (avoids look-ahead via lag features)
tscv_gap = TimeSeriesSplit(n_splits=5, gap=1, test_size=30)
```

### Manual Expanding Window for Forecasting Models

```python
def temporal_cv_expanding(series, model_fn, n_splits=5, forecast_horizon=12):
    """
    Expanding window cross-validation for time series forecasting.
    model_fn: callable(train_series) → forecast array of length forecast_horizon
    """
    n = len(series)
    min_train = n // (n_splits + 1)
    step = (n - min_train - forecast_horizon) // n_splits

    results = []
    for fold in range(n_splits):
        train_end = min_train + fold * step
        test_end = train_end + forecast_horizon
        if test_end > n:
            break

        train = series.iloc[:train_end]
        actual = series.iloc[train_end:test_end].values

        forecast = model_fn(train)
        errors = actual - forecast

        results.append({
            "fold": fold + 1,
            "train_size": len(train),
            "mse": np.mean(errors ** 2),
            "mae": np.mean(np.abs(errors)),
            "rmse": np.sqrt(np.mean(errors ** 2)),
        })

    return pd.DataFrame(results)
```

---

## Evaluation Metrics

All metrics below assume `y_true` and `y_pred` are numpy arrays of actual and predicted values.

```python
import numpy as np

def rmse(y_true, y_pred):
    """Root Mean Squared Error — penalizes large errors. Same units as target."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error — robust to outliers. Same units as target."""
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred, eps=1e-8):
    """
    Mean Absolute Percentage Error — scale-independent percentage error.
    WARNING: undefined (or extreme) when y_true is near zero.
    """
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def smape(y_true, y_pred, eps=1e-8):
    """
    Symmetric MAPE — bounded [0, 200%], more symmetric than MAPE.
    Better than MAPE when actuals can be small, but still sensitive near zero.
    """
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

def mase(y_true, y_pred, y_train, seasonality=1):
    """
    Mean Absolute Scaled Error — scale-independent, compares to naive seasonal forecast.
    MASE < 1: better than naive; MASE > 1: worse than naive. Robust to zero actuals.
    seasonality: 1 for non-seasonal naive, or m for seasonal naive (lag-m forecast).
    """
    naive_errors = np.abs(np.diff(y_train, n=seasonality))
    mae_naive = np.mean(naive_errors)
    if mae_naive == 0:
        return np.nan
    return mae(y_true, y_pred) / mae_naive

# Example usage
y_true = np.array([100, 110, 105, 115, 120])
y_pred = np.array([102, 108, 107, 113, 122])
y_train = np.array([90, 95, 92, 100, 98, 102, 105])

print(f"RMSE:  {rmse(y_true, y_pred):.4f}")
print(f"MAE:   {mae(y_true, y_pred):.4f}")
print(f"MAPE:  {mape(y_true, y_pred):.2f}%")
print(f"SMAPE: {smape(y_true, y_pred):.2f}%")
print(f"MASE:  {mase(y_true, y_pred, y_train, seasonality=1):.4f}")
```

### Metric Selection Guide

| Metric | When to Use | Limitation |
|---|---|---|
| RMSE | Default regression metric; penalizes large errors | Sensitive to outliers; unit-dependent |
| MAE | When outliers should not dominate | Unit-dependent; less sensitive to large errors |
| MAPE | Reporting in percentage terms | Undefined/extreme when actual ≈ 0 |
| SMAPE | Alternative to MAPE with small actuals | Can be misleading; still poor near zero |
| MASE | Multi-series comparison, scale invariance | Requires training data for naive baseline |

**Recommendation**: Always report RMSE + MAE + MASE. MASE is the preferred metric for comparing across series of different scales (e.g., M-competition).

---

## Scripts

Use `scripts/ts_profiler.py` for automated profiling of any CSV time series dataset. See that script for full usage.

---

## Best Practices

1. **Never shuffle time series** — always maintain temporal order in train/test splits.
2. **Test stationarity before fitting ARIMA** — use both ADF and KPSS for a complete picture.
3. **Choose decomposition model type based on variance** — if seasonal variance grows with level, use multiplicative (or log-transform then additive).
4. **Check residuals after fitting** — Ljung-Box test on residuals should not reject white noise.
5. **Use MASE for fair comparison across series** — MAPE breaks down when actuals include zeros.
6. **Use expanding window CV** by default — it simulates the real forecasting scenario.
7. **Set a gap in CV** equal to the forecast horizon to prevent look-ahead via lag features.
8. **Report confidence/prediction intervals** — a point forecast without uncertainty is incomplete.
9. **Use a naive baseline** (last value, seasonal naive) to calibrate MASE and establish a floor.
10. **Plot your series first** — trend, seasonality, and anomalies are often obvious visually.
