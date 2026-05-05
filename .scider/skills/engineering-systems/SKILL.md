---
name: engineering-systems
description: Engineering systems analysis — control theory (PID, transfer functions, Bode plots), signal processing, reliability engineering, engineering optimization (LP/MIP), sensor data processing, and FEA concepts. Use when working with engineering, control, or sensor data.
allowed_agents: [data, experiment]
---

# Engineering Systems

## Overview

This skill covers computational engineering: control systems design and analysis, signal processing, reliability engineering, mathematical optimization, and sensor data processing. Use it for engineering research problems involving dynamic systems, sensor streams, or system optimization.

## When to Use This Skill

- Analyzing or designing control systems (PID, feedback loops)
- Processing sensor time series (filtering, calibration, anomaly detection)
- Running engineering optimization (LP, MIP, scheduling)
- Assessing system reliability or failure analysis
- Working with FEA simulation outputs

---

## 1. Control Systems

### Transfer Functions and System Analysis

```python
import control
import numpy as np
import matplotlib.pyplot as plt

# Define transfer function G(s) = 1 / (s² + 2s + 1)
G = control.tf([1], [1, 2, 1])
print(G)

# Step response
t, y = control.step_response(G)
rise_time = t[np.argmax(y >= 0.1 * y[-1])]  # 10% rise time (approx)
settling_idx = np.where(np.abs(y - y[-1]) > 0.02 * y[-1])[0]
settling_time = t[settling_idx[-1]] if len(settling_idx) > 0 else 0

print(f"Steady-state value: {y[-1]:.3f}")
print(f"Rise time (10%): {rise_time:.3f} s")
print(f"Settling time (2%): {settling_time:.3f} s")
print(f"Overshoot: {(y.max() - y[-1]) / y[-1] * 100:.1f}%")

# Bode plot: gain margin and phase margin
gm, pm, wcg, wcp = control.margin(G)
print(f"Gain margin: {20*np.log10(gm):.1f} dB  (should be > 6 dB for stability)")
print(f"Phase margin: {pm:.1f}°  (should be > 30° for robustness)")

# Poles and zeros (stability: all poles must have Re < 0 for stability)
poles = control.poles(G)
zeros = control.zeros(G)
stable = all(p.real < 0 for p in poles)
print(f"Poles: {poles}  |  Stable: {stable}")
```

### PID Controller Design

```python
# PID controller: C(s) = Kp + Ki/s + Kd*s
# Closed-loop system: T(s) = C(s)*G(s) / (1 + C(s)*G(s))

def tune_pid_ziegler_nichols(Ku: float, Tu: float) -> dict:
    """Ziegler-Nichols tuning from ultimate gain Ku and period Tu.
    Ku: ultimate gain (gain at stability boundary)
    Tu: ultimate period (oscillation period at Ku)
    """
    return {
        "P":   {"Kp": 0.5 * Ku,      "Ki": 0,                 "Kd": 0},
        "PI":  {"Kp": 0.45 * Ku,     "Ki": 0.54 * Ku / Tu,    "Kd": 0},
        "PID": {"Kp": 0.6 * Ku,      "Ki": 1.2 * Ku / Tu,     "Kd": 3 * Ku * Tu / 40},
    }

# Build and simulate closed-loop PID
def make_pid_controller(Kp, Ki, Kd):
    """Return PID transfer function."""
    # C(s) = Kd*s² + Kp*s + Ki) / s
    return control.tf([Kd, Kp, Ki], [1, 0])

G_plant = control.tf([1], [1, 3, 2])  # Example plant
params = tune_ziegler_nichols = {"Kp": 3.0, "Ki": 1.5, "Kd": 0.5}  # Example values
C = make_pid_controller(**params)
T_cl = control.feedback(C * G_plant, 1)  # Closed-loop
t_cl, y_cl = control.step_response(T_cl)
```

### State-Space Representation

```python
import numpy as np

# dx/dt = Ax + Bu
# y = Cx + Du
A = np.array([[0, 1], [-2, -3]])  # system matrix
B = np.array([[0], [1]])           # input matrix
C = np.array([[1, 0]])             # output matrix
D = np.array([[0]])                # feedthrough

sys = control.ss(A, B, C, D)
print(f"Eigenvalues (poles): {np.linalg.eigvals(A)}")
print(f"Controllable: {np.linalg.matrix_rank(control.ctrb(A, B)) == A.shape[0]}")
print(f"Observable: {np.linalg.matrix_rank(control.obsv(A, C)) == A.shape[0]}")
```

---

## 2. Signal Processing

```python
from scipy import signal
import numpy as np

fs = 1000  # Hz
t = np.arange(0, 5, 1/fs)
# Signal: 10 Hz + 120 Hz (noise) + random noise
raw = (np.sin(2*np.pi*10*t) + 0.3*np.sin(2*np.pi*120*t)
       + 0.5*np.random.randn(len(t)))

# ── Filter Design ─────────────────────────────────────────────
# Butterworth lowpass (maximally flat passband)
sos_lp = signal.butter(N=4, Wn=50, btype="low", fs=fs, output="sos")
filtered_lp = signal.sosfiltfilt(sos_lp, raw)  # zero-phase

# Bandpass filter (8–12 Hz, alpha band example)
sos_bp = signal.butter(N=4, Wn=[8, 12], btype="band", fs=fs, output="sos")
filtered_bp = signal.sosfiltfilt(sos_bp, raw)

# ── FFT Analysis ─────────────────────────────────────────────
N = len(raw)
freqs = np.fft.rfftfreq(N, 1/fs)
fft_mag = np.abs(np.fft.rfft(raw * np.hanning(N)))  # Hanning window

# Find dominant frequencies
peak_idx = np.argsort(fft_mag)[-5:][::-1]
print("Top 5 frequencies:")
for i in peak_idx:
    print(f"  {freqs[i]:.1f} Hz: amplitude {fft_mag[i]:.2f}")

# ── Power Spectral Density (Welch) ────────────────────────────
freq_psd, psd = signal.welch(raw, fs=fs, nperseg=256, noverlap=128)
# More reliable than single FFT; averages over segments

# ── Spectrogram (time-frequency) ─────────────────────────────
freq_s, time_s, Sxx = signal.spectrogram(raw, fs=fs, nperseg=128, noverlap=64)
```

**Filter design rules**:
- Always use `sosfiltfilt` (not `lfilter`) for zero-phase filtering (no temporal distortion)
- Apply `sosfilt` when real-time processing is needed (causal, but introduces phase lag)
- Butterworth: flat passband, gentle rolloff — good default
- Chebyshev: steeper rolloff, ripple in passband/stopband — when sharp cutoff matters

---

## 3. Reliability Engineering

```python
import math
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

# Weibull distribution for failure time analysis
# Parameters: shape β (failure mode) and scale η (characteristic life)
# β < 1: decreasing failure rate (infant mortality, defects)
# β = 1: constant failure rate (random failures — exponential distribution)
# β > 1: increasing failure rate (wear-out)

failure_times = np.array([120, 240, 350, 180, 420, 95, 310, 280, 150, 390])  # hours

# Fit Weibull distribution
beta, _, eta = weibull_min.fit(failure_times, floc=0)
print(f"Shape (β): {beta:.2f}  ({'wear-out' if beta>1 else 'early failure' if beta<1 else 'random'})")
print(f"Scale (η): {eta:.1f} hours (characteristic life)")
print(f"MTBF = η·Γ(1+1/β) = {eta * math.gamma(1 + 1/beta):.1f} hours")

# Reliability function R(t) = exp(-(t/η)^β) = P(failure > t)
def reliability(t, beta, eta):
    return np.exp(-(t / eta) ** beta)

t_range = np.linspace(0, 600, 300)
R = reliability(t_range, beta, eta)

# B10 life: time at which 10% of units have failed
from scipy.optimize import brentq
b10 = brentq(lambda t: reliability(t, beta, eta) - 0.90, 0.1, 10000)
print(f"B10 life (10% failure): {b10:.1f} hours")
```

### Fault Tree Analysis (Logic)

```python
# AND gate: system fails if ALL subsystems fail
# OR gate: system fails if ANY subsystem fails

def system_reliability_and(*component_reliabilities):
    """AND gate: all must work for system to work."""
    result = 1.0
    for r in component_reliabilities:
        result *= r
    return result

def system_reliability_or(*component_reliabilities):
    """OR gate: at least one must fail for system to fail (redundant system)."""
    failure_prob = 1.0
    for r in component_reliabilities:
        failure_prob *= (1 - r)
    return 1 - failure_prob  # system reliability

# Example: 2-out-of-3 redundant system
R_comp = 0.95
R_redundant = system_reliability_or(R_comp, R_comp, R_comp)
print(f"2-of-3 redundant system reliability: {R_redundant:.4f}")
```

---

## 4. Engineering Optimization (LP / MIP)

```python
import pulp

# Production planning LP
# Maximize: 25x + 30y (profit)
# Subject to: 2x + y ≤ 100 (labor hours), x + 2y ≤ 80 (materials), x,y ≥ 0

prob = pulp.LpProblem("Production Planning", pulp.LpMaximize)
x = pulp.LpVariable("Product_A", lowBound=0)
y = pulp.LpVariable("Product_B", lowBound=0)

prob += 25*x + 30*y              # objective
prob += 2*x + y <= 100           # labor constraint
prob += x + 2*y <= 80            # materials constraint

prob.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"Status: {pulp.LpStatus[prob.status]}")
print(f"Product A: {x.value():.1f}, Product B: {y.value():.1f}")
print(f"Profit: ${prob.objective.value():.2f}")

# For MIP (integer variables):
# x_int = pulp.LpVariable("x", lowBound=0, cat="Integer")
# y_bin = pulp.LpVariable("y", cat="Binary")  # 0/1 decision

# Google OR-Tools for more complex scheduling/routing:
# from ortools.sat.python import cp_model
```

---

## 5. Sensor Data Processing

```python
import numpy as np
import pandas as pd
from scipy import stats, signal

# ── Calibration ───────────────────────────────────────────────
def linear_calibrate(raw_values: np.ndarray, cal_points: list) -> np.ndarray:
    """Two-point calibration: cal_points = [(raw1, true1), (raw2, true2)]."""
    raw1, true1 = cal_points[0]
    raw2, true2 = cal_points[1]
    slope = (true2 - true1) / (raw2 - raw1)
    intercept = true1 - slope * raw1
    return slope * raw_values + intercept

# ── Outlier Detection ─────────────────────────────────────────
def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """IQR-based outlier detection. Returns boolean mask (True = outlier)."""
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    return (data < Q1 - k*IQR) | (data > Q3 + k*IQR)

def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Z-score outlier detection."""
    z = np.abs(stats.zscore(data))
    return z > threshold

# ── Drift Correction ──────────────────────────────────────────
def polynomial_detrend(signal_values: np.ndarray, degree: int = 2) -> np.ndarray:
    """Remove polynomial trend (drift) from signal."""
    t = np.arange(len(signal_values))
    coeffs = np.polyfit(t, signal_values, degree)
    trend = np.polyval(coeffs, t)
    return signal_values - trend

# ── Synchronization of Multiple Sensors ──────────────────────
def synchronize_sensors(sensor_dfs: list, target_freq_hz: float) -> pd.DataFrame:
    """Resample and align multiple sensor streams to common time axis."""
    period = f"{int(1000/target_freq_hz)}ms"
    resampled = []
    for df in sensor_dfs:
        df.index = pd.to_datetime(df.index)
        resampled.append(df.resample(period).interpolate("linear"))
    return pd.concat(resampled, axis=1).dropna()
```

---

## 6. Finite Element Analysis Concepts

### Key Checks for FEA Results

```python
# Mesh quality metrics (compute from mesh data)
def aspect_ratio(element_lengths):
    """Aspect ratio: max_edge / min_edge. Should be < 5 for good mesh."""
    return max(element_lengths) / min(element_lengths)

# Convergence study: run at multiple mesh refinements
mesh_sizes = [0.1, 0.05, 0.025, 0.01]  # element size
max_stress = [245, 312, 341, 347]       # MPa (example)

# Plot convergence: result should plateau
import numpy as np
rel_change = np.abs(np.diff(max_stress)) / np.abs(max_stress[:-1]) * 100
print(f"Change with mesh refinement (%): {rel_change}")
# Accept when change < 5% with further refinement
```

**FEA best practices**:
- Always run a mesh convergence study (refine until solution changes < 5%)
- Check reaction forces balance applied loads (global equilibrium check)
- Stress singularities at sharp corners → use fillets or accept as mathematical artifact
- Symmetry boundary conditions reduce model size but must match actual symmetry
- Report: element type, mesh size at critical regions, solver tolerance, convergence criterion
