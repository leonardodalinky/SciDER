---
name: physics-simulation
description: Physics simulation and computational analysis — ODE/PDE solvers, molecular dynamics, Monte Carlo methods, signal processing, error propagation, and unit management. Use when working with physical simulation data or running computational physics experiments.
allowed_agents: [data, experiment]
---

# Physics Simulation

## Overview

This skill covers computational physics workflows: from solving differential equations and analyzing simulation trajectories to signal processing and rigorous uncertainty quantification.

## When to Use This Skill

- Analyzing or running physical simulations (MD, Monte Carlo, FEM)
- Solving ODEs or PDEs numerically
- Processing experimental physics data (spectra, time series, sensor data)
- Propagating measurement uncertainties
- Working with physical units and dimensional analysis

---

## 1. ODE Solvers with SciPy

```python
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Example: damped harmonic oscillator
# y'' + 2γy' + ω₀²y = 0  →  state vector [y, y']
def damped_oscillator(t, y, gamma, omega0):
    return [y[1], -2*gamma*y[1] - omega0**2 * y[0]]

# Solve
sol = solve_ivp(
    fun=damped_oscillator,
    t_span=(0, 20),
    y0=[1.0, 0.0],         # initial displacement, velocity
    args=(0.1, 2.0),       # gamma, omega0
    method="RK45",         # default, good for non-stiff
    t_eval=np.linspace(0, 20, 500),
    rtol=1e-8, atol=1e-10,
)

if not sol.success:
    print(f"Solver failed: {sol.message}")
```

### Method Selection

| Method | Use when | Notes |
|--------|----------|-------|
| `RK45` | Non-stiff, smooth solutions | Default, good general purpose |
| `RK23` | Non-stiff, less accuracy needed | Faster than RK45 |
| `DOP853` | Non-stiff, high accuracy required | 8th order, fewer function evaluations |
| `Radau` | Stiff systems | Chemical kinetics, circuit simulation |
| `BDF` | Very stiff, large time spans | Implicit, variable order |
| `LSODA` | Unknown stiffness | Auto-switches between stiff/non-stiff |

**Stiff system detection**: If RK45 takes extremely small steps (t_eval coverage is sparse) or `max_step` warning appears → switch to Radau or BDF.

---

## 2. Molecular Dynamics

### Trajectory Analysis with MDAnalysis
```python
import MDAnalysis as mda
from MDAnalysis.analysis import rms, distances

# Load trajectory
u = mda.Universe("topology.tpr", "trajectory.xtc")

# RMSD relative to first frame
backbone = u.select_atoms("backbone")
R = rms.RMSD(backbone, backbone, ref_frame=0)
R.run()
# R.results.rmsd[:, 2] = RMSD values over time

# Radius of gyration over trajectory
Rg = []
for ts in u.trajectory:
    Rg.append(u.atoms.radius_of_gyration())

# Hydrogen bond analysis
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
hbonds = HydrogenBondAnalysis(u, "protein", "protein")
hbonds.run()
```

### LAMMPS Output Parsing
```python
# Read LAMMPS dump file
import numpy as np

def read_lammps_dump(filename):
    frames = []
    with open(filename) as f:
        while True:
            try:
                next(f)  # ITEM: TIMESTEP
                timestep = int(next(f).strip())
                next(f)  # ITEM: NUMBER OF ATOMS
                n_atoms = int(next(f).strip())
                for _ in range(3): next(f)  # box bounds lines
                next(f)  # ITEM: ATOMS header
                atoms = np.array([next(f).split() for _ in range(n_atoms)], dtype=float)
                frames.append({"timestep": timestep, "atoms": atoms})
            except StopIteration:
                break
    return frames
```

---

## 3. Monte Carlo Methods

### Metropolis Algorithm
```python
import numpy as np

def metropolis_mcmc(energy_func, x_init, n_steps, step_size, temperature=1.0):
    """Metropolis-Hastings MCMC sampler."""
    x = x_init.copy()
    E = energy_func(x)
    samples = [x.copy()]
    n_accepted = 0

    for _ in range(n_steps):
        x_prop = x + np.random.normal(0, step_size, size=x.shape)
        E_prop = energy_func(x_prop)
        delta_E = E_prop - E

        # Accept with probability min(1, exp(-ΔE/kT))
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temperature):
            x = x_prop
            E = E_prop
            n_accepted += 1
        samples.append(x.copy())

    acceptance_rate = n_accepted / n_steps
    print(f"Acceptance rate: {acceptance_rate:.2%} (target: 20-50%)")
    return np.array(samples)

# Convergence diagnostics
def autocorrelation_time(samples):
    """Estimate integrated autocorrelation time."""
    n = len(samples)
    mean = samples.mean()
    var = samples.var()
    acf = np.array([np.mean((samples[:n-k] - mean) * (samples[k:] - mean)) / var
                    for k in range(min(100, n//4))])
    # Integrated autocorr time ≈ 1 + 2 * sum(ACF)
    return 1 + 2 * acf[1:][acf[1:] > 0].sum()
```

**Effective sample size** = N / (2 × autocorrelation_time)

---

## 4. Signal Processing

```python
from scipy import signal
import numpy as np

# Generate example signal
fs = 1000  # Hz sampling rate
t = np.arange(0, 1, 1/fs)
clean = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
noisy = clean + 2*np.random.randn(len(t))

# FFT analysis
freqs = np.fft.rfftfreq(len(noisy), 1/fs)
fft = np.abs(np.fft.rfft(noisy))

# Butterworth lowpass filter (< 100 Hz)
b, a = signal.butter(N=4, Wn=100, btype="low", fs=fs)
filtered = signal.sosfiltfilt(signal.butter(4, 100, "low", fs=fs, output="sos"), noisy)

# Power spectral density (Welch method)
freq_psd, psd = signal.welch(noisy, fs=fs, nperseg=256)

# Spectrogram
freq_spec, t_spec, Sxx = signal.spectrogram(noisy, fs=fs, nperseg=128)
```

**Key rules**:
- Nyquist: sample at ≥ 2× the highest frequency of interest
- Zero-phase filtering: use `sosfiltfilt` (not `lfilter`) to avoid phase shift
- Windowing for FFT of finite signals: apply Hanning/Hamming window to reduce spectral leakage
- Welch PSD: more stable estimate than single-shot FFT (averages over segments)

---

## 5. Error Propagation

### With the `uncertainties` Library
```python
from uncertainties import ufloat, umath
import uncertainties.unumpy as unp

# Define measurements with uncertainties
length = ufloat(10.5, 0.1)   # 10.5 ± 0.1 m
width = ufloat(3.2, 0.05)    # 3.2 ± 0.05 m

area = length * width        # automatic error propagation
print(f"Area = {area}")      # Area = 33.6 ± 0.6 m²

# Works with functions
import math
radius = ufloat(5.0, 0.2)
circumference = 2 * umath.pi * radius
```

### χ² Fitting
```python
from scipy.optimize import curve_fit

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

x_data = np.linspace(0, 5, 50)
y_data = 3 * np.exp(-0.7 * x_data) + 0.5 + 0.1 * np.random.randn(50)
y_err = 0.1 * np.ones_like(y_data)

popt, pcov = curve_fit(model, x_data, y_data, sigma=y_err, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties on parameters

# Reduced χ²
residuals = y_data - model(x_data, *popt)
chi2 = np.sum((residuals / y_err)**2)
n_dof = len(y_data) - len(popt)
chi2_reduced = chi2 / n_dof
print(f"Reduced χ² = {chi2_reduced:.2f}  (good fit ≈ 1.0)")
```

**Interpreting reduced χ²**: ≈1.0 → good fit; << 1 → overfitting or overestimated errors; >> 1 → poor fit or underestimated errors.

---

## 6. Units and Dimensional Analysis

```python
from pint import UnitRegistry

ureg = UnitRegistry()
Q = ureg.Quantity

# Define physical quantities
mass = Q(10.0, "kg")
velocity = Q(3.0, "m/s")
kinetic_energy = 0.5 * mass * velocity**2
print(kinetic_energy.to("joule"))         # 45.0 joule
print(kinetic_energy.to("eV"))            # convert to electron volts

# Physical constants
import scipy.constants as const
print(f"Boltzmann: {const.k} J/K")
print(f"Planck: {const.h} J·s")
print(f"Speed of light: {const.c} m/s")
print(f"Avogadro: {const.N_A} mol⁻¹")

# Natural units conversion (particle physics)
# In natural units: ℏ = c = 1, energies in eV
hbar_eV_s = const.hbar / const.e  # ℏ in eV·s
```

**Best practices**:
- Always specify units in variable names or comments when not using `pint`
- Check dimensional consistency before reporting results
- Be explicit about SI vs. CGS vs. natural units in all publications

---

## 7. Quick Diagnostics Script

Run the `scripts/physics_eda.py` script to inspect simulation output files:
```bash
python <SKILL_BASE_DIR>/scripts/physics_eda.py simulation_output.csv --output physics_report.md
```

It checks for NaN/Inf values, implausible ranges, autocorrelation (non-equilibration), and energy conservation drift.
