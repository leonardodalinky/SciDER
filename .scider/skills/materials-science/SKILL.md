---
name: materials-science
description: Materials science analysis — crystal structure, phase diagrams, mechanical and electronic properties, characterization techniques (XRD, SEM/TEM, AFM, XPS), and high-throughput computational workflows with pymatgen. Use when working with materials data.
allowed_agents: [data, experiment]
---

# Materials Science

## Overview

This skill covers computational and experimental materials science: crystal structure analysis, property prediction, characterization data interpretation, and high-throughput DFT workflows. It complements the `chemistry-analysis` skill (which focuses on molecular systems) — this skill is for extended solids and bulk materials.

## When to Use This Skill

- Analyzing crystal structures (CIF files, XRD data)
- Interpreting characterization data (XRD, SEM, TEM, XPS, AFM)
- Computing or interpreting mechanical/electronic properties from DFT
- High-throughput screening using the Materials Project database

---

## 1. Crystal Structure

### Loading and Analyzing Structures with pymatgen

```python
from pymatgen.core import Structure, Lattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser

# Load from CIF file
parser = CifParser("structure.cif")
structure = parser.parse_structures(primitive=True)[0]

# Basic properties
print(f"Formula: {structure.formula}")
print(f"Lattice: a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f}, c={structure.lattice.c:.3f}")
print(f"Angles: α={structure.lattice.alpha:.2f}, β={structure.lattice.beta:.2f}, γ={structure.lattice.gamma:.2f}")
print(f"Volume: {structure.lattice.volume:.3f} Å³")
print(f"Density: {structure.density:.3f} g/cm³")

# Symmetry analysis
sga = SpacegroupAnalyzer(structure)
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")
print(f"Crystal system: {sga.get_crystal_system()}")
print(f"Point group: {sga.get_point_group_symbol()}")

# Get conventional standard structure
conv_structure = sga.get_conventional_standard_structure()

# Nearest neighbor analysis
for i, site in enumerate(structure.sites[:3]):
    neighbors = structure.get_neighbors(site, r=3.5)
    print(f"Site {i} ({site.specie}): {len(neighbors)} neighbors within 3.5 Å")
```

### Miller Indices and Interplanar Spacing

```python
import numpy as np

def bragg_d_spacing(two_theta_deg: float, wavelength_angstrom: float = 1.5406) -> float:
    """Compute d-spacing from Bragg's law: nλ = 2d sinθ (n=1)."""
    theta_rad = np.deg2rad(two_theta_deg / 2)
    return wavelength_angstrom / (2 * np.sin(theta_rad))

def bragg_two_theta(d_angstrom: float, wavelength_angstrom: float = 1.5406) -> float:
    """Compute 2θ peak position from d-spacing."""
    sin_theta = wavelength_angstrom / (2 * d_angstrom)
    if abs(sin_theta) > 1:
        return None  # no peak at this wavelength
    return 2 * np.rad2deg(np.arcsin(sin_theta))

# Example: Cu Kα radiation (λ = 1.5406 Å)
peaks = [(38.2, "Au (111)"), (44.4, "Au (200)"), (64.6, "Au (220)")]
for two_theta, label in peaks:
    d = bragg_d_spacing(two_theta)
    print(f"{label}: 2θ={two_theta}°, d={d:.3f} Å")
```

---

## 2. Phase Diagrams

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry

# Build a phase diagram from computed DFT energies
entries = [
    ComputedEntry("Li", -1.9),     # Li energy per atom (eV)
    ComputedEntry("Fe", -8.5),     # Fe energy per atom
    ComputedEntry("LiFe", -10.8),  # LiFe (50:50) total energy / 2 atoms
    ComputedEntry("LiFe2", -19.1), # LiFe2 total energy / 3 atoms
]

pd = PhaseDiagram(entries)
plotter = PDPlotter(pd, show_unstable=0.05)
plotter.get_plot().show()

# Check stability of a phase
from pymatgen.core.composition import Composition
comp = Composition("LiFe")
e_hull = pd.get_e_above_hull(ComputedEntry(comp, -10.8))
print(f"Energy above hull: {e_hull:.3f} eV/atom")
print(f"Stable: {e_hull < 0.025}")  # < 25 meV/atom is conventionally stable
```

**Reading phase diagrams**:
- Points on the convex hull → thermodynamically stable phases
- Points above the hull → unstable against decomposition to hull phases
- Two-phase regions: compositions between stable phases are mixtures (Lever rule)

---

## 3. Mechanical Properties

```python
from pymatgen.analysis.elasticity import ElasticTensor
import numpy as np

# If you have the full 6×6 elastic constant tensor (in GPa) from DFT:
# Cij in Voigt notation [C11, C22, C33, C44, C55, C66, C12, C13, C23, ...]
# Example: cubic system
C11, C12, C44 = 250, 150, 120  # GPa (hypothetical)

# For cubic: build full Voigt tensor
Cij = np.zeros((6, 6))
Cij[[0,1,2],[0,1,2]] = C11
Cij[0,1] = Cij[1,0] = Cij[0,2] = Cij[2,0] = Cij[1,2] = Cij[2,1] = C12
Cij[[3,4,5],[3,4,5]] = C44

et = ElasticTensor.from_voigt(Cij)
print(f"Bulk modulus (Voigt): {et.k_voigt:.1f} GPa")
print(f"Shear modulus (Voigt): {et.g_voigt:.1f} GPa")
print(f"Young's modulus: {et.y_mod:.1f} GPa")
print(f"Poisson's ratio: {et.universal_anisotropy:.3f}")

# Mechanical stability criteria (Born criteria for cubic):
born_stable = (C11 > abs(C12)) and (C11 + 2*C12 > 0) and (C44 > 0)
print(f"Mechanically stable: {born_stable}")
```

### Stress-Strain Curve Interpretation
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_stress_strain(strain, stress):
    """Extract Young's modulus and yield strength from stress-strain data."""
    # Find elastic region (initial linear portion)
    # Use linear fit on first 20% of data as estimate
    n_elastic = max(2, len(strain) // 5)
    E_fit = np.polyfit(strain[:n_elastic], stress[:n_elastic], 1)
    E = E_fit[0]  # Young's modulus in GPa

    # 0.2% offset yield strength
    offset_line = E * (strain - 0.002)
    crossing = np.argwhere(np.diff(np.sign(stress - offset_line))).flatten()
    yield_strength = stress[crossing[0]] if len(crossing) > 0 else stress.max()

    return {"youngs_modulus_GPa": E, "yield_strength_MPa": yield_strength}
```

---

## 4. Materials Characterization

### XRD Analysis

```python
import numpy as np

# Scherrer equation: estimate crystallite size from peak broadening
# D = Kλ / (β cos θ)  where β is FWHM in radians, K ≈ 0.9 (Scherrer constant)
def scherrer_size(fwhm_deg: float, two_theta_deg: float,
                  wavelength: float = 1.5406, K: float = 0.9) -> float:
    """Estimate crystallite size (nm) from XRD peak FWHM."""
    fwhm_rad = np.deg2rad(fwhm_deg)
    theta_rad = np.deg2rad(two_theta_deg / 2)
    return K * wavelength / (fwhm_rad * np.cos(theta_rad)) / 10  # Å → nm

# Example: peak at 2θ=38.2° with FWHM=0.3°
size_nm = scherrer_size(fwhm_deg=0.3, two_theta_deg=38.2)
print(f"Crystallite size: {size_nm:.1f} nm")

# Note: Scherrer gives a lower bound — microstrain also broadens peaks
# For accurate size+strain separation, use Williamson-Hall analysis:
# (β cos θ) / λ = 1/D + 4ε sin θ / λ
```

### SEM/TEM Interpretation Guide

| Mode | What it shows | Use for |
|------|---------------|---------|
| **SE (Secondary Electron)** | Surface topography | Morphology, particle shape, fracture surfaces |
| **BSE (Backscattered Electron)** | Atomic number contrast | Phase identification (heavy atoms = bright) |
| **EDX (Energy Dispersive X-ray)** | Elemental composition | Composition maps, point analysis |
| **SAED (Selected Area Electron Diffraction)** | Crystal structure | Phase identification, orientation |
| **HRTEM (High-Resolution TEM)** | Atomic columns | Interface structure, defects, lattice spacing |

### XPS Binding Energies (Common Elements)

```python
# Key XPS binding energies (eV) for oxidation state identification
XPS_REFERENCE = {
    "Fe": {"Fe0 (metal)": 707.0, "Fe²⁺ (FeO)": 709.5, "Fe³⁺ (Fe₂O₃)": 711.0},
    "Cu": {"Cu0 (metal)": 932.7, "Cu⁺ (Cu₂O)": 933.0, "Cu²⁺ (CuO)": 935.0},
    "Ti": {"Ti0": 453.8, "Ti³⁺": 456.5, "Ti⁴⁺ (TiO₂)": 458.8},
    "C":  {"C-C (aliphatic)": 285.0, "C-O": 286.5, "C=O": 288.0, "O-C=O": 289.0},
    "O":  {"O²⁻ (metal oxide)": 529.5, "OH⁻": 531.0, "H₂O": 532.5},
}
```

---

## 5. High-Throughput Computational Workflows

### Materials Project Database

```python
from mp_api.client import MPRester

# Get API key from https://materialsproject.org/api
with MPRester("YOUR_API_KEY") as mpr:
    # Search for stable oxides of Fe
    docs = mpr.summary.search(
        elements=["Fe", "O"],
        energy_above_hull=(0, 0.025),  # stable within 25 meV/atom
        fields=["material_id", "formula_pretty", "band_gap", "formation_energy_per_atom",
                "energy_above_hull", "structure"],
    )

    for doc in docs[:5]:
        print(f"{doc.formula_pretty}: Eg={doc.band_gap:.2f} eV, "
              f"ΔHf={doc.formation_energy_per_atom:.3f} eV/atom")

    # Get band structure
    bs = mpr.get_bandstructure_by_material_id("mp-19306")  # Fe2O3 (hematite)
    print(f"Direct band gap: {bs.get_direct_band_gap():.2f} eV")
```

### Workflow Automation with atomate2

```python
# atomate2 provides pre-built workflows for common DFT calculations
# Example workflow: structure relaxation → static calculation → band structure

# from atomate2.vasp.flows.core import DoubleRelaxMaker, StaticMaker
# from jobflow import run_locally

# maker = DoubleRelaxMaker()
# flow = maker.make(structure)
# run_locally(flow, root_dir="./calculations")

# For structure screening: use pymatgen + ASE interface
from pymatgen.io.ase import AseAtomsAdaptor
# atoms = AseAtomsAdaptor.get_atoms(structure)
# ... attach calculator (VASP, GPAW, etc.)
```
