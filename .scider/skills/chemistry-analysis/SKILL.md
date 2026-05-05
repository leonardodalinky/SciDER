---
name: chemistry-analysis
description: Cheminformatics and computational chemistry — SMILES/InChI parsing, molecular property prediction, spectroscopy interpretation, DFT workflow, materials characterization (XRD, SAXS), and key chemistry databases. Use when analyzing chemical or materials data.
allowed_agents: [data, experiment]
---

# Chemistry Analysis

## Overview

Cheminformatics and computational chemistry span molecular representation, property prediction, spectroscopy interpretation, quantum chemical calculations, and materials characterization. This skill covers the full workflow from parsing SMILES strings through DFT calculations and database queries.

## When to Use This Skill

Use this skill when:
- Parsing, validating, or canonicalizing molecular structures (SMILES, InChI, SDF)
- Computing molecular descriptors (MW, logP, TPSA, fingerprints) for a set of compounds
- Filtering compounds by drug-likeness, ADMET, or structural criteria
- Interpreting spectroscopy data (IR, NMR, MS, UV-Vis)
- Setting up or analyzing DFT/computational chemistry workflows
- Characterizing materials with XRD, SAXS, or BET surface area
- Querying chemistry databases (PubChem, Materials Project, CSD)

For genomics or protein sequence analysis, see the bioinformatics-analysis skill instead.

---

## Cheminformatics with RDKit

### Installation

```bash
conda install -c conda-forge rdkit
# or
pip install rdkit
```

### SMILES/InChI Parsing and Validation

```python
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

# Parse SMILES — returns None if invalid
mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')  # aspirin
if mol is None:
    raise ValueError("Invalid SMILES")

# Canonicalize SMILES (normalize representation)
canonical_smi = Chem.MolToSmiles(mol)

# Convert to InChI and InChIKey
inchi = MolToInchi(mol)
inchikey = Chem.inchi.InchiToInchiKey(inchi)

# Validate a list of SMILES
def validate_smiles(smi_list):
    valid, invalid = [], []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol))  # canonical form
        else:
            invalid.append(smi)
    return valid, invalid
```

### Morgan Fingerprints

Morgan fingerprints (circular fingerprints) are the standard for similarity-based tasks.

```python
from rdkit.Chem import AllChem
from rdkit import DataStructs

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')

# radius=2 ≈ ECFP4, radius=3 ≈ ECFP6; nBits=2048 standard
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Use cases:
# - radius=2 (ECFP4): general similarity, virtual screening
# - radius=3 (ECFP6): more sensitive to local environment differences
# - nBits=1024: faster computation; 2048: fewer bit collisions

# Convert to numpy for ML
import numpy as np
fp_array = np.array(fp)
```

### Molecular Descriptors

```python
from rdkit.Chem import Descriptors, rdMolDescriptors

def compute_descriptors(mol):
    return {
        'MW':             Descriptors.MolWt(mol),
        'ExactMW':        Descriptors.ExactMolWt(mol),
        'logP':           Descriptors.MolLogP(mol),      # Wildman-Crippen
        'TPSA':           Descriptors.TPSA(mol),          # topological polar surface area
        'HBD':            rdMolDescriptors.CalcNumHBD(mol),  # H-bond donors
        'HBA':            rdMolDescriptors.CalcNumHBA(mol),  # H-bond acceptors
        'RotBonds':       rdMolDescriptors.CalcNumRotatableBonds(mol),
        'AromaticRings':  rdMolDescriptors.CalcNumAromaticRings(mol),
        'HeavyAtoms':     mol.GetNumHeavyAtoms(),
        'Rings':          rdMolDescriptors.CalcNumRings(mol),
    }

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
props = compute_descriptors(mol)
```

### Tanimoto Similarity

```python
from rdkit import DataStructs
from rdkit.Chem import AllChem

def tanimoto(smi1, smi2, radius=2, n_bits=2048):
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, n_bits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Bulk similarity against a reference
ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048)
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
sims = DataStructs.BulkTanimotoSimilarity(ref_fp, fps)
```

Rule of thumb: Tanimoto > 0.85 = very similar (often same scaffold); 0.4–0.85 = related; < 0.4 = likely dissimilar.

### Substructure Search

```python
# Search for a functional group using SMARTS
nitro_pattern = Chem.MolFromSmarts('[N+](=O)[O-]')
amide_pattern = Chem.MolFromSmarts('C(=O)N')

mol = Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1')  # paracetamol
has_amide = mol.HasSubstructMatch(amide_pattern)
matches = mol.GetSubstructMatches(amide_pattern)  # atom index tuples
```

### Full Dataset Workflow: Load CSV, Compute Properties, Cluster

```python
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina

# 1. Load and validate
df = pd.read_csv('compounds.csv')
df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
df = df[df['mol'].notna()].copy()
print(f"Valid: {len(df)}/{len(df)} molecules")

# 2. Compute properties
desc_df = df['mol'].apply(lambda m: pd.Series({
    'MW': Descriptors.MolWt(m),
    'logP': Descriptors.MolLogP(m),
    'HBD': rdMolDescriptors.CalcNumHBD(m),
    'HBA': rdMolDescriptors.CalcNumHBA(m),
    'TPSA': Descriptors.TPSA(m),
}))
df = pd.concat([df, desc_df], axis=1)

# 3. Compute fingerprints
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in df['mol']]

# 4. Butina clustering (Taylor-Butina, distance-based)
dists = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])

clusters = Butina.ClusterData(dists, len(fps), distThresh=0.35, isDistData=True)
df['cluster'] = -1
for cluster_id, indices in enumerate(clusters):
    for idx in indices:
        df.loc[df.index[idx], 'cluster'] = cluster_id

print(f"Clusters (cutoff=0.35): {len(clusters)}")
```

---

## Molecular Property Prediction

### Lipinski's Rule of Five (Drug-Likeness)

Oral bioavailability filter for small-molecule drugs:

| Property | Threshold |
|----------|-----------|
| MW | ≤ 500 Da |
| logP (Wildman-Crippen) | ≤ 5 |
| H-bond donors (NH + OH) | ≤ 5 |
| H-bond acceptors (N + O) | ≤ 10 |

Compounds violating ≥ 2 rules are unlikely to be orally bioavailable.

```python
def lipinski_ro5(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)

    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        'MW': mw, 'logP': logp, 'HBD': hbd, 'HBA': hba,
        'violations': violations,
        'drug_like': violations <= 1
    }
```

### Delaney ESOL Solubility Estimate

The Delaney ESOL model predicts aqueous solubility (log S) from simple descriptors:

```
log S = 0.16 - 0.63·clogP - 0.0062·MW + 0.066·RB - 0.74·AP
```

where RB = rotatable bonds, AP = aromatic proportion.

```python
def esol_solubility(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    # aromatic proportion: aromatic atoms / heavy atoms
    n_ar  = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    ap    = n_ar / mol.GetNumHeavyAtoms()
    log_s = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap
    return log_s  # log10(mol/L)
```

### ADMET Properties Overview

| Property | Key Descriptors | Common Models |
|----------|----------------|---------------|
| Absorption (oral) | MW, TPSA, HBD, logP | Lipinski Ro5, Veber rules (TPSA < 140, RB < 10) |
| Distribution (BBB) | MW < 400, logP 1–3, TPSA < 90 | B3DB dataset, DeepSMILES models |
| Metabolism (CYP) | Structural alerts, reactive groups | SMARTS rules, ML classifiers |
| Excretion (t½) | logP, protein binding | QSAR models |
| Toxicity | AMES test, hERG, hepatotox | pkCSM, ADMETlab 2.0 |

For rapid ADMET screening, use `pkCSM` (web API) or the `admet-ai` Python package.

---

## Reaction Informatics

### SMARTS Patterns for Functional Group Detection

```python
# Common SMARTS patterns
patterns = {
    'carboxylic_acid':  '[CX3](=O)[OX2H1]',
    'ester':            '[CX3](=O)[OX2][#6]',
    'primary_amine':    '[NX3;H2;!$(NC=O)]',
    'alcohol':          '[OX2H][#6;!$(C=O)]',
    'aldehyde':         '[CX3H1](=O)',
    'ketone':           '[CX3](=O)([#6])[#6]',
    'amide':            '[CX3](=O)[NX3]',
    'alkene':           '[CX3]=[CX3]',
}

def detect_functional_groups(smi, patterns):
    mol = Chem.MolFromSmiles(smi)
    found = {}
    for name, smarts in patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        found[name] = mol.HasSubstructMatch(patt)
    return found
```

### Retrosynthesis Concepts

Retrosynthetic analysis disconnects a target molecule into simpler precursors:

- **Disconnection strategy**: break at C-C or C-heteroatom bonds formed in known reactions
- **Synthons**: imaginary fragments that map to real reagents (e.g., carbanion → organolithium)
- **FGI** (functional group interconversion): transform one FG into another to reveal a simpler precursor
- **Template-based**: match known reaction SMARTS templates (e.g., ASKCOS, RDKit's `ChemicalReaction`)

```python
from rdkit.Chem import AllChem

# Define a reaction SMARTS (esterification: acid + alcohol → ester)
rxn_smarts = '[CX3:1](=O)[OX2H:2].[OX2H:3][#6:4]>>[CX3:1](=O)[OX2:3][#6:4]'
rxn = AllChem.ReactionFromSmarts(rxn_smarts)

reactants = (Chem.MolFromSmiles('CC(=O)O'), Chem.MolFromSmiles('CCO'))
products = rxn.RunReactants(reactants)
if products:
    print(Chem.MolToSmiles(products[0][0]))
```

---

## Spectroscopy Data Interpretation

### IR Spectroscopy

Key absorption bands (wavenumbers in cm⁻¹):

| Region | Assignment |
|--------|-----------|
| 3200–3600 (broad) | O-H stretch (alcohol, carboxylic acid) |
| 3300–3500 | N-H stretch (amine, amide) |
| 2850–3000 | C-H stretch (alkyl) |
| 2100–2260 | C≡C or C≡N stretch |
| 1700–1750 | C=O stretch (ester ~1735, ketone ~1715, acid ~1710, amide ~1680) |
| 1600–1680 | C=C stretch (alkene, aromatic) |
| 1000–1300 | C-O stretch (fingerprint region) |
| 400–1000 | Fingerprint region — unique to each molecule |

Diagnostic workflow: (1) check 1700-1750 for carbonyl; (2) check 3200-3600 for O-H/N-H; (3) use fingerprint region to confirm identity against library spectra.

### ¹H NMR Chemical Shift Ranges

| δ (ppm) | Proton Type |
|---------|------------|
| 0.0–1.0 | TMS, cyclopropyl |
| 0.8–1.5 | Alkyl (CH₃, CH₂) |
| 1.5–2.5 | α to carbonyl, allylic |
| 2.5–3.5 | α to aromatic, O-CH₃ |
| 3.5–5.0 | O-CH, N-CH, vinyl |
| 5.0–6.5 | Alkene =CH |
| 6.5–8.5 | Aromatic, heteroaromatic |
| 9.0–10.5 | Aldehyde CHO |
| 10–13 | Carboxylic acid, chelated OH |

**Multiplicity (n+1 rule)**: signal splits into n+1 lines where n = number of equivalent neighboring H.
**Integration**: area proportional to number of protons — use to confirm molecular formula.

### Mass Spectrometry

- **M⁺ (molecular ion)**: gives exact molecular weight; may be absent in EI for labile molecules
- **M+1, M+2 peaks**: isotope patterns — 37Cl (M+2 ≈ 1/3 of M⁺), 79Br/81Br (M+2 ≈ M⁺)
- **Base peak**: most abundant fragment; highest m/z considered first
- **Common losses**: 15 (CH₃), 17 (OH), 18 (H₂O), 28 (CO), 29 (CHO), 31 (OCH₃), 35/37 (Cl)
- **High-resolution MS (HRMS)**: gives exact mass to 4+ decimal places → determine molecular formula

```python
# Parse mzML files with pyteomics
from pyteomics import mzml

with mzml.MzML('spectrum.mzML') as reader:
    for spectrum in reader:
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
```

### UV-Vis Spectroscopy

Beer-Lambert law: **A = ε·c·l**
- A = absorbance (dimensionless)
- ε = molar extinction coefficient (L·mol⁻¹·cm⁻¹)
- c = concentration (mol/L)
- l = path length (cm, typically 1 cm)

Common chromophores:

| Chromophore | λ_max (nm) | ε |
|-------------|-----------|---|
| Alkene (π→π*) | ~165 | ~10,000 |
| Conjugated diene | ~217 | ~20,000 |
| Carbonyl (n→π*) | ~280 | ~15 |
| Aromatic | ~254 | ~200 |
| Extended conjugation | 300–500 | 10,000–100,000 |

```python
import numpy as np
import matplotlib.pyplot as plt

# Calculate concentration from absorbance
def beer_lambert(absorbance, epsilon, path_length=1.0):
    """Returns concentration in mol/L"""
    return absorbance / (epsilon * path_length)

# Plot UV-Vis spectrum
wavelength = np.linspace(200, 800, 600)
# absorbance = ...  # from spectrometer
plt.plot(wavelength, absorbance)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('UV-Vis Spectrum')
```

---

## Computational Chemistry Workflow (DFT)

### Step 1: Input Structure Preparation

```python
from rdkit import Chem
from rdkit.Chem import AllChem

smi = 'CC(=O)Oc1ccccc1C(=O)O'
mol = Chem.MolFromSmiles(smi)
mol = Chem.AddHs(mol)  # add explicit hydrogens

# Generate 3D coordinates (ETKDG algorithm, recommended)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

# Force-field optimization (MMFF94 preferred over UFF)
AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94')

# Write to XYZ or SDF
Chem.MolToMolFile(mol, 'input.sdf')
```

### Step 2–5: DFT Calculation Choices

**Functional selection:**
| Functional | Best For |
|-----------|---------|
| B3LYP | General organic chemistry, ground states |
| PBE0 | Improved thermochemistry, UV-Vis |
| M06-2X | Kinetics, non-covalent interactions |
| ωB97X-D | Long-range corrected, weak interactions |
| TPSS | Transition metals |

**Basis set selection:**
| Basis Set | Use Case |
|----------|---------|
| 6-31G* | Quick geometry optimization |
| 6-31+G** | Anions, diffuse functions needed |
| def2-TZVP | High-quality single points |
| def2-QZVP | Benchmark accuracy |
| LANL2DZ | Heavy atoms (ECP for core electrons) |

### ASE Interface for Gaussian/ORCA

```python
from ase import Atoms
from ase.calculators.gaussian import Gaussian
from ase.optimize import BFGS

# Build atoms object
atoms = Atoms('CO2', positions=[[0,0,0],[0,0,1.16],[0,0,-1.16]])

# Set up Gaussian calculator
calc = Gaussian(
    method='B3LYP',
    basis='6-31G*',
    label='co2_opt',
    nprocshared=4,
    mem='4GB',
    opt='',         # geometry optimization keyword
    freq='',        # frequency calculation
)
atoms.calc = calc

# Optimize geometry
opt = BFGS(atoms, logfile='opt.log')
opt.run(fmax=0.01)  # convergence threshold eV/Angstrom

# Extract properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()
```

### Frequency Calculation: Verify Minimum

- No imaginary frequencies (< 0 cm⁻¹): true minimum
- One imaginary frequency: transition state
- Multiple imaginary frequencies: saddle point — re-optimize from perturbed geometry

### Property Extraction

```python
# Parse Gaussian output for HOMO/LUMO
import cclib

data = cclib.io.ccread('calculation.log')
homo_idx = data.homos[0]
homo_energy = data.moenergies[0][homo_idx]        # eV
lumo_energy = data.moenergies[0][homo_idx + 1]    # eV
gap = lumo_energy - homo_energy

dipole = data.moments[1]  # Debye, [x, y, z]
charges = data.atomcharges['mulliken']
```

---

## Materials Characterization

### X-Ray Diffraction (XRD)

**Bragg's law**: nλ = 2d sinθ
- λ: X-ray wavelength (Cu Kα = 1.5406 Å)
- d: interplanar spacing (Å)
- θ: diffraction angle (half of 2θ)

```python
import numpy as np

def bragg_d_spacing(two_theta_deg, wavelength=1.5406):
    """Calculate d-spacing from 2theta (Cu Ka radiation)"""
    theta_rad = np.radians(two_theta_deg / 2)
    return wavelength / (2 * np.sin(theta_rad))

# Scherrer equation: crystallite size from peak broadening
def scherrer_size(two_theta_deg, fwhm_deg, K=0.94, wavelength=1.5406):
    """
    K: Scherrer constant (~0.94 for spherical crystallites)
    fwhm_deg: peak FWHM in degrees
    Returns crystallite size in Angstroms
    """
    theta_rad = np.radians(two_theta_deg / 2)
    fwhm_rad = np.radians(fwhm_deg)
    return (K * wavelength) / (fwhm_rad * np.cos(theta_rad))
```

### pymatgen for Crystal Structure Analysis

```python
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Load crystal structure
struct = Structure.from_file('structure.cif')

# Symmetry analysis
analyzer = SpacegroupAnalyzer(struct)
spacegroup = analyzer.get_space_group_symbol()
crystal_system = analyzer.get_crystal_system()
primitive = analyzer.get_primitive_standard_structure()

# Basic properties
print(f"Formula: {struct.composition.reduced_formula}")
print(f"Volume: {struct.volume:.2f} Å³")
print(f"Space group: {spacegroup}")
print(f"Crystal system: {crystal_system}")
print(f"Lattice: a={struct.lattice.a:.3f} b={struct.lattice.b:.3f} c={struct.lattice.c:.3f}")
```

### SAXS Analysis

Small-angle X-ray scattering probes nanometer-scale structures (1–100 nm).

- **Guinier regime** (low q, qRg < 1.3): `ln I(q) ≈ ln I(0) - Rg²q²/3` → extract Rg (radius of gyration)
- **Porod regime** (high q): I(q) ∝ q⁻⁴ for smooth surfaces; q⁻ᵈ for fractal systems
- **Kratky plot** (q²·I vs q): distinguishes folded (bell-shaped peak) from disordered (plateau) proteins

```python
import numpy as np

def guinier_fit(q, I, q_max_fraction=0.1):
    """Fit Guinier regime to extract Rg and I(0)"""
    from scipy.stats import linregress
    mask = q < q_max_fraction * q.max()
    slope, intercept, r, p, se = linregress(q[mask]**2, np.log(I[mask]))
    Rg = np.sqrt(-3 * slope)
    I0 = np.exp(intercept)
    return Rg, I0, r**2
```

### BET Surface Area

BET (Brunauer-Emmett-Teller) from N₂ adsorption isotherm:

```python
def bet_surface_area(p_p0, n_adsorbed):
    """
    p_p0: relative pressure array (0.05–0.35 for BET linear range)
    n_adsorbed: amount adsorbed (mmol/g or cm³/g STP)
    Returns: BET surface area (m²/g)
    """
    from scipy.stats import linregress
    mask = (p_p0 >= 0.05) & (p_p0 <= 0.35)
    x = p_p0[mask]
    y = x / (n_adsorbed[mask] * (1 - x))  # BET transform

    slope, intercept, *_ = linregress(x, y)
    n_mono = 1 / (slope + intercept)  # monolayer capacity
    c = (slope / intercept) + 1        # BET constant

    # Cross-sectional area of N2 = 0.162 nm²
    # Avogadro = 6.022e23
    sa = n_mono * 6.022e23 * 0.162e-18  # m²/g (if n_mono in mol/g)
    return sa, n_mono, c
```

---

## Thermodynamics

### Gibbs Free Energy

ΔG = ΔH - TΔS

- ΔG < 0: spontaneous at constant T, P
- ΔG > 0: non-spontaneous
- ΔG = 0: equilibrium

```python
def gibbs_free_energy(delta_H_kJ, delta_S_J_per_K, T_K=298.15):
    """
    delta_H: kJ/mol
    delta_S: J/(mol·K)
    Returns delta_G in kJ/mol
    """
    return delta_H_kJ - T_K * (delta_S_J_per_K / 1000)

# Reaction enthalpy from standard formation enthalpies
def reaction_enthalpy(products_hf, reactants_hf):
    """Hess's law: sum(hf_products) - sum(hf_reactants), all in kJ/mol"""
    return sum(products_hf) - sum(reactants_hf)
```

### Free Energy Perturbation (FEP) Basics

FEP estimates binding free energies by alchemically transforming one ligand to another:

- λ goes from 0 (ligand A) to 1 (ligand B)
- ΔΔG_binding = ΔG(B-protein) - ΔG(A-protein) - [ΔG(B-solvent) - ΔG(A-solvent)]
- Use Bennett Acceptance Ratio (BAR) estimator for efficiency
- Requires MD engine (GROMACS, OpenMM, AMBER) and careful system setup (protonation states, water model)

---

## Key Chemistry Databases

### PubChem REST API

```python
import requests

def pubchem_get_properties(smiles, properties='MolecularWeight,IUPACName,XLogP'):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/{properties}/JSON'
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()['PropertyTable']['Properties'][0]

props = pubchem_get_properties('CC(=O)Oc1ccccc1C(=O)O')
print(props)
```

### Materials Project API (`mp-api`)

```python
from mp_api.client import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    # Search for all stable TiO2 phases
    results = mpr.materials.summary.search(
        chemsys='Ti-O',
        energy_above_hull=(0, 0.05),  # eV/atom stability window
        fields=['material_id', 'formula_pretty', 'band_gap', 'formation_energy_per_atom']
    )

    for r in results:
        print(r.material_id, r.formula_pretty, r.band_gap)
```

### ChemSpider

ChemSpider (Royal Society of Chemistry) provides structure search, predicted and experimental spectra, and synonyms. Access via their REST API with a free API key from `chemspider.com`.

### CSD (Cambridge Structural Database)

The CSD contains >1 million small-molecule crystal structures determined by X-ray or neutron diffraction. Access via `ccdc` Python API (license required) or CSD WebCSD for free searches.

---

## Best Practices

1. **Always validate SMILES** before computing properties — `Chem.MolFromSmiles()` returning `None` is a silent failure if not checked
2. **Use canonical SMILES** for storage and comparison (different tools may generate different representations)
3. **Sanitize molecules** (`Chem.SanitizeMol()`) after modifying atoms or bonds
4. **DFT convergence**: always check SCF convergence and geometry optimization convergence criteria
5. **Basis set superposition error (BSSE)**: use counterpoise correction for non-covalent interactions
6. **XRD phase identification**: compare experimental d-spacings to ICDD/JCPDS reference cards

---

## Common Pitfalls

1. **Aromaticity perception**: RDKit may reject structures with incorrect aromaticity — use `Chem.MolFromSmiles(s, sanitize=False)` then `Chem.SanitizeMol()` for fine control
2. **Fingerprint bit collisions**: different substructures can hash to the same bit — higher nBits reduces but does not eliminate this
3. **logP vs logD**: logP is for neutral form; logD = logP adjusted for ionization state at given pH
4. **DFT functional choice**: no single functional is best for all properties — benchmark against experiment or high-level theory
5. **XRD preferred orientation**: peaks may be anomalously intense — check with multiple sample preparations
