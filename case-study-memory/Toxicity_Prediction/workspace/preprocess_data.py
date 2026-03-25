"""Data preprocessing: SMILES -> Morgan fingerprints + RDKit descriptors."""

import logging
import os
import pickle

# Suppress RDKit warnings
logging.getLogger("rdkit").setLevel(logging.ERROR)
os.environ["RDKIT_VERBOSE"] = "0"
from pathlib import Path

import numpy as np
import pandas as pd
from utils import RANDOM_STATE, get_data_path

# RDKit 2D descriptors (≥20, validated to exist)
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "TPSA",
    "LabuteASA",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumHeteroatoms",
    "FractionCSP3",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "NumAromaticRings",
    "RingCount",
    "NumAmideBonds",
    "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles",
]


def _get_descriptor_list():
    from rdkit.Chem import Descriptors

    return [n for n in RDKIT_DESCRIPTOR_NAMES if hasattr(Descriptors, n)]


def _get_morgan_fingerprint(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint. Returns None if invalid."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def _get_rdkit_descriptors(smiles: str, desc_names: list) -> np.ndarray | None:
    """Compute RDKit 2D descriptors. Returns None if invalid."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc = [getattr(Descriptors, n)(mol) for n in desc_names]
        arr = np.array(desc, dtype=np.float32)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return None
        return arr
    except Exception:
        return None


def preprocess() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load CSV, compute Morgan (1024-bit) + descriptors, return (X, y, feature_names).
    Drops invalid SMILES.
    """
    df = pd.read_csv(get_data_path())
    df = df.dropna(subset=["Canonical_Smiles"])
    df = df[df["Canonical_Smiles"].astype(str).str.strip() != ""]

    desc_names = _get_descriptor_list()
    fps_list = []
    descs_list = []
    valid_idx = []

    for i, row in df.iterrows():
        smi = str(row["Canonical_Smiles"]).strip()
        fp = _get_morgan_fingerprint(smi)
        desc = _get_rdkit_descriptors(smi, desc_names)
        if fp is not None and desc is not None:
            fps_list.append(fp)
            descs_list.append(desc)
            valid_idx.append(i)

    fps = np.array(fps_list, dtype=np.float32)
    descs = np.array(descs_list, dtype=np.float32)
    X = np.hstack([fps, descs])
    y = df.loc[valid_idx, "Activity"].values
    feature_names = [f"morgan_{i}" for i in range(1024)] + desc_names

    return X, y, feature_names


def save_preprocessed(output_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocess, save to data/, return (X, y, feature_names)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, names = preprocess()
    np.save(output_dir / "features.npy", X)
    with open(output_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(names, f)
    np.save(output_dir / "labels.npy", y)
    return X, y, names
