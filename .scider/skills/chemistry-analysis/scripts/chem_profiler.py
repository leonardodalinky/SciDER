"""
Chemistry Dataset Profiler — SMILES validity, property distribution, scaffold analysis.

Usage:
    python chem_profiler.py molecules.csv --smiles_col smiles --output chem_report.md
    python chem_profiler.py molecules.csv  # uses 'smiles' column by default
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def check_rdkit():
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False


def compute_properties(smiles_list):
    """Compute molecular properties for a list of SMILES strings."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    results = []
    for smi in smiles_list:
        if not smi or pd.isna(smi):
            results.append(None)
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            results.append(None)
            continue
        results.append({
            "smiles": smi,
            "mw": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "n_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "n_rings": rdMolDescriptors.CalcNumRings(mol),
            "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "lipinski_ok": (
                Descriptors.MolWt(mol) <= 500 and
                Descriptors.MolLogP(mol) <= 5 and
                rdMolDescriptors.CalcNumHBD(mol) <= 5 and
                rdMolDescriptors.CalcNumHBA(mol) <= 10
            ),
        })
    return results


def scaffold_analysis(smiles_list, top_n=10):
    """Murcko scaffold analysis — top scaffolds by frequency."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import Counter

    scaffold_counts = Counter()
    for smi in smiles_list:
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold)
        scaffold_counts[scaffold_smi] += 1

    total = sum(scaffold_counts.values())
    unique_scaffolds = len(scaffold_counts)
    top = scaffold_counts.most_common(top_n)

    return {
        "unique_scaffolds": unique_scaffolds,
        "scaffold_diversity": round(unique_scaffolds / max(total, 1), 3),
        "top_scaffolds": [(smi, count) for smi, count in top],
    }


def similarity_analysis(smiles_list, threshold=0.9, sample_size=500):
    """Sample-based Tanimoto similarity analysis to detect near-duplicates."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    valid = [Chem.MolFromSmiles(str(s)) for s in smiles_list if s and not pd.isna(s)]
    valid = [m for m in valid if m is not None]

    if len(valid) > sample_size:
        import random
        random.seed(42)
        valid = random.sample(valid, sample_size)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in valid]

    similarities = []
    near_dups = 0
    n = len(fps)
    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):  # sample pairs to keep it fast
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
            if sim >= threshold:
                near_dups += 1

    if not similarities:
        return {"error": "insufficient data for similarity analysis"}

    return {
        "mean_similarity": round(float(np.mean(similarities)), 3),
        "median_similarity": round(float(np.median(similarities)), 3),
        "max_similarity": round(float(np.max(similarities)), 3),
        "near_duplicate_pairs": near_dups,
        "threshold_used": threshold,
        "pairs_sampled": len(similarities),
    }


def generate_report(path, df, smiles_col, output):
    """Generate a markdown chemistry profiling report."""
    smiles_list = df[smiles_col].tolist()
    total = len(smiles_list)

    if not check_rdkit():
        report = (
            "# Chemistry Profiler Error\n\n"
            "RDKit is not installed. Install it with:\n"
            "```bash\nuv pip install rdkit\n```\n"
        )
        Path(output).write_text(report)
        print("ERROR: RDKit not installed")
        return report

    print("Computing molecular properties...")
    props = compute_properties(smiles_list)
    valid_props = [p for p in props if p is not None]
    n_valid = len(valid_props)
    n_invalid = total - n_valid

    prop_df = pd.DataFrame(valid_props)

    print("Running scaffold analysis...")
    scaffolds = scaffold_analysis([p["smiles"] for p in valid_props])

    print("Running similarity analysis...")
    sims = similarity_analysis([p["smiles"] for p in valid_props])

    # Lipinski statistics
    n_lipinski_ok = sum(1 for p in valid_props if p["lipinski_ok"])

    lines = [
        f"# Chemistry Dataset Profile: `{Path(path).name}`\n",
        "## Dataset Overview\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Total molecules | {total} |",
        f"| Valid SMILES | {n_valid} ({n_valid/total*100:.1f}%) |",
        f"| Invalid SMILES | {n_invalid} ({n_invalid/total*100:.1f}%) |",
        f"| Lipinski Rule-of-5 compliant | {n_lipinski_ok} ({n_lipinski_ok/max(n_valid,1)*100:.1f}%) |",
        "",
        "## Molecular Property Distributions\n",
        "| Property | Mean | Std | Min | Max |",
        "|----------|------|-----|-----|-----|",
    ]

    for col, label in [
        ("mw", "MW (Da)"), ("logp", "logP"), ("hbd", "HBD"), ("hba", "HBA"),
        ("tpsa", "TPSA (Å²)"), ("n_rotatable", "Rotatable bonds"), ("n_rings", "Rings"),
    ]:
        vals = prop_df[col].dropna()
        lines.append(
            f"| {label} | {vals.mean():.2f} | {vals.std():.2f} | {vals.min():.2f} | {vals.max():.2f} |"
        )

    lines += [
        "",
        "### Lipinski Rule-of-5 Thresholds",
        "- MW ≤ 500 Da, logP ≤ 5, HBD ≤ 5, HBA ≤ 10",
        "",
        "## Scaffold Analysis\n",
        f"- Unique Murcko scaffolds: {scaffolds['unique_scaffolds']}",
        f"- Scaffold diversity: {scaffolds['scaffold_diversity']} (unique/total; higher = more diverse)",
        "",
        "### Top 10 Most Common Scaffolds",
        "| Scaffold SMILES | Count |",
        "|----------------|-------|",
    ]

    for smi, count in scaffolds["top_scaffolds"]:
        display = smi if len(smi) <= 60 else smi[:57] + "..."
        lines.append(f"| `{display}` | {count} |")

    lines += [
        "",
        "## Similarity Analysis (Tanimoto, Morgan r=2)\n",
    ]

    if "error" not in sims:
        lines += [
            f"- Mean pairwise similarity: {sims['mean_similarity']}",
            f"- Median pairwise similarity: {sims['median_similarity']}",
            f"- Max pairwise similarity: {sims['max_similarity']}",
            f"- Near-duplicate pairs (sim ≥ {sims['threshold_used']}): {sims['near_duplicate_pairs']} (from {sims['pairs_sampled']} sampled pairs)",
        ]
        if sims["near_duplicate_pairs"] > 0:
            lines.append(
                f"\n⚠️ **{sims['near_duplicate_pairs']} near-duplicate pairs detected** — consider deduplication before training."
            )
    else:
        lines.append(f"⚠️ {sims['error']}")

    lines += [
        "\n## Recommendations\n",
        "- For ML tasks: verify scaffold split (train/test on different scaffolds) to avoid leakage",
        "- For drug discovery: filter by Lipinski RO5, then apply PAINS filter",
        "- For synthesis planning: check retrosynthetic accessibility with RDKit SA score",
        "- Use `chemistry-analysis` skill for spectroscopy, DFT workflow, and database queries",
    ]

    report = "\n".join(lines)
    Path(output).write_text(report)
    print(f"Report saved to: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Profile a molecular chemistry dataset")
    parser.add_argument("input_file", help="CSV/TSV file path with SMILES column")
    parser.add_argument("--smiles_col", default="smiles", help="Column containing SMILES strings")
    parser.add_argument("--output", default="chem_profile_report.md", help="Output report path")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    if args.smiles_col not in df.columns:
        available = list(df.columns)
        print(f"ERROR: Column '{args.smiles_col}' not found. Available: {available}")
        # Try to find a smiles-like column
        candidates = [c for c in available if "smiles" in c.lower() or "smi" in c.lower()]
        if candidates:
            print(f"Did you mean: {candidates}? Use --smiles_col {candidates[0]}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {args.input_file}")
    generate_report(args.input_file, df, args.smiles_col, args.output)


if __name__ == "__main__":
    main()
