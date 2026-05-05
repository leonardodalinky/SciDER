---
name: bioinformatics-analysis
description: Bioinformatics workflows — RNA-seq and scRNA-seq analysis pipelines, enrichment analysis (GO/KEGG/GSEA), variant interpretation, protein structure analysis, and key database queries. Use when analyzing genomic, transcriptomic, or proteomic data.
allowed_agents: [data, experiment]
---

# Bioinformatics Analysis

## Overview

This skill covers standard bioinformatics analysis workflows for sequencing, single-cell, and structural biology data. It complements the `exploratory-data-analysis` skill (which handles file format detection) with domain-specific analysis guidance.

**Use this skill after**: Running the EDA skill to understand your data's format and structure.

## When to Use This Skill

- Analyzing RNA-seq or scRNA-seq count matrices
- Running differential expression analysis
- Performing gene set enrichment (GO/KEGG)
- Working with VCF variant files
- Analyzing protein structures (PDB files)
- Querying NCBI, Ensembl, UniProt, or STRING databases

---

## 1. RNA-seq Analysis Pipeline

### Step 1: Quality Control
```bash
# FastQC for individual files
fastqc sample.fastq.gz -o qc_reports/
# MultiQC to aggregate
multiqc qc_reports/ -o multiqc_report/
```

**What to check in QC reports**:
- Per-base quality scores: should be > Q30 across most positions
- Adapter contamination: trim with Trimmomatic or fastp if > 5% reads affected
- GC content: should match expected organism GC content; bimodal suggests contamination
- Duplication rate: > 60% for polyA-selected RNA-seq may indicate issues

### Step 2: Alignment
```bash
# STAR alignment (recommended for splice-aware alignment)
STAR --runThreadN 8 \
     --genomeDir /path/to/genome_index \
     --readFilesIn sample_R1.fastq.gz sample_R2.fastq.gz \
     --readFilesCommand zcat \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix results/sample_

# Alternative: Salmon (quasi-mapping, much faster)
salmon quant -i /path/to/salmon_index -l A \
    -1 sample_R1.fastq.gz -2 sample_R2.fastq.gz \
    -p 8 -o results/sample_quant
```

### Step 3: Quantification
```bash
# featureCounts (for STAR BAM files)
featureCounts -T 8 -p -a genome.gtf \
    -o counts.txt results/*.bam
```

### Step 4: Differential Expression with pyDESeq2
```python
import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# Load count matrix (genes × samples)
counts = pd.read_csv("counts.csv", index_col=0)
metadata = pd.read_csv("metadata.csv", index_col=0)
# metadata must have a column matching your design variable

# Create DESeq2 dataset
dds = DeseqDataSet(
    counts=counts.T,  # samples × genes
    metadata=metadata,
    design_factors="condition",  # column in metadata
)
dds.deseq2()

# Run statistical test
stat_res = DeseqStats(dds, contrast=["condition", "treatment", "control"])
stat_res.summary()
results = stat_res.results_df

# Filter significant genes
sig = results[(results["padj"] < 0.05) & (abs(results["log2FoldChange"]) > 1)]
print(f"Significant DEGs: {len(sig)} (padj<0.05, |log2FC|>1)")
sig.to_csv("DEGs.csv")
```

**Key result columns**:
- `baseMean`: average expression across samples
- `log2FoldChange`: effect size (positive = up in treatment)
- `pvalue`: unadjusted p-value
- `padj`: Benjamini-Hochberg adjusted p-value (use this for significance)

---

## 2. scRNA-seq Analysis with Scanpy

```python
import scanpy as sc
import pandas as pd

# Load data
adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
# or from h5ad: adata = sc.read_h5ad("data.h5ad")

# Step 1: Quality control
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# Filter low-quality cells
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"])
adata = adata[adata.obs.n_genes_by_counts > 200, :]
adata = adata[adata.obs.n_genes_by_counts < 5000, :]  # adjust per dataset
adata = adata[adata.obs.pct_counts_mt < 20, :]         # < 20% mitochondrial

# Step 2: Normalization and feature selection
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

# Step 3: Dimensionality reduction
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
sc.pl.pca_variance_ratio(adata, log=True)  # choose n_pcs from elbow

# Step 4: Neighbors + UMAP
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

# Step 5: Clustering
sc.tl.leiden(adata, resolution=0.5)  # increase resolution for more clusters
sc.pl.umap(adata, color=["leiden"])

# Step 6: Marker genes
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
markers = sc.get.rank_genes_groups_df(adata, group="0")
print(markers.head(10))
```

---

## 3. Gene Set Enrichment Analysis (GSEA)

```python
import gseapy as gp

# Option 1: Over-representation analysis (ORA) on a gene list
gene_list = sig.index.tolist()  # from DESeq2 results
enr = gp.enrichr(
    gene_list=gene_list,
    gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human"],
    organism="Human",
    outdir="enrichment_results",
)
print(enr.results[enr.results["Adjusted P-value"] < 0.05].head(20))

# Option 2: Pre-ranked GSEA (uses continuous score like log2FC * -log10(pval))
results_sorted = results.dropna(subset=["padj"])
results_sorted["score"] = results_sorted["log2FoldChange"] * (-np.log10(results_sorted["padj"] + 1e-300))
rnk = results_sorted["score"].sort_values(ascending=False)

pre_res = gp.prerank(
    rnk=rnk,
    gene_sets="KEGG_2021_Human",
    processes=4,
    outdir="gsea_results",
    seed=42,
)
```

**Multiple testing**: Always use adjusted p-values (FDR/BH correction). Report at padj < 0.05 or padj < 0.1 with justification.

---

## 4. Variant Interpretation (VCF)

```python
import pandas as pd

# Read VCF (skip header lines)
vcf_lines = [l for l in open("variants.vcf") if not l.startswith("##")]
# First line after ## is the header (#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT ...)
df = pd.read_csv(pd.io.common.StringIO("".join(vcf_lines)), sep="\t")
df.columns = df.columns.str.lstrip("#")

# Basic filtering
filtered = df[
    (df["QUAL"] > 30) &          # Quality score > 30
    (df["FILTER"] == "PASS")     # Passed variant filters
]

# Parse INFO field
def parse_info(info_str):
    return dict(item.split("=") if "=" in item else (item, True)
                for item in info_str.split(";"))

df["INFO_dict"] = df["INFO"].apply(parse_info)
df["DP"] = df["INFO_dict"].apply(lambda x: int(x.get("DP", 0)))  # Read depth
df["AF"] = df["INFO_dict"].apply(lambda x: float(x.get("AF", 0)))  # Allele frequency

print(f"Total variants: {len(df)}")
print(f"After PASS + QUAL>30: {len(filtered)}")
print(f"SNPs: {len(filtered[filtered.ALT.str.len() == 1])}")
print(f"INDELs: {len(filtered[filtered.ALT.str.len() != 1])}")
```

**Functional annotation tools**: ANNOVAR, VEP (Ensembl), SnpEff — all can be run via command line.

---

## 5. Protein Structure Analysis

```python
from Bio import PDB
import numpy as np

# Load structure
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("protein", "protein.pdb")

# Get CA atoms for RMSD calculation
def get_ca_atoms(struct):
    return [atom for atom in struct.get_atoms() if atom.get_name() == "CA"]

# RMSD between two structures (must have same # of residues)
def rmsd(atoms1, atoms2):
    coords1 = np.array([a.get_coord() for a in atoms1])
    coords2 = np.array([a.get_coord() for a in atoms2])
    diff = coords1 - coords2
    return np.sqrt((diff ** 2).sum(axis=1).mean())

# AlphaFold2 via ColabFold (command line):
# colabfold_batch input.fasta output_dir/ --num-recycle 3

# pLDDT score interpretation (in B-factor column of AlphaFold PDB):
# > 90: very high confidence
# 70–90: high confidence (backbone reliable)
# 50–70: low confidence (treat with caution)
# < 50: very low confidence (disordered region)
```

---

## 6. Key Database Queries

### NCBI Entrez (Biopython)
```python
from Bio import Entrez, SeqIO
Entrez.email = "your@email.com"

# Search PubMed
handle = Entrez.esearch(db="pubmed", term="CRISPR AND cancer[MeSH]", retmax=100)
record = Entrez.read(handle)
pmids = record["IdList"]

# Fetch protein sequence
handle = Entrez.efetch(db="protein", id="NP_000537", rettype="fasta", retmode="text")
record = SeqIO.read(handle, "fasta")
print(record.seq[:50])
```

### UniProt REST API
```python
import requests

# Fetch protein entry
r = requests.get("https://rest.uniprot.org/uniprotkb/P04637.json")
protein = r.json()
print(protein["proteinDescription"]["recommendedName"]["fullName"]["value"])
print("Functions:", [c["texts"][0]["value"] for c in protein.get("comments", [])
                     if c["commentType"] == "FUNCTION"][:1])
```

### Materials Project (for structure databases)
```python
# STRING protein-protein interactions
r = requests.get(
    "https://string-db.org/api/json/network",
    params={"identifiers": "TP53%0DBRCA1", "species": 9606}
)
interactions = r.json()
```
