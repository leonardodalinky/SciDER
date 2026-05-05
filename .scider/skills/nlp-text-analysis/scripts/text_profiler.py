#!/usr/bin/env python3
"""
text_profiler.py — NLP Text Dataset Profiler

Usage:
    python text_profiler.py <input_file> [--text_column text] [--label_column label] [--output report.md]

Supported formats: .csv, .tsv, .jsonl
Outputs a Markdown report with corpus statistics, quality flags, and data notes.
"""

import argparse
import collections
import json
import math
import os
import re
import sys
from pathlib import Path


def load_dataset(input_file: str, text_col: str, label_col: str | None):
    """Load CSV, TSV, or JSONL into a list of dicts with 'text' and optionally 'label'."""
    ext = Path(input_file).suffix.lower()
    records = []

    if ext in (".csv", ".tsv"):
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: pandas is required for CSV/TSV. Install with: pip install pandas")
            sys.exit(1)
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(input_file, sep=sep, dtype=str)

        if text_col not in df.columns:
            print(f"ERROR: Column '{text_col}' not found. Available columns: {list(df.columns)}")
            sys.exit(1)

        for _, row in df.iterrows():
            rec = {"text": str(row[text_col]) if pd.notna(row[text_col]) else ""}
            if label_col and label_col in df.columns:
                rec["label"] = str(row[label_col]) if pd.notna(row[label_col]) else None
            records.append(rec)

    elif ext == ".jsonl":
        with open(input_file, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed JSON on line {lineno}: {e}")
                    continue
                rec = {"text": str(obj.get(text_col, ""))}
                if label_col and label_col in obj:
                    rec["label"] = str(obj[label_col])
                records.append(rec)
    else:
        print(f"ERROR: Unsupported file format '{ext}'. Use .csv, .tsv, or .jsonl")
        sys.exit(1)

    return records


def simple_tokenize(text: str) -> list[str]:
    """Whitespace tokenization with basic cleaning."""
    return text.split()


def compute_corpus_stats(texts: list[str]) -> dict:
    """Compute vocabulary and length statistics over a list of texts."""
    token_lengths = []
    char_lengths = []
    all_tokens = []

    for text in texts:
        tokens = simple_tokenize(text)
        token_lengths.append(len(tokens))
        char_lengths.append(len(text))
        all_tokens.extend(t.lower() for t in tokens)

    vocab = collections.Counter(all_tokens)

    def _stats(values: list[int]) -> dict:
        if not values:
            return {}
        n = len(values)
        total = sum(values)
        mean = total / n
        sorted_v = sorted(values)
        median = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)
        return {
            "total": total,
            "mean": mean,
            "median": median,
            "std": std,
            "min": min(values),
            "max": max(values),
            "p25": sorted_v[int(0.25 * n)],
            "p75": sorted_v[int(0.75 * n)],
            "p95": sorted_v[int(0.95 * n)],
        }

    return {
        "n_docs": len(texts),
        "token_stats": _stats(token_lengths),
        "char_stats": _stats(char_lengths),
        "vocab_size": len(vocab),
        "total_tokens": len(all_tokens),
        "singleton_count": sum(1 for c in vocab.values() if c == 1),
        "top_tokens": vocab.most_common(20),
    }


def detect_quality_issues(records: list[dict], token_lengths: list[int], texts: list[str]) -> dict:
    """Flag potential data quality problems."""
    issues = {}

    # Null / empty
    issues["null_texts"] = sum(1 for r in records if not r.get("text", "").strip())

    # Very short (< 5 tokens)
    issues["very_short"] = sum(1 for l in token_lengths if l < 5)
    issues["very_short_pct"] = issues["very_short"] / max(len(texts), 1)

    # Very long (> 512 tokens — common transformer limit)
    issues["over_512"] = sum(1 for l in token_lengths if l > 512)
    issues["over_512_pct"] = issues["over_512"] / max(len(texts), 1)

    # Duplicates (exact)
    seen = set()
    dupes = 0
    for t in texts:
        if t in seen:
            dupes += 1
        seen.add(t)
    issues["exact_duplicates"] = dupes
    issues["exact_duplicates_pct"] = dupes / max(len(texts), 1)

    # Potential encoding artifacts (mojibake patterns)
    mojibake_re = re.compile(r'[â€™â€œâ€˜Ã©Ã ]')
    issues["potential_mojibake"] = sum(1 for t in texts if mojibake_re.search(t))

    # High non-ASCII ratio (> 30%)
    def non_ascii_ratio(text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if ord(c) > 127) / len(text)

    issues["high_non_ascii"] = sum(1 for t in texts if non_ascii_ratio(t) > 0.3)

    # Likely HTML fragments
    html_re = re.compile(r'<[a-zA-Z][^>]*>')
    issues["contains_html"] = sum(1 for t in texts if html_re.search(t))

    # Likely URLs
    url_re = re.compile(r'https?://')
    issues["contains_urls"] = sum(1 for t in texts if url_re.search(t))

    return issues


def compute_label_distribution(records: list[dict]) -> dict | None:
    """Return label counts and imbalance ratio if label column is present."""
    labels = [r.get("label") for r in records if r.get("label") is not None]
    if not labels:
        return None

    counter = collections.Counter(labels)
    counts = counter.most_common()
    majority = counts[0][1]
    minority = counts[-1][1]
    imbalance_ratio = majority / minority if minority > 0 else float("inf")

    return {
        "n_labeled": len(labels),
        "n_classes": len(counter),
        "counts": counts,
        "imbalance_ratio": imbalance_ratio,
    }


def build_report(
    input_file: str,
    text_col: str,
    label_col: str | None,
    stats: dict,
    issues: dict,
    label_info: dict | None,
) -> str:
    """Render all statistics as a Markdown report string."""
    lines = []
    filename = Path(input_file).name
    ts = stats["token_stats"]
    cs = stats["char_stats"]

    lines.append(f"# Text Dataset Profile: `{filename}`\n")
    lines.append(f"- **Text column**: `{text_col}`")
    if label_col:
        lines.append(f"- **Label column**: `{label_col}`")
    lines.append("")

    # --- Corpus Overview ---
    lines.append("## Corpus Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total documents | {stats['n_docs']:,} |")
    lines.append(f"| Total tokens (whitespace) | {stats['total_tokens']:,} |")
    lines.append(f"| Vocabulary size | {stats['vocab_size']:,} |")
    lines.append(f"| Singleton tokens (freq=1) | {stats['singleton_count']:,} ({stats['singleton_count']/max(stats['vocab_size'],1):.1%} of vocab) |")
    lines.append("")

    # --- Token Length Stats ---
    lines.append("## Token Length Statistics (per document)\n")
    lines.append("| Statistic | Tokens | Characters |")
    lines.append("|-----------|--------|------------|")
    for key, label in [("mean", "Mean"), ("median", "Median"), ("std", "Std dev"),
                        ("min", "Min"), ("p25", "25th percentile"), ("p75", "75th percentile"),
                        ("p95", "95th percentile"), ("max", "Max")]:
        tv = ts.get(key, 0)
        cv = cs.get(key, 0)
        if key in ("mean", "std"):
            lines.append(f"| {label} | {tv:.1f} | {cv:.1f} |")
        else:
            lines.append(f"| {label} | {int(tv):,} | {int(cv):,} |")
    lines.append("")

    # --- Top Tokens ---
    lines.append("## Top 20 Most Frequent Tokens\n")
    lines.append("| Rank | Token | Count |")
    lines.append("|------|-------|-------|")
    for rank, (tok, count) in enumerate(stats["top_tokens"], 1):
        lines.append(f"| {rank} | `{tok}` | {count:,} |")
    lines.append("")

    # --- Label Distribution ---
    if label_info:
        lines.append("## Label / Class Distribution\n")
        lines.append(f"- **Labeled documents**: {label_info['n_labeled']:,}")
        lines.append(f"- **Number of classes**: {label_info['n_classes']}")
        lines.append(f"- **Imbalance ratio** (majority / minority): {label_info['imbalance_ratio']:.1f}x")
        lines.append("")
        lines.append("| Label | Count | Proportion |")
        lines.append("|-------|-------|------------|")
        total_labeled = label_info["n_labeled"]
        for label_val, count in label_info["counts"]:
            lines.append(f"| `{label_val}` | {count:,} | {count/total_labeled:.2%} |")
        lines.append("")

    # --- Data Quality Flags ---
    lines.append("## Data Quality Flags\n")
    lines.append("| Issue | Count | Proportion |")
    lines.append("|-------|-------|------------|")
    n = stats["n_docs"]
    flag_rows = [
        ("Null / empty texts", issues["null_texts"], issues["null_texts"] / max(n, 1)),
        ("Very short texts (< 5 tokens)", issues["very_short"], issues["very_short_pct"]),
        ("Long texts (> 512 tokens)", issues["over_512"], issues["over_512_pct"]),
        ("Exact duplicate texts", issues["exact_duplicates"], issues["exact_duplicates_pct"]),
        ("Potential mojibake / encoding errors", issues["potential_mojibake"], issues["potential_mojibake"] / max(n, 1)),
        ("High non-ASCII ratio (> 30%)", issues["high_non_ascii"], issues["high_non_ascii"] / max(n, 1)),
        ("Contains HTML fragments", issues["contains_html"], issues["contains_html"] / max(n, 1)),
        ("Contains URLs", issues["contains_urls"], issues["contains_urls"] / max(n, 1)),
    ]
    for label_txt, count, pct in flag_rows:
        flag = " ⚠" if count > 0 else ""
        lines.append(f"| {label_txt}{flag} | {count:,} | {pct:.2%} |")
    lines.append("")

    # --- Data Quality Notes ---
    lines.append("## Data Quality Notes\n")
    notes = []

    if issues["null_texts"] > 0:
        notes.append(f"- **{issues['null_texts']:,} null/empty texts** detected. Drop or impute before modeling.")

    if issues["exact_duplicates"] > 0:
        notes.append(
            f"- **{issues['exact_duplicates']:,} exact duplicates** ({issues['exact_duplicates_pct']:.1%}) found. "
            "De-duplicate before train/test split to prevent leakage."
        )

    if issues["very_short"] > 0:
        notes.append(
            f"- **{issues['very_short']:,} texts with fewer than 5 tokens** ({issues['very_short_pct']:.1%}). "
            "Review these — they may be noise, label errors, or truncated entries."
        )

    if issues["over_512"] > 0:
        notes.append(
            f"- **{issues['over_512']:,} texts exceed 512 tokens** ({issues['over_512_pct']:.1%}). "
            "These will be truncated by most transformer models. Consider chunking or using Longformer/BigBird."
        )

    if issues["potential_mojibake"] > 0:
        notes.append(
            f"- **{issues['potential_mojibake']:,} texts contain likely encoding artifacts** (mojibake). "
            "Re-read files with explicit UTF-8 encoding and run `ftfy` to fix."
        )

    if issues["contains_html"] > 0:
        notes.append(
            f"- **{issues['contains_html']:,} texts contain HTML tags**. "
            "Strip with `re.sub(r'<[^>]+>', '', text)` or `BeautifulSoup`."
        )

    if issues["contains_urls"] > 0:
        notes.append(
            f"- **{issues['contains_urls']:,} texts contain URLs**. "
            "Remove with `re.sub(r'https?://\\S+', '', text)` unless URLs are meaningful features."
        )

    if label_info and label_info["imbalance_ratio"] > 5:
        notes.append(
            f"- **Class imbalance ratio is {label_info['imbalance_ratio']:.1f}x**. "
            "Consider stratified sampling, class-weighted loss, oversampling (SMOTE for embeddings), "
            "or undersampling to address imbalance."
        )

    singleton_pct = stats["singleton_count"] / max(stats["vocab_size"], 1)
    if singleton_pct > 0.4:
        notes.append(
            f"- **{singleton_pct:.0%} of vocabulary tokens appear only once** (hapax legomena). "
            "This is normal for raw corpora, but consider subword tokenization or minimum frequency thresholds."
        )

    if not notes:
        notes.append("- No major data quality issues detected. Proceed with standard preprocessing.")

    lines.extend(notes)
    lines.append("")
    lines.append("---")
    lines.append(
        "*Generated by `text_profiler.py` — "
        "SciDER NLP Text Analysis Skill. "
        "Tokenization: whitespace (for profiling only — use a proper tokenizer for modeling).*"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Profile a text dataset and generate a Markdown quality report."
    )
    parser.add_argument("input_file", help="Path to .csv, .tsv, or .jsonl dataset file")
    parser.add_argument(
        "--text_column", default="text",
        help="Column name containing text (default: 'text')"
    )
    parser.add_argument(
        "--label_column", default=None,
        help="Column name containing class labels (optional)"
    )
    parser.add_argument(
        "--output", default="report.md",
        help="Output path for the Markdown report (default: report.md)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"ERROR: File not found: {args.input_file}")
        sys.exit(1)

    print(f"Loading dataset: {args.input_file}")
    records = load_dataset(args.input_file, args.text_column, args.label_column)
    print(f"  Loaded {len(records):,} records")

    texts = [r["text"] for r in records]
    tokenized = [simple_tokenize(t) for t in texts]
    token_lengths = [len(t) for t in tokenized]

    print("Computing corpus statistics...")
    stats = compute_corpus_stats(texts)

    print("Detecting quality issues...")
    issues = detect_quality_issues(records, token_lengths, texts)

    label_info = None
    if args.label_column:
        print("Computing label distribution...")
        label_info = compute_label_distribution(records)

    print("Building report...")
    report = build_report(
        args.input_file, args.text_column, args.label_column,
        stats, issues, label_info
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {args.output}")
    print(f"\nQuick summary:")
    print(f"  Documents:      {stats['n_docs']:,}")
    print(f"  Vocab size:     {stats['vocab_size']:,}")
    print(f"  Avg tokens/doc: {stats['token_stats']['mean']:.1f}")
    print(f"  Duplicates:     {issues['exact_duplicates']:,}")
    print(f"  Over 512 tok:   {issues['over_512']:,} ({issues['over_512_pct']:.1%})")
    if label_info:
        print(f"  Classes:        {label_info['n_classes']}")
        print(f"  Imbalance:      {label_info['imbalance_ratio']:.1f}x")


if __name__ == "__main__":
    main()
