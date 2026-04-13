#!/usr/bin/env python3
"""
snapshot.py — Copy a paper.tex (and optionally paper.pdf) into a refinement
iteration directory, so reverts are real, not symbolic.

The PaperOrchestra refinement halt rules require the loop to roll back to
the previous iteration on overall-score decrease or tied negative sub-axis
delta. To do that physically, every iteration's draft must be preserved.

Usage:
    python snapshot.py --src paper.tex --dst iter2/
    python snapshot.py --src paper.tex --src-pdf paper.pdf --dst iter2/
"""
import argparse
import os
import shutil
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="source paper.tex path")
    p.add_argument("--src-pdf", help="optional source paper.pdf path")
    p.add_argument("--dst", required=True, help="destination iteration directory")
    args = p.parse_args()

    if not os.path.isfile(args.src):
        print(f"ERROR: {args.src} not found", file=sys.stderr)
        return 1

    os.makedirs(args.dst, exist_ok=True)
    dst_tex = os.path.join(args.dst, "paper.tex")
    shutil.copy2(args.src, dst_tex)
    print(f"OK: snapshot {args.src} → {dst_tex}")

    if args.src_pdf:
        if not os.path.isfile(args.src_pdf):
            print(f"WARN: {args.src_pdf} not found, skipping PDF snapshot", file=sys.stderr)
        else:
            dst_pdf = os.path.join(args.dst, "paper.pdf")
            shutil.copy2(args.src_pdf, dst_pdf)
            print(f"OK: snapshot {args.src_pdf} → {dst_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
