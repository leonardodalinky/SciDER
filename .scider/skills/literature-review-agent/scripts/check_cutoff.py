#!/usr/bin/env python3
"""
check_cutoff.py — Strict temporal cutoff check for citation verification.

Implements the paper's Rule 3 (App. D.3): a paper passes only if its
publication date strictly predates the research cutoff. When only the year
is known, assume the worst case (Dec 31). When year + month are known,
assume day-1 of that month (per the paper's "first day of that month"
default).

Exit codes:
    0  paper strictly predates cutoff (PASS)
    1  paper does not strictly predate cutoff (FAIL)
    2  argument error

Usage:
    python check_cutoff.py --paper-year 2024 --paper-month 9 --cutoff 2024-10-01
    python check_cutoff.py --paper-year 2024 --cutoff 2024-10-01
    python check_cutoff.py --paper-date 2024-09-15 --cutoff 2024-10-01
"""
import argparse
import datetime as dt
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-year", type=int, help="Paper publication year")
    p.add_argument("--paper-month", type=int, help="Paper publication month (1-12), optional")
    p.add_argument("--paper-date", help="Full paper date YYYY-MM-DD, overrides year/month")
    p.add_argument("--cutoff", required=True, help="Research cutoff date YYYY-MM-DD")
    args = p.parse_args()

    try:
        cutoff = dt.date.fromisoformat(args.cutoff)
    except ValueError:
        print(f"ERROR: --cutoff must be YYYY-MM-DD, got {args.cutoff}", file=sys.stderr)
        return 2

    if args.paper_date:
        try:
            paper_date = dt.date.fromisoformat(args.paper_date)
        except ValueError:
            print(f"ERROR: --paper-date must be YYYY-MM-DD, got {args.paper_date}", file=sys.stderr)
            return 2
    elif args.paper_year:
        if args.paper_month:
            paper_date = dt.date(args.paper_year, args.paper_month, 1)
        else:
            paper_date = dt.date(args.paper_year, 12, 31)
    else:
        print("ERROR: must provide --paper-date OR --paper-year", file=sys.stderr)
        return 2

    if paper_date < cutoff:
        print(f"PASS  paper={paper_date}  <  cutoff={cutoff}")
        return 0
    print(f"FAIL  paper={paper_date}  not strictly before cutoff={cutoff}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
