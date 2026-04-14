#!/usr/bin/env python3
"""
s2_search.py — Semantic Scholar title-search helper for Phase 2 verification.

Queries the Semantic Scholar Graph API for a paper by title and returns the
top candidate hits as JSON.  Used by the literature-review-agent to verify
each candidate from Phase 1 before adding it to citation_pool.json.

API key (optional):
    If SEMANTIC_SCHOLAR_API_KEY is set in the environment the key is forwarded
    via the ``x-api-key`` header, which raises the rate limit from ~100 req/5 min
    (unauthenticated) to 1 req/s sustained with higher burst headroom.
    If the variable is absent the script falls back to the public unauthenticated
    endpoint — the pipeline works fine without a key; just keep to ≤1 QPS.

    Get a free key at: https://api.semanticscholar.org/
    Then export it once before running the pipeline:
        export SEMANTIC_SCHOLAR_API_KEY="your-key-here"

Usage:
    # check for key and search
    python s2_search.py --query "Attention is All You Need"

    # request more hits and extra fields
    python s2_search.py --query "BERT pre-training" --limit 10 \\
        --fields title,abstract,year,authors,venue,externalIds,citationCount

    # pretty-print raw S2 JSON
    python s2_search.py --query "GPT-4 technical report" --raw

Exit codes:
    0  at least one result returned
    1  HTTP error, network error, or zero results
    2  usage error (bad arguments)
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

S2_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = "title,abstract,year,authors,venue,externalIds"
DEFAULT_LIMIT = 5
MAX_LIMIT = 100
_RETRY_SLEEP = 5  # seconds to wait after a 429 before retrying


def _build_headers() -> dict:
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def search(query: str, limit: int, fields: str, retries: int = 3) -> dict:
    """
    Call /paper/search and return the parsed JSON response.

    Raises SystemExit on unrecoverable errors so the caller (or CLI) gets a
    clean non-zero exit code.
    """
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": limit,
            "fields": fields,
        }
    )
    url = f"{S2_BASE}/paper/search?{params}"
    headers = _build_headers()

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                if attempt < retries:
                    print(
                        f"WARN: S2 rate-limited (429). Sleeping {_RETRY_SLEEP}s "
                        f"before retry {attempt + 1}/{retries}.",
                        file=sys.stderr,
                    )
                    time.sleep(_RETRY_SLEEP)
                    continue
                print(
                    "ERROR: S2 rate-limited (429) and retries exhausted.\n"
                    "Tip: set SEMANTIC_SCHOLAR_API_KEY to get a higher rate limit.\n"
                    "     See https://api.semanticscholar.org/ for a free key.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if exc.code == 404:
                # not found — return an empty result set (caller handles this)
                return {"total": 0, "data": []}
            if exc.code in (500, 502, 503):
                if attempt < retries:
                    print(
                        f"WARN: S2 server error ({exc.code}). Sleeping 30s before "
                        f"retry {attempt + 1}/{retries}.",
                        file=sys.stderr,
                    )
                    time.sleep(30)
                    continue
                print(
                    f"ERROR: S2 server error ({exc.code}) after {retries} attempts.",
                    file=sys.stderr,
                )
                sys.exit(1)
            body = exc.read().decode("utf-8", errors="replace")[:400]
            print(f"ERROR: S2 HTTP {exc.code}: {body}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as exc:
            print(f"ERROR: Network error reaching Semantic Scholar: {exc.reason}", file=sys.stderr)
            sys.exit(1)

    # should never reach here
    sys.exit(1)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--query",
        required=True,
        help="Paper title (or search query) to look up on Semantic Scholar",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max hits to return (default {DEFAULT_LIMIT}, max {MAX_LIMIT})",
    )
    p.add_argument(
        "--fields",
        default=DEFAULT_FIELDS,
        help=f"Comma-separated S2 fields to request (default: {DEFAULT_FIELDS})",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print the full S2 JSON response unmodified instead of normalized output",
    )
    p.add_argument(
        "--check-key",
        action="store_true",
        help="Print whether SEMANTIC_SCHOLAR_API_KEY is set and exit (no network call)",
    )
    args = p.parse_args()

    if args.check_key:
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if key:
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
            print(
                f"SEMANTIC_SCHOLAR_API_KEY is set ({masked}). "
                "Authenticated mode: higher rate limits."
            )
        else:
            print(
                "SEMANTIC_SCHOLAR_API_KEY is NOT set. "
                "Unauthenticated mode: ~100 req/5 min, keep to ≤1 QPS.\n"
                "To enable higher rate limits:\n"
                "  1. Get a free key at https://api.semanticscholar.org/\n"
                '  2. export SEMANTIC_SCHOLAR_API_KEY="your-key-here"'
            )
        return 0

    limit = max(1, min(MAX_LIMIT, args.limit))
    response = search(args.query, limit, args.fields)

    if args.raw:
        json.dump(response, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    data = response.get("data") or []
    if not data:
        print(
            f"WARN: Semantic Scholar returned 0 results for query: {args.query!r}",
            file=sys.stderr,
        )
        json.dump({"total": 0, "data": []}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1

    # Emit normalized output (subset of fields used by pipeline)
    out = {
        "total": response.get("total", len(data)),
        "authenticated": bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()),
        "data": data,
    }
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
