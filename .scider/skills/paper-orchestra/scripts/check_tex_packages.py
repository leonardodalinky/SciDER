#!/usr/bin/env python3
"""
check_tex_packages.py — Probe the local TeX installation for packages used
by common conference templates, and write tex_profile.json so downstream
agents can select the correct LaTeX patterns at generation time rather than
discovering failures during compilation.

Without this probe, the pipeline learns which packages are missing only after
a compile failure — requiring manual edits and re-runs. Running this script
once at pipeline startup eliminates that iteration.

Checks:
  cleveref   → if missing: use Figure~\\ref{} instead of \\cref{}
  nicefrac   → if missing: use a/b instead of \\nicefrac{a}{b}
  microtype  → if missing: omit \\usepackage{microtype}
  fontenc    → if missing: omit \\usepackage[T1]{fontenc} (avoids pcrr8t error)
  url        → if missing: omit \\usepackage{url}
  booktabs   → required for tables; warns if absent
  natbib     → required for plainnat bibliography style
  hyperref   → optional; common in many templates
  times      → optional; some templates request Times fonts
  lmodern    → fallback font package

Output: workspace/tex_profile.json
  {
    "available": ["booktabs", "natbib", ...],
    "missing": ["cleveref", "nicefrac", ...],
    "use_cleveref": false,
    "use_nicefrac": false,
    "use_microtype": false,
    "use_t1_fontenc": false,
    "tex_binary": "/Library/TeX/texbin/pdflatex",
    "checked_at": "2026-04-10T12:00:00"
  }

Usage:
    python check_tex_packages.py --out workspace/tex_profile.json
    python check_tex_packages.py --out workspace/tex_profile.json \\
        --tex-bin /Library/TeX/texbin/pdflatex
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile

PACKAGES_TO_CHECK: list[tuple[str, str]] = [
    # (package_name, load_option)  — empty string means no option
    ("cleveref", "capitalize"),
    ("nicefrac", ""),
    ("microtype", ""),
    ("url", ""),
    ("booktabs", ""),
    ("natbib", ""),
    ("hyperref", ""),
    ("fontenc", "T1"),
    ("times", ""),
    ("lmodern", ""),
]

TEX_BINARY_CANDIDATES = [
    "pdflatex",
    "/Library/TeX/texbin/pdflatex",
    "/usr/local/bin/pdflatex",
    "/usr/bin/pdflatex",
    "/opt/homebrew/bin/pdflatex",
]


def find_tex_binary(hint: str | None) -> str | None:
    candidates = ([hint] if hint else []) + TEX_BINARY_CANDIDATES
    for c in candidates:
        if shutil.which(c) or (os.path.isabs(c) and os.path.isfile(c)):
            return c
    return None


def probe_package(tex_binary: str, package: str, option: str = "") -> bool:
    """Compile a minimal .tex document that loads the package. Returns True if OK."""
    if option:
        use_line = f"\\usepackage[{option}]{{{package}}}"
    else:
        use_line = f"\\usepackage{{{package}}}"

    minimal = "\\documentclass{article}\n" f"{use_line}\n" "\\begin{document}x\\end{document}\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "probe.tex")
        with open(tex_path, "w") as f:
            f.write(minimal)
        try:
            result = subprocess.run(
                [
                    tex_binary,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    tmpdir,
                    tex_path,
                ],
                capture_output=True,
                timeout=20,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def build_fallback_profile(reason: str) -> dict:
    """Profile assuming nothing is available — used when pdflatex not found."""
    return {
        "available": [],
        "missing": [pkg for pkg, _ in PACKAGES_TO_CHECK],
        "use_cleveref": False,
        "use_nicefrac": False,
        "use_microtype": False,
        "use_t1_fontenc": False,
        "tex_binary": None,
        "error": reason,
        "checked_at": datetime.datetime.utcnow().isoformat(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output tex_profile.json path")
    p.add_argument("--tex-bin", help="Path to pdflatex (auto-detected if omitted)")
    args = p.parse_args()

    tex_binary = find_tex_binary(args.tex_bin)
    if not tex_binary:
        print("WARN: pdflatex not found — writing fallback profile (all missing)", file=sys.stderr)
        profile = build_fallback_profile("pdflatex not found in PATH or known locations")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(profile, f, indent=2)
        return 1

    print(f"TeX binary: {tex_binary}")
    print("Probing packages...")
    available: list[str] = []
    missing: list[str] = []

    for pkg, opt in PACKAGES_TO_CHECK:
        ok = probe_package(tex_binary, pkg, option=opt)
        label = f"[{opt}]{pkg}" if opt else pkg
        status = "ok" if ok else "MISSING"
        print(f"  {label:30s} {status}")
        (available if ok else missing).append(pkg)

    profile = {
        "available": available,
        "missing": missing,
        "use_cleveref": "cleveref" in available,
        "use_nicefrac": "nicefrac" in available,
        "use_microtype": "microtype" in available,
        "use_t1_fontenc": "fontenc" in available,
        "tex_binary": tex_binary,
        "checked_at": datetime.datetime.utcnow().isoformat(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile written → {args.out}")
    print(f"  use_cleveref   = {profile['use_cleveref']}")
    print(f"  use_nicefrac   = {profile['use_nicefrac']}")
    print(f"  use_microtype  = {profile['use_microtype']}")
    print(f"  use_t1_fontenc = {profile['use_t1_fontenc']}")
    if missing:
        print(f"  missing pkgs   = {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
