#!/usr/bin/env bash
# End-to-end DSBench SFT data pipeline (data_analysis + data_modeling):
#   1. Generate (data + experiment) trajectories with FullWorkflow (skip-existing)
#   2. Score modeling submissions with upstream's per-comp eval scripts
#      (analysis tasks: no eval — quality gate is just ok=true)
#   3. Emit a workspace.list of absolute paths for keeper workspaces:
#        - analysis:  ok=true
#        - modeling:  passed=true (eval ran and produced a finite score)
#
# Re-runs are safe: --skip-existing on generate. Re-launch the script anytime
# to top up.
#
# Override defaults via env vars:
#   BENCH_ROOT        unzipped dsbench-data dir holding data_analysis/, data_modeling/
#                     (default: /sciclone/proj-ds/ai4scientist/kelin/SciDER/dsbench/dsbench-data)
#   OUTPUT_ROOT       where workspaces go (default: data_generation/dsbench/dataset)
#   FAMILY            analysis | modeling | both (default: both)
#   LIMIT             cap how many tasks to attempt this run
#   MAX_REVISIONS     critic retry budget per agent (default 1)
#   DATA_RECURSION    data agent recursion limit (default 80)
#   EXP_RECURSION     experiment agent recursion limit (default 128)
#   EVAL_TIMEOUT      per-task eval subprocess timeout in seconds (default 60)
#   WORKSPACE_LIST    path for the workspace list output
#                     (default: <OUTPUT_ROOT>/workspace.list)
#
# Examples:
#   bash data_generation/dsbench/generate.sh
#   FAMILY=analysis LIMIT=5 bash data_generation/dsbench/generate.sh
#   FAMILY=modeling LIMIT=3 bash data_generation/dsbench/generate.sh

set -euo pipefail

# --- Notify on exit (success OR failure, including set -e abort) -------------
notify_on_exit() {
  local rc
  rc=$?
  if command -v apprise >/dev/null 2>&1; then
    if [ $rc -eq 0 ]; then
      apprise -b "DSBench Generation Succeed" || true
    else
      apprise -b "DSBench Generation Failed (exit code: $rc)" || true
    fi
  fi
}
trap notify_on_exit EXIT

# --- Resolve project root from this script's location ------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults (env override) -------------------------------------------------
BENCH_ROOT="${BENCH_ROOT:-/sciclone/proj-ds/ai4scientist/kelin/SciDER/dsbench/dsbench-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data_generation/dsbench/dataset}"
WORKSPACE_LIST="${WORKSPACE_LIST:-${OUTPUT_ROOT}/workspace.list}"
FAMILY="${FAMILY:-both}"
MAX_REVISIONS="${MAX_REVISIONS:-2}"
DATA_RECURSION="${DATA_RECURSION:-128}"
EXP_RECURSION="${EXP_RECURSION:-256}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-120}"

# --- Build optional CLI args from env ---------------------------------------
GEN_EXTRA=()
[[ -n "${LIMIT:-}" ]] && GEN_EXTRA+=(--limit "$LIMIT")

mkdir -p "$OUTPUT_ROOT"

# --- Pick interpreter --------------------------------------------------------
PY="${PY:-python}"
if command -v uv >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
fi
echo "[generate.sh] interpreter:    $PY"
echo "[generate.sh] bench_root:     $BENCH_ROOT"
echo "[generate.sh] output_root:    $OUTPUT_ROOT"
echo "[generate.sh] workspace_list: $WORKSPACE_LIST"
echo "[generate.sh] family:         $FAMILY"
echo "[generate.sh] max_revisions:  $MAX_REVISIONS"
echo "[generate.sh] data_recursion: $DATA_RECURSION"
echo "[generate.sh] exp_recursion:  $EXP_RECURSION"
echo "[generate.sh] eval_timeout:   $EVAL_TIMEOUT"
[[ ${#GEN_EXTRA[@]} -gt 0 ]] && echo "[generate.sh] gen extras:     ${GEN_EXTRA[*]}"

cd "$PROJECT_ROOT"

# --- 1. Generate -------------------------------------------------------------
echo
echo "==> [1/3] Generating trajectories (FullWorkflow, no ideation)"
"$PY" -m data_generation.dsbench.generation \
    --bench-root "$BENCH_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --family "$FAMILY" \
    --skip-existing \
    --max-revisions "$MAX_REVISIONS" \
    --data-recursion-limit "$DATA_RECURSION" \
    --experiment-recursion-limit "$EXP_RECURSION" \
    "${GEN_EXTRA[@]}"

# --- 2. Score modeling submissions ------------------------------------------
echo
echo "==> [2/3] Scoring modeling submissions (timeout=${EVAL_TIMEOUT}s)"
"$PY" -m data_generation.dsbench.eval \
    --bench-root "$BENCH_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --timeout "$EVAL_TIMEOUT"

# --- 3. Emit workspace list --------------------------------------------------
echo
echo "==> [3/3] Building workspace list at $WORKSPACE_LIST"

# Filter rule depends on family:
#   analysis: keep iff ok=true (no scoring)
#   modeling: keep iff passed=true (eval ran and produced a finite score)
"$PY" - "$OUTPUT_ROOT" "$WORKSPACE_LIST" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out  = Path(sys.argv[2])

n_analysis_ok = n_analysis_total = 0
n_modeling_pass = n_modeling_total = 0
lines: list[str] = []
for ws in sorted(root.iterdir()):
    if not (ws.is_dir() and ws.name.startswith("dsbench_")):
        continue
    out_json = ws / "output.json"
    if not out_json.is_file():
        continue
    try:
        rec = json.loads(out_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    family = rec.get("family")
    if family == "analysis":
        n_analysis_total += 1
        if rec.get("ok"):
            n_analysis_ok += 1
            lines.append(str(ws.resolve()))
    elif family == "modeling":
        n_modeling_total += 1
        if rec.get("passed"):
            n_modeling_pass += 1
            lines.append(str(ws.resolve()))

out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
ar = (n_analysis_ok / n_analysis_total * 100) if n_analysis_total else 0.0
mr = (n_modeling_pass / n_modeling_total * 100) if n_modeling_total else 0.0
print(f"[generate.sh] analysis: {n_analysis_ok}/{n_analysis_total} ok ({ar:.1f}%)")
print(f"[generate.sh] modeling: {n_modeling_pass}/{n_modeling_total} passed ({mr:.1f}%)")
print(f"[generate.sh] {len(lines)} workspace paths written to {out}")
PY

echo
echo "[generate.sh] Done."
echo "Use the workspace list with train/prepare_data.py:"
echo "  python train/prepare_data.py \\"
echo "      --workspace-list $WORKSPACE_LIST \\"
echo "      --out raw_datafiles/dsbench.jsonl \\"
echo "      --id-level 1"
