#!/usr/bin/env bash
# End-to-end DS-1000 SFT data pipeline:
#   1. Generate trajectories with the coding subagent  (skip-existing)
#   2. Evaluate every workspace against DS-1000's test_execution harness
#   3. Print pass-rate summary + emit a workspace.list of absolute paths for
#      every passing workspace (caller can feed this to train/prepare_data.py)
#
# Re-runs are safe: --skip-existing on generate, default skip-already-evaluated
# on eval. Re-launch the script anytime to top up.
#
# Override defaults via env vars:
#   OUTPUT_ROOT          where workspaces go (default: data_generation/ds1000/dataset)
#   LIBRARY              filter to a single DS-1000 library (Pandas/NumPy/...)
#   LIMIT                cap how many tasks to attempt this run
#   EVAL_TIMEOUT         per-task subprocess timeout in eval (default 15s)
#   WORKSPACE_LIST       path for the passing-workspace list output
#                        (default: <OUTPUT_ROOT>/workspace.list)
#
# Examples:
#   bash data_generation/ds1000/generate.sh
#   LIBRARY=Pandas LIMIT=20 bash data_generation/ds1000/generate.sh
#   OUTPUT_ROOT=/scratch/me/ds1000 bash data_generation/ds1000/generate.sh

set -euo pipefail

# --- Notify on exit (success OR failure, including set -e abort) -------------
# Trap fires on any exit path; $? holds the exit code of whatever caused it.
# `command -v apprise` guards a missing apprise binary so the trap itself
# never fails (which would mask the real error).
notify_on_exit() {
  local rc
  rc=$?
  if command -v apprise >/dev/null 2>&1; then
    if [ $rc -eq 0 ]; then
      apprise -b "DS-1000 Generation Succeed" || true
    else
      apprise -b "DS-1000 Generation Failed (exit code: $rc)" || true
    fi
  fi
}
trap notify_on_exit EXIT

# --- Resolve project root from this script's location ------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults (env override) -------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data_generation/ds1000/dataset}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-15}"
WORKSPACE_LIST="${WORKSPACE_LIST:-${OUTPUT_ROOT}/workspace.list}"

# --- Build optional CLI args from env ---------------------------------------
GEN_EXTRA=()
[[ -n "${LIBRARY:-}" ]]   && GEN_EXTRA+=(--library "$LIBRARY")
[[ -n "${LIMIT:-}"   ]]   && GEN_EXTRA+=(--limit "$LIMIT")

mkdir -p "$OUTPUT_ROOT"

# --- Pick interpreter --------------------------------------------------------
PY="${PY:-python}"
if command -v uv >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
fi
echo "[generate.sh] interpreter:    $PY"
echo "[generate.sh] output_root:    $OUTPUT_ROOT"
echo "[generate.sh] workspace_list: $WORKSPACE_LIST"
[[ ${#GEN_EXTRA[@]} -gt 0 ]] && echo "[generate.sh] gen extras:   ${GEN_EXTRA[*]}"

cd "$PROJECT_ROOT"

# --- 1. Generate -------------------------------------------------------------
echo
echo "==> [1/3] Generating trajectories"
"$PY" -m data_generation.ds1000.generation \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing \
    "${GEN_EXTRA[@]}"

# --- 2. Score correctness ----------------------------------------------------
echo
echo "==> [2/3] Scoring with DS-1000 test harness (timeout=${EVAL_TIMEOUT}s)"
"$PY" -m data_generation.ds1000.eval \
    --output-root "$OUTPUT_ROOT" \
    --timeout "$EVAL_TIMEOUT"

# --- 3. Filter passing workspaces -------------------------------------------
echo
echo "==> [3/3] Building passed-workspace list at $WORKSPACE_LIST"

# Walk every output.json, emit the absolute workspace dir for runs with
# passed=true. We use python (not jq) so the script has zero non-stdlib deps
# beyond what generate/eval already required.
"$PY" - "$OUTPUT_ROOT" "$WORKSPACE_LIST" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out  = Path(sys.argv[2])

total = passed = 0
lines: list[str] = []
for ws in sorted(root.iterdir()):
    if not (ws.is_dir() and ws.name.startswith("ds1000_")):
        continue
    out_json = ws / "output.json"
    if not out_json.is_file():
        continue
    try:
        rec = json.loads(out_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if "passed" not in rec:
        continue
    total += 1
    if rec.get("passed"):
        passed += 1
        lines.append(str(ws.resolve()))

out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
rate = (passed / total * 100) if total else 0.0
print(f"[generate.sh] eval summary: {passed}/{total} passed  ({rate:.1f}%)")
print(f"[generate.sh] {len(lines)} workspace paths written to {out}")
PY

echo
echo "[generate.sh] Done."
echo "Use the workspace list with train/prepare_data.py:"
echo "  python train/prepare_data.py \\"
echo "      --workspace-list $WORKSPACE_LIST \\"
echo "      --out raw_datafiles/ds1000.jsonl \\"
echo "      --id-level 1"
