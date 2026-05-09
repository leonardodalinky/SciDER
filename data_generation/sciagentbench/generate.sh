#!/usr/bin/env bash
# End-to-end ScienceAgentBench SFT data pipeline:
#   1. Generate (data + experiment) trajectories with FullWorkflow (skip-existing)
#   2. Emit a workspace.list of absolute paths for every ok=true workspace
#      (no eval/filter — every workflow run that completes is kept)
#
# Re-runs are safe: --skip-existing on generate. Re-launch the script anytime
# to top up.
#
# Override defaults via env vars:
#   BENCH_ROOT        unzipped ScienceAgentBench dir holding datasets/, gold_programs/, ...
#                     (default: /sciclone/proj-ds/ai4scientist/kelin/SciDER/sciagentbench/benchmark)
#   OUTPUT_ROOT       where workspaces go (default: data_generation/sciagentbench/dataset)
#   USE_KNOWLEDGE     "1" to inject domain_knowledge into the prompt (default: off)
#   LIMIT             cap how many tasks to attempt this run
#   MAX_REVISIONS     critic retry budget per agent (default 1)
#   DATA_RECURSION    data agent recursion limit (default 80)
#   EXP_RECURSION     experiment agent recursion limit (default 128)
#   WORKSPACE_LIST    path for the workspace list output
#                     (default: <OUTPUT_ROOT>/workspace.list)
#
# Examples:
#   bash data_generation/sciagentbench/generate.sh
#   LIMIT=5 USE_KNOWLEDGE=1 bash data_generation/sciagentbench/generate.sh
#   BENCH_ROOT=/scratch/me/sciagentbench bash data_generation/sciagentbench/generate.sh

set -euo pipefail

# --- Notify on exit (success OR failure, including set -e abort) -------------
notify_on_exit() {
  local rc
  rc=$?
  if command -v apprise >/dev/null 2>&1; then
    if [ $rc -eq 0 ]; then
      apprise -b "ScienceAgentBench Generation Succeed" || true
    else
      apprise -b "ScienceAgentBench Generation Failed (exit code: $rc)" || true
    fi
  fi
}
trap notify_on_exit EXIT

# --- Resolve project root from this script's location ------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults (env override) -------------------------------------------------
BENCH_ROOT="${BENCH_ROOT:-/sciclone/proj-ds/ai4scientist/kelin/SciDER/sciagentbench/benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data_generation/sciagentbench/dataset}"
WORKSPACE_LIST="${WORKSPACE_LIST:-${OUTPUT_ROOT}/workspace.list}"
MAX_REVISIONS="${MAX_REVISIONS:-2}"
DATA_RECURSION="${DATA_RECURSION:-128}"
EXP_RECURSION="${EXP_RECURSION:-512}"

# --- Build optional CLI args from env ---------------------------------------
GEN_EXTRA=()
[[ "${USE_KNOWLEDGE:-0}" == "1" ]] && GEN_EXTRA+=(--use-knowledge)
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
echo "[generate.sh] max_revisions:  $MAX_REVISIONS"
echo "[generate.sh] data_recursion: $DATA_RECURSION"
echo "[generate.sh] exp_recursion:  $EXP_RECURSION"
[[ ${#GEN_EXTRA[@]} -gt 0 ]] && echo "[generate.sh] gen extras:     ${GEN_EXTRA[*]}"

cd "$PROJECT_ROOT"

# --- 1. Generate -------------------------------------------------------------
echo
echo "==> [1/2] Generating trajectories (FullWorkflow, no ideation)"
"$PY" -m data_generation.sciagentbench.generation \
    --bench-root "$BENCH_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing \
    --max-revisions "$MAX_REVISIONS" \
    --data-recursion-limit "$DATA_RECURSION" \
    --experiment-recursion-limit "$EXP_RECURSION" \
    "${GEN_EXTRA[@]}"

# --- 2. Emit workspace list --------------------------------------------------
echo
echo "==> [2/2] Building workspace list at $WORKSPACE_LIST"

"$PY" - "$OUTPUT_ROOT" "$WORKSPACE_LIST" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out  = Path(sys.argv[2])

total = ok = 0
lines: list[str] = []
for ws in sorted(root.iterdir()):
    if not (ws.is_dir() and ws.name.startswith("sciagentbench_")):
        continue
    out_json = ws / "output.json"
    if not out_json.is_file():
        continue
    try:
        rec = json.loads(out_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    total += 1
    if not rec.get("ok"):
        continue
    ok += 1
    lines.append(str(ws.resolve()))

out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
rate = (ok / total * 100) if total else 0.0
print(f"[generate.sh] gen summary: {ok}/{total} ok  ({rate:.1f}%)")
print(f"[generate.sh] {len(lines)} workspace paths written to {out}")
PY

echo
echo "[generate.sh] Done."
echo "Use the workspace list with train/prepare_data.py:"
echo "  python train/prepare_data.py \\"
echo "      --workspace-list $WORKSPACE_LIST \\"
echo "      --out raw_datafiles/sciagentbench.jsonl \\"
echo "      --id-level 1"
