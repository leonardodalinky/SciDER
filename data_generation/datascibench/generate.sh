#!/usr/bin/env bash
# End-to-end DataSciBench SFT data pipeline:
#   1. Generate (data + experiment) trajectories with FullWorkflow (skip-existing)
#   2. Emit a path list of every produced trajectory pair (no eval/filter —
#      every workflow run is kept regardless of "quality")
#
# Re-runs are safe: --skip-existing on generate. Re-launch the script anytime
# to top up.
#
# Override defaults via env vars:
#   BENCH_ROOT           DataSciBench-data dir holding csv_excel_*/, dl_*/, human_*/
#                        (default: /sciclone/proj-ds/ai4scientist/kelin/SciDER/DataSciBench/data/DataSciBench-data)
#   OUTPUT_ROOT          where workspaces go (default: data_generation/datascibench/dataset)
#   FAMILY               csv_excel | dl | human (omit = all)
#   LIMIT                cap how many tasks to attempt this run
#   MAX_REVISIONS        critic retry budget per agent (default 1)
#   DATA_RECURSION       data agent recursion limit (default 80)
#   EXP_RECURSION        experiment agent recursion limit (default 128)
#   TRAJ_LIST            path for the trajectory list output
#                        (default: <OUTPUT_ROOT>/all_trajectories.txt)
#
# Examples:
#   bash data_generation/datascibench/generate.sh
#   FAMILY=human LIMIT=5 bash data_generation/datascibench/generate.sh
#   BENCH_ROOT=/scratch/me/DataSciBench-data bash data_generation/datascibench/generate.sh

set -euo pipefail

# --- Notify on exit (success OR failure, including set -e abort) -------------
notify_on_exit() {
  local rc=$?
  if command -v apprise >/dev/null 2>&1; then
    if [ $rc -eq 0 ]; then
      apprise -b "DataSciBench Generation Succeed" || true
    else
      apprise -b "DataSciBench Generation Failed (exit code: $rc)" || true
    fi
  fi
}
trap notify_on_exit EXIT

# --- Resolve project root from this script's location ------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults (env override) -------------------------------------------------
BENCH_ROOT="${BENCH_ROOT:-/sciclone/proj-ds/ai4scientist/kelin/SciDER/datascibench/data/DataSciBench-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data_generation/datascibench/dataset}"
TRAJ_LIST="${TRAJ_LIST:-${OUTPUT_ROOT}/all_trajectories.txt}"
MAX_REVISIONS="${MAX_REVISIONS:-1}"
DATA_RECURSION="${DATA_RECURSION:-80}"
EXP_RECURSION="${EXP_RECURSION:-128}"

# --- Build optional CLI args from env ---------------------------------------
GEN_EXTRA=()
[[ -n "${FAMILY:-}" ]] && GEN_EXTRA+=(--family "$FAMILY")
[[ -n "${LIMIT:-}" ]]  && GEN_EXTRA+=(--limit "$LIMIT")

mkdir -p "$OUTPUT_ROOT"

# --- Pick interpreter --------------------------------------------------------
PY="${PY:-python}"
if command -v uv >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
fi
echo "[generate.sh] interpreter:    $PY"
echo "[generate.sh] bench_root:     $BENCH_ROOT"
echo "[generate.sh] output_root:    $OUTPUT_ROOT"
echo "[generate.sh] traj_list:      $TRAJ_LIST"
echo "[generate.sh] max_revisions:  $MAX_REVISIONS"
echo "[generate.sh] data_recursion: $DATA_RECURSION"
echo "[generate.sh] exp_recursion:  $EXP_RECURSION"
[[ ${#GEN_EXTRA[@]} -gt 0 ]] && echo "[generate.sh] gen extras:     ${GEN_EXTRA[*]}"

cd "$PROJECT_ROOT"

# --- 1. Generate -------------------------------------------------------------
echo
echo "==> [1/2] Generating trajectories (FullWorkflow, no ideation)"
"$PY" -m data_generation.datascibench.generation \
    --bench-root "$BENCH_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing \
    --max-revisions "$MAX_REVISIONS" \
    --data-recursion-limit "$DATA_RECURSION" \
    --experiment-recursion-limit "$EXP_RECURSION" \
    "${GEN_EXTRA[@]}"

# --- 2. Emit trajectory list -------------------------------------------------
echo
echo "==> [2/2] Building trajectory list at $TRAJ_LIST"

# Walk every output.json, pull both history paths for runs with ok=true.
# We use python (not jq) so the script has zero non-stdlib deps beyond what
# generate already required. Each line is one history path; both data_agent
# and experiment_agent histories are emitted (separately) so train/prepare_data
# picks them all up.
"$PY" - "$OUTPUT_ROOT" "$TRAJ_LIST" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out  = Path(sys.argv[2])

total = ok = 0
lines: list[str] = []
for ws in sorted(root.iterdir()):
    if not (ws.is_dir() and ws.name.startswith("datascibench_")):
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
    for fname in ("data_agent_history.json", "experiment_agent_history.json"):
        p = ws / fname
        if p.is_file():
            lines.append(str(p.resolve()))

out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
rate = (ok / total * 100) if total else 0.0
print(f"[generate.sh] gen summary: {ok}/{total} ok  ({rate:.1f}%)")
print(f"[generate.sh] {len(lines)} trajectory paths written to {out}")
PY

echo
echo "[generate.sh] Done."
echo "Use the path list with train/prepare_data.py:"
echo "  python train/prepare_data.py \\"
echo "      --workspace-list $TRAJ_LIST \\"
echo "      --out raw_datafiles/datascibench.jsonl \\"
echo "      --id-level 1"
