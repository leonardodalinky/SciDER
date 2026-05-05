#!/usr/bin/env bash
# End-to-end AI-Idea-Bench SFT data pipeline:
#   1. Generate ideation trajectories with the IdeationAgent (skip-existing)
#   2. Emit a path list of every produced trajectory (no eval/filter — every
#      ideation run is kept regardless of "quality")
#
# Re-runs are safe: --skip-existing on generate. Re-launch the script anytime
# to top up.
#
# Override defaults via env vars:
#   OUTPUT_ROOT          where workspaces go (default: data_generation/aiidea/dataset)
#   LIMIT                cap how many tasks to attempt this run
#   RECURSION_LIMIT      IdeationAgent recursion limit (default 50)
#   TRAJ_LIST            path for the trajectory list output
#                        (default: <OUTPUT_ROOT>/all_trajectories.txt)
#
# Examples:
#   bash data_generation/aiidea/generate.sh
#   LIMIT=20 bash data_generation/aiidea/generate.sh
#   OUTPUT_ROOT=/scratch/me/aiidea bash data_generation/aiidea/generate.sh

set -euo pipefail

# --- Notify on exit (success OR failure, including set -e abort) -------------
# Trap fires on any exit path; $? holds the exit code of whatever caused it.
# `command -v apprise` guards a missing apprise binary so the trap itself
# never fails (which would mask the real error).
notify_on_exit() {
  local rc=$?
  if command -v apprise >/dev/null 2>&1; then
    if [ $rc -eq 0 ]; then
      apprise -b "AIIdea Generation Succeed" || true
    else
      apprise -b "AIIdea Generation Failed (exit code: $rc)" || true
    fi
  fi
}
trap notify_on_exit EXIT

# --- Resolve project root from this script's location ------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults (env override) -------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data_generation/aiidea/dataset}"
TRAJ_LIST="${TRAJ_LIST:-${OUTPUT_ROOT}/all_trajectories.txt}"
RECURSION_LIMIT="${RECURSION_LIMIT:-128}"

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
echo "[generate.sh] output_root:    $OUTPUT_ROOT"
echo "[generate.sh] traj_list:      $TRAJ_LIST"
echo "[generate.sh] recursion_lim:  $RECURSION_LIMIT"
[[ ${#GEN_EXTRA[@]} -gt 0 ]] && echo "[generate.sh] gen extras:     ${GEN_EXTRA[*]}"

cd "$PROJECT_ROOT"

# --- 1. Generate -------------------------------------------------------------
echo
echo "==> [1/2] Generating ideation trajectories"
"$PY" -m data_generation.aiidea.generation \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing \
    --recursion-limit "$RECURSION_LIMIT" \
    "${GEN_EXTRA[@]}"

# --- 2. Emit trajectory list -------------------------------------------------
echo
echo "==> [2/2] Building trajectory list at $TRAJ_LIST"

# Walk every output.json, pull the absolute trajectory path for runs with
# ok=true. We use python (not jq) so the script has zero non-stdlib deps
# beyond what generate already required.
"$PY" - "$OUTPUT_ROOT" "$TRAJ_LIST" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out  = Path(sys.argv[2])

total = ok = 0
lines: list[str] = []
for ws in sorted(root.iterdir()):
    if not (ws.is_dir() and ws.name.startswith("aiidea_")):
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
    history = ws / Path(rec.get("history_path", "ideation_agent_history.json")).name
    if history.is_file():
        lines.append(str(history.resolve()))

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
echo "      --out raw_datafiles/aiidea.jsonl \\"
echo "      --id-level 1"
