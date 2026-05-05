"""Prepare an SFT dataset from SciDER trajectory JSONs.

Walks every workspace listed in a text file (one workspace path per line),
collects agent and subagent trajectories, and emits one JSONL row per
trajectory.

Each emitted row::

    {
      "id": "<workspace_id>/<agent_id>/<trajectory_id>",
      "workspace_id": "<basename of the workspace dir>",
      "agent_id":     "<data|ideation|experiment|writing|approval|critic|...>",
      "trajectory_id":"<main | 001 | 002 | ...>",
      "source_path":  "<absolute path to the source JSON>",
      "messages":     [ ...the array stored by save_conversation_history... ]
    }

Trajectory layout per workspace (see scider/workflows/history_export.py)::

    <workspace>/
        <agent>_agent_history.json        # main agent trajectory (trajectory_id="main")
        subagents/<type>_NNN.json         # one per subagent invocation

Usage::

    python train/prepare_data.py \\
        --workspace-list workspaces.txt \\
        [--out data.jsonl]

The workspaces file uses ``#``-prefixed lines as comments; blank lines are
ignored. Workspace paths can be absolute or relative to the current working
directory. Default output is ``./data.jsonl`` in the current working
directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

# Pattern for main agent trajectory files at workspace root.
_MAIN_TRAJ_RE = re.compile(r"^(?P<agent>[A-Za-z0-9_]+)_agent_history\.json$")
# Pattern for subagent trajectory files under <workspace>/subagents/.
_SUBAGENT_TRAJ_RE = re.compile(r"^(?P<agent>[A-Za-z0-9_]+)_(?P<idx>\d{3})\.json$")


def _read_workspace_list(list_path: Path) -> list[Path]:
    """Read a workspace list file. Each non-empty, non-``#`` line is a path."""
    workspaces: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        workspaces.append(Path(line).expanduser())
    return workspaces


def _iter_trajectory_files(workspace: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(agent_id, trajectory_id, file_path)`` for every trajectory
    JSON found under a workspace.

    - Main agent JSONs at the workspace root use ``trajectory_id="main"``.
    - Subagent JSONs under ``subagents/`` use the 3-digit numeric suffix
      (e.g. ``"001"``) as their ``trajectory_id``.
    """
    if not workspace.is_dir():
        return

    for entry in sorted(workspace.iterdir()):
        if not entry.is_file():
            continue
        m = _MAIN_TRAJ_RE.match(entry.name)
        if m:
            yield m.group("agent"), "main", entry

    sub_dir = workspace / "subagents"
    if sub_dir.is_dir():
        for entry in sorted(sub_dir.iterdir()):
            if not entry.is_file():
                continue
            m = _SUBAGENT_TRAJ_RE.match(entry.name)
            if m:
                yield m.group("agent"), m.group("idx"), entry


def _load_messages(traj_path: Path) -> list | None:
    """Load a trajectory JSON. Returns the messages array or None on failure."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[skip] {traj_path}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, list):
        print(
            f"[skip] {traj_path}: top-level must be a list, got {type(data).__name__}",
            file=sys.stderr,
        )
        return None
    return data


def _workspace_id(ws: Path, id_level: int) -> str:
    """Return the directory segment ``id_level`` levels from the end.

    ``id_level=1`` → basename (default), ``id_level=2`` → parent dir name, etc.
    Falls back to basename if the path is shallower than ``id_level``.
    """
    parts = ws.resolve().parts
    if id_level < 1:
        raise ValueError(f"--id-level must be >= 1, got {id_level}")
    if id_level > len(parts):
        print(
            f"[warn] {ws}: path has only {len(parts)} segments, "
            f"falling back to basename for workspace_id",
            file=sys.stderr,
        )
        return ws.name or str(ws)
    return parts[-id_level]


def build_records(workspaces: list[Path], id_level: int = 1) -> Iterator[dict]:
    """Walk each workspace and yield one SFT record per trajectory file."""
    for ws in workspaces:
        if not ws.is_dir():
            print(f"[skip] workspace not a directory: {ws}", file=sys.stderr)
            continue
        ws_id = _workspace_id(ws, id_level)
        for agent_id, traj_id, path in _iter_trajectory_files(ws):
            messages = _load_messages(path)
            if not messages:
                continue
            yield {
                "id": f"{ws_id}/{agent_id}/{traj_id}",
                "workspace_id": ws_id,
                "agent_id": agent_id,
                "trajectory_id": traj_id,
                "source_path": str(path.resolve()),
                "messages": messages,
            }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workspace-list",
        type=Path,
        required=True,
        help="Text file with one workspace path per line (# comments allowed).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data.jsonl"),
        help="Output JSONL path. Default: ./data.jsonl (current working directory).",
    )
    parser.add_argument(
        "--id-level",
        type=int,
        default=1,
        help=(
            "Which directory segment of the workspace path to use as "
            "workspace_id, counted from the end. 1 = basename (default), "
            "2 = parent directory name, etc."
        ),
    )
    args = parser.parse_args()

    if not args.workspace_list.is_file():
        print(f"--workspace-list not found: {args.workspace_list}", file=sys.stderr)
        return 2

    workspaces = _read_workspace_list(args.workspace_list)
    if not workspaces:
        print(f"No workspaces listed in {args.workspace_list}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_records = 0
    n_messages = 0
    with args.out.open("w", encoding="utf-8") as fp:
        for record in build_records(workspaces, id_level=args.id_level):
            fp.write(json.dumps(record, ensure_ascii=False))
            fp.write("\n")
            n_records += 1
            n_messages += len(record["messages"])

    print(
        f"Wrote {n_records} trajectories ({n_messages} messages total) "
        f"from {len(workspaces)} workspace(s) → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
