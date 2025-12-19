"""
SciEvo Workflow Module (Compatibility Layer)

This module is kept for backward compatibility.
For new code, please use: from scievo.workflows import FullWorkflow

The main workflow implementation has been moved to scievo.workflows.full_workflow
"""

# Re-export for backward compatibility
from scievo.workflows.full_workflow import (
    FullWorkflow as SciEvoWorkflow,
    run_full_workflow as run_workflow,
)

__all__ = [
    "SciEvoWorkflow",
    "run_workflow",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python -m scievo.workflow <data_path> <workspace_path> <user_query> [repo_source]"
        )
        sys.exit(1)

    result = run_workflow(
        data_path=sys.argv[1],
        workspace_path=sys.argv[2],
        user_query=sys.argv[3],
        repo_source=sys.argv[4] if len(sys.argv) > 4 else None,
    )

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nStatus: {result.final_status}")
    print(f"\nFinal Summary:\n{result.final_summary}")
