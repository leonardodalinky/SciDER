"""
SciEvo Workflow Module (Compatibility Layer)

The main workflow implementations have been moved to scievo.workflows package
"""

# Re-export for backward compatibility
from scievo.workflows.data_workflow import run_data_workflow
from scievo.workflows.experiment_workflow import run_experiment_workflow
from scievo.workflows.full_workflow import get_separator
from scievo.workflows.full_workflow import run_full_workflow as run_workflow

__all__ = [
    "run_workflow",
    "run_data_workflow",
    "run_experiment_workflow",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SciEvo Workflow CLI - Run different workflow types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (default)
  python -m scievo.run_workflow full data.csv workspace "Train SVR model"
  
  # Data analysis only
  python -m scievo.run_workflow data data.csv workspace
  
  # Experiment only
  python -m scievo.run_workflow experiment workspace "Train SVR"
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Workflow mode to run")
    subparsers.required = True

    # Full workflow subcommand
    full_parser = subparsers.add_parser("full", help="Run complete workflow")
    full_parser.add_argument("data_path", help="Path to the data file or directory")
    full_parser.add_argument(
        "workspace_path", help="Workspace directory for the experiment"
    )
    full_parser.add_argument("user_query", help="User's experiment objective")
    full_parser.add_argument(
        "repo_source",
        nargs="?",
        default=None,
        help="Optional repository source (local path or git URL)",
    )
    full_parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for DataAgent (default: 100)",
    )
    full_parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for ExperimentAgent (default: 100)",
    )
    full_parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (otherwise uses timestamp)",
    )

    # Data workflow subcommand
    data_parser = subparsers.add_parser("data", help="Run only DataAgent")
    data_parser.add_argument("data_path", help="Path to the data file or directory")
    data_parser.add_argument("workspace_path", help="Workspace directory")
    data_parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for DataAgent (default: 100)",
    )
    data_parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (otherwise uses timestamp)",
    )

    # Experiment workflow subcommand
    exp_parser = subparsers.add_parser("experiment", help="Run only ExperimentAgent")
    exp_parser.add_argument("workspace_path", help="Workspace directory")
    exp_parser.add_argument("user_query", help="User's experiment objective")
    exp_parser.add_argument(
        "data_analysis_path",
        nargs="?",
        default=None,
        help="Path to existing data_analysis.md file",
    )
    exp_parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for ExperimentAgent (default: 100)",
    )

    args = parser.parse_args()

    try:
        if args.mode == "full":
            result = run_workflow(
                data_path=args.data_path,
                workspace_path=args.workspace_path,
                user_query=args.user_query,
                repo_source=args.repo_source,
                data_agent_recursion_limit=args.data_recursion_limit,
                experiment_agent_recursion_limit=args.experiment_recursion_limit,
                session_name=args.session_name,
            )

            print("\n" + get_separator())
            print("FULL WORKFLOW COMPLETE")
            print(get_separator())
            print(f"\nStatus: {result.final_status}")
            print(f"\nFinal Summary:\n{result.final_summary}")

        elif args.mode == "data":
            result = run_data_workflow(
                data_path=args.data_path,
                workspace_path=args.workspace_path,
                recursion_limit=args.recursion_limit,
                session_name=args.session_name,
            )

            print("\n" + get_separator())
            print("DATA WORKFLOW COMPLETE")
            print(get_separator())
            print(f"\nStatus: {result.final_status}")
            print(f"\nData Summary:\n{result.data_summary}")

        elif args.mode == "experiment":
            result = run_experiment_workflow(
                workspace_path=args.workspace_path,
                user_query=args.user_query,
                data_analysis_path=args.data_analysis_path,
                recursion_limit=args.recursion_limit,
            )

            print("\n" + get_separator())
            print("EXPERIMENT WORKFLOW COMPLETE")
            print(get_separator())
            print(f"\nStatus: {result.final_status}")
            print(f"\nFinal Summary:\n{result.final_summary}")

    except Exception as e:
        print(f"\nError running workflow: {e}")
        import traceback

        traceback.print_exc()
        parser.exit(1)
