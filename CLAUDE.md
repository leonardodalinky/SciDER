# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SciEvo is a multi-agent framework for automated scientific experimentation. It orchestrates data analysis and experimental code generation through specialized agents that can search papers, generate code, execute experiments, and maintain long-term memory of insights.

## Setup and Environment

### Initial Setup

```bash
# Install dependencies (choose based on your platform)
# For macOS
uv sync --extra mac

# For CPU-only
uv sync --extra cpu

# For CUDA 12.8
uv sync --extra cu128

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Environment Configuration

Copy `.env.template` to `.env` and configure:

```bash
cp .env.template .env
```

Required environment variables:
- `OPENAI_API_KEY` - OpenAI API access
- `GEMINI_API_KEY` - Google Gemini API access
- `BRAIN_DIR` - Session storage location (default: `./tmp_brain`)

Optional configurations (see `.env.template` for full list):
- `REASONING_BANK_ENABLED` - Enable long-term memory consolidation
- `HISTORY_AUTO_COMPRESSION` - Auto-compress conversation history
- `CRITIC_ENABLED` - Enable agent output critique
- `CODING_AGENT_VERSION` - v2 or v3
- `AIDER_*` - Aider code editor configuration
- `OPENHANDS_MODEL` - Model for OpenHands integration

### Code Formatting

This project uses:
- **black** (line length: 100, target: py310)
- **isort** (profile: black, line length: 100)
- **nbstripout** for cleaning notebooks

Run formatting manually:
```bash
pre-commit run --all-files
```

## Running Workflows

### Full Workflow (Data Analysis + Experiment)

```bash
python -m scievo.run_workflow full <data_path> <workspace_path> "<user_query>" [repo_source]

# Example
python -m scievo.run_workflow full data.csv ./workspace "Train SVR model for regression"

# With options
python -m scievo.run_workflow full data.csv ./workspace "Train model" \
    --data-recursion-limit 100 \
    --experiment-recursion-limit 100 \
    --session-name my_experiment
```

### Data Analysis Only

```bash
python -m scievo.run_workflow data <data_path> <workspace_path> [--recursion-limit N] [--session-name NAME]

# Example
python -m scievo.run_workflow data data.csv ./workspace --session-name my_analysis
```

### Experiment Only (Requires Existing Analysis)

```bash
python -m scievo.run_workflow experiment <workspace_path> "<user_query>" [data_analysis_path] [--recursion-limit N]

# Example (uses data_analysis.md from workspace)
python -m scievo.run_workflow experiment ./workspace "Train SVR model"

# With custom analysis file
python -m scievo.run_workflow experiment ./workspace "Train model" ./my_analysis.md
```

## Architecture Overview

### Core Components

**`scievo/core/`** - Infrastructure and shared utilities
- `types.py` - Core message types, state management (ToolsetState, HistoryState, RBankState, ExecState)
- `brain.py` - Singleton session manager coordinating shared application state
- `llms.py` - Model registry with completion/response API wrappers (supports rate limiting, embeddings)
- `exec/` - Command execution sessions (SessionManager, PTYSession)
- `code_env.py` - Workspace context manager (LocalEnv)
- `utils.py` - TOON/JSON parsing, markdown extraction
- `constant.py` - Configuration flags and defaults

**`scievo/tools/`** - 20+ tool integrations
- Core: `fs_tool`, `shell_tool`, `exec_tool`
- Search: `arxiv_tool`, `dataset_search_tool`, `metric_search_tool`, `web_tool`
- Code: `coder_tool`, `cursor_tool`, `claude_code_tool`, `claude_agent_sdk_tool`, `openhands_tool`
- Other: `github_tool`, `ideation_tool`, `history_tool`, `state_tool`, `todo_tool`, `env_tool`
- Registry: `Tool` base class with JSON schemas, `ToolRegistry` singleton

**`scievo/agents/`** - Agent implementations using LangGraph
- `data_agent/` - Analyzes data, generates `data_analysis.md`, searches papers/datasets
  - Flow: START → planner → gateway (router) → llm_chat/tool_calling/mem_extraction → replanner → finalize → END
  - Sub-agents: `paper_subagent/` for academic search
- `experiment_agent/` - Generates and executes experimental code
  - Flow: START → init → coding → exec → summary → analysis → revision_judge → END
  - Sub-agents: CodingSubagent, ExecSubagent, SummarySubagent
- `ideation_agent/` - Research idea generation
- `critic_agent/` - Output quality review

**`scievo/workflows/`** - Workflow orchestration
- `full_workflow.py` - Chains DataAgent → ExperimentAgent
- `data_workflow.py` - Standalone DataAgent execution
- `experiment_workflow.py` - Standalone ExperimentAgent execution
- `run_workflow.py` - CLI entry point with three subcommands (backward compatibility layer)

**`scievo/prompts/`** - Prompt management
- `prompt_data.py` - Dataclass-based organization (DataPrompts, ExperimentPrompts, etc.)
- YAML files with Jinja2 templating for dynamic content

**`scievo/rbank/`** - ReasoningBank (Long-term Memory)
- `memo.py` - Persistent memory with embeddings for similarity search
- `subgraph/` - Memory consolidation subgraph
- Three memory tiers: short-term (session), long-term (cross-project), project-specific

### Key Architectural Patterns

1. **Singleton Pattern** - Brain, ModelRegistry, SessionManager, ToolRegistry ensure single instances
2. **State Graph Pattern** (LangGraph) - Agents as stateful graphs with nodes (steps) and edges (transitions)
3. **Sub-agent Composition** - Complex agents orchestrate specialized sub-agents
4. **History Compression** - Automatic message summarization to manage token usage
5. **Tool Registry** - Self-registering tools with JSON schemas for LLM consumption
6. **Memory Consolidation** - Periodic extraction of insights into long-term, project, and short-term memory

### Data Flow

```
run_workflow.py CLI
    ↓
FullWorkflow
    ├─→ DataWorkflow
    │   ├─→ DataAgent (planner → execution loop → finalize)
    │   │   └─→ PaperSubagent (searches papers/datasets)
    │   └─→ Output: data_analysis.md
    │
    └─→ ExperimentWorkflow
        ├─→ ExperimentAgent (init → coding → exec → summary → revision loop)
        │   ├─→ CodingSubagent
        │   ├─→ ExecSubagent
        │   └─→ SummarySubagent
        └─→ Output: metrics, final_summary

All agents use: Brain, ModelRegistry, ToolRegistry, Prompts, ReasoningBank
```

## Development Guidelines

### Agent State Management

Agents use LangGraph state objects that extend core state types:
- `HistoryState` - Message history with compression support
- `ToolsetState` - Available tools
- `RBankState` - Memory directories
- `ExecState` - Execution sessions

State is passed through node functions and updated via returns.

### Adding New Tools

1. Create tool in `scievo/tools/` directory
2. Inherit from `Tool` base class
3. Define `json_schema` property
4. Implement tool logic
5. Tool auto-registers on import via `ToolRegistry`

### Working with Memory

- Enable via `REASONING_BANK_ENABLED=true` in `.env`
- Extraction frequency controlled by `MEM_EXTRACTION_ROUND_FREQ`
- Three directories: short-term, long-term (MEM_LONG_TERM_DIR), project (MEM_PROJECT_DIR)
- Memories stored as markdown with embeddings for retrieval

### History Management

- Auto-compression enabled via `HISTORY_AUTO_COMPRESSION=true`
- Triggers at `HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD` (default: 64000)
- Keeps `HISTORY_AUTO_COMPRESSION_KEEP_RATIO` (default: 0.33) of messages
- Compression patches stored in `HistoryState.history_patches`

## File Locations

- Workflow implementations: `scievo/workflows/`
- Agent logic: `scievo/agents/{agent_name}/`
- Tool definitions: `scievo/tools/`
- Prompts: `scievo/prompts/` (YAML files) + `prompt_data.py` (dataclasses)
- Core infrastructure: `scievo/core/`
- Memory: Configured via `BRAIN_DIR`, `MEM_LONG_TERM_DIR`, `MEM_PROJECT_DIR`
- Generated outputs: Within workspace directory specified in CLI

## Testing and Debugging

### Jupyter Notebooks

Development notebooks are prefixed with `tmp_*`:
- `tmp_workflow_w_ideation.ipynb` - Full workflow with ideation
- `tmp_ideation_test.ipynb` - Ideation agent testing
- `tmp_paper_agent_test.ipynb` - Paper search testing
- Other `tmp_*.ipynb` files for component testing

### Logging

Control verbosity via `.env`:
```bash
LOGURU_LEVEL=DEBUG          # or INFO
LOG_MEM_SUBGRAPH=true       # Memory consolidation logs
LOG_SYSTEM_PROMPT=false     # Show system prompts
```

### Running Partial Workflows

Use mode-specific commands for testing individual components:
```bash
# Test only data analysis
python -m scievo.run_workflow data test_data/sample.csv ./debug_workspace

# Test experiment with existing analysis
python -m scievo.run_workflow experiment ./debug_workspace "Test query"
```

## Important Notes

- **Python Version**: Requires Python >=3.13 (see `pyproject.toml`)
- **Package Manager**: Uses `uv` for dependency management
- **PyTorch**: Platform-specific installation via custom indices (see `pyproject.toml` [tool.uv.sources])
- **Optional Dependencies**: OpenHands (`openhands-sdk`, `openhands-tools`) - enable via `SCIEVO_ENABLE_OPENHANDS`
- **Pre-commit Hooks**: Always run before committing to maintain code style
- **Temporary Files**: `tmp_*` directories and notebooks are for development, not production
- **Brain Directory**: Session state persists in `BRAIN_DIR` - can accumulate over time
