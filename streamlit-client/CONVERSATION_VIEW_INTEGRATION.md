# Conversation View Integration Guide

This document explains how to integrate agent message logging with the unified conversation view in the Streamlit interface.

## Overview

The enhanced Streamlit app (`app_enhanced.py`) now displays all agent and subagent messages in a unified conversation log, similar to ChatGPT. Messages are captured in real-time during workflow execution and displayed with clear agent labels.

## Architecture

1. **WorkflowMonitor** - Central message logging system (in `workflow_monitor.py`)
2. **Conversation Log** - Session state storage for all messages
3. **Display Function** - Renders messages in a chat-like interface

## How It Works

### Message Flow

```
Agent/Subagent → WorkflowMonitor.log_update() → Callback → Session State → Display
```

### Message Structure

Each message has:
- `timestamp`: When the message was logged
- `agent_name`: Name of the agent/subagent (e.g., "Data Agent", "Coding Subagent")
- `message`: The actual message content
- `message_type`: Type of message (status, thought, action, result, error)
- `phase`: Workflow phase (from PhaseType enum)
- `status`: Message status (started, progress, completed, error)

## Integration Steps

### 1. In Agent Code

To log messages from agents, import the monitor and call `log_update()`:

```python
from workflow_monitor import get_monitor, PhaseType

class DataAgent:
    def __init__(self):
        self.monitor = get_monitor()

    def analyze_data(self, data_path):
        # Log that we're starting analysis
        self.monitor.log_update(
            phase=PhaseType.DATA_EXECUTION,
            status="started",
            message=f"Starting analysis of {data_path}",
            agent_name="Data Agent",
            message_type="action"
        )

        # ... do the analysis ...

        # Log thoughts/reasoning
        self.monitor.log_update(
            phase=PhaseType.DATA_EXECUTION,
            status="progress",
            message="Dataset has 1000 rows and 20 features. Detecting column types...",
            agent_name="Data Agent",
            message_type="thought"
        )

        # Log results
        self.monitor.log_update(
            phase=PhaseType.DATA_EXECUTION,
            status="completed",
            message="Analysis complete: Found 15 numeric and 5 categorical features",
            agent_name="Data Agent",
            message_type="result"
        )
```

### 2. In Subagents

Subagents should identify themselves clearly:

```python
class PaperSubagent:
    def search_papers(self, query):
        monitor = get_monitor()

        monitor.log_update(
            phase=PhaseType.DATA_PAPER_SEARCH,
            status="progress",
            message=f"Searching arXiv for: {query}",
            agent_name="Paper Search Subagent",
            message_type="action"
        )

        # ... search logic ...

        monitor.log_update(
            phase=PhaseType.DATA_PAPER_SEARCH,
            status="completed",
            message=f"Found {len(papers)} relevant papers",
            agent_name="Paper Search Subagent",
            message_type="result"
        )
```

### 3. In LangGraph Nodes

For LangGraph-based agents, log at key decision points:

```python
def planner_node(state):
    monitor = get_monitor()

    monitor.log_update(
        phase=PhaseType.DATA_PLANNING,
        status="progress",
        message="Planning analysis strategy based on data characteristics",
        agent_name="Data Agent - Planner",
        message_type="thought"
    )

    # Generate plan with LLM
    plan = llm.invoke(messages)

    monitor.log_update(
        phase=PhaseType.DATA_PLANNING,
        status="completed",
        message=f"Generated plan with {len(plan.steps)} steps",
        agent_name="Data Agent - Planner",
        message_type="result"
    )

    return {"plan": plan}
```

### 4. Error Handling

Always log errors:

```python
try:
    result = execute_code(code)
except Exception as e:
    monitor.log_update(
        phase=PhaseType.EXPERIMENT_EXEC,
        status="error",
        message=f"Execution failed: {str(e)}",
        agent_name="Execution Subagent",
        message_type="error"
    )
    raise
```

## Message Type Guidelines

Use appropriate message types for better visual organization:

- **status**: High-level phase transitions
  - "Starting data analysis workflow"
  - "Moving to experiment phase"

- **thought**: Agent reasoning and planning
  - "Based on the data distribution, I'll use log transformation"
  - "This error suggests we need to add exception handling"

- **action**: Concrete actions being taken
  - "Calling arXiv API with query: transformer models"
  - "Executing code in sandbox environment"
  - "Writing results to workspace/output.json"

- **result**: Outcomes and findings
  - "Found 15 relevant papers"
  - "Model achieved 0.85 accuracy"
  - "Generated 3 experiment variations"

- **error**: Errors and warnings
  - "API rate limit exceeded, retrying in 60s"
  - "Code execution failed: NameError"

## Best Practices

1. **Be Specific**: Include relevant details (file names, counts, values)
2. **Use Clear Agent Names**: Help users understand which component is active
3. **Log Key Decisions**: Capture important reasoning steps
4. **Balance Detail**: Not too verbose, but enough for transparency
5. **Group Related Actions**: Use consistent agent names for related operations

## Example: Full Workflow Integration

Here's how a complete workflow might generate conversation messages:

```
[System] Starting workflow for: train SVR model for regression

[Ideation Agent] Searching literature for SVR regression approaches
[Ideation Agent] Found 12 relevant papers on SVR methods
[Ideation Agent] Generated 3 novel research ideas

[Data Agent - Planner] Analyzing dataset structure
[Data Agent - Planner] Planning analysis: EDA → Feature Engineering → Baseline
[Data Agent] Loading data from data/housing.csv
[Data Agent] Dataset: 506 rows, 13 features, 1 target
[Paper Search Subagent] Searching for SVR regression papers
[Paper Search Subagent] Found 8 papers with implementation details

[Experiment Agent] Initializing experiment setup
[Coding Subagent] Generating SVR training code
[Coding Subagent] Code includes: data loading, preprocessing, model training
[Execution Subagent] Running generated code in sandbox
[Execution Subagent] Code executed successfully
[Summary Subagent] Model RMSE: 3.47, R²: 0.82
[Experiment Agent] Results look promising, finalizing experiment

[System] Workflow completed successfully!
```

## Current Status

✅ **Completed:**
- Unified conversation display in Streamlit UI
- Message logging infrastructure (WorkflowMonitor)
- Real-time message updates during execution
- Color-coded message types
- Agent name labeling

⏳ **To Do:**
- Instrument agents in `scievo/agents/` to log messages
- Add logging to LangGraph nodes
- Capture tool use events
- Log LLM reasoning steps

## Testing

To test the conversation view without full agent integration:

```python
# In any agent or test code
from workflow_monitor import get_monitor, PhaseType

monitor = get_monitor()
monitor.log_update(
    phase=PhaseType.DATA_EXECUTION,
    status="progress",
    message="Test message",
    agent_name="Test Agent",
    message_type="thought"
)
```

The message will appear in the Streamlit conversation view immediately.
