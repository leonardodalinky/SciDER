# SciDER Streamlit Interface

Web UI for running SciDER research workflows with real-time progress and interactive approval.

## Quick Start

From the project root:

```bash
bash streamlit-client/run.sh
```

Or manually:

```bash
uv sync --extra streamlit
uv run python -m streamlit run streamlit-client/app.py --server.port 7860
```

Open http://localhost:7860 in your browser.

## First-Time Setup

On first launch, a settings page will appear. Configure:

- **Model Provider** (Gemini / OpenAI) and API key (required)
- **Anthropic API Key** (optional — for Claude coding agent)
- **OpenAI API Key for Embeddings** (optional — enables memory features)
- **Per-role model assignments** (which model to use for each agent role)

Settings (including API keys) are stored in the browser's localStorage and never saved on the server.

## Workflows

| Workflow | Description |
|----------|-------------|
| **Ideation** | Generate research ideas from literature review |
| **Data Analysis** | Analyze data files, search related papers, produce insights |
| **Experiment** | Generate code, execute experiments, iterate with revisions |
| **Full Workflow** | Chain ideation → data analysis → experiment |

## User Approval

When `USER_APPROVAL_ENABLED=true` (default), the UI pauses at key checkpoints for user review:

- **Approve** — continue to next step
- **Reject** — retry the current step
- **Feedback** — provide guidance and retry

For ideation, users can also select a specific research idea to pass to the experiment agent.

## Docker

From the project root:

```bash
docker compose up --build
```

## File Structure

```
streamlit-client/
├── app.py                  # Main entry point
├── settings.py             # Persistent settings (browser localStorage)
├── utils.py                # Shared utilities (upload, chat history)
├── forms/                  # Workflow form UIs
│   ├── ideation.py
│   ├── data.py
│   ├── experiment.py
│   ├── full.py
│   └── settings.py         # Settings form
├── components/
│   └── display.py          # Approval UI, rendering helpers
├── workflow/
│   ├── approval.py         # StreamlitApprovalHandler
│   ├── runner.py            # Background thread executor
│   ├── monitor.py           # Progress tracking
│   ├── node_monitor.py      # Node-level monitoring
│   └── observable_list.py
├── log_utils/
│   └── handler.py           # Loguru → Streamlit bridge
├── run.sh / run.bat         # Launch scripts
└── case-study-memory/       # Saved chat histories
```
