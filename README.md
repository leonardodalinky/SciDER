# SciEvo

## Installation

You can install the project using `pip`:

```shell
# from git
pip install git+https://github.com/leonardodalinky/SciEvo
# locally
pip install -e .
```

## Configuration

The project is configured using environment variables. You can set these variables in a `.env` file at the root of the project. A template `.env.template` is provided for reference.

Also, you can set environment variables directly in your shell or terminal session.

## Coding framwork

Currently we supports "OpenHands", "Claude Code" and "Claude Agent SDK" (Recommended) as coding framework. You can choose to install one or more of them.

### Optional: install Claude Code (for `claude_code` toolset):

- Ensure the `claude` CLI is installed and authenticated on your machine.
- If your `claude` command needs extra flags, set `CLAUDE_CODE_CMD`, e.g.:

```shell
export CLAUDE_CODE_CMD="claude"
```

Optional: install Claude Agent SDK (for `claude_agent_sdk` toolset):

- Docs: `https://platform.claude.com/docs/en/agent-sdk/overview`
- Install:

```shell
pip install claude-agent-sdk
export ANTHROPIC_API_KEY="..."
```

## Development Guide

First, install `pre-commit`:
```shell
pip install pre-commit
```

Install `pre-commit` to format code:
```shell
pre-commit install
```

Then, copy `.env.template` to `.env` and fill in the necessary values.

Finally, run the following command to sync dependencies:
```shell
# for cpu
uv sync --extra cpu

# for mac
uv sync --extra mac

# for gpu
uv sync --extra cu128
```
