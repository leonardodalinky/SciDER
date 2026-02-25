# SciDER: Scientific Data-centric End-to-end Researcher

<div align="center">
    <a href="https://harryluumn.github.io/scider-proj-page/"><img src="https://img.shields.io/badge/Project Page-blue?style=for-the-badge&color=1a1a2e&logo=homepage&logoColor=orange" alt="Project Page"></a>
    <a href="https://huggingface.co/spaces/AI4Research/scider"><img src="https://img.shields.io/badge/Live DEMO-1a1a2e?logo=huggingface&style=for-the-badge" alt="Live Demo"></a>
    <br/>
    <a href="TODO"><img src="https://img.shields.io/badge/arXiv-TODO.TODO-brightred?color=B31B1B&logo=arXiv&style=for-the-badge" alt="ArXiv"></a>
    <br/>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.12-blue?logo=python&style=for-the-badge" alt="Python 3.12">
    </a>
    <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&style=for-the-badge" alt="pre-commit">
    </a>
</div>

## Installation

You can install the project using `pip`:

```shell
# from git
pip install git+https://github.com/leonardodalinky/SciDER
# locally
pip install -e .
```

## Configuration

The project is configured using environment variables. You can set these variables in a `.env` file at the root of the project. A template `.env.template` is provided for reference.

Also, you can set environment variables directly in your shell or terminal session.

## UI Deployment

The web UI is a Streamlit application. Deploy it using the `Dockerfile` at the project root.

1. Create a `.env` file at the project root (copy from `.env.template`) and fill in your API keys.

2. Build the image:

```shell
docker build -t scider:latest .
```

3. Run the container:

```shell
docker run -d \
  --name scider \
  -p 7860:7860 \
  --env-file .env \
  scider:latest
```

4. Access the UI at `http://localhost:7860`.

## Coding framework

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

## Benchmarks

See [BENCHMARKS](./benchmarks) for details on the benchmarks we have conducted to evaluate SciDER's performance.

## Feedback and Contributions

We welcome contributions to improve SciDER. Please open an issue or submit a pull request on our GitHub repository.

Also, any feedback on the project is greatly appreciated. You can fill the [feedback form](https://forms.gle/Vz4K55J8ePTEs6TU8) to rate this app and help to improve the project.
