# SciEvo

```shell
# for cpu
uv sync --extra cpu

# for mac
uv sync --extra mac

# for gpu
uv sync --extra cu128
```

After install uv's dependencies, install `aider`:
```shell
uv tool install --force --python python3.12 aider-chat@latest
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
