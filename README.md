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
uv pip install aider-chat
```

Since the `aider` break the tool system, the running command should be now:
```shell
uv run --no-sync <python_script>
```
