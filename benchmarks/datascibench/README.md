# Benchmark for DataSciBench

We assume all commands are run from the same dir as this README file.

First, create a new uv env:

```bash
uv init --python 3.12
```

Then, install the dependencies:

```bash
uv pip install -e ./DataSciBench/MetaGPT
uv pip install -r ../../requirements.txt
uv pip install -r ./DataSciBench/requirements.txt
```
