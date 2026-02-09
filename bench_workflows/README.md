# Benchmark for MLEBench

We assume all commands are run from the same dir as this README file.

First, create a new uv env:

```bash
uv init --python 3.12
```

Then, install the dependencies:

```bash
uv pip install -e ./mle-bench
```

Then, git-lfs all the mle-bench:

```bash
pushd mle-bench
git lfs fetch --all
git lfs pull
popd
```

Create `mlebench-env` docker image:

```bash
pushd mle-bench
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
popd
```

Create the agent docker images:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
export AGENT_NAME=scievo

pushd ../..
docker build --platform=linux/amd64 -t $AGENT_NAME -f benchmarks/mlebench/agent_docker/Dockerfile --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR
popd
```

Finally, you can run the benchmark:

```bash
cd mle-bench
python -m mlebench.run --model <model_name> --tasks <task_name>
```
