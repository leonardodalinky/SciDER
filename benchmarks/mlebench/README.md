# Benchmark for MLEBench

We assume all commands are run from the same dir as this README file.

First, create a new uv env:

```bash
uv init --python 3.12
uv venv --seed
source .venv/bin/activate
```

Then, install the dependencies:

```bash
pip install -e ./mle-bench
```

Then, git-lfs all the mle-bench:

```bash
pushd mle-bench
git lfs fetch --all
git lfs pull
popd
```

Later, prepare the data:

```bash
pushd mle-bench; mlebench prepare --lite; popd
```

Create `mlebench-env` docker image:

```bash
pushd mle-bench; docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .; popd
```

Create the agent docker images:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
export AGENT_NAME=scider

pushd ../..; docker build --platform=linux/amd64 --no-cache -t $AGENT_NAME -f benchmarks/mlebench/mle-bench/agents/scider/Dockerfile --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR .; popd
```

Finally, you can run the benchmark:

```bash
# Set agent ID from `agents/scider/config.yaml`
export AGENT_ID=scider/gemini-low-medium

# GPU
pushd mle-bench/; python run_agent.py --agent-id $AGENT_ID --competition-set experiments/splits/low.txt --container-config environment/config/container_configs/gpu.json; popd
# CPU only
pushd mle-bench/; python run_agent.py --agent-id $AGENT_ID --competition-set experiments/splits/low.txt; popd
```

Grading:

```bash
RUN_GROUP=runs/<run-group>

pushd mle-bench/
python experiments/make_submission.py --metadata $RUN_GROUP/metadata.json --output $RUN_GROUP/submission.jsonl
mlebench grade --submission $RUN_GROUP/submission.jsonl --output-dir $RUN_GROUP
popd
```
