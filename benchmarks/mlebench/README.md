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
# x86_64 host
pushd mle-bench; docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .; popd

# aarch64 host (e.g. Grace Hopper, Jetson, Graviton)
pushd mle-bench; docker build --platform=linux/arm64 -t mlebench-env -f environment/Dockerfile_aarch64 .; popd

# DGX Spark (GB10 / Blackwell sm_121) — NVIDIA NGC-based variant
# (``Dockerfile_aarch64`` only compiles torch kernels up to sm_90, so on
# GB10 CUDA kernels don't match the physical GPU. The DGX Spark build
# uses nvcr.io/nvidia/pytorch:25.12-py3 as base, which ships an
# sm_121-capable torch + CUDA 13.0.)
pushd mle-bench; docker build --platform=linux/arm64 -t mlebench-env -f environment/Dockerfile_dgx_spark .; popd
```

Notes for the base image (both amd64 and aarch64):

- TensorFlow was bumped from **2.17.0 → 2.19.0**. Reason: TF 2.19 is the
  first release whose `[and-cuda]` extra pins
  `nvidia-cudnn-cu12==9.3.0.75` and `nvidia-nccl-cu12==2.23.4` — both of
  which ship `manylinux2014_aarch64` wheels on PyPI. TF 2.17/2.18's
  cuDNN/NCCL pins were x86_64-only, making GPU-TF unreachable on aarch64.
- `tf-keras` was bumped to `2.19.0` in lockstep.
- `tensorpack==0.11` was **dropped** — the library has been unmaintained
  since 2021 and is not compatible with TF 2.19. If a competition really
  needs it, add it back to its own project's deps.
- The torch trio pins were bumped to `torch==2.5.1` /
  `torchvision==0.20.1` / `torchaudio==2.5.1` in
  `environment/requirements.txt` — the earliest set with full aarch64
  cu124 coverage.

aarch64-specific:

- `torch` / `torchvision` / `torchaudio` are installed from pytorch.org's
  cu124 aarch64 index, so **PyTorch runs on GPU** (SBSA hosts: Grace
  Hopper, Graviton-with-GPU, etc.).
- `tensorflow[and-cuda]==2.19` is now installed on aarch64 too — **TF
  also runs on GPU**.
- See the comments at the top of `Dockerfile_aarch64` for details.

Create the agent docker images:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
export AGENT_NAME=scider

# x86_64 host
pushd ../..; docker build --platform=linux/amd64 -t $AGENT_NAME -f benchmarks/mlebench/mle-bench/agents/scider/Dockerfile --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR .; popd

# aarch64 host
pushd ../..; docker build --platform=linux/arm64 -t $AGENT_NAME -f benchmarks/mlebench/mle-bench/agents/scider/Dockerfile --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR .; popd
```

Model assignments for this benchmark live in
`bench_workflows/model_configs/mlebench_roles.yaml` (in the SciDER source
tree). Edit that yaml to swap models — there is no longer a `--models` CLI
flag or per-preset agent id, so `config.yaml` has a single `scider` entry.

Finally, you can run the benchmark:

```bash
export AGENT_ID=scider

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
