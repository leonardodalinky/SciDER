# Training the OpenSciDER

LoRA SFT of a Qwen-family base model on SciDER agent trajectories using
[LLaMA-Factory](LlamaFactory/).

## 1. Install dependencies

All dependencies are declared in [`pyproject.toml`](pyproject.toml). From
this `train/` directory:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` installs:

- **LlamaFactory** (vendored under [`LlamaFactory/`](LlamaFactory/), editable; carries a
  local patch — see §4)
- **deepspeed**, **liger-kernel** — distributed training + faster kernels
- **tiktoken** — used by `preprocess_data.py`
- **flash-linear-attention ≥ 0.4.1** — Qwen3.5 hybrid linear-attention kernels
- **tilelang** — workaround for the Triton ≥ 3.4 backward bug on Hopper (H100/H200)

`flash-linear-attention` and `tilelang` are only needed for `Qwen3.5-*`
models. They install fast and don't break other model families, so they
stay in the pinned set.

`transformers >= 4.57.0.dev` (required by Qwen3.5) is pulled in
transitively via LlamaFactory.

## 2. Prepare the SFT dataset

Two-stage pipeline:

### 2a. Collect trajectories from SciDER workspaces

```bash
python prepare_data.py \
    --workspace-list workspaces.txt \
    --out raw_datafiles/<run_name>.jsonl \
    [--id-level 1]
```

`workspaces.txt` is one workspace path per line (`#` for comments).
`--id-level N` picks the Nth-from-end directory segment as `workspace_id`
(default 1 = basename). Each output row is one agent or subagent
trajectory: `{id, workspace_id, agent_id, trajectory_id, source_path, messages}`.

### 2b. Preprocess for SFT

```bash
python preprocess_data.py \
    --input  raw_datafiles/ \
    --out    dataset/datafiles/ \
    --merge-consecutive \
    --max-tool-tokens 512 \
    --max-message-tokens 16384 \
    --minimal
```

Per-step behavior:

- `--merge-consecutive` — collapse adjacent same-role messages (assistant
  text + assistant tool_calls become one combined turn)
- `--max-tool-tokens 512` — head/tail-truncate any tool result longer than 512 tokens
- `--max-message-tokens 16384` — split rows whose messages exceed 16k tokens
  at user-turn boundaries; pieces emit as `{id}#part01`, `#part02`...
- `--minimal` — drop everything but `id` / `messages` / `datasource` (the
  source filename without `.jsonl`)

Internally the script also normalizes message schema (uniform `role`,
`content`, `tool_calls`, `tool_call_id`), trims leading non-user / trailing
non-assistant junk, drops orphan tool calls, and skips rows that won't
satisfy LF's strict `user/tool ↔ assistant` alternation rule.

The output directory `dataset/datafiles/` is what `dataset/README.md`
points to as the HuggingFace dataset; both `huggingface_hub` and LF can
load it directly.

## 3. Run training

The `SciderTraj` dataset is registered in
[LlamaFactory/data/dataset_info.json](LlamaFactory/data/dataset_info.json)
with `formatting: openai`, pointing at `../../dataset/datafiles/`.

```bash
bash train.sh
```

Defaults: 2× GPU (`CUDA_VISIBLE_DEVICES=0,1`), config
`examples/train_lora/qwen3_5_9b_scider_lora_sft.yaml`.

Override via env vars (the script materializes a temp YAML; LF refuses raw
CLI overrides when launched with a config file):

```bash
CUDA_VISIBLE_DEVICES=2,3 bash train.sh
CONFIG=examples/train_lora/qwen3_lora_sft.yaml bash train.sh
OUTPUT_DIR=/scratch/me/scider_lora bash train.sh
NUM_TRAIN_EPOCHS=2 LEARNING_RATE=5e-5 bash train.sh
```

Recognized env overrides: `OUTPUT_DIR`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`,
`PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `CUTOFF_LEN`,
`LORA_RANK`, `LORA_ALPHA`. To override anything else, edit the YAML or
extend the `OVERRIDES` map in [train.sh](train.sh).

## 4. Merge LoRA into the base model

The training output is a LoRA adapter (~150 MB), not a full model.
[`merge.sh`](merge.sh) folds it back into the base weights so the result
loads with plain `AutoModelForCausalLM.from_pretrained(...)`:

```bash
ADAPTER_DIR=/path/to/saves/qwen3.5-9b-scider/lora/sft \
EXPORT_DIR=/path/to/scider_qwen3_5_9b_merged \
    bash train/merge.sh
```

Both env vars are required. Optional overrides: `MODEL_NAME` (base model
id), `EXPORT_SIZE` (GB per shard, default 5), `EXPORT_DEVICE`
(`cpu` default; `auto` is faster but needs ≥ 25 GB free VRAM), `TEMPLATE`,
`CONFIG` (point at a different merge YAML).

To merge a specific intermediate checkpoint instead of the final adapter,
point `ADAPTER_DIR` at it directly:

```bash
ADAPTER_DIR=/path/.../sft/checkpoint-800 ...
```

After merging, the resulting directory contains
`config.json`, `model-*.safetensors`, `tokenizer*` — load it like any HF model.

The provided YAML uses LoRA r=32 / α=64, `cutoff_len=16384`, ZeRO-2,
flash-attn-2, gradient checkpointing — fits comfortably on 2× H200.

## 4. Notes on the LF formatter patch

[`LlamaFactory/src/llamafactory/data/formatter.py`](LlamaFactory/src/llamafactory/data/formatter.py)
has a small local patch in `_parse_functions` that accepts both string
and dict `arguments`. Without it, LF re-serializes our string `arguments`
into a doubly-encoded JSON, which Qwen's tool formatter can't decode.
Keep this patch when pulling LF upstream changes.
