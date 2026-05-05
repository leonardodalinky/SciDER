---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- agent
- sft
- trajectories
- scider
configs:
- config_name: default
  data_files:
  - split: train
    path: datafiles/*.jsonl
---

# SciDER SFT Trajectories

Supervised fine-tuning dataset of agent message trajectories collected from
SciDER workflow runs. Each row is a single trajectory (one main agent or one
subagent invocation) extracted from a workspace's saved history JSON.

## Generation

Built with [`train/prepare_data.py`](../prepare_data.py) — see that script
for the full extraction logic.

## Schema

Each line in `datafiles/*.jsonl` is one trajectory:

| field           | type         | description |
|-----------------|--------------|-------------|
| `id`            | `string`     | `<workspace_id>/<agent_id>/<trajectory_id>` |
| `workspace_id`  | `string`     | Workspace directory segment (controlled by `--id-level`) |
| `agent_id`      | `string`     | Agent name, e.g. `data`, `experiment`, `approval`, `critic` |
| `trajectory_id` | `string`     | `main` for top-level agents, zero-padded `001` / `002` / ... for subagents |
| `source_path`   | `string`     | Absolute path of the source JSON the row was extracted from |
| `messages`      | `list[dict]` | Message array as saved by `save_conversation_history` |

### Per-message keys

Inside `messages`, each entry follows the SciDER history format:

- `role`: `user` / `assistant` / `tool` / `system`
- `content`: text content (restored to pre-snip / pre-persist form)
- `agent_sender`: which agent emitted the message
- `tool_name` / `tool_call_id`: present when relevant
- `tool_calls`: present on assistant turns that invoke tools
- `is_meta`: `true` for programmatically injected messages
- `is_compact_boundary`, `compact_metadata`: present at autocompact breakpoints

## Loading

```python
from datasets import load_dataset
ds = load_dataset("<this-repo>", split="train")
```
