---
name: huggingface-model-search
description: Find pretrained models on HuggingFace Hub by task, keyword, size, or downloads. Use when you need a base checkpoint for fine-tuning (LoRA/PEFT/AFT) or a zero-shot baseline. Returns model IDs ready for `AutoModel.from_pretrained(...)`.
allowed_agents: [experiment, native_coding]
preload_for: []
---

# HuggingFace Model Search

Find a pretrained model on the HuggingFace Hub to fine-tune or use zero-shot.
Use this when the task spec names a dataset/metric and you need a strong
backbone — e.g. AIRS-Bench tasks where training from scratch is off the table.

## When to use

- The task asks you to beat a SOTA paper on a named dataset — find the model the
  paper (or a close descendant) released on HF.
- You need a backbone of a particular size (e.g. ~7B, ≤16B) for a task family
  (text classification, QA, code gen, molecular property, time series, ...).
- You want to check whether a candidate model actually exists + has the
  expected files (config, tokenizer, weights) before writing training code.

## Method 1 — HF Hub API (preferred)

`huggingface_hub` is already available in most workspaces (or installable via
`uv add huggingface_hub`). Use `HfApi().list_models(...)` — it supports filters
for task, library, language, and sorts by downloads/likes/trending.

```python
from huggingface_hub import HfApi
api = HfApi()

# Example: top-downloaded text-classification models, sorted
models = api.list_models(
    task="text-classification",           # pipeline_tag filter
    sort="downloads",                     # or "likes", "trending", "lastModified"
    direction=-1,
    limit=30,
    search="sentiment",                   # free-text name/tag search (optional)
)
for m in models:
    print(m.id, m.downloads, m.tags[:6])
```

Common `task=` values: `text-classification`, `text-generation`,
`question-answering`, `token-classification`, `translation`, `summarization`,
`sentence-similarity`, `image-classification`, `time-series-forecasting`,
`graph-ml`. The full list is at https://huggingface.co/tasks.

### Filter by size (rough — use tags + model card)

`list_models` doesn't expose a numeric param count, so filter by the `tags`
convention (`7b`, `13b`, `mistral`, `llama`) or by the org/name prefix, then
confirm by reading the model card:

```python
models = api.list_models(task="text-generation", tags=["7b"], sort="downloads", limit=20)
```

For size-precise filtering, fetch the model card and parse:

```python
info = api.model_info("meta-llama/Llama-3.1-8B")
print(info.safetensors.total)   # total param count (when safetensors is used)
```

### Verify the checkpoint has what you need

```python
info = api.model_info("mistralai/Mistral-7B-v0.3", files_metadata=True)
files = [f.rfilename for f in info.siblings]
print(files)   # expect config.json, tokenizer.json, *.safetensors, ...
```

## Method 2 — WebSearch / WebFetch

When the API returns nothing useful or you want to find which model a paper
released, fall back to:

- `WebSearch` with `"<dataset name> huggingface model"` or `"<paper title>
  huggingface"`.
- `WebFetch` `https://huggingface.co/models?search=<keyword>&pipeline_tag=<task>`
  — parse the HTML result list.

## Loading the chosen model

Use the right task-specific class (not bare `AutoModel`) unless you're adding
a custom head:

```python
# Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mdl = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Causal LM (for instruction-tuned LLMs used zero/few-shot)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
mdl = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="bfloat16", # save VRAM
    device_map="auto",
)

# QA
from transformers import AutoModelForQuestionAnswering
```

For **LoRA / QLoRA** on a ~7B checkpoint:

```python
# uv add peft bitsandbytes accelerate
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb,
    device_map="auto",
)
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base, lora)
```

## Gated models

Some HF checkpoints (Llama, Gemma, Mistral) require accepting a license on the
web UI + a token. If you hit `GatedRepoError`:

1. Pick an open alternative of similar size/quality (e.g. `Qwen/Qwen2.5-7B`,
   `mistralai/Mistral-7B-v0.3` once accepted, `google/gemma-2-9b` once
   accepted, `HuggingFaceTB/SmolLM2-*`, `tiiuae/falcon-7b`).
2. Or set `HF_TOKEN` if the user has already accepted the license off-band.

## Reference table — strong open backbones (updated for 2026)

Use this as a **fallback** when your own search turns up nothing better.
Always prefer the most recent model that the task leaderboard / recent papers
point to — by the time you read this, something newer may exist. Verify every
candidate with `api.model_info(...)` before committing.

**Size / architecture constraints for this benchmark:**
- **No MoE models.** Mixture-of-Experts (e.g. `Qwen3-30B-A3B`, `Llama-4-Scout`,
  `Mixtral`, any `*-A3B*` / `*-MoE*` variant) are unstable to fine-tune with a
  modest single-GPU budget — stick to dense models only.
- **Hard cap: ≤16B parameters.** Skip anything labeled 17B+, 27B+, 32B, 70B,
  etc. — won't fit a single-GPU LoRA run here. Dense 4B / 7B / 8B / 12B / 14B
  are the sweet spot.

| Family | Recommended (recent) | Notes |
|---|---|---|
| General causal LM (dense, newest) | `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-2B` | **Qwen3.5 released 2026-02/03** — current Qwen generation, dense variants. Prefer `Qwen3.5-9B` as the default ≤16B dense option. |
| General causal LM (dense, previous gen) | `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B` | Released 2025-04-29. Has an `enable_thinking` switch; set `enable_thinking=False` for Qwen2.5-Instruct-style behavior. Fallback if Qwen3.5 misbehaves. |
| General causal LM (dense, budget) | `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3.5-0.8B` | Instruct variants, good when VRAM is tight. |
| General causal LM (Llama family) | `meta-llama/Llama-3.1-8B` | Gated. |
| General causal LM (Google) | `google/gemma-3-4b-it`, `google/gemma-3-12b-it` | Gemma 3, gated. Multimodal variants available. Stop at 12B — the 27B is borderline. |
| Legacy ~7B fallback (no gating) | `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-v0.3` (once license accepted) | Use only if newer options are unavailable. |
| Code | `Qwen/Qwen2.5-Coder-7B-Instruct`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b` | Stick to dense checkpoints. |
| Math | `Qwen/Qwen2.5-Math-7B-Instruct`, `Qwen/Qwen3-4B-Thinking-2507` | Thinking-mode Qwen3 is strong on math. |
| Encoder (classification / QA / NLI, <1B → do AFT) | `microsoft/deberta-v3-large`, `answerdotai/ModernBERT-large`, `FacebookAI/roberta-large` | **DeBERTa-v3-large still wins on benchmark accuracy and sample efficiency for classification/NLI.** ModernBERT (Dec 2024) wins on long-context (8192) + throughput; pick it only when input length exceeds RoBERTa's 512 window or when throughput matters. |
| Sentence embeddings | `Qwen/Qwen3-Embedding-8B`, `Qwen/Qwen3-Embedding-4B`, `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`, `BAAI/bge-large-en-v1.5` | **Qwen3-Embedding (2025-06) is the current open embedding SOTA**; pick the size to match your latency budget. `bge-m3` for multilingual fallback, `bge-large-en-v1.5` for English-only. |
| Rerankers | `Qwen/Qwen3-Reranker-4B`, `Qwen/Qwen3-Reranker-0.6B`, `BAAI/bge-reranker-v2-m3` | Use after embedding retrieval for a precision boost. |
| Vision-language | `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `google/gemma-3-12b-it` | Qwen3-VL is current SOTA for its size. All dense. |
| Time-series forecasting | `Salesforce/moirai-2.0-R-small`, `amazon/chronos-bolt-base`, `google/timesfm-2.0-500m-pytorch`, `Datadog/Toto-Open-Base-1.0` | All foundation models for zero-shot / fine-tuned forecasting. Moirai 2.0 (Nov 2025) is the current Salesforce release; Toto (Datadog, 151M) tops the BOOM/observability benchmark. |
| Graph / molecular | Check `graph-ml` pipeline tag; domain-specific models (e.g. `DeepChem/*`, `UnifiedMol/*`) are usually better than general LMs. | |

Treat this as a starting point — always confirm with `list_models` + model card
for the specific task, and prefer models released AFTER any prior SOTA paper
the task metadata cites.

## By task family — AIRS-Bench-specific recipes

Match your task's `category` / `research_problem` (from metadata.yaml) to one
of the rows below. These are dense, trainable-on-one-GPU recommendations.

### Text Classification / NLI / similarity / coreference
*(e.g. SICK classification, Yelp sentiment, WSC, Winogrande, SICK STS)*
- **First choice**: `microsoft/deberta-v3-large` (~400M) — recent 2025 head-
  to-head studies confirm it still beats ModernBERT on classification/NLI
  accuracy and sample efficiency. Do AFT.
- **Alternative when long context matters**: `answerdotai/ModernBERT-large`
  (~400M, 8192 ctx, 2024-12 release). Pick this if inputs exceed ~512 tokens.
- For tasks that benefit from reasoning (Winogrande, WSC): try `Qwen/Qwen3.5-9B`
  or `Qwen/Qwen3-8B` zero-shot or with LoRA.

### Question Answering
*(SQuAD extractive, DuoRC, FinQA, ELI5 abstractive)*
- **Extractive QA (SQuAD / DuoRC)**: fine-tune `microsoft/deberta-v3-large`
  (preferred) or `answerdotai/ModernBERT-large` with
  `AutoModelForQuestionAnswering`. Do AFT.
- **Abstractive / long-form (ELI5, FinQA)**: fine-tune a 4B–9B causal LM with
  LoRA. `Qwen/Qwen3.5-9B` or `Qwen/Qwen3.5-4B` work well. For FinQA
  specifically, CoT prompting + LoRA helps; Qwen2.5-Math-7B-Instruct is also
  strong on numerical reasoning.

### Math (word problems, numerical reasoning)
*(SVAMP)*
- **First choice**: `Qwen/Qwen2.5-Math-7B-Instruct` — purpose-built for math,
  frequently tops SVAMP/GSM8K. Use zero-shot with CoT prompting first; add
  LoRA only if zero-shot underperforms SOTA.
- **Alternative (thinking mode)**: `Qwen/Qwen3-4B-Thinking-2507` or
  `Qwen/Qwen3.5-9B` with extended CoT prompting.
- **Dense math specialist**: `deepseek-ai/deepseek-math-7b-instruct` (V1; the
  newer `DeepSeek-Math-V2` is based on DeepSeek-V3 MoE and out of scope).

### Code generation / retrieval
*(APPS Pass@5, CodeXGLUE MRR)*
- **First choice (generation)**: `Qwen/Qwen2.5-Coder-7B-Instruct` — best dense
  code model ≤16B as of 2026. (Qwen3-Coder / Qwen3-Coder-Next are all MoE —
  avoid per this benchmark's constraints.) Qwen2.5-Coder-32B exists but is
  over the 16B cap.
- **Alternative**: `bigcode/starcoder2-15b` (just at the 16B cap).
- **Code retrieval / embedding**: `Qwen/Qwen3-Embedding-4B` (2025 SOTA,
  general-purpose), `Salesforce/SFR-Embedding-Code-400M`,
  `jinaai/jina-embeddings-v3`.

### Time-Series Forecasting
*(Monash: KaggleWebTraffic, Rideshare, SolarWeekly)*
- **Zero-shot first**: time-series foundation models work well without
  fine-tuning. Try in this order (all confirmed on HF, 2025–2026):
  - `Salesforce/moirai-2.0-R-small` — Nov 2025 release. Newer replacement for
    Moirai 1.1; smaller but better on GIFT-Eval. Use via the `uni2ts` library
    (`github.com/SalesforceAIResearch/uni2ts`).
  - `amazon/chronos-bolt-base` (~200M) — Amazon's 2024 Bolt variant, very
    fast, remains strong on Monash. Install `chronos-forecasting`.
  - `google/timesfm-2.0-500m-pytorch` — Google's v2, 500M.
  - `Datadog/Toto-Open-Base-1.0` — Datadog's 151M decoder-only model (2025),
    tops observability benchmarks; worth trying on high-frequency signals.
- **Fine-tune only if zero-shot underperforms.** Most Monash splits are well
  within these models' training distributions.
- Install: `uv add gluonts chronos-forecasting uni2ts` (different models use
  different libraries — check each model card for per-family import paths).

### Molecular Property Prediction (QM9) / Graph Regression (ZINC)
- HuggingFace Hub is **weaker** for graph/molecular models — most SOTA still
  lives in `torch_geometric` / `dgl` / domain-specific repos, but a few
  pretrained checkpoints are on HF now:
  - **Pretrained molecular on HF**: `dptech/Uni-Mol-Models` (DeepModeling's
    Uni-Mol collection; covers QM9-like tasks with 3D conformers),
    `katielink/MoLFormer-XL` / `ibm-research/MoLFormer-XL-both-10pct` (SMILES
    transformer, ~30M, good zero-shot baseline for property regression).
  - **Train-from-scratch GNN (preferred for QM9/ZINC)**: 3D equivariant GNNs
    via `torch_geometric` — `GemNet`, `PaiNN`, `SchNet` for QM9; `GraphGPS`
    or `GRIT` for ZINC. Models are <20M params and converge in minutes.
- Do NOT try to use a 7B causal LM for these — wrong inductive bias.
