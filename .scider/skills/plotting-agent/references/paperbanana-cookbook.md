# PaperBanana Cookbook

Optional backbone for the plotting-agent's render step.

PaperBanana (Zhu et al., 2026 — <https://github.com/dwzhu-pku/PaperBanana>)
is the default figure-generation backbone described in the PaperOrchestra paper
(arXiv:2604.05018, §4 Step 2).  It runs a
**Retriever → Planner → Stylist → Visualizer → Critic** multi-agent loop,
producing publication-quality diagrams and plots that are grounded in a curated
reference collection.

The plotting-agent works fine without PaperBanana — it falls back to the
bundled matplotlib renderer (`render_matplotlib.py` / `render_diagram.py`).
PaperBanana is best for:

| Use case | Recommendation |
|---|---|
| Complex architectural / framework overview diagrams (`plot_type == "diagram"`) | **PaperBanana** (Retriever grounds style in real paper examples) |
| Plots backed by numeric data in `experimental_log.md` | Either; matplotlib gives more deterministic axis values |
| Hosts with no vision capability | Matplotlib (PaperBanana's Critic loop needs a VLM) |
| Batch / non-interactive runs | Either; PaperBanana adds API cost per figure |

---

## 1. One-time setup

```bash
# Clone PaperBanana
git clone https://github.com/dwzhu-pku/PaperBanana /path/to/PaperBanana

# Install dependencies (Python 3.12 required)
cd /path/to/PaperBanana
uv pip install -r requirements.txt      # or: pip install -r requirements.txt

# Configure API key — fill at least one, you do not need both
cp configs/model_config.template.yaml configs/model_config.yaml
```

Open `configs/model_config.yaml` and paste your key:

| Provider | Where to get a key | Field to fill |
|---|---|---|
| **Google (Gemini)** | [aistudio.google.com](https://aistudio.google.com/) (free) | `api_keys.google_api_key` |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai/) | `api_keys.openrouter_api_key` |

If both are set, OpenRouter is preferred. For OpenRouter, set the model names
to `"openrouter/<model>"` (e.g. `"openrouter/google/gemini-pro-1.5"`).

```bash
# Point paper-orchestra at your clone
export PAPERBANANA_PATH="/path/to/PaperBanana"
```

Optional model overrides (take precedence over `model_config.yaml`):

```bash
export PAPERBANANA_MAIN_MODEL="gemini-3.1-pro-preview"
export PAPERBANANA_IMAGE_MODEL="gemini-3.1-flash-image-preview"
```

Verify the setup before running the pipeline:

```bash
python skills/plotting-agent/scripts/paperbanana_render.py --check-backend
# Expected output:
#   PaperBanana found at: /path/to/PaperBanana
#   Backend is ready.
```

---

## 2. Rendering a single figure

```bash
# Diagram (e.g. Figure 1 overview)
python skills/plotting-agent/scripts/paperbanana_render.py \
    --figure-id   fig_overview \
    --caption     "Figure 1: Overview of our proposed PaperOrchestra framework." \
    --content-file workspace/inputs/idea.md \
    --task        diagram \
    --aspect-ratio 16:9 \
    --out         workspace/figures/fig_overview.png

# Plot (e.g. main results bar chart)
python skills/plotting-agent/scripts/paperbanana_render.py \
    --figure-id   fig_main_results \
    --caption     "Figure 3: Comparison of our method against baselines." \
    --content-file workspace/inputs/experimental_log.md \
    --task        plot \
    --aspect-ratio 5:4 \
    --max-critic-rounds 2 \
    --out         workspace/figures/fig_main_results.png
```

Exit codes:
- `0` — success, image saved at `--out`
- `1` — PaperBanana pipeline error (see stderr)
- `2` — `PAPERBANANA_PATH` not set or invalid → fall back to matplotlib

---

## 3. Input format (PaperBanana)

The wrapper converts the plotting-agent's figure spec to PaperBanana's input dict:

| plotting-agent field | PaperBanana field | Notes |
|---|---|---|
| `objective` | `caption` + `visual_intent` | Used as generation prompt |
| `idea.md` or `experimental_log.md` | `content` | Method/data context |
| `aspect_ratio` | `additional_info.rounded_ratio` | Same string format |
| `plot_type` | `task_name` | `diagram` or `plot` |

---

## 4. Output format (PaperBanana)

PaperBanana stores all intermediate and final images as base64-encoded JPEG strings
in the result dictionary.  The wrapper selects the best image using this priority:

1. Latest critic round: `target_{task}_critic_descN_base64_jpg` (N = 0…3)
2. `eval_image_field` pointer
3. Stylist output: `target_{task}_stylist_desc0_base64_jpg`
4. Planner output: `target_{task}_desc0_base64_jpg`
5. Vanilla baseline: `vanilla_{task}_base64_jpg`

The selected image is decoded and saved as a 300-DPI PNG to `--out`.

---

## 5. Pipeline modes

PaperBanana's `exp_mode` controls which agents run.  The wrapper uses
`demo_full` (full pipeline, no benchmark evaluation):

| Mode | Pipeline |
|---|---|
| `demo_full` *(default)* | Retriever → Planner → Stylist → Visualizer → Critic (3×) |
| `demo_planner_critic` | Retriever → Planner → Visualizer → Critic (3×) |

To override, set `--exp-mode` (future flag — hardcoded to `demo_full` today).

---

## 6. Cost and rate limits

- PaperBanana makes multiple LLM calls per figure (~5–10 in `demo_full` mode).
- API cost depends on the model chosen; Gemini Flash is cheapest for image gen.
- No S2 / Exa calls — PaperBanana uses its own reference collection for retrieval.
- Runs one figure at a time when invoked via `paperbanana_render.py`; parallel
  batch processing is available via PaperBanana's native `main.py` if you have
  many figures.

---

## 7. Security notes

- `PAPERBANANA_PATH`, `PAPERBANANA_MAIN_MODEL`, and `PAPERBANANA_IMAGE_MODEL`
  are read from the environment only.  This repo never commits these values.
- API keys for PaperBanana live in `{PAPERBANANA_PATH}/configs/model_config.yaml`,
  which is `.gitignore`'d in the PaperBanana repo.  Never commit that file.

---

## 8. Attribution

If you use PaperBanana as the plotting backbone, cite:

```bibtex
@article{zhu2026paperbanana,
  title={PaperBanana: A Reference-Driven Multi-Agent Framework for
         Automated Academic Illustration Generation},
  author={Zhu, Dawei and others},
  year={2026},
  url={https://github.com/dwzhu-pku/PaperBanana}
}
```
