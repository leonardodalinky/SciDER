"""Settings form for API key and model configuration."""

import os

import streamlit as st

# Available models per provider
GEMINI_MODELS = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3-pro-preview",
]

OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5-pro",
    "gpt-5-chat",
    "o1-mini",
    "o3-mini",
    "o3",
    "o4-mini",
]

# Claude models for coding agent (Claude SDK)
CLAUDE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-6",
    "claude-opus-4-5",
]

# Model roles grouped by category (experiment_coding excluded — handled separately)
MODEL_ROLE_GROUPS = {
    "Ideation": {
        "ideation": "Idea generation",
        "paper_search": "Paper search",
        "metric_search": "Metric search",
    },
    "Data Analysis": {
        "data": "Data analysis",
        "critic": "Critic evaluation",
    },
    "Experiment": {
        "experiment": "Experiment agent",
    },
    "System": {
        "history": "History compression",
    },
}

# Default model assignments (low-cost for most, higher for key roles)
GEMINI_ROLE_DEFAULTS = {
    "ideation": "gemini/gemini-2.5-flash",
    "paper_search": "gemini/gemini-2.5-flash-lite",
    "metric_search": "gemini/gemini-2.5-flash-lite",
    "data": "gemini/gemini-2.5-flash-lite",
    "critic": "gemini/gemini-2.5-flash-lite",
    "history": "gemini/gemini-2.5-flash-lite",
    "experiment": "gemini/gemini-2.5-flash",
}

OPENAI_ROLE_DEFAULTS = {
    "ideation": "gpt-5-mini",
    "paper_search": "gpt-5-nano",
    "metric_search": "gpt-5-nano",
    "data": "gpt-5-nano",
    "critic": "gpt-5-nano",
    "history": "gpt-5-nano",
    "experiment": "gpt-5-mini",
}


def render_settings_form(current_settings: dict | None = None) -> dict | None:
    """Render settings form. Returns new settings dict on submit, None otherwise."""
    st.markdown("### Settings")
    st.caption(
        "Your settings are stored locally on this machine only and are never uploaded to the cloud."
    )

    current = current_settings or {}
    current_provider = current.get("model_provider", "Gemini")
    current_roles = current.get("model_roles", {})

    # Provider selector outside form so switching triggers immediate rerun
    model_provider = st.selectbox(
        "Model Provider",
        ["Gemini", "OpenAI"],
        index=(
            ["Gemini", "OpenAI"].index(current_provider)
            if current_provider in ("Gemini", "OpenAI")
            else 0
        ),
        key="settings_model_provider",
    )

    with st.form("settings_form"):
        # --- API Keys ---
        st.markdown("#### API Keys")

        api_key = st.text_input(
            f"{model_provider} API Key",
            type="password",
            placeholder=f"Enter your {model_provider} API key",
            value=current.get("api_key", ""),
            help="Required.",
        )

        st.divider()

        anthropic_api_key = st.text_input(
            "Anthropic (Claude) API Key",
            type="password",
            placeholder="Optional — needed for Claude coding agent",
            value=current.get("anthropic_api_key", ""),
        )

        openai_api_key = st.text_input(
            "OpenAI API Key (for embeddings)",
            type="password",
            placeholder="Optional — needed for embedding features",
            value=current.get("openai_api_key", ""),
        )
        st.caption("Embedding model (text-embedding-3-small) requires an OpenAI key.")

        s2_api_key = st.text_input(
            "Semantic Scholar API Key",
            type="password",
            placeholder="Optional — enables Semantic Scholar paper search",
            value=current.get("s2_api_key", ""),
        )
        st.caption(
            "Optional. If provided, paper search will also query Semantic Scholar "
            "in addition to arXiv. Get a key at https://www.semanticscholar.org/product/api"
        )

        # --- HuggingFace Dataset Download ---
        st.divider()
        st.markdown("#### HuggingFace Dataset Download")

        from scider.core import constant as _c

        if _c.HF_DATASET_DOWNLOAD_ENABLED:
            st.success(f"Enabled — max dataset size: {_c.HF_DATASET_MAX_SIZE_MB} MB")
        else:
            st.info("Disabled. Set `HF_DATASET_DOWNLOAD_ENABLED=true` in `.env` to enable.")
        st.caption(
            "When enabled, you can enter a HuggingFace dataset repo name "
            "(e.g. `google/fleurs`) instead of uploading a local file. "
            "Datasets are downloaded and cached automatically. "
            "Configure size limit via `HF_DATASET_MAX_SIZE_MB` in `.env`."
        )

        # --- Coding Agent ---
        st.divider()
        st.markdown("#### Coding Agent")

        coding_version = os.getenv("CODING_AGENT_VERSION", "claude_sdk")
        if coding_version in ("v3", "claude_sdk"):
            version_label = "Claude Agent SDK"
        elif coding_version in ("v2", "openhands"):
            version_label = "OpenHands"
        else:
            version_label = coding_version
        st.text_input(
            "Coding Agent Backend",
            value=version_label,
            disabled=True,
            key="coding_agent_version_display",
        )
        st.caption(
            "To change the coding agent backend, set the `CODING_AGENT_VERSION` "
            "environment variable (`claude_sdk` or `openhands`) in `.env`."
        )

        if coding_version in ("v3", "claude_sdk"):
            coding_models = CLAUDE_MODELS
            coding_default = current_roles.get("experiment_coding", "claude-haiku-4-5")
        else:
            coding_models = GEMINI_MODELS if model_provider == "Gemini" else OPENAI_MODELS
            fallback_defaults = (
                GEMINI_ROLE_DEFAULTS if model_provider == "Gemini" else OPENAI_ROLE_DEFAULTS
            )
            coding_default = current_roles.get(
                "experiment_coding", fallback_defaults.get("experiment_coding", coding_models[0])
            )

        if coding_default not in coding_models:
            coding_default = coding_models[0]
        coding_idx = coding_models.index(coding_default)

        coding_model = st.selectbox(
            "Code generation model",
            coding_models,
            index=coding_idx,
            key="model_role_experiment_coding",
        )

        # --- Per-role model selection ---
        st.divider()
        st.markdown("#### Model Assignments")
        st.caption("Choose which model to use for each agent role.")

        models_list = GEMINI_MODELS if model_provider == "Gemini" else OPENAI_MODELS
        defaults = GEMINI_ROLE_DEFAULTS if model_provider == "Gemini" else OPENAI_ROLE_DEFAULTS

        role_selections = {}
        max_cols = 3
        for group_name, roles in MODEL_ROLE_GROUPS.items():
            st.markdown(f"**{group_name}**")
            role_items = list(roles.items())
            for row_start in range(0, len(role_items), max_cols):
                row = role_items[row_start : row_start + max_cols]
                cols = st.columns(max_cols)
                for col, (role, label) in zip(cols, row):
                    with col:
                        saved = current_roles.get(role)
                        default = (
                            saved if saved in models_list else defaults.get(role, models_list[0])
                        )
                        idx = models_list.index(default) if default in models_list else 0
                        role_selections[role] = st.selectbox(
                            label,
                            models_list,
                            index=idx,
                            key=f"model_role_{role}",
                        )

        # Include coding model in role_selections
        role_selections["experiment_coding"] = coding_model

        # --- Submit ---
        submitted = st.form_submit_button("Save Settings", type="primary")

        if submitted:
            final_api_key = api_key.strip()
            final_anthropic = anthropic_api_key.strip()
            final_openai = openai_api_key.strip()
            final_s2 = s2_api_key.strip()

            if not final_api_key:
                st.error("API key is required.")
                return None

            return {
                "api_key": final_api_key,
                "model_provider": model_provider,
                "anthropic_api_key": final_anthropic,
                "openai_api_key": final_openai,
                "s2_api_key": final_s2,
                "model_roles": role_selections,
            }

    return None
