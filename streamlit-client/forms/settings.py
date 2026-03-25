"""Settings form for API key and model configuration."""

import streamlit as st

# Available models per provider
GEMINI_MODELS = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
]

OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.2",
]

# Model roles grouped by category
MODEL_ROLE_GROUPS = {
    "Ideation": {
        "ideation": "Idea generation",
        "paper_search": "Paper search",
        "metric_search": "Metric search",
    },
    "Data Analysis": {
        "data": "Data analysis",
        "plan": "Planning",
        "critic": "Critic evaluation",
    },
    "Experiment": {
        "experiment_agent": "Experiment orchestration",
        "experiment_coding": "Code generation",
        "experiment_execute": "Code execution",
        "experiment_monitor": "Execution monitoring",
        "experiment_summary": "Result summarization",
    },
    "System": {
        "history": "History compression",
        "mem": "Memory extraction",
    },
}

# Default model assignments (low-cost for most, higher for key roles)
GEMINI_ROLE_DEFAULTS = {
    "ideation": "gemini/gemini-2.5-flash",
    "paper_search": "gemini/gemini-2.5-flash-lite",
    "metric_search": "gemini/gemini-2.5-flash-lite",
    "data": "gemini/gemini-2.5-flash-lite",
    "plan": "gemini/gemini-2.5-flash",
    "critic": "gemini/gemini-2.5-flash-lite",
    "mem": "gemini/gemini-2.5-flash-lite",
    "history": "gemini/gemini-2.5-flash-lite",
    "experiment_agent": "gemini/gemini-2.5-flash",
    "experiment_coding": "gemini/gemini-2.5-flash",
    "experiment_execute": "gemini/gemini-2.5-flash",
    "experiment_monitor": "gemini/gemini-2.5-flash-lite",
    "experiment_summary": "gemini/gemini-2.5-flash-lite",
}

OPENAI_ROLE_DEFAULTS = {
    "ideation": "gpt-5-mini",
    "paper_search": "gpt-5-nano",
    "metric_search": "gpt-5-nano",
    "data": "gpt-5-nano",
    "plan": "gpt-5-mini",
    "critic": "gpt-5-nano",
    "mem": "gpt-5-nano",
    "history": "gpt-5-nano",
    "experiment_agent": "gpt-5-mini",
    "experiment_coding": "gpt-5-mini",
    "experiment_execute": "gpt-5-mini",
    "experiment_monitor": "gpt-5-nano",
    "experiment_summary": "gpt-5-nano",
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
            value="",
            help="Required.",
        )
        if current.get("api_key"):
            st.caption("A key is already saved. Leave blank to keep it.")

        st.divider()

        anthropic_api_key = st.text_input(
            "Anthropic (Claude) API Key",
            type="password",
            placeholder="Optional — needed for Claude coding agent",
            value="",
        )
        if current.get("anthropic_api_key"):
            st.caption("A key is already saved. Leave blank to keep it.")

        openai_api_key = st.text_input(
            "OpenAI API Key (for embeddings)",
            type="password",
            placeholder="Optional — needed for memory/embedding features",
            value="",
        )
        if current.get("openai_api_key"):
            st.caption("A key is already saved. Leave blank to keep it.")
        st.caption(
            "Embedding model (text-embedding-3-small) requires an OpenAI key. "
            "Without it, memory features will be disabled."
        )

        # --- Memory toggle ---
        st.divider()
        st.markdown("#### Memory")
        memory_enabled = st.checkbox(
            "Enable memory (reasoning bank)",
            value=current.get("memory_enabled", False),
            help="Requires OpenAI API key for embeddings. Extracts and retrieves "
            "long-term memory from conversations.",
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

        # --- Submit ---
        submitted = st.form_submit_button("Save Settings", type="primary")

        if submitted:
            final_api_key = api_key.strip() or current.get("api_key", "")
            final_anthropic = anthropic_api_key.strip() or current.get("anthropic_api_key", "")
            final_openai = openai_api_key.strip() or current.get("openai_api_key", "")

            if not final_api_key:
                st.error("API key is required.")
                return None

            if memory_enabled and not final_openai:
                st.error("Memory requires an OpenAI API key for embeddings.")
                return None

            return {
                "api_key": final_api_key,
                "model_provider": model_provider,
                "anthropic_api_key": final_anthropic,
                "openai_api_key": final_openai,
                "memory_enabled": memory_enabled,
                "model_roles": role_selections,
            }

    return None
