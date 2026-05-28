"""Settings form for API keys and per-role model assignments.

The frontend uses the unified ModelCatalog (model_settings/catalog.yaml) so
users can mix-and-match providers per role. Models whose required env vars
aren't set are shown but greyed out based on the keys actually entered in the
form (not the process environment).
"""

import os

import streamlit as st

from scider.default.models import CAP_COMPLETION, ModelCatalog, ModelEntry

# Model roles grouped by category. experiment_coding is handled separately
# because it's tied to CODING_AGENT_VERSION.
MODEL_ROLE_GROUPS = {
    "Ideation": {
        "ideation": "Idea generation",
        "paper_search": "Paper search",
        "metric_search": "Metric search",
    },
    "Data Analysis": {
        "data": "Data analysis",
    },
    "Experiment": {
        "experiment": "Experiment agent",
    },
    "Critic": {
        "critic": "Critic evaluation",
    },
    "Paper Writing": {
        "writing": "Writing agent",
    },
    "System": {
        "history": "History compression",
    },
}

# Mapping: env var name -> (settings key, env var name)
_KEY_ENV_MAP = {
    "GEMINI_API_KEY": "gemini_api_key",
    "OPENAI_API_KEY": "openai_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
}


def _initial_key_value(settings_key: str, current: dict) -> str:
    """Resolve initial key value from saved settings only (never from env)."""
    return current.get(settings_key, "")


def _entry_available_with_keys(entry: ModelEntry, provided_keys: dict[str, str]) -> bool:
    """Check if a model entry is available given the keys the user actually entered."""
    for env_var in entry.requires_env:
        settings_key = _KEY_ENV_MAP.get(env_var, "")
        if not provided_keys.get(settings_key, ""):
            return False
    return True


def _make_format_func(provided_keys: dict[str, str]):
    """Build a format_func that checks availability against provided keys, not os.environ."""

    def _format(model_id: str) -> str:
        entry = ModelCatalog.get(model_id)
        if entry is None:
            return f"{model_id} (unknown)"
        if not _entry_available_with_keys(entry, provided_keys):
            missing_env = [
                k for k in entry.requires_env if not provided_keys.get(_KEY_ENV_MAP.get(k, ""), "")
            ]
            missing_labels = ", ".join(missing_env)
            return f"{entry.id}  \u26a0 missing {missing_labels}"
        return f"{entry.id}  ({entry.provider})"

    return _format


def _completion_model_ids() -> list[str]:
    return [e.id for e in ModelCatalog.by_capability(CAP_COMPLETION)]


def _claude_completion_ids() -> list[str]:
    return [e.id for e in ModelCatalog.by_capability(CAP_COMPLETION) if e.provider == "anthropic"]


def _select_model(
    label: str,
    options: list[str],
    saved: str | None,
    fallback: str | None,
    key: str,
    format_func=None,
) -> str:
    default = saved if saved in options else (fallback if fallback in options else options[0])
    idx = options.index(default)
    kwargs = {}
    if format_func is not None:
        kwargs["format_func"] = format_func
    return st.selectbox(
        label,
        options,
        index=idx,
        key=key,
        **kwargs,
    )


def _ping_provider(provider: str, api_key: str) -> tuple[bool, str]:
    """Make a 1-token completion call to validate a provider key.

    Returns (ok, message). Costs a fraction of a cent per provider.
    """
    import litellm

    # Cheap, widely-available models per provider. If a model goes EOL the
    # error message still tells the user something useful (key works, model
    # unavailable) so they can investigate.
    model = {
        "gemini": "gemini/gemini-flash-latest",
        "openai": "openai/gpt-5-mini",
        "anthropic": "anthropic/claude-haiku-4-5",
    }[provider]
    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            api_key=api_key,
            timeout=15,
        )
        return True, "Connected"
    except Exception as e:
        # Surface the most diagnostic part of the litellm error. Auth errors
        # tend to come back as AuthenticationError / 401 — short and clear.
        msg = type(e).__name__ + ": " + str(e)
        if len(msg) > 160:
            msg = msg[:160] + "…"
        return False, msg


def _render_connection_tester(current: dict) -> None:
    """Render the 'Test API Connections' panel (outside the form so it can run on
    click). Reads typed values from session_state so the user can test keys
    *before* saving.
    """
    init_gemini = current.get("gemini_api_key", "")
    init_openai = current.get("openai_api_key", "")
    init_anthropic = current.get("anthropic_api_key", "")
    # Settings form writes typed values to these session_state keys live.
    g_key = st.session_state.get("_sk_gemini", init_gemini).strip()
    o_key = st.session_state.get("_sk_openai", init_openai).strip()
    a_key = st.session_state.get("_sk_anthropic", init_anthropic).strip()

    any_key = bool(g_key or o_key or a_key)

    with st.container(border=True):
        col_btn, col_label = st.columns([1, 4])
        with col_btn:
            clicked = st.button(
                "🔌 Test Connections",
                key="btn_test_connections",
                disabled=not any_key,
                use_container_width=True,
                help=(
                    "Sends one 1-token request per provider whose key is filled in. "
                    "Costs a fraction of a cent. Use before saving to catch bad keys early."
                ),
            )
        with col_label:
            if not any_key:
                st.caption(
                    "Fill in at least one API key below, then come back here to verify it works."
                )
            else:
                st.caption(
                    "Validates that each provided key reaches its provider. "
                    "Results are cached until you click again."
                )

        if clicked:
            results: dict[str, tuple[bool, str]] = {}
            with st.spinner("Pinging providers…"):
                if g_key:
                    results["Gemini"] = _ping_provider("gemini", g_key)
                if o_key:
                    results["OpenAI"] = _ping_provider("openai", o_key)
                if a_key:
                    results["Anthropic"] = _ping_provider("anthropic", a_key)
            st.session_state["_conn_test_results"] = results

        results = st.session_state.get("_conn_test_results", {})
        if results:
            for provider, (ok, msg) in results.items():
                if ok:
                    st.success(f"**{provider}** — ✅ {msg}")
                else:
                    st.error(f"**{provider}** — ❌ {msg}")


def render_settings_form(current_settings: dict | None = None) -> dict | None:
    """Render settings form. Returns new settings dict on submit, None otherwise."""
    st.markdown("### Configure SciDER")
    st.caption(
        "API keys and model selections are stored locally on this machine only — "
        "they are never uploaded to the cloud or shared between users."
    )

    # Make sure the catalog is loaded once before we render anything.
    ModelCatalog.load()

    current = current_settings or {}
    current_roles = current.get("model_roles", {})

    completion_ids = _completion_model_ids()
    claude_ids = _claude_completion_ids() or completion_ids

    # --- API Keys (outside form so we can read their values for rendering) ---
    # Streamlit forms capture widget values only on submit, so we use
    # session_state keys to read the *current* typed values for the
    # format_func, falling back to initial defaults.

    # Compute initial defaults: saved setting > env var > empty
    init_gemini = _initial_key_value("gemini_api_key", current)
    init_openai = _initial_key_value("openai_api_key", current)
    init_anthropic = _initial_key_value("anthropic_api_key", current)

    # Build a snapshot of provided keys for the format_func.
    # On first render we use initial values; after user types, session_state
    # updates on the next rerun (Streamlit forms only update on submit, but
    # since model dropdowns are inside the same form, the availability display
    # reflects the *initial* keys — which is correct: if you just opened
    # settings, the keys you already saved / have in env are "provided".)
    provided_keys = {
        "gemini_api_key": st.session_state.get("_sk_gemini", init_gemini),
        "openai_api_key": st.session_state.get("_sk_openai", init_openai),
        "anthropic_api_key": st.session_state.get("_sk_anthropic", init_anthropic),
    }

    format_func = _make_format_func(provided_keys)

    # Connection tester sits OUTSIDE the form so its button can run on click
    # (st.form only allows the submit button inside). It reads typed values
    # from session_state, so the user can verify keys before pressing Save.
    _render_connection_tester(current)

    with st.form("settings_form"):
        # --- API Keys ---
        st.markdown("#### API Keys")
        st.caption(
            "Enter keys for the providers you want to use. Gemini or OpenAI keys enable most workflows. "
            "An Anthropic key is required for the Claude coding agent used in Experiment and Full Pipeline. "
            "Models without a matching key will appear with a warning in the dropdowns below."
        )

        # Gemini + OpenAI side by side
        key_col1, key_col2 = st.columns(2)
        with key_col1:
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Enter your Gemini API key",
                value=init_gemini,
                key="_sk_gemini",
            )
        with key_col2:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key",
                value=init_openai,
                key="_sk_openai",
            )

        # Anthropic full-width — most important for coding agent
        ant_col, s2_col = st.columns(2)
        with ant_col:
            anthropic_api_key = st.text_input(
                "Anthropic (Claude) API Key",
                type="password",
                placeholder="Required for Experiment and Full Pipeline coding agent",
                value=init_anthropic,
                key="_sk_anthropic",
            )
        with s2_col:
            s2_api_key = st.text_input(
                "Semantic Scholar API Key",
                type="password",
                placeholder="Optional — improves paper search rate limits",
                value=_initial_key_value("s2_api_key", current),
            )
        st.caption(
            "Gemini or OpenAI enables most workflows. Anthropic is required for the coding agent. "
            "Semantic Scholar is optional — get a key at https://www.semanticscholar.org/product/api"
        )

        # --- System status (compact 3-column row) ---
        st.divider()
        st.markdown("#### System Status")
        st.caption("Read-only. Configure these settings via environment variables in `.env`.")

        from scider.core import constant as _c

        mem_read = os.getenv("SCIDER_MEMORY_READ", "true").lower() in {"1", "true", "yes", "y"}
        mem_write = os.getenv("SCIDER_MEMORY_WRITE", "true").lower() in {"1", "true", "yes", "y"}
        coding_version = os.getenv("CODING_AGENT_VERSION", "claude_sdk")
        if coding_version in ("v3", "claude_sdk"):
            version_label = "Claude Agent SDK"
        elif coding_version == "native":
            version_label = "Native (SciDER)"
        else:
            version_label = coding_version

        sys_col1, sys_col2, sys_col3 = st.columns(3)
        with sys_col1:
            st.markdown("**HuggingFace Datasets**")
            if _c.HF_DATASET_DOWNLOAD_ENABLED:
                st.success(f"✓ Enabled ({_c.HF_DATASET_MAX_SIZE_MB} MB max)")
            else:
                st.info("Disabled")
        with sys_col2:
            st.markdown("**Cross-Session Memory**")
            if mem_read and mem_write:
                st.success("✓ Read + Write")
            elif mem_read:
                st.info("Read only")
            elif mem_write:
                st.info("Write only")
            else:
                st.warning("Disabled")
        with sys_col3:
            st.markdown("**Coding Agent**")
            st.info(version_label)

        # --- Coding Agent model ---
        st.divider()
        st.markdown("#### Coding Agent Model")
        st.caption(
            "The model used by the coding subagent during Experiment and Full Pipeline workflows. "
            "Set `CODING_AGENT_VERSION` in `.env` to switch between `claude_sdk` and `native`."
        )

        if coding_version in ("v3", "claude_sdk"):
            coding_options = claude_ids
            coding_fallback = "claude-haiku-4-5"
        else:
            coding_options = completion_ids
            coding_fallback = "gemini-flash-latest"

        coding_model = _select_model(
            "Code generation model",
            coding_options,
            saved=current_roles.get("experiment_coding"),
            fallback=coding_fallback,
            key="model_role_experiment_coding",
            format_func=format_func,
        )

        # --- Per-role model selection ---
        st.divider()
        st.markdown("#### Model Assignments")
        st.caption(
            "Choose which model to use for each agent role. Models from any provider "
            "can be mixed freely."
        )

        role_selections: dict[str, str] = {}
        max_cols = 3
        for group_name, roles in MODEL_ROLE_GROUPS.items():
            st.markdown(f"**{group_name}**")
            role_items = list(roles.items())
            for row_start in range(0, len(role_items), max_cols):
                row = role_items[row_start : row_start + max_cols]
                # Use exactly as many columns as there are roles in this row — avoids blank columns
                cols = st.columns(len(row))
                for col, (role, label) in zip(cols, row):
                    with col:
                        role_selections[role] = _select_model(
                            label,
                            completion_ids,
                            saved=current_roles.get(role),
                            fallback=None,
                            key=f"model_role_{role}",
                            format_func=format_func,
                        )

        role_selections["experiment_coding"] = coding_model

        # --- Submit ---
        submitted = st.form_submit_button("Save Settings", type="primary")

        if submitted:
            final_gemini = gemini_api_key.strip()
            final_openai = openai_api_key.strip()
            final_anthropic = anthropic_api_key.strip()
            final_s2 = s2_api_key.strip()

            if not (final_gemini or final_openai or final_anthropic):
                st.error("Provide at least one provider API key (Gemini, OpenAI, or Anthropic).")
                return None

            # Build final provided keys for availability check.
            final_keys = {
                "gemini_api_key": final_gemini,
                "openai_api_key": final_openai,
                "anthropic_api_key": final_anthropic,
            }

            # Validate that selected models have their keys filled in.
            unavailable = []
            for role, mid in role_selections.items():
                entry = ModelCatalog.get(mid)
                if entry and not _entry_available_with_keys(entry, final_keys):
                    missing_env = [
                        k
                        for k in entry.requires_env
                        if not final_keys.get(_KEY_ENV_MAP.get(k, ""), "")
                    ]
                    unavailable.append((role, mid, missing_env))

            if unavailable:
                lines = "\n".join(
                    f"- **{role}** \u2192 `{mid}` (missing: {', '.join(missing)})"
                    for role, mid, missing in unavailable
                )
                st.error("Some selected models are still missing API keys:\n" + lines)
                return None

            return {
                "gemini_api_key": final_gemini,
                "openai_api_key": final_openai,
                "anthropic_api_key": final_anthropic,
                "s2_api_key": final_s2,
                "model_roles": role_selections,
            }

    return None
