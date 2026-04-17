"""Tests for WorkspaceInitConfig, LocalEnv integration, and
detect_python_runtime config dispatch.

These are the Phase-1 acceptance tests for the workspace-init refactor: they
pin the default behaviour (backward compatibility) and the three new modes
(no uv init, PATH injection, config-aware runtime text).
"""

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from scider.core import utils as core_utils
from scider.core.code_env import DEFAULT_WORKSPACE_INIT_CONFIG, LocalEnv, WorkspaceInitConfig


class TestWorkspaceInitConfig:
    def test_default_values_match_historical_behavior(self):
        cfg = WorkspaceInitConfig()
        assert cfg.env_manager == "uv"
        assert cfg.init_uv is True
        assert cfg.venv_path is None

    def test_frozen(self):
        cfg = WorkspaceInitConfig()
        with pytest.raises((TypeError, ValueError)):
            cfg.env_manager = "python"  # type: ignore[misc]

    def test_cache_key_is_hashable_and_stable(self):
        a = WorkspaceInitConfig(env_manager="python", init_uv=False, venv_path=Path("/tmp/v"))
        b = WorkspaceInitConfig(env_manager="python", init_uv=False, venv_path=Path("/tmp/v"))
        c = WorkspaceInitConfig(env_manager="python", init_uv=False, venv_path=Path("/tmp/other"))
        assert a.cache_key() == b.cache_key()
        assert a.cache_key() != c.cache_key()
        # Usable as a dict key.
        d = {a.cache_key(): 1}
        assert d[b.cache_key()] == 1

    def test_default_singleton_is_equal_to_fresh_default(self):
        assert DEFAULT_WORKSPACE_INIT_CONFIG == WorkspaceInitConfig()


class TestLocalEnvInitUv:
    """Verify `init_uv` + `env_manager` gate the `uv init` call."""

    def test_skips_uv_init_when_env_manager_python(self, tmp_path: Path):
        ws = tmp_path / "ws"
        LocalEnv(
            working_dir=ws,
            init_config=WorkspaceInitConfig(env_manager="python", init_uv=False),
        )
        assert ws.exists()
        # No uv init should have happened.
        assert not (ws / "pyproject.toml").exists()

    def test_skips_uv_init_when_init_uv_false_even_with_uv_manager(self, tmp_path: Path):
        ws = tmp_path / "ws"
        LocalEnv(
            working_dir=ws,
            init_config=WorkspaceInitConfig(env_manager="uv", init_uv=False),
        )
        assert not (ws / "pyproject.toml").exists()

    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
    def test_runs_uv_init_by_default(self, tmp_path: Path):
        ws = tmp_path / "ws"
        LocalEnv(working_dir=ws)  # default config
        # uv init should have created a pyproject.toml
        assert (ws / "pyproject.toml").exists()
        # boilerplate stubs should have been cleaned up
        assert not (ws / "hello.py").exists()
        assert not (ws / "main.py").exists()


class TestLocalEnvPathInjection:
    """Verify venv_path is prepended to PATH inside the context and restored."""

    def test_venv_path_prepended_inside_context(self, tmp_path: Path):
        ws = tmp_path / "ws"
        venv = tmp_path / "myvenv"
        (venv / "bin").mkdir(parents=True)
        cfg = WorkspaceInitConfig(env_manager="python", init_uv=False, venv_path=venv)
        env = LocalEnv(working_dir=ws, init_config=cfg)

        original_path = os.environ.get("PATH", "")
        with env:
            new_path = os.environ["PATH"]
            assert new_path.startswith(f"{(venv / 'bin').resolve()}{os.pathsep}")
            # Original entries are preserved after the prefix.
            assert original_path in new_path or new_path.endswith(original_path)
        # On exit, PATH is restored exactly.
        assert os.environ.get("PATH", "") == original_path

    def test_no_path_change_when_venv_path_unset(self, tmp_path: Path):
        ws = tmp_path / "ws"
        env = LocalEnv(
            working_dir=ws,
            init_config=WorkspaceInitConfig(env_manager="python", init_uv=False),
        )
        original_path = os.environ.get("PATH", "")
        with env:
            assert os.environ.get("PATH", "") == original_path
        assert os.environ.get("PATH", "") == original_path

    def test_cwd_still_restored_even_with_path_injection(self, tmp_path: Path):
        ws = tmp_path / "ws"
        venv = tmp_path / "myvenv"
        (venv / "bin").mkdir(parents=True)
        env = LocalEnv(
            working_dir=ws,
            init_config=WorkspaceInitConfig(env_manager="python", init_uv=False, venv_path=venv),
        )
        original_cwd = Path.cwd()
        with env:
            assert Path.cwd() == ws.resolve()
        assert Path.cwd() == original_cwd


class TestDetectPythonRuntime:
    """detect_python_runtime dispatches on config and caches per-config."""

    def setup_method(self):
        # Wipe the module-level cache so each test starts fresh.
        core_utils._python_runtime_cache.clear()

    def test_default_config_returns_uv_text_when_uv_present(self):
        if shutil.which("uv") is None:
            pytest.skip("uv not installed")
        text = core_utils.detect_python_runtime()
        assert "python_runtime:" in text
        assert "uv run python" in text
        assert "uv add" in text

    def test_python_manager_returns_bare_python_text(self):
        cfg = WorkspaceInitConfig(env_manager="python", init_uv=False)
        text = core_utils.detect_python_runtime(cfg)
        assert "python_runtime: python" in text
        assert "pip install" in text
        assert "uv add" not in text

    def test_python_manager_with_venv_forbids_installs(self):
        cfg = WorkspaceInitConfig(
            env_manager="python", init_uv=False, venv_path=Path("/tmp/ab_venv")
        )
        text = core_utils.detect_python_runtime(cfg)
        assert "prebuilt venv at /tmp/ab_venv" in text
        assert "preinstalled" in text
        # Agent is told explicitly NOT to install.
        assert "Do NOT run" in text

    def test_uv_manager_falls_back_to_python_text_when_uv_missing(self):
        # Force shutil.which to return None for uv.
        with patch.object(core_utils, "__name__", core_utils.__name__):
            with patch("shutil.which", return_value=None):
                cfg = WorkspaceInitConfig(env_manager="uv", init_uv=True)
                text = core_utils.detect_python_runtime(cfg)
                assert "uv requested but not found" in text

    def test_cache_is_per_config(self):
        cfg1 = WorkspaceInitConfig(env_manager="python", init_uv=False)
        cfg2 = WorkspaceInitConfig(env_manager="python", init_uv=False, venv_path=Path("/tmp/a"))
        t1 = core_utils.detect_python_runtime(cfg1)
        t2 = core_utils.detect_python_runtime(cfg2)
        # Different configs produce different text.
        assert t1 != t2
        # Both are now cached.
        assert cfg1.cache_key() in core_utils._python_runtime_cache
        assert cfg2.cache_key() in core_utils._python_runtime_cache

    def test_rejects_non_config_input(self):
        with pytest.raises(TypeError):
            core_utils.detect_python_runtime("not a config")  # type: ignore[arg-type]
