---
name: uv-package-manager
description: UV package manager for fast Python dependency management and virtual environments. Use when installing packages, managing dependencies, or setting up virtual environments.
allowed_agents: [experiment, native_coding]
preload_for: [experiment, native_coding]
---

# UV Package Manager

uv is an extremely fast Python package installer (10-100x faster than pip), written in Rust.

## Essential Commands

```bash
# Virtual environment
uv venv                         # Create .venv
uv venv --python 3.12           # With specific Python version
uv run python script.py         # Run in venv (no activation needed)
uv run pytest                   # Run CLI tool in venv

# Dependency management (pyproject.toml based)
uv add requests pandas          # Add packages
uv add --dev pytest black       # Add dev dependencies
uv remove requests              # Remove package
uv sync                         # Install all dependencies from lockfile
uv lock                         # Create/update lockfile

# pip-compatible interface
uv pip install -r requirements.txt   # Install from requirements
uv pip install package_name          # Install single package
uv pip list                          # List installed packages

# Python version management
uv python install 3.12          # Install Python version
uv python pin 3.12              # Pin version for project
```

## Key Patterns

- **New project**: `uv init . && uv add <deps> && uv run python main.py`
- **Existing project**: `uv sync` to install all deps from lockfile
- **Run without activating venv**: Always use `uv run <command>` instead of activating
- **Prefer `uv add`** over `uv pip install` — it updates pyproject.toml and lockfile
