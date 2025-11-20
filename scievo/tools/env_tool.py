import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from shell_tool import run_shell_cmd

from .registry import register_tool, register_toolset_desc


@register_tool(
    "environment",
    {
        "type": "function",
        "function": {
            "name": "create_virtualenv",
            "description": "Create a Python virtual environment using python3 -m venv",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory of the venv"},
                },
                "required": ["path"],
            },
        },
    },
)
def create_virtualenv(path: str):
    return run_shell_cmd(f"python3 -m venv {path}")


@register_tool(
    "environment",
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install a Python package via pip inside the environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name"},
                    "version": {
                        "type": "string",
                        "description": "Package version (optional)",
                        "default": "",
                    },
                    "venv": {
                        "type": "string",
                        "description": "Path to virtual environment (optional)",
                        "default": "",
                    },
                },
                "required": ["package"],
            },
        },
    },
)
def pip_install(package: str, version: str = "", venv: str = ""):
    if version:
        pkg = f"{package}=={version}"
    else:
        pkg = package

    if venv:
        pip = f"{venv}/bin/pip"
    else:
        pip = "pip"

    return run_shell_cmd(f"{pip} install {pkg}")


@register_tool(
    "environment",
    {
        "type": "function",
        "function": {
            "name": "pip_install_requirements",
            "description": "Install dependencies from a requirements.txt file",
            "parameters": {
                "type": "object",
                "properties": {
                    "requirements_path": {"type": "string"},
                    "venv": {"type": "string", "default": ""},
                },
                "required": ["requirements_path"],
            },
        },
    },
)
def pip_install_requirements(requirements_path: str, venv: str = ""):
    pip = f"{venv}/bin/pip" if venv else "pip"
    return run_shell_cmd(f"{pip} install -r {requirements_path}")


@register_tool(
    "environment",
    {
        "type": "function",
        "function": {
            "name": "check_python_import",
            "description": "Check whether a Python module can be imported",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "python": {"type": "string", "default": "python3"},
                },
                "required": ["module"],
            },
        },
    },
)
def check_python_import(module: str, python: str = "python3"):
    code = f"{python} -c 'import {module}; print(\"OK\")'"
    return run_shell_cmd(code)


@register_tool(
    "environment",
    {
        "type": "function",
        "function": {
            "name": "clone_repo",
            "description": "Clone a Git repository using git",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "dest": {"type": "string", "default": ""},
                },
                "required": ["repo_url"],
            },
        },
    },
)
def clone_repo(repo_url: str, dest: str = ""):
    if dest:
        return run_shell_cmd(f"git clone {repo_url} {dest}")
    return run_shell_cmd(f"git clone {repo_url}")


@register_tool(
    "environment",
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
)
def write_file(path: str, content: str):
    Path(path).write_text(content)
    return f"File written to {path}"
