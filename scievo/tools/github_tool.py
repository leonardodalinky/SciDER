import os
import shutil
from pathlib import Path

from .registry import register_tool, register_toolset_desc

register_toolset_desc("github", "Tools for interacting with GitHub repositories on the local system.")

###############################################################################
# Tool 1: clone_repo
###############################################################################

@register_tool(
    "github",
    {
        "type": "function",
        "function": {
            "name": "clone_repo",
            "description": "Clone a GitHub repository to a target local directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "HTTP(S) URL of the GitHub repository to clone."
                    },
                    "dest_dir": {
                        "type": "string",
                        "description": "Local directory path where the repository will be cloned."
                    },
                },
                "required": ["repo_url", "dest_dir"],
            },
        },
    },
)
def clone_repo(repo_url: str, dest_dir: str) -> str:
    """
    Clone a GitHub repository into a given local directory.

    Workflow:
    1. Expand user and environment variables in dest_dir.
    2. Ensure directory exists; create it if needed.
    3. Run `git clone` using shutil and system git.
    4. Return success or error message.
    """
    try:
        dest_path = Path(os.path.expandvars(dest_dir)).expanduser()
        dest_path.mkdir(parents=True, exist_ok=True)

        # Determine repository name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        destination = dest_path / repo_name

        # Remove existing directory to avoid git conflict
        if destination.exists():
            shutil.rmtree(destination)

        # Perform clone
        result = os.system(f"git clone {repo_url} {destination}")

        if result != 0:
            return f"Error: Failed to clone repository from {repo_url}"

        return f"Repository cloned to: {destination}"

    except Exception as e:
        return f"Error cloning repository: {e}"


###############################################################################
# Tool 2: read_readme
###############################################################################

@register_tool(
    "github",
    {
        "type": "function",
        "function": {
            "name": "read_readme",
            "description": "Read README.md from a repository directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_dir": {
                        "type": "string",
                        "description": "Local directory where repository was cloned."
                    },
                },
                "required": ["repo_dir"],
            },
        },
    },
)
def read_readme(repo_dir: str) -> str:
    """
    Read the README.md file inside a GitHub repository.

    Workflow:
    1. Identify repository directory path.
    2. Search for README files with common name patterns.
    3. Return file content if found.
    """
    try:
        repo_path = Path(os.path.expandvars(repo_dir)).expanduser()
        if not repo_path.exists():
            return f"Error: Repository directory '{repo_dir}' does not exist"

        candidates = ["README.md", "readme.md", "Readme.md", "README.MD"]

        for filename in candidates:
            file_path = repo_path / filename
            if file_path.exists():
                return file_path.read_text(errors="ignore")

        return "No README.md file found in the repository."

    except Exception as e:
        return f"Error reading README: {e}"


###############################################################################
# Tool 3: list_repo_files
###############################################################################

@register_tool(
    "github",
    {
        "type": "function",
        "function": {
            "name": "list_repo_files",
            "description": "Recursively list all files inside a cloned GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_dir": {
                        "type": "string",
                        "description": "Path to the local repository folder.",
                    },
                },
                "required": ["repo_dir"],
            },
        },
    },
)
def list_repo_files(repo_dir: str) -> str:
    """
    Recursively list all files inside a repository.

    Workflow:
    1. Expand repo path.
    2. Walk through directories using Path.rglob().
    3. Return file list as newline-separated string.
    """
    try:
        repo_path = Path(os.path.expandvars(repo_dir)).expanduser()
        if not repo_path.exists():
            return f"Error: Repository directory '{repo_dir}' does not exist"

        files = [str(p) for p in repo_path.rglob("*") if p.is_file()]

        if not files:
            return "No files found inside repository."

        return "\n".join(files)

    except Exception as e:
        return f"Error listing repository files: {e}"


###############################################################################
# Tool 4: get_file_content
###############################################################################

@register_tool(
    "github",
    {
        "type": "function",
        "function": {
            "name": "get_file_content",
            "description": "Retrieve the content of a file inside a cloned GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_dir": {
                        "type": "string",
                        "description": "Local path to the repository.",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "Relative path of the file inside the repository.",
                    },
                },
                "required": ["repo_dir", "relative_path"],
            },
        },
    },
)
def get_file_content(repo_dir: str, relative_path: str) -> str:
    """
    Return the content of a file inside a repository.

    Workflow:
    1. Compute the absolute file path.
    2. Ensure file exists.
    3. Read and return the file's content.
    """
    try:
        repo_path = Path(os.path.expandvars(repo_dir)).expanduser()
        file_path = repo_path / relative_path

        if not file_path.exists():
            return f"Error: File '{relative_path}' does not exist in repository"

        return file_path.read_text(errors="ignore")

    except Exception as e:
        return f"Error reading file: {e}"
