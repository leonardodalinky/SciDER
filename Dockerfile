# ============================================================================
# SciDER — Multi-agent research automation system
#
# Base: NVIDIA CUDA (for PyTorch GPU support)
# Includes: texlive-full (LaTeX compilation), uv (Python package manager)
# ============================================================================

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # uv installs to ~/.local/bin by default
    PATH="/root/.local/bin:$PATH"

# --------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools & basics
    build-essential \
    curl \
    git \
    # Python 3.12
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    # LaTeX (full distribution for PaperOrchestra compilation)
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Install uv
# --------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --------------------------------------------------------------------------
# Copy project metadata first (for layer caching)
# --------------------------------------------------------------------------
COPY pyproject.toml uv.lock /app/

# --------------------------------------------------------------------------
# Install Python dependencies via uv
#   --extra cu128     : PyTorch with CUDA 12.8 support
#   --extra streamlit : Streamlit frontend dependencies
# --------------------------------------------------------------------------
RUN uv sync --extra cu128 --extra streamlit --no-dev --no-cache

# --------------------------------------------------------------------------
# Copy application code
# --------------------------------------------------------------------------
COPY scider/ /app/scider/
COPY .scider/ /app/.scider/
COPY model_settings/ /app/model_settings/
COPY streamlit-client/ /app/streamlit-client/
COPY static/ /app/static/
# Case studies for browse-only mode (no API key needed)
COPY case-study-memory/ /app/streamlit-client/case-study-memory/

# --------------------------------------------------------------------------
# Runtime setup
# --------------------------------------------------------------------------
RUN mkdir -p workspace

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "streamlit-client/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
