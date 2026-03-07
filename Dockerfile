FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
COPY requirements-space.txt /app/requirements-space.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -v && \
    pip install --no-cache-dir -r requirements-space.txt -v

# Copy application code
COPY scievo/ /app/scievo/
COPY streamlit-client/ /app/streamlit-client/
# Copy case studies for Case Study mode (browse saved chats without API)
COPY case-study-memory/ /app/streamlit-client/case-study-memory/

WORKDIR /app/streamlit-client

# Runtime writable dirs (ensure exist even if case-study-memory is empty)
RUN mkdir -p case-study-memory workspace tmp_brain

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
