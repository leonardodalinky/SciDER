FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml /app/
COPY scider/ /app/scider/
COPY streamlit-client/ /app/streamlit-client/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e /app && \
    pip install --no-cache-dir -r /app/streamlit-client/requirements.txt

# Set working directory to streamlit-client
WORKDIR /app/streamlit-client

# Create necessary directories
RUN mkdir -p /app/streamlit-client/case-study-memory && \
    mkdir -p /app/streamlit-client/workspace && \
    mkdir -p /app/streamlit-client/tmp_brain

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
