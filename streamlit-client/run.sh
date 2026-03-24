#!/bin/bash
# Quick start script for SciDER Streamlit Interface

set -e
cd "$(dirname "$0")/.."

# Check if parent .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found in project root"
    echo "Please copy .env.template to .env and configure your API keys"
    echo ""
    read -p "Press enter to continue anyway or Ctrl+C to exit..."
fi

# Sync streamlit dependency
uv sync --extra streamlit

echo "Starting SciDER Streamlit Interface..."
uv run python -m streamlit run streamlit-client/app.py --server.port 7860
