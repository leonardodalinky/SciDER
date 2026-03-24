@echo off
REM Quick start script for SciDER Streamlit Interface (Windows)

cd /d "%~dp0\.."

REM Check if parent .env exists
if not exist ".env" (
    echo Warning: .env file not found in project root
    echo Please copy .env.template to .env and configure your API keys
    echo.
    pause
)

REM Sync streamlit dependency
uv sync --extra streamlit

echo Starting SciDER Streamlit Interface...
uv run python -m streamlit run streamlit-client/app.py --server.port 7860
