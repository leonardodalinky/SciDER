@echo off
REM Quick start script for SciEvo Streamlit Interface (Windows)

REM Check if streamlit is installed
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing Streamlit...
    pip install streamlit
)

REM Check if parent .env exists
if not exist "..\\.env" (
    echo Warning: .env file not found in parent directory
    echo Please copy .env.template to .env and configure your API keys
    echo.
    pause
)

echo Starting SciEvo Streamlit Interface...
echo.
echo Choose version:
echo 1) Enhanced (recommended) - Real-time progress tracking
echo 2) Basic - Simple interface
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo Starting enhanced version...
    streamlit run app_enhanced.py
) else if "%choice%"=="2" (
    echo Starting basic version...
    streamlit run app.py
) else (
    echo Invalid choice. Starting enhanced version...
    streamlit run app_enhanced.py
)
