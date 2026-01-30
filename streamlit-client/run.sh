#!/bin/bash
# Quick start script for SciEvo Streamlit Interface

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed. Installing..."
    pip install streamlit
fi

# Check if parent .env exists
if [ ! -f "../.env" ]; then
    echo "⚠️  Warning: .env file not found in parent directory"
    echo "Please copy .env.template to .env and configure your API keys"
    echo ""
    read -p "Press enter to continue anyway or Ctrl+C to exit..."
fi

echo "🚀 Starting SciEvo Streamlit Interface..."
echo ""
echo "Choose version:"
echo "1) Enhanced (recommended) - Real-time progress tracking"
echo "2) Basic - Simple interface"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "Starting enhanced version..."
        streamlit run app_enhanced.py
        ;;
    2)
        echo "Starting basic version..."
        streamlit run app.py
        ;;
    *)
        echo "Invalid choice. Starting enhanced version..."
        streamlit run app_enhanced.py
        ;;
esac
