# Quick Start Guide - SciEvo Streamlit Interface

## 🚀 3-Step Quick Start

### Step 1: Install Streamlit
```bash
pip install streamlit
```

### Step 2: Configure Environment
```bash
# From parent directory
cp .env.template .env
# Edit .env and add your API keys
```

### Step 3: Launch
```bash
# From streamlit-client directory
./run.sh          # Mac/Linux
run.bat           # Windows
```

## 📋 First-Time Checklist

- [ ] Install streamlit (`pip install streamlit`)
- [ ] Configure `.env` file with API keys
- [ ] Verify setup (`python test_setup.py`)
- [ ] Launch interface (`./run.sh` or `run.bat`)

## 🎯 Simple Usage Example

1. **Launch the interface**
   ```bash
   streamlit run app_enhanced.py
   ```

2. **Fill in the sidebar:**
   - Research Query: "transformer models for time series"
   - Research Domain: "machine learning"
   - Workspace: ./workspace

3. **Click "🚀 Start Research Workflow"**

4. **Watch progress in real-time:**
   - 💡 Ideation: Generates research ideas
   - 📊 Data Analysis: (if enabled)
   - 🧪 Experiments: (if enabled)

5. **View results in tabs:**
   - Summary
   - Ideation
   - Data Analysis
   - Experiments
   - Raw Data

## 🔧 Common Commands

### Test Your Setup
```bash
python test_setup.py
```

### Launch Basic Interface
```bash
streamlit run app.py
```

### Launch Enhanced Interface (Recommended)
```bash
streamlit run app_enhanced.py
```

### Launch on Different Port
```bash
streamlit run app_enhanced.py --server.port 8502
```

## 📖 Need More Help?

- **Full Documentation**: See `README.md`
- **Example Configs**: See `example_config.yaml`
- **Development Info**: See `DEVELOPMENT_SUMMARY.md`

## 💡 Quick Tips

1. Start with **Ideation Only** to understand the workflow
2. Use **Enhanced version** (`app_enhanced.py`) for better experience
3. Check `example_config.yaml` for workflow examples
4. Save important results using "💾 Save Summary" button
5. Use custom session names for organized research

## ⚠️ Troubleshooting

**Streamlit not found?**
```bash
pip install streamlit
```

**Import errors?**
```bash
# Ensure you're in the right directory
cd streamlit-client
export PYTHONPATH=..:$PYTHONPATH
streamlit run app_enhanced.py
```

**API key errors?**
```bash
# Check .env file in parent directory
cat ../.env | grep API_KEY
```

## 🎉 You're Ready!

Open your browser to `http://localhost:8501` and start your AI-powered research!
