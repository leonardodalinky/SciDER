# SciDER Streamlit Chat Interface

A ChatGPT-like web interface for running the complete SciDER workflow with real-time progress tracking and intermediate output visualization.

## Features

- 🎨 **Modern Chat Interface**: Clean, intuitive interface similar to ChatGPT
- 💡 **Full Workflow Support**: Run the complete research pipeline with ideation agent
- 📊 **Real-time Progress**: Live updates showing progress for each agent and sub-agent
- 🔍 **Intermediate Outputs**: View outputs from each stage:
  - Literature search results
  - Paper analysis
  - Research idea generation
  - Novelty assessment
  - Data analysis
  - Experiment execution and revisions
- ⚙️ **Configurable Settings**: Control all workflow parameters through the UI
- 📁 **Session Management**: Save and manage research sessions

## Installation

### 1. Install Dependencies

From the `streamlit-client` directory:

```bash
pip install -r requirements.txt
```

Or install streamlit directly:

```bash
pip install streamlit
```

### 2. Ensure SciDER is Set Up

Make sure you've completed the main SciDER setup from the parent directory:

```bash
cd ..
uv sync --extra mac  # or cpu/cu128 depending on your platform
```

### 3. Configure Environment

Ensure your `.env` file in the parent directory is properly configured with API keys:

```bash
cp ../.env.template ../.env
# Edit .env and add your API keys:
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...
```

## Usage

### Running the Application

#### Basic Version (Simple Interface)

```bash
streamlit run app.py
```

#### Enhanced Version (With Real-time Progress)

```bash
streamlit run app_enhanced.py
```

The application will open in your default web browser at `http://localhost:8501`

### Configuring Your Research

1. **Research Query** (Required)
   - Enter your research topic or experimental objective
   - Example: "transformer models for time series forecasting"

2. **Research Domain** (Optional)
   - Specify the domain for better context
   - Example: "machine learning", "computational biology"

3. **Workspace Settings**
   - Set the workspace path where results will be saved
   - Optionally provide a custom session name

4. **Enable Workflow Stages**
   - **Ideation**: Always enabled - generates research ideas
   - **Data Analysis**: Optional - analyzes your data and finds relevant papers/datasets
   - **Experiment Execution**: Optional - generates and executes experimental code

5. **Advanced Settings**
   - Adjust recursion limits for each agent
   - Configure revision limits for experiments

### Understanding the Output

#### Phase 1: Research Ideation (Always runs)

- **Literature Search**: Searches for relevant papers
- **Paper Analysis**: Analyzes found papers for insights
- **Idea Generation**: Generates novel research directions
- **Novelty Check**: Assesses novelty of generated ideas (0-10 score)
- **Ideation Report**: Comprehensive research ideation summary

**Outputs:**
- List of reviewed papers with abstracts
- Generated research ideas
- Novelty score and feedback
- Research ideation summary

#### Phase 2: Data Analysis (Optional)

- **Data Exploration**: Analyzes structure and characteristics of your data
- **Paper Subagent**: Searches for relevant papers, datasets, and metrics
- **Summary Generation**: Creates comprehensive data analysis report

**Outputs:**
- Data structure analysis
- Statistical summaries
- Found papers (with relevance scores)
- Found datasets
- Found evaluation metrics
- `data_analysis.md` file in workspace

#### Phase 3: Experiment Execution (Optional)

- **Coding Subagent**: Generates experimental code
- **Execution Subagent**: Runs the generated code
- **Summary Subagent**: Analyzes results and metrics
- **Revision Loop**: Iterates to improve experiments (up to max_revisions)

**Outputs:**
- Generated code
- Execution logs
- Metrics and results
- Revision summaries
- Final experiment summary

### Viewing Results

After workflow completion, results are displayed in organized tabs:

- **Summary**: Complete workflow summary
- **Ideation**: Research ideas, novelty assessment, papers reviewed
- **Data Analysis**: Data insights, found papers/datasets/metrics
- **Experiments**: Execution results, code, metrics from all revisions
- **Raw Data**: Complete workflow state information

### Saving Results

Click the "💾 Save Summary" button to save the complete workflow summary to:
```
<workspace_path>/workflow_summary.md
```

## File Structure

```
streamlit-client/
├── app.py                    # Basic Streamlit interface
├── app_enhanced.py           # Enhanced interface with real-time progress
├── display_components.py     # Reusable UI components for displaying results
├── workflow_monitor.py       # Progress monitoring and callback system
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Configuration Options

### Workspace Settings

- **workspace_path**: Directory where all outputs are saved
- **session_name**: Custom name for the session (auto-generated if not provided)

### Data Workflow Settings

- **data_path**: Path to your data file (CSV, JSON, etc.)
- **data_desc**: Additional description or context about your data

### Experiment Workflow Settings

- **repo_source**: Local path or git URL to code repository
- **max_revisions**: Maximum number of revision loops (1-10)

### Advanced Settings

- **ideation_agent_recursion_limit**: Max steps for ideation agent (default: 50)
- **data_agent_recursion_limit**: Max steps for data agent (default: 100)
- **experiment_agent_recursion_limit**: Max steps for experiment agent (default: 100)

## Example Workflows

### 1. Research Ideation Only

```
Research Query: "self-supervised learning for protein structure prediction"
Research Domain: "computational biology"
Workspace: "./workspace/protein_research"

Enable Data Analysis: ☐
Enable Experiment Execution: ☐
```

This will:
- Search literature on the topic
- Analyze relevant papers
- Generate novel research ideas
- Assess novelty of ideas

### 2. Full Pipeline: Ideation + Data + Experiments

```
Research Query: "improve time series forecasting accuracy"
Research Domain: "machine learning"
Workspace: "./workspace/timeseries_exp"
Data Path: "./data/stock_prices.csv"

Enable Data Analysis: ☑
Enable Experiment Execution: ☑
Max Revisions: 5
```

This will:
- Generate research ideas for improvement
- Analyze your time series data
- Search for relevant papers and benchmarks
- Generate forecasting code
- Execute experiments
- Iterate to improve results

### 3. Data Analysis + Experiments (Skip Ideation for established research)

While ideation always runs, you can focus on data and experiments by providing a specific, well-defined query:

```
Research Query: "train LSTM model on this data with standard metrics"
Data Path: "./data/sensor_data.csv"

Enable Data Analysis: ☑
Enable Experiment Execution: ☑
```

## Troubleshooting

### Streamlit Not Found

```bash
pip install streamlit
```

### Import Errors

Ensure you're running from the correct directory and that the parent SciDER package is accessible:

```bash
cd streamlit-client
export PYTHONPATH=..:$PYTHONPATH
streamlit run app_enhanced.py
```

### API Key Errors

Check that your `.env` file in the parent directory contains valid API keys:

```bash
cat ../.env | grep API_KEY
```

### Workflow Hangs

- Check recursion limits - increase if the agent needs more steps
- Monitor logs in terminal for errors
- Check that data file path is correct and accessible

### Port Already in Use

If port 8501 is already in use:

```bash
streamlit run app_enhanced.py --server.port 8502
```

## Tips

1. **Start Simple**: Begin with ideation only to understand the workflow
2. **Iterate**: Use the revision system for experiments to improve results
3. **Save Sessions**: Use custom session names for important research
4. **Review Intermediate Outputs**: Check paper searches and data analysis before experiments
5. **Monitor Progress**: Watch the real-time updates to understand agent behavior

## Advanced Usage

### Custom Workflow Monitoring

You can extend `workflow_monitor.py` to add custom callbacks:

```python
from workflow_monitor import get_monitor, PhaseType

monitor = get_monitor()

def my_callback(update):
    print(f"Phase: {update.phase}, Status: {update.status}")
    # Your custom logic here

monitor.add_callback(my_callback)
```

### Integrating with Other Tools

The workflow results can be exported and used with other analysis tools:

```python
from scider.workflows.full_workflow_with_ideation import run_full_workflow_with_ideation

result = run_full_workflow_with_ideation(
    user_query="your query",
    workspace_path="./workspace",
)

# Access structured outputs
print(result.ideation_papers)
print(result.novelty_score)
print(result.data_summary)
```

## Contributing

To improve the interface:

1. Add new visualization components in `display_components.py`
2. Enhance progress tracking in `workflow_monitor.py`
3. Improve the UI in `app_enhanced.py`

## License

Same as parent SciDER project.
