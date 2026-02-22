# Docker Setup for SciDER Streamlit Client

This directory contains Docker configuration files for running the SciDER Streamlit client in a containerized environment.

## Quick Start

### Using Docker Compose (Recommended)

1. **Create a `.env` file** in the project root directory with your API keys:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   EMBED_API_KEY=your_embed_api_key
   SCIDER_DEFAULT_MODEL=gemini/gemini-2.5-flash-lite
   ```

2. **Build and run**:
   ```bash
   cd streamlit-client
   docker-compose up --build
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

### Using Docker directly

1. **Build the image**:
   ```bash
   cd streamlit-client
   docker build -t scider-streamlit:latest -f Dockerfile ..
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name scider-streamlit \
     -p 8501:8501 \
     -v $(pwd)/case-study-memory:/app/streamlit-client/case-study-memory \
     -v $(pwd)/workspace:/app/streamlit-client/workspace \
     -v $(pwd)/tmp_brain:/app/streamlit-client/tmp_brain \
     -v ../.env:/app/.env:ro \
     -e BRAIN_DIR=/app/streamlit-client/tmp_brain \
     -e WORKSPACE_PATH=/app/streamlit-client/workspace \
     scider-streamlit:latest
   ```

## Environment Variables

The following environment variables can be set:

- `GEMINI_API_KEY`: Gemini API key for LLM models
- `OPENAI_API_KEY`: OpenAI API key for LLM models
- `EMBED_API_KEY`: API key for embedding models
- `SCIDER_DEFAULT_MODEL`: Default model to use (default: `gemini/gemini-2.5-flash-lite`)
- `BRAIN_DIR`: Directory for Brain storage (default: `/app/streamlit-client/tmp_brain`)
- `WORKSPACE_PATH`: Workspace directory path (default: `/app/streamlit-client/workspace`)
- `CODING_AGENT_VERSION`: Coding agent version (default: `v3`)
- `SCIDER_ENABLE_OPENHANDS`: Enable OpenHands (default: `0`)

## Volumes

The following directories are mounted as volumes to persist data:

- `case-study-memory/`: Stores chat history and case studies
- `workspace/`: Workspace for data analysis and experiments
- `tmp_brain/`: Brain storage directory

## Troubleshooting

### Port already in use

If port 8501 is already in use, change it in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use 8502 instead
```

### API keys not working

Make sure your `.env` file is correctly mounted and contains valid API keys. Check the container logs:
```bash
docker logs scider-streamlit
```

### Permission issues

If you encounter permission issues with volumes, you may need to adjust permissions:
```bash
sudo chown -R $USER:$USER case-study-memory workspace tmp_brain
```

## Development

To rebuild after code changes:
```bash
docker-compose up --build
```

To view logs:
```bash
docker-compose logs -f streamlit
```

To stop the container:
```bash
docker-compose down
```
