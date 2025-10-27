# Hybrid Filter Selector - Step-by-Step Implementation Guide

## Prerequisites

Before starting, ensure you have:
- Python 3.11+
- Ollama installed locally
- 8GB+ RAM
- [uv](https://github.com/astral-sh/uv) installed (`uv.lock` is present in this project)

## Step 1: Install Ollama and Pull Models

```bash
# Install Ollama (if not already installed)
# For macOS:
brew install ollama

# For Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# In a new terminal, pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Step 2: Project Structure

Your project should look like:

```
smart-filter-selector_2/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── filter_routes.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── hybrid_selector.py
│   │   ├── level_detector.py
│   │   └── translation_service.py
│   └── utils/
│       ├── __init__.py
│       ├── similarity.py
│       └── filter_loader.py
├── data/
│   ├── values_with_context.json
│   ├── embeddings.json
│   ├── levels.json
│   └── values_with_context_mini.json
├── scripts/
│   └── generate_embeddings.py
├── requirements.txt
├── .env
├── run.py
├── pyproject.toml
├── uv.lock
├── quickstart.sh
├── test_api.py
└── README.md
```

## Step 3: Install Python Dependencies

Create `requirements.txt`:

```txt
flask==3.0.0
flask-cors==4.0.0
langchain==0.1.0
langchain-community==0.0.10
numpy==1.26.2
scikit-learn==1.3.2
requests==2.31.0
pydantic==2.5.0
python-dotenv==1.0.0
```

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
# Install uv if not already installed
pip install uv

# Sync dependencies and create virtual environment using uv.lock/requirements.txt
uv sync

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

## Step 4: Configuration

**Skip `.env` creation.**  
All configuration is managed via `app/config.py`.  
Edit `app/config.py` to set parameters such as:
- OLLAMA_URL
- OLLAMA_EMBEDDING_MODEL
- OLLAMA_LLM_MODEL
- FLASK_PORT
- FLASK_DEBUG
- MAX_FILTERS_PER_CATEGORY
- MIN_CONFIDENCE_THRESHOLD
- TOP_K_SIMILARITY

Example:
```python
# app/config.py
OLLAMA_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3.2:3b"
FLASK_PORT = 8000
FLASK_DEBUG = True
MAX_FILTERS_PER_CATEGORY = 10
MIN_CONFIDENCE_THRESHOLD = 0.6
TOP_K_SIMILARITY = 30
```

## Step 5: Prepare Filter Data

Copy `values_with_context.json` to the `data/` directory.

## Step 6: Generate Embeddings

```bash
# Make sure Ollama is running
uv run scripts/generate_embeddings.py
```

This will:
- Load filter values from `data/values_with_context.json`
- Generate embeddings via Ollama
- Save to `data/embeddings.json`

## Step 7: Run the Application

```bash
uv run run.py
```

You should see:

```
🚀 Starting Smart Filter Selector Service...
📊 Ollama URL: http://localhost:11434
 * Running on http://127.0.0.1:8000
```

## Step 8: Test the API

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "ollama_connected": true,
  "embeddings_loaded": true
}
```

### Analyze Query

```bash
curl -X POST http://localhost:8000/api/filter/analyze-query \
  -H "Content-Type: application/json" \
  -d '{"query": "railway signaling expert with ERTMS and SCADE experience"}'
```

Expected response structure:

```json
{
  "query": "...",
  "reducedFilters": {...},
  "confidence": {...},
  "reasoning": {...},
  "processingTime": "..."
}
```

## Step 9: Customization

- Adjust `app/config.py` for filter counts, confidence, and models.
- Use different LLM models by updating `OLLAMA_LLM_MODEL`.

## Step 10: Troubleshooting

- **Ollama not running:** `ollama serve`
- **Model not found:** `ollama pull nomic-embed-text`
- **Embeddings missing:** `uv run scripts/generate_embeddings.py`
- **Slow response:** First query loads models; subsequent queries are faster.

## Step 11: Hybrid Approach Overview

1. **Embedding-based filtering:** Fast similarity search.
2. **LLM-based refinement:** Contextual selection and reasoning.
3. **Structured output:** Consistent JSON response.

## Step 12: Integration Examples

### Python

```python
import requests
def get_smart_filters(query):
    response = requests.post(
        'http://localhost:8000/api/filter/analyze-query',
        json={'query': query}
    )
    return response.json()
```

### Node.js

```javascript
const axios = require('axios');
async function getSmartFilters(query) {
  const response = await axios.post(
    'http://localhost:8000/api/filter/analyze-query',
    { query }
  );
  return response.data;
}
```

## Step 13: Service Descriptions

### EmbeddingService
The `EmbeddingService` is responsible for managing embeddings and performing similarity searches. It:
- Loads precomputed embeddings from `data/embeddings.json`.
- Generates embeddings for user queries using the Ollama API.
- Finds the most similar filters to a query based on cosine similarity.
- Ensures embeddings are loaded and ready for use.

### LLMService
The `LLMService` refines filter candidates using a large language model (LLM) integrated with LangChain. It:
- Uses a structured prompt to analyze user queries and filter candidates.
- Selects the most relevant filters per category.
- Provides confidence scores and reasoning for each selection.
- Falls back to embedding-based results if the LLM fails.

### TranslationService
The `TranslationService` detects the language of user queries and translates them to English if necessary. It:
- Uses the LangChain framework and Ollama API for language detection and translation.
- Preserves technical terms, acronyms, and proper nouns during translation.
- Provides a fallback mechanism for basic translation if the LLM fails.

### HybridFilterSelector
The `HybridFilterSelector` orchestrates the entire query analysis pipeline. It:
- Combines the `EmbeddingService`, `LLMService`, and `TranslationService` to process queries.
- Detects and translates the query language.
- Performs embedding-based filtering to find top candidates.
- Refines the candidates using the LLM.
- Detects expertise or proficiency levels in the query.
- Applies a confidence threshold to filter out low-confidence results.

### OllamaClient
The `OllamaClient` interacts with the Ollama API to:
- Generate embeddings for text inputs.
- Generate text completions using the LLM.
- Check the connection status of the Ollama service.

### LevelDetector
The `LevelDetector` identifies expertise or proficiency levels in user queries. It:
- Analyzes the query to detect levels such as beginner, intermediate, or expert.
- Provides confidence scores and reasoning for the detected levels.

### Utility Modules
- **`similarity.py`**: Contains functions for calculating cosine similarity between embeddings.
- **`filter_loader.py`**: Handles loading and preprocessing of filter data from JSON files.
