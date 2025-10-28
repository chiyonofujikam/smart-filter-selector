# Intelligent Filter Selection System

This repository contains a Python-based microservice designed to provide intelligent filter recommendations. It leverages embeddings, large language models (LLMs), and hybrid approaches to analyze natural language queries and return optimized subsets of filtering options.

## Project Overview

The system is built to:
- Analyze user queries using NLP techniques.
- Generate embeddings for semantic similarity matching.
- Refine results using LLMs for contextual understanding.
- Provide structured JSON responses with confidence scores and reasoning.

## Key Components

### 1. **Embedding Service**
- Generates embeddings for filter values and user queries.
- Performs similarity matching using cosine similarity.

### 2. **LLM Service**
- Refines filter candidates using structured prompts.
- Provides reasoning and confidence scores for selections.

### 3. **Hybrid Filter Selector**
- Combines embedding-based filtering and LLM-based refinement.
- Ensures high accuracy and contextual relevance.

### 4. **Translation Service**
- Detects and translates non-English queries to English.
- Preserves technical terms and proper nouns.

### 5. **Level Detector**
- Identifies expertise levels (e.g., beginner, expert) in queries.

## Architecture

The system follows a modular architecture with the following layers:
- **API Layer**: Exposes RESTful endpoints for client applications.
- **Service Layer**: Implements core functionalities like embedding generation, LLM refinement, and query analysis.
- **Data Layer**: Manages embeddings, filter configurations, and caching.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ikos-lab/smart-filter-selector.git
   cd smart-filter-selector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file (refer to `.env.example`).

4. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

### Generate Embeddings
Run the script to generate embeddings for filter values:
```bash
python scripts/generate_embeddings.py
```

### Test the API
- Health Check:
  ```bash
  curl http://localhost:8000/health
  ```
- Analyze Query:
  ```bash
  curl -X POST http://localhost:8000/api/filter/analyze-query \
    -H "Content-Type: application/json" \
    -d '{"query": "railway signaling expert with ERTMS"}'
  ```

## Explanation

This project is designed to address the challenge of intelligently selecting filters based on user queries. By combining embeddings for fast similarity matching and LLMs for contextual refinement, the system ensures accurate and relevant results. The modular design allows for easy integration and scalability.

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

## Step 14: Redis Setup & Connection Guide

This guide explains how to install, start, and connect to **Redis** for embedding caching in the project.

### 1️⃣ Install Redis

Run the following commands in your terminal:

```bash
sudo apt update
sudo apt install redis-server redis-tools -y
```

This installs:

* `redis-server`: the Redis database
* `redis-cli`: the command-line interface to test Redis

---

### 2️⃣ Start and Enable the Redis Service

Start Redis and make sure it runs automatically on boot:

```bash
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

Check that it’s running:

```bash
sudo systemctl status redis-server
```

✅ You should see:

```
Active: active (running)
```

---

### 3️⃣ Test Redis Connection

Run:

```bash
redis-cli ping
```

If Redis is working, it will return:

```
PONG
```

---

### 4️⃣ Configure Your App

In your `config.py`, ensure these settings are correct:

```python
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_TTL = 86400  # Cache expiry in seconds (24h)
```

Your app will automatically connect and log:

```
✅ Connected to Redis cache
```

---

### ⚠️ Common Error: "Error 111 connecting to localhost:6379. Connection refused."

If you see this message in your logs:

```
⚠️ Redis connection failed: Error 111 connecting to localhost:6379. Connection refused.
```

It means Redis is **not running** or **not reachable**.

#### ✅ Fix:

1. Start Redis:

   ```bash
   sudo systemctl start redis-server
   ```
2. Check its status:

   ```bash
   sudo systemctl status redis-server
   ```
3. Test again:

   ```bash
   redis-cli ping
   ```

   You should see `PONG`.

If you’re using Docker instead of a local installation, you can start Redis with:

```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

---

### 5️⃣ Verify in Your App

When Redis is running and your app starts, you should see:

```
✅ Connected to Redis cache
```

and embedding queries will be cached efficiently.
