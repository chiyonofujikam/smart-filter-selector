# Smart Filter Selector

## Project Overview
The Smart Filter Selector is a Python-based microservice designed to intelligently select filters using embeddings and large language models (LLMs). It is built with Flask and integrates various services for embedding generation, hybrid selection, and language translation.

## Features
- Embedding-based filter selection.
- LLM-based refinement for intelligent filtering.
- Language detection and translation.
- Expertise/proficiency level detection.
- Modular architecture for scalability and maintainability.

## Installation

### Prerequisites
- Python >= 3.11
- Redis server
- Ollama server for embeddings and LLMs

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd smart-filter-selector
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Configure environment variables in `app/config.py`:
   - Update parameters such as `OLLAMA_URL`, `FLASK_PORT`, and `REDIS_HOST` as needed.

4. Start Redis and Ollama servers.

5. Generate embeddings:
   ```bash
   uv run scripts/generate_embeddings.py
   ```

## Usage

### Running the Service
Start the Flask application:
```bash
uv run run.py
```

### API Endpoints
- **Health Check**: `GET /health`
- **Analyze Query**: `POST /api/filter/analyze-query`
  - Request Body:
    ```json
    {
      "query": "Your natural language query",
      "options": {
        "maxFiltersPerCategory": 10,
        "minConfidence": 0.6
      }
    }
    ```
- **List Embeddings**: `GET /api/filter/embeddings`

## Project Structure
```
smart-filter-selector/
├── app/
│   ├── config.py          # Configuration settings
│   ├── main.py            # Flask app creation and setup
│   ├── models/            # Request and response models
│   ├── routes/            # API routes
│   ├── services/          # Core services (embedding, LLM, etc.)
│   └── utils/             # Utility modules
├── data/                  # Data files (e.g., levels.json, embeddings)
├── scripts/               # Scripts for generating embeddings
├── run.py                 # Entry point for the Flask app
├── test_api.py            # API testing script
└── pyproject.toml         # Project dependencies and metadata
```

## Key Components

### Services
- **Embedding Service**: Manages embeddings and similarity search.
- **Hybrid Selector**: Combines embeddings and LLM for filter selection.
- **Translation Service**: Detects and translates non-English queries.
- **Level Detector**: Identifies expertise/proficiency levels.

### Utilities
- **Filter Loader**: Loads and manages filter configuration data.
- **Similarity**: Computes cosine similarity between vectors.

## Testing
Run the API tests:
```bash
uv run test/test_api.py
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
