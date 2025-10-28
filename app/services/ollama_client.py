from typing import List

import requests

from app.config import config


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.OLLAMA_URL
        self.embedding_model = config.OLLAMA_EMBEDDING_MODEL
        self.llm_model = config.OLLAMA_LLM_MODEL

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text,
            "use_gpu": True
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["embedding"]

    def generate_completion(self, prompt: str, format: str = None) -> str:
        """
        Generate text completion using LLM.

        Args:
            prompt: Input prompt
            format: Optional format specification (e.g., "json")

        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "use_gpu": True
        }

        if format:
            payload["format"] = format

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["response"]

    def check_connection(self) -> bool:
        """
        Check if Ollama service is accessible.

        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False