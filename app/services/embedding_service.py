import json
import os
from typing import Dict, List, Any
from app.services.ollama_client import OllamaClient
from app.utils.similarity import cosine_similarity
from app.config import config
import logging
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for managing embeddings and similarity search."""

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.embeddings_cache = {}
        self.filter_embeddings = []
        self.load_embeddings()

    def is_loaded(self) -> bool:
        """Check if embeddings are loaded."""
        return len(self.filter_embeddings) > 0

    def load_embeddings(self):
        """Load pre-computed embeddings from file."""
        if os.path.exists(config.EMBEDDINGS_PATH):
            with open(config.EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
                self.filter_embeddings = json.load(f)
            logger.info(f"✅ Loaded {len(self.filter_embeddings)} pre-computed embeddings")
        else:
            logger.warning(f"⚠️  No embeddings file found at {config.EMBEDDINGS_PATH}")
            logger.warning(f"   Run: python scripts/generate_embeddings.py")

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query (with caching).

        Args:
            query: User query

        Returns:
            Embedding vector
        """
        if query in self.embeddings_cache:
            return self.embeddings_cache[query]

        embedding = self.ollama_client.generate_embedding(query)
        self.embeddings_cache[query] = embedding
        return embedding

    def find_similar_filters(self, query: str, top_k: int = None) -> Dict[str, List[Dict]]:
        """
        Find most similar filters using embedding similarity.

        Args:
            query: User query
            top_k: Number of top results per category

        Returns:
            Dictionary with top similar filters per category
        """
        if not self.filter_embeddings:
            logger.warning("⚠️  No embeddings loaded!")
            return {}

        top_k = top_k or config.TOP_K_SIMILARITY

        # Get query embedding
        query_embedding = self.get_query_embedding(query)

        # Calculate similarities for all filters
        grouped_results = {}

        for filter_data in self.filter_embeddings:
            result = {
                'category': filter_data['category'],
                'subcategory': filter_data['subcategory'],
                'value': filter_data['value'],
                'score': cosine_similarity(
                    query_embedding,
                    filter_data['embedding']
                )
            }

            if result['category'] not in grouped_results:
                grouped_results[result['category']] = []
            grouped_results[result['category']].append(result)

        # Sort each category by score and take top-K
        for category in grouped_results:
            grouped_results[category] = sorted(
                grouped_results[category],
                key=lambda x: x['score'],
                reverse=True
            )[:top_k]

        return grouped_results
