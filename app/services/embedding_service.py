import json
import logging
import os
from typing import Any, Dict, List

import redis
from chromadb import PersistentClient

from app.config import config
from app.services.ollama_client import OllamaClient
from app.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for managing embeddings and similarity search."""

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.embeddings_cache = {}
        self.filter_embeddings = []
        self.redis_client = self._init_redis()
        self.chroma_client = self._init_chroma()
        self.load_embeddings()

    def is_loaded(self) -> bool:
        """Check if embeddings are loaded."""
        return len(self.filter_embeddings) > 0

    def _init_chroma(self):
        """Initialize ChromaDB client."""
        client = PersistentClient(path=config.PERSIST_DIRECTORY)
        logger.info("✅ Connected to ChromaDB")
        return client

    def _init_redis(self):
        """Initialize Redis client."""
        try:
            client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=0,
                decode_responses=True  # Store as string for JSON serialization
            )
            client.ping()
            logger.info("✅ Connected to Redis cache")
            return client
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            return None

    # def load_embeddings(self):
    #     """Load pre-computed embeddings from file."""
    #     if os.path.exists(config.EMBEDDINGS_PATH):
    #         with open(config.EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
    #             self.filter_embeddings = json.load(f)
    #         logger.info(f"✅ Loaded {len(self.filter_embeddings)} pre-computed embeddings")
    #     else:
    #         logger.warning(f"⚠️  No embeddings file found at {config.EMBEDDINGS_PATH}")
    #         logger.warning(f"   Run: python scripts/generate_embeddings.py")

    # def get_query_embedding(self, query: str) -> List[float]:
    #     """
    #     Get embedding for query (with caching).

    #     Args:
    #         query: User query

    #     Returns:
    #         Embedding vector
    #     """
    #     if query in self.embeddings_cache:
    #         return self.embeddings_cache[query]

    #     embedding = self.ollama_client.generate_embedding(query)
    #     self.embeddings_cache[query] = embedding
    #     return embedding

    def load_embeddings(self):
        """Load pre-computed embeddings from file."""
        if not os.path.exists(config.PERSIST_DIRECTORY):
            logger.warning(f"⚠️  No embeddings directory found at {config.PERSIST_DIRECTORY}")
            logger.warning(f"   Run: python scripts/generate_embeddings.py")
            return

        results = self.chroma_client.get_collection(
            name=config.EMBEDDINGS_COLLECTION_NAME
        ).get(include=["metadatas", "embeddings"])
        self.filter_embeddings = list(
            {
                "category": meta.get("category", ""),
                "subcategory": meta.get("subcategory", ""),
                "value": {
                    "name": meta.get("name", ""),
                    "description": meta.get("description", "")
                },
                "embedding": embedding
            }
            for meta, embedding in zip(results["metadatas"], results["embeddings"])
        )

        logger.info(f"✅ Loaded {len(self.filter_embeddings)} pre-computed embeddings")

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query (with Redis caching).

        Args:
            query: User query

        Returns:
            Embedding vector
        """
        # 1️⃣ Try Redis cache
        if self.redis_client:
            cached = self.redis_client.get(f"embedding:{query}")
            if cached:
                logger.debug(f"🧠 Redis hit for query: {query}")
                return json.loads(cached)

        # 2️⃣ Try in-memory cache (fallback)
        if query in self.embeddings_cache:
            logger.debug(f"💾 Local cache hit for query: {query}")
            return self.embeddings_cache[query]

        # 3️⃣ Generate new embedding
        logger.debug(f"🚀 Generating embedding for new query: {query}")
        embedding = self.ollama_client.generate_embedding(query)

        # 4️⃣ Save to both caches
        self.embeddings_cache[query] = embedding
        if self.redis_client:
            self.redis_client.setex(
                f"embedding:{query}",
                config.REDIS_TTL,
                json.dumps(embedding)
            )

        return embedding

    def find_similar_filters(self, query: str) -> Dict[str, List[Dict]]:
        """
        Find most similar filters using embedding similarity.

        Args:
            query: User query

        Returns:
            Dictionary with top similar filters per category
        """
        if not self.filter_embeddings:
            logger.warning("⚠️  No embeddings loaded!")
            return {}

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
            )[:config.TOP_K_SIMILARITY]

        return grouped_results
