import logging
import os
import sys
import uuid

import chromadb
from chromadb.config import Settings

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import config
from app.services.ollama_client import OllamaClient
from app.utils.filter_loader import FilterLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("generate_embeddings")

def generate_embeddings():
    """Generate and store embeddings for all filter values in ChromaDB."""

    logger.info("🔄 Generating Embeddings for Filter Values")

    # Initialize services
    filter_loader = FilterLoader(config.FILTER_CONFIG_PATH)
    ollama_client = OllamaClient()

    # Check Ollama connection
    logger.info("1️⃣  Checking Ollama connection...")
    if not ollama_client.check_connection():
        logger.error(f"❌ Cannot connect to Ollama! Make sure Ollama is running at {config.OLLAMA_URL}. Run: ollama serve")
        return
    logger.info(f"✅ Connected to Ollama at {config.OLLAMA_URL}")

    # Load filter data
    logger.info("2️⃣  Loading filter configuration...")
    try:
        flattened_filters = filter_loader.flatten_filter_values()
        logger.info(f"✅ Loaded {len(flattened_filters)} filter values")
    except Exception as e:
        logger.error(f"❌ Error loading filters: {e}")
        return

    # Initialize Chroma client
    logger.info("3️⃣  Initializing ChromaDB client...")
    os.makedirs(config.PERSIST_DIRECTORY, exist_ok=True)

    client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=config.EMBEDDINGS_COLLECTION_NAME)

    # Generate embeddings
    logger.info(f"4️⃣  Generating embeddings using {config.OLLAMA_EMBEDDING_MODEL}...")
    total = len(flattened_filters)

    for idx, (category, subcategory, value) in enumerate(flattened_filters, 1):
        name = value.get('name', '')
        description = value.get('description', '')
        text = description.strip() if description.strip() and description.strip() != '.' else name

        try:
            embedding = ollama_client.generate_embedding(text)

            # Unique ID for each record
            uid = str(uuid.uuid4())

            # Ensure all metadata values are safe (convert None → "")
            metadata = {
                "category": str(category),
                "subcategory": str(subcategory),
                "name": str(name),
                "description": str(description)
            }
            # Store in ChromaDB
            collection.add(
                ids=[uid],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )

            # Progress indicator
            if idx % 10 == 0 or idx == total:
                progress = (idx / total) * 100
                logger.info(f"Progress: {idx}/{total} ({progress:.1f}%) - Last: {name[:50]}")

        except Exception as e:
            logger.warning(f"⚠️  Error generating embedding for '{name}': {e}")
            continue

    # Persist and summarize
    logger.info("✅ Embeddings successfully stored in ChromaDB!")
    logger.info(f"Total embeddings: {total}")
    logger.info(f"Database path: {config.PERSIST_DIRECTORY}")
    logger.info(f"Collection name: {config.EMBEDDINGS_COLLECTION_NAME}")
    logger.info("🚀 You can now query the embeddings from ChromaDB!")

if __name__ == '__main__':
    generate_embeddings()
