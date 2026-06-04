"""Vectorization script for Phase 2: GraphRAG.

Embeds community summaries using sentence-transformers and saves to a local
FAISS index for rapid semantic retrieval in Phase 5.
"""
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from phase2.config import EMBEDDING_MODEL, FAISS_INDEX_DIR, SUMMARIES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_index() -> None:
    if not SUMMARIES_FILE.exists():
        logger.error(f"Summaries file not found: {SUMMARIES_FILE}")
        return

    logger.info(f"Loading summaries from {SUMMARIES_FILE}")
    with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    if not summaries:
        logger.warning("No summaries to vectorize.")
        return

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts_to_embed = [s["summary"] for s in summaries]
    
    logger.info(f"Encoding {len(texts_to_embed)} summaries...")
    # encode() returns a numpy array
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    
    # FAISS requires float32
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")

    # Create L2 FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    logger.info(f"FAISS index built with {index.ntotal} vectors.")

    # Save index and mapping
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss_file = FAISS_INDEX_DIR / "community_index.faiss"
    mapping_file = FAISS_INDEX_DIR / "community_mapping.json"
    
    faiss.write_index(index, str(faiss_file))
    
    # Save the mapping from FAISS integer ID (index in array) to community data
    mapping = {
        i: {
            "community_id": s["community_id"],
            "level": s.get("level"),
            "nodes": s["nodes"],
            "summary": s["summary"]
        }
        for i, s in enumerate(summaries)
    }
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved FAISS index to {faiss_file}")
    logger.info(f"Saved mapping to {mapping_file}")


if __name__ == "__main__":
    build_index()
