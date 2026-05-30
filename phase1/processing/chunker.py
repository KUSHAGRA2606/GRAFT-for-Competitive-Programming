"""
chunker.py — LangChain-based text chunking with metadata enrichment.

Splits cleaned documents into 600-char chunks with 100-char overlap,
prioritizing splits at Markdown headers for topical coherence.
Each chunk is enriched with a unique ID, index, and nearest header.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from phase1.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS
from phase1.processing.text_cleaner import find_nearest_header


def _make_splitter() -> RecursiveCharacterTextSplitter:
    """Create the configured text splitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


_SPLITTER = _make_splitter()


def _generate_chunk_id(source: str, doc_id: str, chunk_index: int) -> str:
    """
    Generate a unique chunk ID.
    Examples: cph_ch07_003, cpalgo_graph_dijkstra_002, cf_1234C_001
    """
    return f"{source}_{doc_id}_{chunk_index:03d}"


def chunk_document(
    text: str,
    metadata: dict[str, Any],
    source: str,
    doc_id: str,
) -> list[dict[str, Any]]:
    """
    Split a single cleaned document into chunks with enriched metadata.

    Args:
        text:     The cleaned document text.
        metadata: Base metadata dict (source-specific fields).
        doc_id:   Short identifier for the document (e.g., "ch07", "dijkstra").
        source:   Source name (cph, usaco_guide, cp_algorithms, codeforces).

    Returns:
        List of chunk dicts, each with 'chunk_id', 'text', and 'metadata'.
    """
    if not text or not text.strip():
        return []

    chunks_text = _SPLITTER.split_text(text)
    result = []

    # Pre-compute header positions for nearest-header lookup
    for idx, chunk_text in enumerate(chunks_text):
        # Find the position of this chunk in the original text
        # (approximate — find first occurrence starting from expected position)
        chunk_start = text.find(chunk_text[:50])
        if chunk_start == -1:
            chunk_start = 0

        nearest = find_nearest_header(text, chunk_start)

        chunk = {
            "chunk_id": _generate_chunk_id(source, doc_id, idx),
            "text": chunk_text.strip(),
            "metadata": {
                **metadata,
                "chunk_index": idx,
                "nearest_header": nearest,
            },
        }
        result.append(chunk)

    return result


def chunk_documents(
    documents: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    """
    Chunk a list of documents from a single source.

    Each document dict must have:
        - "text": str
        - "metadata": dict  (must include "doc_id")

    Returns:
        Flat list of all chunks across all documents.
    """
    all_chunks = []
    for doc in documents:
        doc_id = doc["metadata"].get("doc_id", "unknown")
        chunks = chunk_document(
            text=doc["text"],
            metadata=doc["metadata"],
            source=source,
            doc_id=doc_id,
        )
        all_chunks.extend(chunks)
    return all_chunks
