"""Orchestrator for Phase 2: GraphRAG Construction.

Runs the complete pipeline:
1. LLM Extraction (Entities & Relationships)
2. Graph Construction (NetworkX)
3. Community Detection (Leiden)
4. LLM Summarization
5. FAISS Vectorization
"""
import argparse
import asyncio
import logging
import sys

from phase2.extraction import run_extraction
from phase2.graphing import build_graph
from phase2.clustering import detect_communities
from phase2.summarization import run_summarization
from phase2.vectorization import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase2")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2: GraphRAG Construction")
    parser.add_argument("--test", action="store_true", help="Run in test mode (extract only a small sample of chunks).")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  GRAFT Phase 2: GraphRAG Pipeline")
    logger.info("=" * 60)

    # 1. Extraction
    logger.info("\n--- Step 1: Entity & Relationship Extraction ---")
    sample_size = 5 if args.test else None
    await run_extraction(sample_size=sample_size)

    # 2. Graph Construction
    logger.info("\n--- Step 2: Graph Construction ---")
    build_graph()

    # 3. Community Detection
    logger.info("\n--- Step 3: Community Detection ---")
    detect_communities()

    # 4. Summarization
    logger.info("\n--- Step 4: Community Summarization ---")
    await run_summarization()

    # 5. Vectorization
    logger.info("\n--- Step 5: FAISS Vectorization ---")
    build_index()

    logger.info("\n" + "=" * 60)
    logger.info("  Phase 2 Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(1)
