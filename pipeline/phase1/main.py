"""
main.py — Phase 1 orchestrator for GRAFT data ingestion & chunking.

Usage:
    python -m phase1.main                    # Run full pipeline (all sources)
    python -m phase1.main --source cph       # Run only CPH scraper
    python -m phase1.main --source usaco     # Run only USACO Guide scraper
    python -m phase1.main --source cpalgo    # Run only cp-algorithms scraper
    python -m phase1.main --source codeforces  # Run only Codeforces scraper
    python -m phase1.main --chunk-only       # Re-chunk from cached raw documents
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from phase1.config import OUTPUT_FILE, RAW_DIR

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")


# ──────────────────────────────────────────────
# Scraper registry
# ──────────────────────────────────────────────

SCRAPERS = {
    "cph": {
        "label": "Competitive Programmer's Handbook",
        "module": "phase1.scrapers.cph_scraper",
        "source_key": "cph",
    },
    "usaco": {
        "label": "USACO Guide (Silver + Gold)",
        "module": "phase1.scrapers.usaco_scraper",
        "source_key": "usaco_guide",
    },
    "cpalgo": {
        "label": "cp-algorithms",
        "module": "phase1.scrapers.cpalgo_scraper",
        "source_key": "cp_algorithms",
    },
    "codeforces": {
        "label": "Codeforces Editorials",
        "module": "phase1.scrapers.codeforces_scraper",
        "source_key": "codeforces",
    },
}

# Path for cached intermediate documents (before chunking)
_CACHED_DOCS_FILE = RAW_DIR / "_all_documents.json"


def _import_scraper(module_path: str):
    """Dynamically import a scraper module and return it."""
    import importlib
    return importlib.import_module(module_path)


# ──────────────────────────────────────────────
# Scraping phase
# ──────────────────────────────────────────────

def run_scraper(name: str) -> list[dict]:
    """Run a single scraper by name, return list of document dicts."""
    info = SCRAPERS[name]
    log.info("━" * 60)
    log.info(f"Scraping: {info['label']}")
    log.info("━" * 60)

    mod = _import_scraper(info["module"])
    documents = mod.scrape()

    log.info(f"  ✓ {info['label']}: {len(documents)} documents scraped")
    return documents


def run_all_scrapers(sources: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Run scrapers for the specified sources (or all if None).
    Returns {source_key: [document_dicts]}.
    """
    targets = sources or list(SCRAPERS.keys())
    all_docs: dict[str, list[dict]] = {}

    for name in targets:
        if name not in SCRAPERS:
            log.warning(f"Unknown source '{name}', skipping")
            continue
        try:
            docs = run_scraper(name)
            source_key = SCRAPERS[name]["source_key"]
            all_docs[source_key] = docs
        except Exception:
            log.exception(f"Failed to scrape {name}")
            all_docs[SCRAPERS[name]["source_key"]] = []

    return all_docs


# ──────────────────────────────────────────────
# Chunking phase
# ──────────────────────────────────────────────

def chunk_all_documents(all_docs: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """
    Chunk all documents from all sources.
    Returns (all_chunks, source_stats).
    """
    from phase1.processing.chunker import chunk_documents

    all_chunks: list[dict] = []
    source_stats: dict[str, dict] = {}

    for source_key, documents in all_docs.items():
        if not documents:
            source_stats[source_key] = {"documents": 0, "chunks": 0}
            continue

        log.info(f"Chunking {source_key}: {len(documents)} documents...")
        chunks = chunk_documents(documents, source=source_key)
        all_chunks.extend(chunks)

        source_stats[source_key] = {
            "documents": len(documents),
            "chunks": len(chunks),
        }
        log.info(f"  ✓ {source_key}: {len(chunks)} chunks")

    return all_chunks, source_stats


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def save_corpus(
    chunks: list[dict],
    source_stats: dict[str, dict],
    output_path: Path,
) -> None:
    """Save the final algorithmic_corpus.json."""
    from phase1.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

    corpus = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(chunks),
            "sources": source_stats,
            "chunk_config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "separators": CHUNK_SEPARATORS,
            },
        },
        "chunks": chunks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"✅ Saved {len(chunks)} chunks to {output_path} ({size_mb:.1f} MB)")


def save_cached_docs(all_docs: dict[str, list[dict]]) -> None:
    """Cache scraped documents for --chunk-only reruns."""
    _CACHED_DOCS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHED_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    log.info(f"Cached documents to {_CACHED_DOCS_FILE}")


def load_cached_docs() -> dict[str, list[dict]]:
    """Load cached documents from a previous scrape run."""
    if not _CACHED_DOCS_FILE.exists():
        log.error(f"No cached documents found at {_CACHED_DOCS_FILE}")
        log.error("Run the full pipeline first (without --chunk-only)")
        sys.exit(1)
    with open(_CACHED_DOCS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: GRAFT Data Ingestion & Chunking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m phase1.main                     # Full pipeline
  python -m phase1.main --source cph        # CPH only
  python -m phase1.main --source codeforces # Codeforces only
  python -m phase1.main --chunk-only        # Re-chunk cached data
        """,
    )
    parser.add_argument(
        "--source",
        choices=list(SCRAPERS.keys()),
        nargs="+",
        default=None,
        help="Run only specific scrapers (default: all)",
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Skip scraping, re-chunk from cached documents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output file path (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or OUTPUT_FILE

    log.info("=" * 60)
    log.info("  GRAFT Phase 1: Data Ingestion & Chunking Pipeline")
    log.info("=" * 60)

    # ── Phase A: Scrape or load cached docs ──
    if args.chunk_only:
        log.info("Loading cached documents (--chunk-only mode)...")
        all_docs = load_cached_docs()
    else:
        all_docs = run_all_scrapers(sources=args.source)
        save_cached_docs(all_docs)

    # ── Summary ──
    total_docs = sum(len(docs) for docs in all_docs.values())
    log.info(f"\nTotal documents across all sources: {total_docs}")
    for source, docs in all_docs.items():
        log.info(f"  {source}: {len(docs)} documents")

    if total_docs == 0:
        log.warning("No documents scraped. Exiting.")
        sys.exit(1)

    # ── Phase B: Chunk ──
    all_chunks, source_stats = chunk_all_documents(all_docs)

    # ── Phase C: Save ──
    save_corpus(all_chunks, source_stats, output_path)

    # ── Final summary ──
    log.info("\n" + "=" * 60)
    log.info("  Pipeline Complete!")
    log.info("=" * 60)
    log.info(f"  Total chunks: {len(all_chunks)}")
    for source, stats in source_stats.items():
        log.info(f"  {source}: {stats['documents']} docs → {stats['chunks']} chunks")
    log.info(f"  Output: {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
