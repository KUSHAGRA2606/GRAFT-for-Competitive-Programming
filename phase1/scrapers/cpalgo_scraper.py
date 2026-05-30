"""
cpalgo_scraper.py — Scrape cp-algorithms Markdown articles from GitHub.

Workflow
--------
1. Shallow-clone the cp-algorithms repo into ``data/raw/cpalgo/`` (skip if
   the directory already exists).
2. Walk each topic directory listed in ``CPALGO_TOPIC_DIRS`` under the
   repo's ``src/`` tree, collecting every ``.md`` file recursively.
3. Clean each file with :func:`clean_markdown`, extract a title and build
   a canonical URL, then return the results as a list of document dicts.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from phase1.config import CPALGO_REPO_URL, CPALGO_TOPIC_DIRS, RAW_CPALGO_DIR
from phase1.processing.text_cleaner import clean_markdown

logger = logging.getLogger(__name__)

# Files that should never be treated as articles.
_SKIP_FILENAMES = {"navigation.md", "index.md", "readme.md"}

# Base URL for the live cp-algorithms website.
_BASE_URL = "https://cp-algorithms.com"


# ──────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────

def _clone_repo(dest: Path) -> None:
    """Shallow-clone the cp-algorithms repo into *dest* (no-op if exists)."""
    if dest.exists():
        logger.info("Repo already present at %s — skipping clone.", dest)
        return

    logger.info("Cloning %s → %s (shallow) …", CPALGO_REPO_URL, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", CPALGO_REPO_URL, str(dest)],
        check=True,
    )
    logger.info("Clone complete.")


def _should_skip(path: Path) -> bool:
    """Return *True* if the file should be excluded from scraping."""
    return path.name.lower() in _SKIP_FILENAMES or path.name.startswith("_")


def _extract_title(text: str, stem: str) -> str:
    """Extract the article title from the first ``# `` header in *text*.

    Falls back to a prettified version of *stem* (e.g. ``min_cost_flow``
    → ``Min Cost Flow``) when no header is found.
    """
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return stem.replace("_", " ").replace("-", " ").title()


def _build_url(rel_path: Path) -> str:
    """Build the canonical cp-algorithms URL from a path relative to ``src/``.

    For example ``graph/flows/min_cost_flow.md`` →
    ``https://cp-algorithms.com/graph/flows/min_cost_flow.html``.
    """
    # Replace the .md extension with .html and use forward slashes.
    return f"{_BASE_URL}/{rel_path.with_suffix('.html').as_posix()}"


def _read_file(path: Path) -> str | None:
    """Read *path* as UTF-8 text, gracefully handling encoding errors."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Could not read %s — skipping.", path)
        return None


# ──────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────

def scrape() -> list[dict]:
    """Scrape all cp-algorithms articles and return a list of document dicts.

    Each dict has the form::

        {
            "text": "<cleaned article text>",
            "metadata": {
                "source":        "cp_algorithms",
                "doc_id":        "<category>_<article_stem>",
                "category":      "<topic directory name>",
                "article_title": "<first # header or prettified stem>",
                "url":           "https://cp-algorithms.com/…/article.html",
            },
        }

    Returns
    -------
    list[dict]
        One entry per successfully cleaned article.
    """
    _clone_repo(RAW_CPALGO_DIR)

    src_dir: Path = RAW_CPALGO_DIR / "src"
    if not src_dir.is_dir():
        logger.error("Expected source directory %s not found.", src_dir)
        return []

    documents: list[dict] = []

    for category in CPALGO_TOPIC_DIRS:
        topic_dir = src_dir / category
        if not topic_dir.is_dir():
            logger.warning("Topic directory %s does not exist — skipping.", topic_dir)
            continue

        md_files = sorted(topic_dir.rglob("*.md"))
        count = 0

        for md_path in md_files:
            if _should_skip(md_path):
                continue

            raw_text = _read_file(md_path)
            if raw_text is None:
                continue

            cleaned = clean_markdown(raw_text)
            if not cleaned.strip():
                logger.debug("Empty after cleaning: %s — skipping.", md_path)
                continue

            # Path relative to src/ (e.g. ``graph/flows/min_cost_flow.md``).
            rel_path = md_path.relative_to(src_dir)
            stem = md_path.stem  # e.g. ``min_cost_flow``

            doc_id = f"{category}_{stem}"
            title = _extract_title(cleaned, stem)
            url = _build_url(rel_path)

            documents.append(
                {
                    "text": cleaned,
                    "metadata": {
                        "source": "cp_algorithms",
                        "doc_id": doc_id,
                        "category": category,
                        "article_title": title,
                        "url": url,
                    },
                }
            )
            count += 1

        logger.info("%-25s → %d articles", category, count)

    logger.info("Total cp-algorithms articles scraped: %d", len(documents))
    return documents
