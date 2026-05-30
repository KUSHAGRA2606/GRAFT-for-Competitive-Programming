"""
usaco_scraper.py — Scrape USACO Guide MDX modules (Silver & Gold).

Clones the cpinitiative/usaco-guide repo (shallow, --depth 1) and parses
every .mdx file under the configured division directories (e.g.
``content/3_Silver/``, ``content/4_Gold/``).  Each module's YAML
frontmatter is extracted for metadata, and the body is cleaned via
``clean_mdx()`` to produce plain-text suitable for chunking.

Public API
----------
scrape() -> list[dict]
    Returns a list of document dicts ready for the chunking pipeline.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from phase1.config import RAW_USACO_DIR, USACO_DIVISIONS, USACO_REPO_URL
from phase1.processing.text_cleaner import clean_mdx, extract_mdx_frontmatter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _clone_repo() -> None:
    """Shallow-clone the USACO Guide repo if the target directory is missing."""
    if RAW_USACO_DIR.exists():
        logger.info("USACO repo already exists at %s — skipping clone.", RAW_USACO_DIR)
        return

    RAW_USACO_DIR.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning USACO Guide repo into %s …", RAW_USACO_DIR)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", USACO_REPO_URL, str(RAW_USACO_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Clone complete.")
    except subprocess.CalledProcessError as exc:
        logger.error("git clone failed (rc=%d): %s", exc.returncode, exc.stderr.strip())
        raise


def _collect_mdx_files() -> list[tuple[str, Path]]:
    """Return a list of (division_name, mdx_path) for every .mdx file in the
    configured USACO division directories.
    """
    results: list[tuple[str, Path]] = []
    content_root = RAW_USACO_DIR / "content"

    for division, folder_name in USACO_DIVISIONS.items():
        division_dir = content_root / folder_name
        if not division_dir.is_dir():
            logger.warning(
                "Division directory not found: %s — skipping '%s'.",
                division_dir,
                division,
            )
            continue

        mdx_files = sorted(division_dir.rglob("*.mdx"))
        logger.info("Found %d .mdx files in %s.", len(mdx_files), division_dir)
        results.extend((division, p) for p in mdx_files)

    return results


def _parse_mdx_file(division: str, filepath: Path) -> dict | None:
    """Parse a single MDX file into a document dict.

    Returns ``None`` if the file is unreadable or produces empty text after
    cleaning.
    """
    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s: %s", filepath, exc)
        return None

    frontmatter, body = extract_mdx_frontmatter(raw_text)
    cleaned = clean_mdx(body)

    if not cleaned.strip():
        logger.debug("Empty text after cleaning — skipping %s.", filepath.name)
        return None

    # Resolve module identifiers with graceful fallbacks.
    module_id: str = frontmatter.get("id") or filepath.stem
    module_title: str = frontmatter.get("title", module_id)
    prerequisites: list[str] = frontmatter.get("prerequisites") or []

    # Normalise prerequisites — the field is sometimes a list of dicts in the
    # USACO Guide frontmatter (e.g. ``- slug: complete-search``).  We flatten
    # it to a plain list of strings.
    if prerequisites and isinstance(prerequisites[0], dict):
        prerequisites = [
            str(p.get("slug") or p.get("id") or p.get("name", ""))
            for p in prerequisites
            if isinstance(p, dict)
        ]
        prerequisites = [p for p in prerequisites if p]

    doc_id = f"{division}_{module_id}"

    return {
        "text": cleaned,
        "metadata": {
            "source": "usaco_guide",
            "doc_id": doc_id,
            "division": division,
            "module_id": module_id,
            "module_title": module_title,
            "prerequisites": prerequisites,
            "url": f"https://usaco.guide/{division}/{module_id}",
        },
    }


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def scrape() -> list[dict]:
    """Clone the USACO Guide repo (if needed) and return cleaned documents.

    Returns
    -------
    list[dict]
        Each dict contains ``"text"`` (cleaned module body) and ``"metadata"``
        with fields ``source``, ``doc_id``, ``division``, ``module_id``,
        ``module_title``, ``prerequisites``, and ``url``.
    """
    _clone_repo()

    mdx_files = _collect_mdx_files()
    if not mdx_files:
        logger.warning("No .mdx files found — returning empty list.")
        return []

    documents: list[dict] = []
    for division, filepath in mdx_files:
        doc = _parse_mdx_file(division, filepath)
        if doc is not None:
            documents.append(doc)

    logger.info(
        "USACO scraper finished: %d documents from %d files.",
        len(documents),
        len(mdx_files),
    )
    return documents
