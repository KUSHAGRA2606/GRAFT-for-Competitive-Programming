"""
cph_scraper.py — Scraper for the Competitive Programmer's Handbook (CPH).

Clones the LaTeX source from GitHub (shallow), reads each chapter*.tex file,
cleans the LaTeX markup via `clean_latex()`, and returns a list of document
dicts ready for downstream chunking.
"""

import subprocess
import logging
import re
from pathlib import Path

from phase1.config import CPH_REPO_URL, RAW_CPH_DIR, CPH_CHAPTERS
from phase1.processing.text_cleaner import clean_latex

logger = logging.getLogger(__name__)


def _clone_repo(url: str, dest: Path) -> None:
    """Shallow-clone *url* into *dest* if it doesn't already contain .tex files."""
    if dest.exists() and list(dest.glob("chapter*.tex")):
        logger.info("CPH repo already present at %s — skipping clone.", dest)
        return

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning CPH repo %s → %s …", url, dest)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("git clone failed (exit %d): %s", result.returncode, result.stderr.strip())
        raise RuntimeError(f"Failed to clone CPH repo: {result.stderr.strip()}")

    logger.info("Clone complete.")


def _read_tex_file(path: Path) -> str:
    """Read a .tex file with UTF-8 encoding, replacing undecodable bytes."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return ""


def scrape() -> list[dict]:
    """
    Scrape all 27 CPH chapters and return cleaned documents.

    Returns
    -------
    list[dict]
        Each dict has keys ``"text"`` (cleaned Markdown-ish string) and
        ``"metadata"`` (source information including chapter number, title,
        and a stable ``doc_id``).
    """
    # ── 1. Clone the repo (no-op if already present) ──
    _clone_repo(CPH_REPO_URL, RAW_CPH_DIR)

    # ── 2. Discover chapter files in sorted order ──
    tex_files: list[Path] = sorted(
        RAW_CPH_DIR.glob("chapter*.tex"),
        key=lambda p: p.name,
    )

    if not tex_files:
        logger.warning("No chapter*.tex files found in %s", RAW_CPH_DIR)
        return []

    logger.info("Found %d chapter file(s) in %s.", len(tex_files), RAW_CPH_DIR)

    # ── 3. Process each chapter ──
    documents: list[dict] = []

    for tex_path in tex_files:
        # Extract chapter number from filename (e.g. "chapter07.tex" → 7)
        match = re.search(r"chapter(\d+)\.tex$", tex_path.name)
        if match is None:
            logger.warning("Skipping unexpected filename: %s", tex_path.name)
            continue

        chapter_num: int = int(match.group(1))
        chapter_title: str = CPH_CHAPTERS.get(chapter_num, f"Chapter {chapter_num}")
        doc_id: str = f"ch{chapter_num:02d}"

        logger.info("Processing %s — %s", doc_id, chapter_title)

        raw_latex: str = _read_tex_file(tex_path)
        if not raw_latex.strip():
            logger.warning("Empty or unreadable file: %s — skipping.", tex_path.name)
            continue

        cleaned_text: str = clean_latex(raw_latex)

        documents.append(
            {
                "text": cleaned_text,
                "metadata": {
                    "source": "cph",
                    "doc_id": doc_id,
                    "chapter_num": chapter_num,
                    "chapter_title": chapter_title,
                    "url": f"https://github.com/pllk/cphb/blob/master/{tex_path.name}",
                },
            }
        )

    logger.info("CPH scrape complete — %d document(s) collected.", len(documents))
    return documents
