"""
codeforces_scraper.py — Scrape Codeforces editorial text for competition problems.

Three-step pipeline:
  1. Fetch problem lists via the CF API (tags: dp, graphs, greedy; rating 1300-1800).
  2. Discover editorial blog IDs by scraping each contest's HTML page.
  3. Fetch editorial HTML via the blog API, split by problem, and clean.

All intermediate results are cached to disk so the pipeline can resume on failure.
"""

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from phase1.config import (
    CF_API_BASE,
    CF_TAGS,
    CF_RATING_MIN,
    CF_RATING_MAX,
    CF_REQUEST_DELAY,
    RAW_CF_DIR,
)
from phase1.processing.text_cleaner import clean_html

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
_HEADERS = {"User-Agent": "GRAFT-Scraper/1.0 (research)"}
_MAX_RETRIES = 3
_EDITORIAL_MAP_PATH = RAW_CF_DIR / "editorial_map.json"
_EDITORIALS_DIR = RAW_CF_DIR / "editorials"
_CHECKPOINT_BATCH_SIZE = 10

# ──────────────────────────────────────────────
#  HTTP helpers
# ──────────────────────────────────────────────

def _request_with_backoff(url: str, params: dict | None = None) -> requests.Response:
    """Make a GET request with exponential backoff on 429/503 errors.

    Raises ``requests.HTTPError`` after *_MAX_RETRIES* consecutive failures
    on retryable status codes, or immediately on other 4xx/5xx errors.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if resp.status_code in (429, 503):
                wait = CF_REQUEST_DELAY * (2 ** attempt)
                logger.warning(
                    "HTTP %d from %s — retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, url, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = CF_REQUEST_DELAY * (2 ** attempt)
                logger.warning(
                    "Request error for %s: %s — retrying in %.1fs (attempt %d/%d)",
                    url, exc, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise
    # Final retry exhausted (shouldn't normally reach here)
    raise requests.HTTPError(f"Max retries exceeded for {url}")


# ──────────────────────────────────────────────
#  Step 1 — Fetch problem lists
# ──────────────────────────────────────────────

def _fetch_problems() -> tuple[dict[tuple[int, str], dict], dict[int, list[dict]]]:
    """Return ``(problems_by_key, problems_by_contest)`` for the configured tags/ratings.

    Each *problem dict* carries the fields returned by the CF API plus the
    ``"tags"`` list already attached.  ``problems_by_contest`` groups them by
    ``contestId`` for downstream processing.
    """
    problems: dict[tuple[int, str], dict] = {}

    for tag in CF_TAGS:
        url = f"{CF_API_BASE}/problemset.problems"
        logger.info("Fetching problem list for tag=%s", tag)
        time.sleep(CF_REQUEST_DELAY)

        try:
            resp = _request_with_backoff(url, params={"tags": tag})
            data = resp.json()
        except Exception:
            logger.warning("Failed to fetch problems for tag=%s — skipping", tag, exc_info=True)
            continue

        if data.get("status") != "OK":
            logger.warning("CF API returned status=%s for tag=%s", data.get("status"), tag)
            continue

        for prob in data["result"]["problems"]:
            rating = prob.get("rating")
            if rating is None:
                continue
            if not (CF_RATING_MIN <= rating <= CF_RATING_MAX):
                continue

            key = (prob["contestId"], prob["index"])
            if key not in problems:
                problems[key] = {
                    "contestId": prob["contestId"],
                    "index": prob["index"],
                    "name": prob.get("name", ""),
                    "rating": rating,
                    "tags": prob.get("tags", []),
                }

    # Group by contest
    by_contest: dict[int, list[dict]] = defaultdict(list)
    for prob in problems.values():
        by_contest[prob["contestId"]].append(prob)

    logger.info(
        "Step 1 complete: %d unique problems across %d contests",
        len(problems), len(by_contest),
    )
    return problems, dict(by_contest)


# ──────────────────────────────────────────────
#  Step 2 — Find editorial blog IDs
# ──────────────────────────────────────────────

def _load_editorial_map() -> dict[str, int | None]:
    """Load the cached contestId → blogId mapping from disk."""
    if _EDITORIAL_MAP_PATH.exists():
        with open(_EDITORIAL_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_editorial_map(emap: dict[str, int | None]) -> None:
    """Persist the editorial map to disk."""
    _EDITORIAL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_EDITORIAL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(emap, f, indent=2)


def _find_editorial_blog_ids(contest_ids: list[int]) -> dict[int, int]:
    """Scrape contest pages to discover editorial blog entry IDs.

    Returns a mapping ``{contestId: blogId}`` for contests that have an
    editorial link.  The results are cached in ``editorial_map.json`` so
    subsequent runs skip already-processed contests.
    """
    emap = _load_editorial_map()
    result: dict[int, int] = {}

    # Populate result from cache for contest_ids we already know about
    for cid in contest_ids:
        cached = emap.get(str(cid))
        if cached is not None and cached != -1:
            result[cid] = cached

    # Determine which contests still need scraping
    to_scrape = [] # FORCE SKIP SCRAPING DUE TO 403 BLOCK [cid for cid in contest_ids if str(cid) not in emap]
    if not to_scrape:
        logger.info("Skipping further HTML scraping due to 403 block. Using %d contest editorial mappings loaded from cache", len(result))
        return result

    logger.info("Scraping editorial links for %d contests (%d cached)", len(to_scrape), len(result))
    scraped_count = 0

    for cid in tqdm(to_scrape, desc="Finding editorials", unit="contest"):
        time.sleep(CF_REQUEST_DELAY)
        url = f"https://codeforces.com/contest/{cid}"
        try:
            resp = _request_with_backoff(url)
            soup = BeautifulSoup(resp.text, "lxml")
            blog_id = _extract_editorial_link(soup)
        except Exception:
            logger.warning("Failed to scrape contest %d — skipping", cid, exc_info=True)
            emap[str(cid)] = -1  # mark as attempted
            scraped_count += 1
            if scraped_count % _CHECKPOINT_BATCH_SIZE == 0:
                _save_editorial_map(emap)
            continue

        if blog_id is not None:
            emap[str(cid)] = blog_id
            result[cid] = blog_id
        else:
            emap[str(cid)] = -1  # no editorial found
            logger.debug("No editorial link found for contest %d", cid)

        scraped_count += 1
        if scraped_count % _CHECKPOINT_BATCH_SIZE == 0:
            _save_editorial_map(emap)

    # Final save
    _save_editorial_map(emap)
    logger.info("Step 2 complete: %d contests have editorials", len(result))
    return result


def _extract_editorial_link(soup: BeautifulSoup) -> int | None:
    """Find a link labelled "Tutorial" or "Editorial" and return the blog ID."""
    for a_tag in soup.find_all("a", href=True):
        text = a_tag.get_text(strip=True)
        if re.search(r"tutorial|editorial", text, re.IGNORECASE):
            m = re.search(r"/blog/entry/(\d+)", a_tag["href"])
            if m:
                return int(m.group(1))
    return None


# ──────────────────────────────────────────────
#  Step 3 — Fetch editorial content
# ──────────────────────────────────────────────

def _editorial_cache_path(blog_id: int) -> Path:
    """Return the cache file path for a single editorial."""
    return _EDITORIALS_DIR / f"{blog_id}.json"


def _fetch_editorial_html(blog_id: int) -> str | None:
    """Fetch editorial HTML from the API (or cache). Returns HTML string or None."""
    cache = _editorial_cache_path(blog_id)
    if cache.exists():
        with open(cache, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("content")

    time.sleep(CF_REQUEST_DELAY)
    url = f"{CF_API_BASE}/blogEntry.view"
    try:
        resp = _request_with_backoff(url, params={"id": blog_id})
        data = resp.json()
    except Exception:
        logger.warning("Failed to fetch editorial blog %d", blog_id, exc_info=True)
        return None

    if data.get("status") != "OK":
        logger.warning("CF API error for blog %d: %s", blog_id, data.get("comment", ""))
        return None

    content = data["result"].get("content", "")

    # Cache raw HTML
    _EDITORIALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"blog_id": blog_id, "content": content}, f, ensure_ascii=False)

    return content


# ──────────────────────────────────────────────
#  Editorial splitting
# ──────────────────────────────────────────────

# Regex patterns that identify the start of a per-problem section in an
# editorial.  Ordered from most specific to least specific.
_PROBLEM_HEADER_PATTERNS = [
    # <b>A</b>, <h3>A</h3>, <h2>Problem A</h2>, etc.
    re.compile(
        r"<(?:b|strong|h[1-6])[^>]*>\s*(?:Problem\s+)?([A-Z]\d?)\s*[.\-—:]?\s*"
        r"(?:.*?)</(?:b|strong|h[1-6])>",
        re.IGNORECASE | re.DOTALL,
    ),
    # Markdown-style: ## A., **A.**, **Problem A**
    re.compile(
        r"(?:#{1,4}\s+|(?:\*\*))(?:Problem\s+)?([A-Z]\d?)\s*[.\-—:]?",
        re.IGNORECASE,
    ),
    # Plain text line: "A - Problem Name" or "A. Problem Name"
    re.compile(
        r"(?:^|\n)\s*([A-Z]\d?)\s*[.\-—:]\s+\S",
        re.MULTILINE,
    ),
]


def _split_editorial_by_problem(
    html: str, problem_indices: set[str],
) -> dict[str, str]:
    """Split editorial HTML into per-problem sections.

    Returns ``{index: html_section}`` for each problem index found in the
    editorial.  If splitting fails (no recognisable headers), returns an
    empty dict so the caller can fall back to whole-editorial mode.
    """
    # Collect all header positions and the index letter they refer to.
    markers: list[tuple[int, str]] = []

    for pattern in _PROBLEM_HEADER_PATTERNS:
        for m in pattern.finditer(html):
            idx = m.group(1).upper()
            # Only consider indices we actually care about
            if idx in problem_indices or idx.rstrip("0123456789") in problem_indices:
                markers.append((m.start(), idx))

    if not markers:
        return {}

    # Deduplicate by position and sort
    markers = sorted(set(markers), key=lambda x: x[0])

    # Deduplicate by index — keep first occurrence
    seen: set[str] = set()
    unique_markers: list[tuple[int, str]] = []
    for pos, idx in markers:
        if idx not in seen:
            seen.add(idx)
            unique_markers.append((pos, idx))
    markers = unique_markers

    sections: dict[str, str] = {}
    for i, (pos, idx) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(html)
        sections[idx] = html[pos:end]

    return sections


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def scrape() -> list[dict]:
    """Run the full Codeforces editorial scraping pipeline.

    Returns a list of document dicts, each with ``"text"`` and ``"metadata"``
    keys, suitable for downstream chunking and embedding.
    """
    logger.info("=== Codeforces scraper starting ===")

    # Ensure output directories exist
    RAW_CF_DIR.mkdir(parents=True, exist_ok=True)
    _EDITORIALS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch & filter problems
    problems_by_key, problems_by_contest = _fetch_problems()
    if not problems_by_key:
        logger.warning("No problems found — aborting")
        return []

    contest_ids = sorted(problems_by_contest.keys())

    # Step 2: Discover editorial blog IDs
    editorial_map = _find_editorial_blog_ids(contest_ids)
    if not editorial_map:
        logger.warning("No editorial blog IDs found — aborting")
        return []

    # Step 3: Fetch editorials and build documents
    # Collect unique blog IDs and reverse-map to contests
    blog_to_contests: dict[int, list[int]] = defaultdict(list)
    for cid, bid in editorial_map.items():
        blog_to_contests[bid].append(cid)

    unique_blog_ids = sorted(blog_to_contests.keys())
    logger.info("Fetching %d unique editorials", len(unique_blog_ids))

    documents: list[dict] = []

    for blog_id in tqdm(unique_blog_ids, desc="Fetching editorials", unit="blog"):
        html = _fetch_editorial_html(blog_id)
        if not html:
            continue

        # Gather all problems from all contests that share this editorial
        contest_ids_for_blog = blog_to_contests[blog_id]
        all_problems: list[dict] = []
        for cid in contest_ids_for_blog:
            all_problems.extend(problems_by_contest.get(cid, []))

        if not all_problems:
            continue

        # Build set of expected problem indices
        problem_indices = {p["index"] for p in all_problems}

        # Try to split editorial into per-problem sections
        sections = _split_editorial_by_problem(html, problem_indices)

        if sections:
            # Matched per-problem sections
            for prob in all_problems:
                idx = prob["index"]
                section_html = sections.get(idx)
                if not section_html:
                    continue
                text = clean_html(section_html)
                if not text.strip():
                    continue
                documents.append(_build_document(prob, blog_id, text))
        else:
            # Fallback: emit the whole editorial as one document per contest
            text = clean_html(html)
            if not text.strip():
                continue
            for cid in contest_ids_for_blog:
                doc_id = f"{cid}_full"
                # Use the first problem from this contest for representative metadata
                contest_probs = problems_by_contest.get(cid, [])
                if not contest_probs:
                    continue
                rep = contest_probs[0]
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": "codeforces",
                        "doc_id": doc_id,
                        "contest_id": cid,
                        "problem_index": "full",
                        "problem_name": f"Contest {cid} (full editorial)",
                        "rating": rep["rating"],
                        "tags": rep["tags"],
                        "editorial_blog_id": blog_id,
                        "url": f"https://codeforces.com/blog/entry/{blog_id}",
                    },
                })

    logger.info(
        "=== Codeforces scraper complete: %d documents from %d editorials ===",
        len(documents), len(unique_blog_ids),
    )
    return documents


def _build_document(prob: dict, blog_id: int, text: str) -> dict:
    """Construct a single document dict from a problem and its cleaned text."""
    contest_id = prob["contestId"]
    index = prob["index"]
    return {
        "text": text,
        "metadata": {
            "source": "codeforces",
            "doc_id": f"{contest_id}{index}",
            "contest_id": contest_id,
            "problem_index": index,
            "problem_name": prob["name"],
            "rating": prob["rating"],
            "tags": prob["tags"],
            "editorial_blog_id": blog_id,
            "url": f"https://codeforces.com/contest/{contest_id}/problem/{index}",
        },
    }
