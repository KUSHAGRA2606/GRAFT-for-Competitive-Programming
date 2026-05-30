"""
Central configuration for Phase 1: Data Ingestion & Chunking.
All paths, URLs, and tunable parameters live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Repository URLs (theory sources)
# ──────────────────────────────────────────────
CPH_REPO_URL = "https://github.com/pllk/cphb.git"
USACO_REPO_URL = "https://github.com/cpinitiative/usaco-guide.git"
CPALGO_REPO_URL = "https://github.com/cp-algorithms/cp-algorithms.git"

# ──────────────────────────────────────────────
# Codeforces API
# ──────────────────────────────────────────────
CF_API_BASE = "https://codeforces.com/api"
CF_TAGS = ["dp", "graphs", "greedy"]
CF_RATING_MIN = 1300
CF_RATING_MAX = 1800
CF_REQUEST_DELAY = 2.5  # seconds between HTTP requests (rate-limit safe)

# ──────────────────────────────────────────────
# Chunking (LangChain RecursiveCharacterTextSplitter)
# ──────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CPH_DIR = RAW_DIR / "cph"
RAW_USACO_DIR = RAW_DIR / "usaco"
RAW_CPALGO_DIR = RAW_DIR / "cpalgo"
RAW_CF_DIR = RAW_DIR / "codeforces"

OUTPUT_FILE = PROJECT_ROOT / "data" / "algorithmic_corpus.json"

# ──────────────────────────────────────────────
# CPH chapter mapping (for metadata)
# ──────────────────────────────────────────────
CPH_CHAPTERS = {
    1: "Introduction",
    2: "Programming techniques",
    3: "Efficiency",
    4: "Data structures",
    5: "Complete search",
    6: "Greedy algorithms",
    7: "Dynamic programming",
    8: "Amortized analysis",
    9: "Range queries",
    10: "Bit manipulation",
    11: "Basics of graphs",
    12: "Shortest paths",
    13: "Tree algorithms",
    14: "Spanning trees",
    15: "Directed graphs",
    16: "Strong connectivity",
    17: "Tree queries",
    18: "Paths and circuits",
    19: "Flows and cuts",
    20: "Number theory",
    21: "Combinatorics",
    22: "Matrices",
    23: "Probability",
    24: "Game theory",
    25: "Suffix structures",
    26: "Geometry",
    27: "Sweep line algorithms",
}

# ──────────────────────────────────────────────
# USACO Guide divisions to scrape
# ──────────────────────────────────────────────
USACO_DIVISIONS = {
    "silver": "3_Silver",
    "gold": "4_Gold",
}

# ──────────────────────────────────────────────
# cp-algorithms topic directories
# ──────────────────────────────────────────────
CPALGO_TOPIC_DIRS = [
    "algebra",
    "combinatorics",
    "data_structures",
    "dynamic_programming",
    "geometry",
    "graph",
    "linear_algebra",
    "misc",
    "string",
    "num_methods",
    "schedules",
    "sequences",
]
