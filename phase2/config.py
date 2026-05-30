"""Configuration constants and prompts for Phase 2: GraphRAG Construction."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# API & Model Configuration
# ==============================================================================
# Fireworks AI Configuration
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-flash"

# Gemini Configuration (Used for Summarization)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Rate Limiting (Gemini Free Tier: 15 RPM, 1M TPM)
MAX_REQUESTS_PER_MINUTE = 14
MAX_TOKENS_PER_MINUTE = 900000 
CONCURRENCY_LIMIT = 2

# ==============================================================================
# File Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Inputs
CORPUS_FILE = DATA_DIR / "algorithmic_corpus.json"

# Outputs
GRAPH_DIR = DATA_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTIONS_FILE = GRAPH_DIR / "extractions.json"
CHECKPOINT_FILE = GRAPH_DIR / "extractions_checkpoint.json"
NETWORKX_GRAPH_FILE = GRAPH_DIR / "algorithmic_graph.graphml"
COMMUNITIES_FILE = GRAPH_DIR / "communities.json"
SUMMARIES_FILE = GRAPH_DIR / "community_summaries.json"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# ==============================================================================
# Prompts
# ==============================================================================

# Few-shot prompt for extracting entities and relationships, strictly enforcing JSON.
EXTRACTION_SYSTEM_PROMPT = """You are an expert algorithm data extractor specializing in competitive programming.
Extract the core entities and relationships from the text. 
Return ONLY a valid JSON object matching this exact schema:
{
  "relationships": [
    {"source": "Entity1", "target": "Entity2", "relationship": "optimizes"}
  ]
}

Guidelines:
- Entities should be clear, normalized concepts (e.g., "Segment Tree", "Time Complexity", "Dynamic Programming").
- Relationships should be concise verbs/actions.
- Do not include markdown codeblocks (```json). Just output the raw JSON object.
- If no meaningful algorithmic relationships are found, return {"relationships": []}.

Example:
> Text: 'A Segment Tree allows for efficient range queries, updating in O(log N) time.'
> Output: {
  "relationships": [
    {"source": "Segment Tree", "target": "Range Queries", "relationship": "computes efficiently"}, 
    {"source": "Segment Tree", "target": "O(log N)", "relationship": "has time complexity"}
  ]
}"""

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert competitive programming AI.
You will be given a list of interconnected algorithmic concepts that form a 'community' within a knowledge graph.
Your task is to write a concise, 2-paragraph summary explaining what this cluster of concepts represents, how they relate to each other, and their typical application in competitive programming.

Format guidelines:
- Exactly two paragraphs.
- Keep it highly technical but clear.
- Do NOT output any markdown headers, just the text.
"""
