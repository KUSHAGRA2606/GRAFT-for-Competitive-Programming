import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# API Configuration (Fireworks AI for Dataset Generation)
# ==============================================================================
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-flash"

# Rate Limiting (Fireworks Free/Trial limits or whatever is safe)
MAX_REQUESTS_PER_MINUTE = 60
MAX_TOKENS_PER_MINUTE = 200000 
CONCURRENCY_LIMIT = 5

# ==============================================================================
# File Paths
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
GRAPH_DIR = DATA_DIR / "graph"
PHASE3_DIR = DATA_DIR / "raft"

# Input Files
# We use the extractions from Phase 1/2 as the oracle chunks
ORACLE_CHUNKS_FILE = GRAPH_DIR / "extractions_checkpoint.json"

# Output Files
PHASE3_DIR.mkdir(exist_ok=True, parents=True)
RAFT_DATA_FILE = PHASE3_DIR / "raft_training_data.jsonl"

# ==============================================================================
# Prompts
# ==============================================================================
RAFT_SYSTEM_PROMPT = """You are an expert AI dataset generator for Retrieval-Augmented Generation (RAG).
Your goal is to generate a highly complex, multi-hop question that requires synthesizing information from the provided text, and then provide a Chain-of-Thought (CoT) reasoning path that strictly uses the text to answer the question.

You will be provided with a single ORACLE context chunk.

Return a JSON object with EXACTLY these two keys:
{
    "question": "<A complex question that can only be answered using the oracle context>",
    "answer_cot": "<A detailed step-by-step reasoning path that answers the question, explicitly citing information from the context. Start by establishing facts from the context, then draw a conclusion.>"
}
"""
