"""
GRAFT Agent - GraphRAG Query-Time Orchestration using LangGraph
================================================================
A Map-Reduce pipeline that routes user queries through community summaries
using a fine-tuned Qwen2.5-3B model with LoRA adapters.

Hardware: Designed for i5 laptop without GPU (CPU-only, 4-bit quantized).
"""

import json
import warnings
import logging
from typing import TypedDict, List
from pathlib import Path

import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Paths
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index" / "community_index.faiss"
FAISS_MAPPING_PATH = DATA_DIR / "faiss_index" / "community_mapping.json"
ADAPTER_PATH = ROOT_DIR / "graft_qwen-adapters"

# Base model must match what the adapters were trained on
BASE_MODEL_ID = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"

# Ultra-strict limits for 4GB VRAM
TOP_COMMUNITIES = 5  # Fetch top 5 from FAISS
TOP_SENTENCES = 7    # Extract top 7 sentences using Reranker

# ==============================================================================
# Global State (For Streamlit/API caching)
# ==============================================================================
model = None
tokenizer = None
faiss_index = None
community_mapping = None
embed_model = None
reranker = None

# ==============================================================================
# 1. Model Initialization
# ==============================================================================
def load_model_and_tokenizer():
    """Load base Qwen model in 4-bit and attach LoRA adapters."""
    print("\n" + "=" * 60)
    print("  GRAFT Agent - Loading Fine-Tuned Model")
    print("=" * 60)

    print("[1/4] Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("[2/4] Loading base model (Qwen2.5-3B-Instruct) in 4-bit on GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa", # 🚀 HARDWARE ACCELERATION: ~20% faster inference
        trust_remote_code=True,
    )

    print("[3/4] Attaching LoRA adapters from ./graft_qwen-adapters...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()

    print("[4/4] Loading tokenizer with ChatML template...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!\n")
    return model, tokenizer


# ==============================================================================
# 2. Retrieval Engine (FAISS + Sentence Transformers)
# ==============================================================================
def load_retrieval_engine():
    """Load FAISS index and community mapping for semantic search."""
    print("[Retrieval] Loading FAISS index and community summaries...")

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_raw = json.load(f)

    # The mapping file is a dict with string keys ("0", "1", ...).
    # Convert to a list ordered by key so FAISS integer indices map correctly.
    mapping = [mapping_raw[str(i)] for i in range(len(mapping_raw))]

    # Load embedding model (same one used during indexing)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    print(f"[Retrieval] Loaded {index.ntotal} community vectors.\n")
    return index, mapping, embed_model, reranker


# ==============================================================================
# 3. Inference Helper
# ==============================================================================
def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 512) -> str:
    """Run inference on the fine-tuned model using ChatML formatting."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the NEW tokens (skip the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ==============================================================================
# 4. LangGraph State Machine
# ==============================================================================
class GraphState(TypedDict):
    query: str
    community_summaries: List[str]
    community_answers: List[str]
    global_answer: str


def receive_node(state: GraphState) -> GraphState:
    """Node 1: Accepts the user query."""
    print(f"\n  [Receive] Query: \"{state['query']}\"")
    return state


def retrieve_node(state: GraphState) -> GraphState:
    """Node 2: Retrieves communities, extracts sentences, and reranks them."""
    query = state["query"]
    print(f"  [Retrieve] Searching FAISS for Top {TOP_COMMUNITIES} communities...")

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = faiss_index.search(query_embedding, TOP_COMMUNITIES)

    # 1. Gather all sentences from the top communities
    all_sentences = []
    for idx in indices[0]:
        if idx < len(community_mapping):
            entry = community_mapping[idx]
            summary_text = entry.get("summary", str(entry))
            # Rough sentence splitting
            sentences = [s.strip() + "." for s in summary_text.replace("?", ".").replace("!", ".").split(".") if len(s.strip()) > 10]
            all_sentences.extend(sentences)

    print(f"  [Rerank] Scoring {len(all_sentences)} sentences against the query using CrossEncoder...")
    
    # 2. Score every sentence against the query
    pairs = [[query, sent] for sent in all_sentences]
    scores = reranker.predict(pairs)

    # 3. Sort by score and take the top N sentences
    ranked = sorted(zip(scores, all_sentences), key=lambda x: x[0], reverse=True)
    top_sentences = [sent for score, sent in ranked[:TOP_SENTENCES]]

    print(f"  [Rerank] Extracted the {TOP_SENTENCES} most relevant sentences to form the dense context.")
    state["community_summaries"] = top_sentences
    return state


def answer_node(state: GraphState) -> GraphState:
    """Node 3: Direct Answer Generation (Bypassing Map for 2x Speed)"""
    print(f"  [Generating] Writing final answer directly from context...")

    # We feed the highly-dense extracted sentences directly!
    combined_context = " ".join(state["community_summaries"])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert algorithmic assistant. "
                "Answer the user's question using ONLY the provided context. "
                "Be concise and get straight to the point."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{combined_context}\n\n"
                f"Question: {state['query']}\n\n"
                f"Provide a clear, fast answer."
            ),
        },
    ]

    final_answer = generate_response(model, tokenizer, messages, max_new_tokens=150)
    state["global_answer"] = final_answer
    return state


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("receive", receive_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)

    # Define edges (linear pipeline)
    workflow.set_entry_point("receive")
    workflow.add_edge("receive", "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


def init_system():
    """Initializes globals for Streamlit caching and returns the compiled LangGraph."""
    global model, tokenizer, faiss_index, community_mapping, embed_model, reranker
    
    if model is None:
        model, tokenizer = load_model_and_tokenizer()
        faiss_index, community_mapping, embed_model, reranker = load_retrieval_engine()
    
    return build_graph()


# ==============================================================================
# 5. Main CLI Loop
# ==============================================================================
if __name__ == "__main__":
    # Initialize all components
    graph = init_system()

    print("=" * 60)
    print("  GRAFT Agent Ready!")
    print("  Type your algorithmic question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Run the LangGraph pipeline
        print("\n" + "-" * 50)
        print("  GRAFT Pipeline Executing...")
        print("-" * 50)

        initial_state: GraphState = {
            "query": user_input,
            "community_summaries": [],
            "community_answers": [],
            "global_answer": "",
        }

        result = graph.invoke(initial_state)

        print("\n" + "=" * 50)
        print("  GRAFT Answer:")
        print("=" * 50)
        print(result["global_answer"])
        print("=" * 50)
