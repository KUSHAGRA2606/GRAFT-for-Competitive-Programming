import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

QUESTIONS_PATH = "data/raft/test_questions.json"
BASELINE_PATH = "data/benchmark_baseline_answers.json"
GRAFT_PATH = "data/benchmark_graft_answers.json"
CORPUS_PATH = "data/algorithmic_corpus.json"
FAISS_INDEX_PATH = "data/faiss_index/community_index.faiss"
FAISS_MAPPING_PATH = "data/faiss_index/community_mapping.json"

def calculate_answer_relevance():
    print("\n[1] Calculating Answer Relevance (Semantic Embedding Similarity)...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        ans_a = json.load(f)
    with open(GRAFT_PATH, "r", encoding="utf-8") as f:
        ans_b = json.load(f)
        
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    q_embs = embed_model.encode(questions)
    a_embs = embed_model.encode(ans_a)
    b_embs = embed_model.encode(ans_b)
    
    # Calculate cosine similarities
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    baseline_scores = [cos_sim(q_embs[i], a_embs[i]) for i in range(len(questions))]
    graft_scores = [cos_sim(q_embs[i], b_embs[i]) for i in range(len(questions))]
    
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    graft_avg = sum(graft_scores) / len(graft_scores)
    
    print(f"  -> Baseline Answer Relevance: {baseline_avg*100:.1f}%")
    print(f"  -> GRAFT Answer Relevance: {graft_avg*100:.1f}%")
    return baseline_avg, graft_avg

def _grade_context_relevance(question, context):
    client = OpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url="https://api.fireworks.ai/inference/v1"
    )
    
    prompt = f"""Evaluate if the following context contains ANY relevant information to help answer the user's question.
Question: {question}
Context: {context}

Return ONLY a strict JSON object: {{"relevant": true}} or {{"relevant": false}}"""
    
    try:
        response = client.chat.completions.create(
            model="accounts/fireworks/models/deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("relevant", False)
    except Exception as e:
        return False

def calculate_context_precision():
    print("\n[2] Calculating Context Precision...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    # Baseline Retrieval
    print("  -> Loading Baseline FAISS Index...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
        corpus = corpus_data.get("chunks", corpus_data) if isinstance(corpus_data, dict) else corpus_data
    subset = corpus[:1000]
    raw_texts = [c.get("text", c.get("content", str(c))) for c in subset]
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(raw_texts, show_progress_bar=False)
    base_index = faiss.IndexFlatL2(embeddings.shape[1])
    base_index.add(np.array(embeddings, dtype=np.float32))
    
    # GRAFT Retrieval
    print("  -> Loading GRAFT FAISS Index & CrossEncoder...")
    graft_index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
        graft_mapping = json.load(f)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    baseline_relevant = 0
    baseline_total = 0
    graft_relevant = 0
    graft_total = 0
    
    for q in tqdm(questions, desc="Grading Retrieved Contexts"):
        # Baseline context
        q_emb = embed_model.encode([q])
        _, idxs = base_index.search(np.array(q_emb, dtype=np.float32), 3)
        for i in idxs[0]:
            chunk = raw_texts[i][:500]
            if _grade_context_relevance(q, chunk): baseline_relevant += 1
            baseline_total += 1
            
        # GRAFT context
        _, g_idxs = graft_index.search(np.array(q_emb, dtype=np.float32), 5)
        retrieved_summaries = [graft_mapping[str(i)] for i in g_idxs[0] if str(i) in graft_mapping]
        
        # Rerank
        pairs = [[q, summary["summary"]] for summary in retrieved_summaries]
        scores = reranker.predict(pairs)
        ranked = [s for _, s in sorted(zip(scores, retrieved_summaries), reverse=True)]
        
        for summary in ranked[:3]:
            if _grade_context_relevance(q, summary["summary"]): graft_relevant += 1
            graft_total += 1
            
    base_precision = baseline_relevant / baseline_total
    graft_precision = graft_relevant / graft_total
    
    print(f"  -> Baseline Context Precision: {base_precision*100:.1f}%")
    print(f"  -> GRAFT Context Precision: {graft_precision*100:.1f}%")
    return base_precision, graft_precision

if __name__ == "__main__":
    a_b, a_g = calculate_answer_relevance()
    c_b, c_g = calculate_context_precision()
    print("\n=== FINAL METRICS ===")
    print(f"Answer Relevance (Baseline): {a_b*100:.1f}%")
    print(f"Answer Relevance (GRAFT):    {a_g*100:.1f}%")
    print(f"Context Precision (Baseline):{c_b*100:.1f}%")
    print(f"Context Precision (GRAFT):   {c_g*100:.1f}%")
