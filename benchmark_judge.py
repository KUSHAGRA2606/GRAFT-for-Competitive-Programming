import os
import json
import random
import torch
import gc
import numpy as np
import faiss
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import graft_agent

# ==============================================================================
# Configuration
# ==============================================================================
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

NUM_QUESTIONS = 50
CORPUS_PATH = "data/algorithmic_corpus.json"
QUESTIONS_PATH = "data/raft/test_questions.json"
BASELINE_PATH = "data/benchmark_baseline_answers.json"
GRAFT_PATH = "data/benchmark_graft_answers.json"
RESULTS_PATH = "data/benchmark_results.json"
PLOT_PATH = "graft_benchmark_results.png"

BASE_MODEL_ID = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"

# ==============================================================================
# Helpers
# ==============================================================================
def call_judge(prompt, json_mode=False):
    client = OpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url="https://api.fireworks.ai/inference/v1"
    )
    
    kwargs = {
        "model": "accounts/fireworks/models/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are an impartial AI judge. Return strict JSON if requested."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "{}"

# ==============================================================================
# Step 1: Test Generation
# ==============================================================================
def step1_generate_questions():
    if os.path.exists(QUESTIONS_PATH):
        with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
            q = json.load(f)
            print(f"[Step 1] Loaded {len(q)} existing questions from {QUESTIONS_PATH}.")
            return q
    else:
        print(f"ERROR: {QUESTIONS_PATH} not found. Please provide the questions.")
        exit(1)

# ==============================================================================
# Step 2: Head-to-Head Inference
# ==============================================================================
def step2_baseline_inference(questions):
    print("\n[Step 2A] Running Baseline Vector RAG (Answer A)...")
    
    print("  Building naive FAISS index on raw chunks...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
        corpus = corpus_data.get("chunks", corpus_data) if isinstance(corpus_data, dict) else corpus_data
    
    subset = corpus[:1000]
    raw_texts = [c.get("text", c.get("content", str(c))) for c in subset]
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(raw_texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    
    print("  Loading Base Qwen2.5-3B-Instruct Model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto", 
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    baseline_answers = []
    for q in tqdm(questions, desc="Baseline Generation"):
        q_emb = embed_model.encode([q])
        dist, idxs = index.search(np.array(q_emb, dtype=np.float32), 3)
        context = "\n---\n".join([raw_texts[i][:500] for i in idxs[0]])
        
        messages = [
            {"role": "system", "content": "Answer the user's question using the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=150)
        ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        baseline_answers.append(ans)
        
    # Free VRAM to prevent OOM before loading GRAFT
    print("  Freeing GPU VRAM...")
    del model
    del tokenizer
    del index
    del embed_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return baseline_answers

def step2_graft_inference(questions):
    print("\n[Step 2B] Running GRAFT Engine (Answer B)...")
    graph = graft_agent.init_system()
    
    graft_answers = []
    for q in tqdm(questions, desc="GRAFT Generation"):
        initial_state = {"query": q, "community_summaries": [], "community_answers": [], "global_answer": ""}
        try:
            result = graph.invoke(initial_state)
            ans = result.get("global_answer", "Error")
        except Exception as e:
            ans = f"Error: {str(e)}"
        graft_answers.append(ans)
        
    return graft_answers

# ==============================================================================
# Step 3: LLM-as-a-Judge Evaluation
# ==============================================================================
def step3_evaluate(questions, baseline_answers, graft_answers):
    print("\n[Step 3] Running LLM-as-a-Judge Evaluation...")
    results = []
    
    for i in tqdm(range(len(questions)), desc="Judging"):
        q = questions[i]
        ans_a = baseline_answers[i]
        ans_b = graft_answers[i]
        
        prompt = f"""You are an impartial judge evaluating two AI systems on their ability to answer complex programming questions based on retrieved context.

Question: {q}

[System A]
{ans_a}

[System B]
{ans_b}

Evaluate which system is better across 4 metrics:
1. Comprehensiveness (Which answer provides more detail?)
2. Diversity (Which answer covers more perspectives/concepts?)
3. Empowerment (Which answer is more helpful for a developer?)
4. Directness (Which answer gets to the point faster?)

Return ONLY a strict JSON object with this exact structure:
{{
  "comprehensiveness": {{"winner": "A" or "B" or "Tie", "reason": "short reason"}},
  "diversity": {{"winner": "A" or "B" or "Tie", "reason": "short reason"}},
  "empowerment": {{"winner": "A" or "B" or "Tie", "reason": "short reason"}},
  "directness": {{"winner": "A" or "B" or "Tie", "reason": "short reason"}}
}}
"""
        res = call_judge(prompt, json_mode=True)
        try:
            data = json.loads(res)
            results.append({"question": q, "System A": ans_a, "System B": ans_b, "evaluation": data})
        except:
            pass # Skip failed evals
            
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results

# ==============================================================================
# Step 4: Statistical Visualization
# ==============================================================================
def step4_visualize(results):
    print("\n[Step 4] Visualizing Results...")
    metrics = ["comprehensiveness", "diversity", "empowerment", "directness"]
    graft_wins = {m: 0 for m in metrics}
    baseline_wins = {m: 0 for m in metrics}
    ties = {m: 0 for m in metrics}
    
    for r in results:
        evals = r.get("evaluation", {})
        for m in metrics:
            if m in evals:
                winner = evals[m].get("winner", "")
                if winner == "B": graft_wins[m] += 1
                elif winner == "A": baseline_wins[m] += 1
                else: ties[m] += 1
                
    total_graft = sum(graft_wins.values())
    total_baseline = sum(baseline_wins.values())
    total_ties = sum(ties.values())
    total_matches = total_graft + total_baseline + total_ties
    
    overall_win_rate = (total_graft / total_matches) * 100 if total_matches > 0 else 0
    print(f"\n========================================")
    print(f" FINAL CV METRIC SCORE: GRAFT Win Rate = {overall_win_rate:.1f}%")
    print(f"========================================\n")
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, [graft_wins[m] for m in metrics], width, label='GRAFT (System B)', color='#2ca02c')
    ax.bar(x, [ties[m] for m in metrics], width, label='Tie', color='#7f7f7f')
    ax.bar(x + width, [baseline_wins[m] for m in metrics], width, label='Baseline (System A)', color='#1f77b4')
    
    ax.set_ylabel('Number of Wins')
    ax.set_title(f'Head-to-Head Win Rate: GRAFT vs Baseline RAG\n(Overall Win Rate: {overall_win_rate:.1f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Saved benchmark plot to {PLOT_PATH}")

if __name__ == "__main__":
    if not FIREWORKS_API_KEY:
        print("ERROR: FIREWORKS_API_KEY not found in environment.")
        exit(1)
        
    questions = step1_generate_questions()
    
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r") as f:
            ans_a = json.load(f)
    else:
        ans_a = step2_baseline_inference(questions)
        with open(BASELINE_PATH, "w") as f:
            json.dump(ans_a, f)
            
    if os.path.exists(GRAFT_PATH):
        with open(GRAFT_PATH, "r") as f:
            ans_b = json.load(f)
    else:
        ans_b = step2_graft_inference(questions)
        with open(GRAFT_PATH, "w") as f:
            json.dump(ans_b, f)
            
    results = step3_evaluate(questions, ans_a, ans_b)
    step4_visualize(results)
