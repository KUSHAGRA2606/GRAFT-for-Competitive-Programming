import json
import asyncio
import logging
import random
from typing import List, Dict, Any
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

from phase3.config import (
    FIREWORKS_API_KEY,
    FIREWORKS_BASE_URL,
    FIREWORKS_MODEL,
    MAX_REQUESTS_PER_MINUTE,
    CONCURRENCY_LIMIT,
    ORACLE_CHUNKS_FILE,
    RAFT_DATA_FILE,
    RAFT_SYSTEM_PROMPT,
    DATA_DIR
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for Fireworks
client = AsyncOpenAI(
    api_key=FIREWORKS_API_KEY,
    base_url=FIREWORKS_BASE_URL
)

# Limiters
rate_limiter = AsyncLimiter(MAX_REQUESTS_PER_MINUTE, 60.0)
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


def load_valid_chunks() -> Dict[str, str]:
    """Loads the original chunk texts, but ONLY for the chunks that successfully made it into the graph."""
    logger.info("Loading corpus...")
    corpus_file = DATA_DIR / "algorithmic_corpus.json"
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    all_chunks = {c["chunk_id"]: c["text"] for c in corpus["chunks"]}
    
    logger.info("Loading valid extracted chunk IDs...")
    with open(ORACLE_CHUNKS_FILE, "r", encoding="utf-8") as f:
        extractions = json.load(f)
        
    valid_ids = set(extractions.keys())
    
    # Filter
    valid_chunks = {k: v for k, v in all_chunks.items() if k in valid_ids}
    logger.info(f"Loaded {len(valid_chunks)} valid chunks out of {len(all_chunks)} total chunks.")
    return valid_chunks


async def generate_raft_data_for_chunk(
    chunk_id: str, 
    oracle_text: str, 
    all_chunk_texts: List[str]
) -> Dict[str, Any]:
    """Uses LLM to generate a question and CoT answer, then formats the distractor context."""
    
    # 1. Select 3 random distractors
    distractors = []
    while len(distractors) < 3:
        cand = random.choice(all_chunk_texts)
        if cand != oracle_text and cand not in distractors:
            distractors.append(cand)
            
    # 2. Call LLM to generate Question and CoT Answer based ONLY on Oracle
    messages = [
        {"role": "system", "content": RAFT_SYSTEM_PROMPT},
        {"role": "user", "content": f"<oracle_context>\n{oracle_text}\n</oracle_context>"}
    ]
    
    max_retries = 50
    for attempt in range(max_retries):
        try:
            async with rate_limiter:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=FIREWORKS_MODEL,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=1000
                    )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # 3. Format the final context string (shuffle oracle with distractors)
            documents = distractors + [oracle_text]
            random.shuffle(documents)
            
            context_string = ""
            for i, doc in enumerate(documents):
                context_string += f"<document id=\"{i+1}\">\n{doc}\n</document>\n\n"
            
            return {
                "chunk_id": chunk_id,
                "question": result.get("question", ""),
                "context": context_string.strip(),
                "answer": result.get("answer_cot", "")
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to generate for {chunk_id} after 50 attempts: {e}")
                return None
            
            # Use a capped exponential backoff so it doesn't sleep for years
            sleep_time = min(60, 2 ** attempt)
            await asyncio.sleep(sleep_time)


async def main():
    chunks = load_valid_chunks()
    all_chunk_texts = list(chunks.values())
    
    # Load completed chunks to resume if interrupted
    completed_ids = set()
    if RAFT_DATA_FILE.exists():
        with open(RAFT_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_ids.add(data["chunk_id"])
                except json.JSONDecodeError:
                    continue
                    
    logger.info(f"Found {len(completed_ids)} already processed chunks.")
    
    tasks = []
    chunk_items = list(chunks.items())
    
    # Filter out already completed items
    incomplete_items = [(k, v) for k, v in chunk_items if k not in completed_ids]
    
    # Calculate how many more we need to reach 150 total chunks
    needed = max(0, 150 - len(completed_ids))
    
    if needed == 0:
        logger.info(f"Already reached {len(completed_ids)} chunks (target is 150). Done!")
        return
        
    logger.info(f"Need to generate {needed} more chunks to reach 150 total.")
    random.seed(42)
    random.shuffle(incomplete_items)
    chunk_items_to_process = incomplete_items[:needed]
    
    for chunk_id, text in chunk_items_to_process:
        tasks.append(generate_raft_data_for_chunk(chunk_id, text, all_chunk_texts))
        
    if not tasks:
        logger.info("All chunks processed!")
        return
        
    logger.info(f"Starting generation for {len(tasks)} chunks...")
    
    # Process and append to JSONL file
    with open(RAFT_DATA_FILE, "a", encoding="utf-8") as f:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating RAFT Data"):
            result = await coro
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    logger.info(f"RAFT Data Generation Complete! Saved to {RAFT_DATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
