"""Extraction script for Phase 2: GraphRAG using Fireworks AI (via OpenAI SDK).

Processes the algorithmic chunks through DeepSeek-V4-Flash to extract
entities and relationships, saving checkpoints to prevent data loss.
"""
import asyncio
import json
import logging
from pathlib import Path

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from phase2.config import (
    CHECKPOINT_FILE,
    CONCURRENCY_LIMIT,
    CORPUS_FILE,
    EXTRACTIONS_FILE,
    EXTRACTION_SYSTEM_PROMPT,
    FIREWORKS_MODEL,
    FIREWORKS_BASE_URL,
    FIREWORKS_API_KEY,
    MAX_REQUESTS_PER_MINUTE,
    MAX_TOKENS_PER_MINUTE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize rate limiters
req_limiter = AsyncLimiter(MAX_REQUESTS_PER_MINUTE, 60)
token_limiter = AsyncLimiter(MAX_TOKENS_PER_MINUTE, 60)
concurrency_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


async def _extract_chunk(
    client: AsyncOpenAI, chunk: dict, max_retries: int = 5
) -> dict | None:
    """Extract relationships from a single chunk using Fireworks AI."""
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]
    
    prompt_len = len(EXTRACTION_SYSTEM_PROMPT) + len(text)
    estimated_tokens = (prompt_len // 4) + 200

    for attempt in range(1, max_retries + 1):
        try:
            async with concurrency_semaphore:
                await token_limiter.acquire(estimated_tokens)
                async with req_limiter:
                    response = await client.chat.completions.create(
                        model=FIREWORKS_MODEL,
                        messages=[
                            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Chunk Text:\n{text}"}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                return None
            
            parsed = json.loads(raw_content)
            
            if "relationships" in parsed and isinstance(parsed["relationships"], list):
                return {
                    "chunk_id": chunk_id,
                    "metadata": chunk.get("metadata", {}),
                    "relationships": parsed["relationships"]
                }
            else:
                logger.warning(f"Invalid schema for chunk {chunk_id}: {raw_content}")
                return None
                
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error for chunk {chunk_id}. Attempt {attempt}/{max_retries}")
        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"API Error for chunk {chunk_id}: {str(e)}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
    logger.error(f"Failed to process chunk {chunk_id} after {max_retries} attempts.")
    return None


async def run_extraction(sample_size: int | None = None) -> None:
    """Run the asynchronous extraction pipeline."""
    logger.info("Initializing Fireworks client...")
    client = AsyncOpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
    )

    logger.info(f"Loading corpus from {CORPUS_FILE}")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    chunks = corpus["chunks"]
    if sample_size:
        chunks = chunks[:sample_size]
        logger.info(f"Running TEST mode with {sample_size} chunks.")
    else:
        logger.info(f"Running FULL extraction on {len(chunks)} chunks.")

    extracted_data = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            try:
                extracted_data = json.load(f)
                logger.info(f"Loaded checkpoint with {len(extracted_data)} completed chunks.")
            except json.JSONDecodeError:
                logger.warning("Checkpoint file corrupted. Starting fresh.")

    chunks_to_process = [c for c in chunks if c["chunk_id"] not in extracted_data]
    logger.info(f"Chunks remaining to process: {len(chunks_to_process)}")

    if not chunks_to_process:
        logger.info("All chunks already processed.")
        return

    progress = tqdm(total=len(chunks_to_process), desc="Extracting Graph Data")

    async def process_and_save(chunk: dict):
        result = await _extract_chunk(client, chunk)
        if result is not None:
            extracted_data[chunk["chunk_id"]] = result
        
        progress.update(1)
        
        if progress.n % 1 == 0:
            logger.info(f"Live Progress: {progress.n} chunks processed.")
            _save_checkpoint(extracted_data)

    tasks = [process_and_save(c) for c in chunks_to_process]
    await asyncio.gather(*tasks)
    
    progress.close()

    _save_checkpoint(extracted_data)
    
    final_output = list(extracted_data.values())
    with open(EXTRACTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Extraction complete! Saved {len(final_output)} records to {EXTRACTIONS_FILE}")


def _save_checkpoint(data: dict) -> None:
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Retry loop for Windows PermissionError (if file is temporarily locked by antivirus or read scripts)
    import time
    for _ in range(5):
        try:
            temp_file.replace(CHECKPOINT_FILE)
            break
        except PermissionError:
            time.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(run_extraction(sample_size=10))
