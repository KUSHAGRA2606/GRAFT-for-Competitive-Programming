"""Community summarization script for Phase 2: GraphRAG using Fireworks AI.

Passes each detected community back to DeepSeek-V4-Flash to generate
a conceptual summary of the algorithmic cluster.
"""
import asyncio
import json
import logging
from pathlib import Path

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from phase2.config import (
    COMMUNITIES_FILE,
    CONCURRENCY_LIMIT,
    FIREWORKS_MODEL,
    FIREWORKS_BASE_URL,
    FIREWORKS_API_KEY,
    MAX_REQUESTS_PER_MINUTE,
    MAX_TOKENS_PER_MINUTE,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIES_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

req_limiter = AsyncLimiter(MAX_REQUESTS_PER_MINUTE, 60)
token_limiter = AsyncLimiter(MAX_TOKENS_PER_MINUTE, 60)
concurrency_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


def _build_community_context(community: dict) -> str:
    nodes = ", ".join(community["nodes"])
    edges = []
    sorted_edges = sorted(community["edges"], key=lambda x: x["weight"], reverse=True)
    for edge in sorted_edges[:50]:
        edges.append(f"{edge['source']} -> {edge['target']} ({edge['relationship_types']}) [weight: {edge['weight']}]")
    
    edge_str = "\n".join(edges)
    return f"Community Level: {community.get('level', 'N/A')}\nNodes: {nodes}\n\nTop Relationships:\n{edge_str}"


async def _summarize_community(
    client: AsyncOpenAI, community: dict, max_retries: int = 15
) -> dict | None:
    comm_id = community["id"]
    context = _build_community_context(community)
    
    prompt_len = len(SUMMARIZATION_SYSTEM_PROMPT) + len(context)
    estimated_tokens = (prompt_len // 4) + 400

    for attempt in range(1, max_retries + 1):
        try:
            async with concurrency_semaphore:
                await token_limiter.acquire(estimated_tokens)
                async with req_limiter:
                    response = await client.chat.completions.create(
                        model=FIREWORKS_MODEL,
                        messages=[
                            {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Community Context:\n{context}"}
                        ],
                        temperature=0.3,
                    )
            
            summary_text = response.choices[0].message.content
            if not summary_text:
                return None
            
            return {
                "community_id": comm_id,
                "level": community.get("level"),
                "nodes": community["nodes"],
                "summary": summary_text.strip()
            }
                
        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"API Error for community {comm_id}: {str(e)}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
    logger.error(f"Failed to summarize community {comm_id} after {max_retries} attempts.")
    return None


async def run_summarization() -> None:
    if not COMMUNITIES_FILE.exists():
        logger.error(f"Communities file not found: {COMMUNITIES_FILE}")
        return

    logger.info("Initializing Fireworks client...")
    client = AsyncOpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
    )

    logger.info(f"Loading communities from {COMMUNITIES_FILE}")
    with open(COMMUNITIES_FILE, "r", encoding="utf-8") as f:
        all_communities = json.load(f)

    existing_summaries = []
    if SUMMARIES_FILE.exists():
        try:
            with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
                existing_summaries = json.load(f)
        except Exception:
            existing_summaries = []
    
    completed_ids = {s["community_id"] for s in existing_summaries}
    communities_to_process = [c for c in all_communities if c["id"] not in completed_ids]

    logger.info(f"Found {len(existing_summaries)} already completed communities. Generating summaries for the remaining {len(communities_to_process)}.")

    progress = tqdm(total=len(communities_to_process), desc="Summarizing Communities")
    new_summaries = []

    async def process_community(comm: dict):
        result = await _summarize_community(client, comm)
        if result is not None:
            new_summaries.append(result)
        progress.update(1)

    tasks = [process_community(c) for c in communities_to_process]
    await asyncio.gather(*tasks)
    
    progress.close()

    all_summaries = existing_summaries + new_summaries
    with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summarization complete! Saved {len(all_summaries)} total summaries to {SUMMARIES_FILE}")


if __name__ == "__main__":
    asyncio.run(run_summarization())
