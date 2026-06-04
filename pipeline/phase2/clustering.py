"""Community detection script for Phase 2: GraphRAG.

Loads the NetworkX graph and applies the Leiden algorithm via graspologic
to detect hierarchical communities of algorithmic concepts.
"""
import json
import logging
import networkx as nx
from graspologic.partition import hierarchical_leiden

from phase2.config import COMMUNITIES_FILE, NETWORKX_GRAPH_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def detect_communities() -> None:
    if not NETWORKX_GRAPH_FILE.exists():
        logger.error(f"Graph file not found: {NETWORKX_GRAPH_FILE}")
        return

    logger.info(f"Loading graph from {NETWORKX_GRAPH_FILE}")
    G = nx.read_graphml(NETWORKX_GRAPH_FILE)

    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty. Cannot detect communities.")
        return

    logger.info("Running hierarchical Leiden algorithm...")
    
    # Graspologic Leiden algorithm requires a NetworkX graph.
    # It returns a list of Partition objects. We'll use the top level partitions
    # or all hierarchical partitions depending on the granularity needed.
    # For Phase 2, we will use hierarchical_leiden to get multi-level communities.
    
    partitions = hierarchical_leiden(G, random_seed=42)
    
    # Restructure into a clean JSON format
    # partitions is a list of hierarchical partitions. 
    # Each partition has a 'node' and a 'cluster' (community ID).
    
    communities = {}
    
    for partition in partitions:
        cluster_id = str(partition.cluster)
        node_name = str(partition.node)
        
        if cluster_id not in communities:
            communities[cluster_id] = {
                "id": cluster_id,
                "level": partition.level,
                "nodes": [],
                "edges": []
            }
        communities[cluster_id]["nodes"].append(node_name)

    # Now populate internal edges for each community to give context to the summarizer
    for cluster_id, comm_data in communities.items():
        subgraph = G.subgraph(comm_data["nodes"])
        
        for u, v, data in subgraph.edges(data=True):
            comm_data["edges"].append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "relationship_types": data.get("relationship_types", "")
            })

    # Filter out trivial communities (e.g., single orphaned nodes with no edges)
    filtered_communities = [
        c for c in communities.values()
        if len(c["nodes"]) > 1 or len(c["edges"]) > 0
    ]

    logger.info(f"Detected {len(filtered_communities)} significant communities.")

    with open(COMMUNITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_communities, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved communities to {COMMUNITIES_FILE}")


if __name__ == "__main__":
    detect_communities()
