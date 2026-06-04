"""Graph construction script for Phase 2: GraphRAG.

Reads extracted entities and relationships and builds an undirected, weighted
NetworkX graph. Edge weights are based on co-occurrence frequencies.
"""
import json
import logging
import networkx as nx

from phase2.config import EXTRACTIONS_FILE, NETWORKX_GRAPH_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_graph() -> None:
    if not EXTRACTIONS_FILE.exists():
        logger.error(f"Extractions file not found: {EXTRACTIONS_FILE}")
        return

    logger.info(f"Loading extractions from {EXTRACTIONS_FILE}")
    with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
        extractions = json.load(f)

    G = nx.Graph()
    edge_counts = {}
    node_sources = {}  # Keep track of which chunks mentioned which nodes

    for chunk_data in extractions:
        chunk_id = chunk_data.get("chunk_id", "unknown")
        rels = chunk_data.get("relationships", [])

        for r in rels:
            # Basic validation
            if not isinstance(r, dict) or "source" not in r or "target" not in r:
                continue

            # Normalize names slightly
            source = str(r["source"]).strip().title()
            target = str(r["target"]).strip().title()
            rel_type = str(r.get("relationship", "")).strip().lower()

            if not source or not target or source == target:
                continue

            # Ensure nodes exist
            if not G.has_node(source):
                G.add_node(source)
                node_sources[source] = set()
            if not G.has_node(target):
                G.add_node(target)
                node_sources[target] = set()

            node_sources[source].add(chunk_id)
            node_sources[target].add(chunk_id)

            # We use an undirected graph. Sort to make edge key canonical
            edge_key = tuple(sorted([source, target]))
            
            if edge_key not in edge_counts:
                edge_counts[edge_key] = {"weight": 0, "types": set()}
            
            edge_counts[edge_key]["weight"] += 1
            if rel_type:
                edge_counts[edge_key]["types"].add(rel_type)

    # Add edges to the graph
    for (u, v), data in edge_counts.items():
        G.add_edge(
            u, v, 
            weight=data["weight"], 
            relationship_types="; ".join(data["types"])
        )

    logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save to GraphML
    nx.write_graphml(G, NETWORKX_GRAPH_FILE)
    logger.info(f"Saved graph to {NETWORKX_GRAPH_FILE}")


if __name__ == "__main__":
    build_graph()
