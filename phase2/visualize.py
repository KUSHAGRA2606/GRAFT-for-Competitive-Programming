"""Visualization script using PyVis for Phase 2: GraphRAG.

Loads the NetworkX graph and generates an interactive HTML file.
"""
import logging
import networkx as nx
from pyvis.network import Network

from phase2.config import NETWORKX_GRAPH_FILE, GRAPH_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VISUALIZATION_FILE = GRAPH_DIR / "interactive_graph.html"


def visualize_graph():
    if not NETWORKX_GRAPH_FILE.exists():
        logger.error(f"Graph file not found: {NETWORKX_GRAPH_FILE}")
        return

    logger.info(f"Loading graph from {NETWORKX_GRAPH_FILE}")
    G = nx.read_graphml(NETWORKX_GRAPH_FILE)

    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty. Cannot visualize.")
        return

    logger.info("Initializing PyVis Network...")
    
    # Create the PyVis network
    net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
    
    # Use barnes hut algorithm to space out the nodes
    net.barnes_hut()

    # Convert NetworkX to PyVis
    net.from_nx(G)

    # Save to HTML
    logger.info("Generating HTML visualization...")
    net.save_graph(str(VISUALIZATION_FILE))
    logger.info(f"Interactive graph saved to {VISUALIZATION_FILE}")


if __name__ == "__main__":
    visualize_graph()
