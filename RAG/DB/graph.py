import networkx as nx
import os

# Define the local path to save your graph persistently
GRAPH_PATH = "./local_graph_db.graphml"

def initialize_graph():
    """
    Initializes the in-memory NetworkX graph. 
    Loads from disk if it already exists, otherwise creates a fresh one.
    """
    # We use a Directed Graph (DiGraph) because course prerequisites are directional 
    # (Course A -> Course B) and staff directories are hierarchical (Dean -> Professor).
    if os.path.exists(GRAPH_PATH):
        print(f"Loading existing graph from {GRAPH_PATH}...")
        return nx.read_graphml(GRAPH_PATH)
    else:
        print("Initializing a new empty directed graph...")
        return nx.DiGraph()

def save_graph(graph):
    """Saves the in-memory graph to disk."""
    nx.write_graphml(graph, GRAPH_PATH)
    print(f"Graph successfully saved to {os.path.abspath(GRAPH_PATH)}")

if __name__ == "__main__":
    print("Setting up NetworkX Graph Database...")
    psut_graph = initialize_graph()
    save_graph(psut_graph)
    print("Graph initialization complete.")