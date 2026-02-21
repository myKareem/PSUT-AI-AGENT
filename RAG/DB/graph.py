import networkx as nx
import os

# Paths based on your folder structure screenshot
GRAPH_PATH = "local_graph_db.graphml"
STAFF_FILE = "C:\\Users\\Kareem\\Desktop\\GP\\PSUT-AI-AGENT\\RAG\\KB\\staff_directory.md"

def populate_staff_graph():
    print(f"Loading empty graph from {GRAPH_PATH}...")
    try:
        graph = nx.read_graphml(GRAPH_PATH)
    except Exception:
        graph = nx.DiGraph()

    if not os.path.exists(STAFF_FILE):
        print(f"Error: Could not find {STAFF_FILE}")
        return

    with open(STAFF_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the document by '### ' which indicates a new staff member
    profiles = content.split('### ')[1:]
    
    # Add a root node for the university
    graph.add_node("PSUT", type="University")

    for profile in profiles:
        lines = profile.strip().split('\n')
        if not lines:
            continue
            
        name = lines[0].strip()
        node_data = {}
        
        # Extract attributes
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('- Title:'):
                node_data['title'] = line.replace('- Title:', '').strip()
            elif line.startswith('- Email:'):
                node_data['email'] = line.replace('- Email:', '').strip()
            elif line.startswith('- Phone:'):
                node_data['phone'] = line.replace('- Phone:', '').strip()
            elif line.startswith('**Context:**'):
                node_data['context'] = line.replace('**Context:**', '').strip()
        
        # Add the person to the graph
        graph.add_node(name, **node_data)
        
        # Link the person to the PSUT root node
        graph.add_edge(name, "PSUT", relation="works at")
        
        # Optional: If the title mentions a specific college (كلية), link them to it
        title = node_data.get('title', '')
        if 'كلية' in title:
            # Simple extraction: just use the title text as the department node for now
            dept_node = title
            graph.add_node(dept_node, type="Department")
            graph.add_edge(name, dept_node, relation="belongs to")

    # Save the populated graph back to the file
    nx.write_graphml(graph, GRAPH_PATH)
    print(f"Success! Graph populated with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Saved to {GRAPH_PATH}")

if __name__ == "__main__":
    populate_staff_graph()