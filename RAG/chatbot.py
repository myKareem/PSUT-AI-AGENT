import os
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dhakira.cache.semantic import SemanticCache
from dhakira.config import CacheConfig
import networkx as nx

# Configuration
DB_PATH = "C:\\Users\\Kareem\\Desktop\\GP\\local_qdrant_db"
MODEL_NAME = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
LLM_MODEL = "qwen3:8b" # Ensure this matches what you ran in Ollama

# Initialize Dhakira's Semantic Cache
# This will save ~50% of LLM calls by caching identical or highly similar Arabic queries
cache = SemanticCache(CacheConfig(
    enabled=True,
    max_size=32000,
    ttl_seconds=86400 # Cache for 24 hours
))

print("Loading local embedding model (CPU/GPU)...")
embedder = SentenceTransformer(MODEL_NAME)

print("Connecting to local Qdrant database...")
qdrant = QdrantClient(path=DB_PATH)

print("Loading local Graph database...")
try:
    # Use NetworkX's built-in graphml reader instead of pickle
    staff_graph = nx.read_graphml('C:\\Users\\Kareem\\Desktop\\GP\\local_graph_db.graphml')
    print(f"Graph loaded successfully with {staff_graph.number_of_nodes()} nodes.")
except Exception as e:
    print(f"Warning: Graph DB not found or error loading. Initializing empty graph. Error: {e}")
    staff_graph = nx.DiGraph()

def load_cag_context():
    """
    Loads frequently accessed, highly structured data directly into memory.
    This acts as the Cache-Augmented Generation (CAG) layer.
    """
    cag_text = "=== general_faq ===\n"
    
    staff_path = "C:\\Users\\Kareem\\Desktop\\GP\\PSUT-AI-AGENT\\RAG\\KB\\general_faq.md"
    if os.path.exists(staff_path):
        with open(staff_path, 'r', encoding='utf-8') as f:
            cag_text += f.read() + "\n\n"
    
                
    return cag_text

print("Building CAG system prompt...")
SYSTEM_CAG_CONTEXT = load_cag_context()

def classify_intent(query: str) -> str:
    """
    Uses the local LLM to classify the query intent to route to the correct database.
    """
    routing_prompt = f"""Analyze the user query and classify it into EXACTLY ONE of the following categories. Output NOTHING ELSE except the category name.

Categories:
- study_plans: Questions about specific courses, prerequisites, or curriculum tables.
- major_overview: Questions about tuition prices, major descriptions, or credit hour totals.
- student_guide: Questions about university rules, penalties, code of conduct, or graduation policies.
- staff_directory: Questions about deans, professors, doctors, or their contact information.
- general_faq: Questions about procedural steps, ID cards, or portal access.

User Query: {query}
Category:"""

    response = ollama.generate(model=LLM_MODEL, prompt=routing_prompt)
    intent = response['response'].strip().lower()
    
    # Fallback to searching all if the LLM hallucinates an invalid category
    valid_collections = ["study_plans", "major_overview", "student_guide","staff_directory"]
    if intent in valid_collections:
        return [intent]
    elif "general" in intent:
        return [] # Empty list means don't search Qdrant, rely entirely on the CAG (System Prompt)
    else:
        return valid_collections # Fallback: search all

# ==========================================
# GRAPH RETRIEVAL LAYER
# ==========================================
def search_staff_graph(query: str) -> str:
    """
    Searches the in-memory NetworkX graph for a staff member using robust Arabic keyword matching.
    """
    # Common words that might accidentally match the wrong person
    stop_words = ["اعطني", "ايميل", "رقم", "هاتف", "دكتور", "دكتورة", "استاذ", "عميد", "عن", "من", "هو", "هي"]
    
    # Clean the query and filter out stop words and short words
    query_words = [w for w in query.strip().split() if len(w) > 2 and w not in stop_words]
    
    found_profiles = []

    for node_name, node_data in staff_graph.nodes(data=True):
        if node_data.get('type') == 'University':
            continue
            
        name_str = str(node_name)
        match_found = False
        
        # Check if any significant name word from the query exists in the actual graph node name
        for word in query_words:
            if word in name_str:
                match_found = True
                break
                
        if match_found:
            profile = [f"### {node_name}"]
            for key, value in node_data.items():
                if key not in ['type', 'id']: # Hide internal graph metadata
                    profile.append(f"- {key.capitalize()}: {value}")
            
            # Retrieve graph relationships (e.g., Dean of X)
            for target in staff_graph.successors(node_name):
                relation = staff_graph.edges[node_name, target].get('relation', 'connected to')
                profile.append(f"- Role Context: {relation} {target}")
                
            found_profiles.append("\n".join(profile))

    # Return top 3 matches to avoid flooding the context window if a common name like "محمد" is queried
    return "\n\n".join(found_profiles[:3]) if found_profiles else ""

def retrieve_rag_context(query: str, target_collections: list, top_k: int = 2) -> str:
    # Your new logic: Skip vector search entirely if it's a general FAQ or Staff query
    if not target_collections:
        print("[DEBUG] No target collections specified. Skipping vector search.")
        return "" 
        
    query_vector = embedder.encode(query).tolist()
    retrieved_texts = []
    
    for collection in target_collections:
        try:
            print(f"\n[DEBUG] Attempting to search collection: {collection}...")
            
            # The working modern Qdrant syntax (prevents the AttributeError)
            response = qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                using="dense", 
                limit=top_k
            )
            
            results = response.points
            print(f"[DEBUG] Found {len(results)} chunks in {collection}.")
            
            for i, hit in enumerate(results):
                chunk_text = hit.payload.get("text", "")
                if chunk_text:
                    # Printing the chunk to the terminal for debugging
                    print(f"\n>>> PRINTING CHUNK {i+1} FROM {collection} <<<")
                    print(chunk_text)
                    print(">" * 50)
                    retrieved_texts.append(f"Source ({collection}):\n{chunk_text}")
                    
        except Exception as e:
            # Keeping the error visible so it never fails silently again
            print(f"\n[CRITICAL ERROR] Failed searching collection '{collection}': {e}\n")
            continue 
            
    return "\n\n".join(retrieved_texts)

def generate_response(query: str) -> str:
    # 1. Check Cache
    cached_response = cache.get(query)
    if cached_response:
        print("[Served from Dhakira Semantic Cache]")
        return cached_response.get("response", "")

    # 2. Classify Intent
    collections_to_search = classify_intent(query)
    print(f"[Router decision: Searching {collections_to_search if collections_to_search else 'CAG Only'}]")

    # 3. Retrieve specific RAG context
    rag_context = ""
    
    # If the router identified it as a staff query, search the Graph DB
    if "staff_directory" in collections_to_search:
        graph_results = search_staff_graph(query)
        if graph_results:
            rag_context += f"=== STAFF GRAPH RESULTS ===\n{graph_results}\n\n"
        # Remove it from the list so Qdrant doesn't try to search a non-existent vector collection
        collections_to_search.remove("staff_directory")
    
    # Search Qdrant Vector DB for the remaining collections
    vector_results = retrieve_rag_context(query, collections_to_search)
    if vector_results:
        rag_context += f"=== VECTOR SEARCH RESULTS ===\n{vector_results}\n\n"
    # 4. Construct the final prompt for Ollama
    system_prompt = f"""You are the official Smart Assistant for Princess Sumaya University for Technology (PSUT). 
Your role is to help students and answer their academic and administrative inquiries professionally, clearly, and objectively and you must answer in Arabic jordanian dialect.

**Core Operational Rules (You MUST follow these strictly):**

1. **Strict Context Reliance (Handling Uncertainty):** Answer **ONLY** based on the information provided in the "Context" sections below. If a student asks about something not found in the context, you are strictly forbidden from guessing or making up an answer. In such cases, clearly state: "I apologize, but I do not currently have this information. Please refer to the relevant department or the Deanship of Admissions and Registration."

2. **Step-by-Step Reasoning (Chain-of-Thought):**
When a student asks about procedures (such as registration steps) or disciplinary rules, explain the answer in a logical, step-by-step manner to make it easy to understand.

3. **Formatting and Clarity:**
- Use bullet points when listing conditions or steps.
- Use **bold text** to highlight course names, course codes, staff names, or financial amounts.
- Keep your answers concise and direct without long, unnecessary introductions.
- The use of emojis is strictly prohibited in all responses.

4. **General Guidelines:**
- you might be asked questions that are not clear enough or lack specific details. In such cases, do not make assumptions. Instead, ask the student for clarification or additional information to better assist them.
- when you are asked for example: "احكيلي عن تخصص علم البيانات" assume its asking for the batchelores "علم البيانات والذكاء الاصطناعي" not the masters, and answer accordingly. If the user wants information about the masters they will ask specifically about it.
- Never print emojies in you answers.
- if you didnt retrieve any relevant context from the RAG search, do not say "Based on the retrieved information..." or anything similar. Just answer based on the CAG context and if you dont find the answer there say you dont have the information.

=== STATIC CONTEXT (CAG) ===
{SYSTEM_CAG_CONTEXT}

=== DYNAMIC RETRIEVED CONTEXT (RAG) ===
{rag_context}
"""

    # 4. Stream response from local Ollama
    response_stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ],
        stream=True
    )
    
    print("\nPSUT Bot: ", end="", flush=True)
    full_response = ""
    for chunk in response_stream:
        token = chunk['message']['content']
        print(token, end="", flush=True)
        full_response += token
    print("\n")
    
    # 5. Save the generated response to Dhakira Cache for future identical questions
    cache.put(query, {"response": full_response})
    
    return full_response

if __name__ == "__main__":
    
    print("\nPSUT Local Support Bot Initialized. Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        generate_response(user_input)