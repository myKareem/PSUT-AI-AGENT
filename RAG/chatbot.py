import os
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dhakira.cache.semantic import SemanticCache
from dhakira.config import CacheConfig

# Configuration
DB_PATH = "C:\\Users\\20220458\\Desktop\\GP\\PSUT-AI-AGENT\\local_qdrant_db"
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

def load_cag_context():
    """
    Loads frequently accessed, highly structured data directly into memory.
    This acts as the Cache-Augmented Generation (CAG) layer.
    """
    cag_text = "=== UNIVERSITY STAFF DIRECTORY & FAQS ===\n"
    
    # 1. Load Staff Directory 
    staff_path = "C:\\Users\\20220458\\Desktop\\GP\\PSUT-AI-AGENT\\RAG\\KB\\staff_directory.md"
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
- study_plans: Questions about courses, prerequisites, or curriculum tables.
- major_overview: Questions about tuition prices, major descriptions, or credit hour totals.
- student_guide: Questions about university rules, penalties, code of conduct, or graduation policies.
- general_faq: Questions about procedural steps, ID cards, portal access, or staff contact info.

User Query: {query}
Category:"""

    response = ollama.generate(model=LLM_MODEL, prompt=routing_prompt)
    intent = response['response'].strip().lower()
    
    # Fallback to searching all if the LLM hallucinates an invalid category
    valid_collections = ["study_plans", "major_overview", "student_guide"]
    if intent in valid_collections:
        return [intent]
    elif "general" in intent:
        return [] # Empty list means don't search Qdrant, rely entirely on the CAG (System Prompt)
    else:
        return valid_collections # Fallback: search all

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
    rag_context = retrieve_rag_context(query, collections_to_search)
    # 4. Construct the final prompt for Ollama
    system_prompt = f"""You are the official student support AI for Princess Sumaya University for Technology (PSUT). 
    You must answer questions accurately in Arabic based ONLY on the provided context. If the answer is not in the context, say exactly "I do not know" and nothing else.

{SYSTEM_CAG_CONTEXT}

=== DYNAMIC RETRIEVED RULES AND PLANS ===

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