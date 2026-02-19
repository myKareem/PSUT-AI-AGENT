import os
import json
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dhakira.cache.semantic import SemanticCache
from dhakira.config import CacheConfig

# Configuration
DB_PATH = "C:\\Users\\Kareem\\Desktop\\GP\\local_qdrant_db"
MODEL_NAME = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
LLM_MODEL = "qwen3:8b" # Ensure this matches what you ran in Ollama

# Initialize Dhakira's Semantic Cache
# This will save ~50% of LLM calls by caching identical or highly similar Arabic queries
cache = SemanticCache(CacheConfig(
    enabled=True,
    max_size=1000,
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
    
    # 1. Load Staff Directory (Assuming it's a raw markdown file in your directory)
    staff_path = "C:\\Users\\Kareem\\Desktop\\GP\\PSUT-AI-AGENT\\RAG\\KB\\staff_directory.md"
    if os.path.exists(staff_path):
        with open(staff_path, 'r', encoding='utf-8') as f:
            cag_text += f.read() + "\n\n"
    
                
    return cag_text

print("Building CAG system prompt...")
SYSTEM_CAG_CONTEXT = load_cag_context()

def retrieve_rag_context(query: str, top_k: int = 2) -> str:
    """
    Dynamically searches Qdrant collections for heavy, dense text 
    (Student Guide, Study Plans, Major Overviews).
    """
    query_vector = embedder.encode(query).tolist()
    collections_to_search = ["student_guide", "major_overview", "study_plans"]
    
    retrieved_texts = []
    
    for collection in collections_to_search:
        # Check if collection has data before searching to avoid errors
        try:
            results = qdrant.search(
                collection_name=collection,
                query_vector=("dense", query_vector),
                limit=top_k
            )
            for hit in results:
                # hit.payload contains the chunk dictionary we ingested earlier
                chunk_text = hit.payload.get("text", "")
                if chunk_text:
                    retrieved_texts.append(f"Source ({collection}):\n{chunk_text}")
        except Exception as e:
            continue # Skip if collection is empty or not initialized properly
            
    return "\n\n".join(retrieved_texts)

def generate_response(query: str) -> str:
    """
    Orchestrates the cache, RAG retrieval, and Ollama LLM generation.
    """
    # 1. Check Dhakira Semantic Cache first
    cached_response = cache.get(query)
    if cached_response:
        print("[Served from Dhakira Semantic Cache]")
        return cached_response.get("response", "")

    # 2. If not cached, retrieve dynamic RAG context
    rag_context = retrieve_rag_context(query)
    
    # 3. Construct the final prompt for Ollama
    system_prompt = f"""You are the official student support AI for Princess Sumaya University for Technology (PSUT). 
You must answer questions accurately in Arabic based ONLY on the provided context. If the answer is not in the context, say you do not know.

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