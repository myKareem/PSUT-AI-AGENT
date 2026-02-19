import json
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# Paths and Model
DB_PATH = "./local_qdrant_db"
CHUNKS_DIR = "C:\\Users\\Kareem\\Desktop\\GP\\PSUT-AI-AGENT\\RAG\\chunks"
MODEL_NAME = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"

# Map collections to your exact JSON filenames
COLLECTIONS_MAP = {
    "general_faq": "parsed_faq.json",
    "student_guide": "parsed_guide.json",
    "major_overview": "parsed_majors.json",
    "study_plans": "parsed_plans.json"
}

def load_json_chunks(filename):
    file_path = os.path.join(CHUNKS_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def run_ingestion():
    print("1. Loading Local Embedding Model...")
    embedder = SentenceTransformer(MODEL_NAME)
    
    print("2. Connecting to Local Qdrant Database...")
    client = QdrantClient(path=DB_PATH)

    for collection_name, filename in COLLECTIONS_MAP.items():
        print(f"\nProcessing {filename} into '{collection_name}'...")
        chunks = load_json_chunks(filename)
        
        if not chunks:
            print(f"Warning: No data found in {filename}. Skipping.")
            continue

        points = []
        for chunk in chunks:
            # Note: Change "text" to "content" or whatever key your ChunkWise script used to store the main text
            text_content = chunk.get("text", "") 
            if not text_content:
                continue
            
            # Generate the vector embedding
            dense_vector = embedder.encode(text_content).tolist()
            
            # Create a unique ID and structure the point for Qdrant
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={"dense": dense_vector}, 
                    payload=chunk  # Stores the text and metadata (like headers, URLs)
                )
            )

        # Upsert data into Qdrant
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Success: Ingested {len(points)} vectors into '{collection_name}'.")

if __name__ == "__main__":
    run_ingestion()
    print("\nAll vector data successfully ingested into Qdrant.")