import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams

# 1. Initialize a local Qdrant client
# This will create a directory named 'local_qdrant_db' in your current working folder to store the vectors persistently.
DB_PATH = "./local_qdrant_db"
client = QdrantClient(path=DB_PATH)

# 2. Define the embedding dimension
# Arabic-Triplet-Matryoshka-V2 base dimension is typically 768. 
DENSE_EMBEDDING_DIM = 768

def setup_psut_collections(qdrant_client: QdrantClient, collections: list):
    """
    Initializes Qdrant collections with hybrid search capabilities (Dense + BM25).
    """
    for collection_name in collections:
        # Check if the collection already exists to avoid overwriting
        if not qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # Dense vector configuration for semantic search
                    "dense": VectorParams(
                        size=DENSE_EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    # Sparse vector configuration for exact keyword/BM25 search
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")

# 3. Target collections based on your chunked files
collections_to_initialize = ["general_faq", "student_guide", "major_overview", "study_plans"]

if __name__ == "__main__":
    print("Initializing local Qdrant Vector Database...")
    setup_psut_collections(client, collections_to_initialize)
    print(f"Qdrant initialization complete. Data is stored at: {os.path.abspath(DB_PATH)}")