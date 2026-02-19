from sentence_transformers import SentenceTransformer

# The exact HuggingFace repository name for the model
MODEL_NAME = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"

def load_embedding_model():
    print(f"Loading embedding model: {MODEL_NAME}...")
    # This automatically detects your GPU (CUDA) if available, otherwise falls back to CPU
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully!")
    return model

if __name__ == "__main__":
    embedder = load_embedding_model()
    
    sample_text = "كيف أسجل موادي في جامعة الأميرة سمية؟"
    vector = embedder.encode(sample_text)
    
    print(f"Test successful. Vector dimension size: {len(vector)}")