import os
import json
from chunkwise import ChunkConfig, ParentDocumentChunker

def parse_student_guide():
    # 1. Define Paths
    base_dir = os.getcwd()
    input_path = os.path.join(base_dir, "kb", "student_guide.md")
    output_path = os.path.join(base_dir, "parsed_guide.json")

    print(f"Reading file from: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Configure Parent Document Strategy
    # Child Chunk (400 chars): Optimized for vector search precision (finding specific rules).
    # Parent Chunk (2000 chars): Optimized for LLM context (understanding the full section).
    
    config = ChunkConfig(
        language="ar",  # Activates Arabic-specific splitters (commas, full stops)
        chunk_overlap=50 # Slight overlap for children to maintain sentence continuity
    )

    chunker = ParentDocumentChunker(
        config=config,
        child_chunk_size=400,   # Small chunks for indexing
        parent_chunk_size=2000, # Large chunks for retrieval context
        parent_overlap=200      # Ensure context flows between parents
    )

    print("Processing Student Guide (Small-to-Big Strategy)...")
    
    # 3. Generate Chunks
    # This creates small 'child' chunks, each containing a reference to its 'parent'
    chunks = chunker.chunk(content)
    
    structured_data = []
    
    for chunk in chunks:
        # Retrieve the parent content from the chunk's metadata
        # ChunkWise automatically stores the larger context in 'parent_content'
        parent_text = chunk.metadata.get("parent_content", "")
        
        entry = {
            # The 'text' field is the small child chunk (used for Embedding)
            "text": chunk.content,
            
            # The metadata holds the large parent chunk (used for Generation)
            "metadata": {
                "source": "student_guide.md",
                "type": "legal_regulation",
                "strategy": "parent_document",
                "parent_content": parent_text, 
                "chunk_index": chunk.index
            }
        }
        structured_data.append(entry)

    # 4. Save Output
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        print("-" * 30)
        print(f"Success! Generated {len(structured_data)} searchable child chunks.")
        print(f"Output saved to: {output_path}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    parse_student_guide()