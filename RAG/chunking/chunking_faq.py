import os
import json
import re

def parse_general_faq():
    base_dir = os.getcwd()
    input_path = r"C:\Users\karee\OneDrive\Desktop\Gp\RAG\kb\general_faq.md"
    output_path = os.path.join(base_dir, "parsed_faq.json")

    print(f"Reading file from: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()


    raw_segments = re.split(r'(?=### FAQ:)', content)

    structured_data = []
    
    for index, segment in enumerate(raw_segments):
        text = segment.strip()
        
        if not text or "### FAQ:" not in text:
            continue

        entry = {
            "text": text,
            "metadata": {
                "source": "general_faq.md",
                "type": "faq",
                "chunk_index": index,
                "char_length": len(text)
            }
        }
        structured_data.append(entry)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        print("-" * 30)
        print(f" Success. Parsed {len(structured_data)} individual FAQ entries.")
        print(f" Output saved to: {output_path}")
        print("-" * 30)
        
        
       
            
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    parse_general_faq()