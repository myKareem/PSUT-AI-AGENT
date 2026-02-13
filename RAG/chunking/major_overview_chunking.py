import os
import json
import re
# We keep the imports as requested, though we rely on standard parsing 
# to achieve the "single massive chunk" requirement effectively.
from chunkwise import ChunkConfig, MarkdownChunker 

def parse_majors():
    # 1. Define Paths
    base_dir = os.getcwd()
    input_path = os.path.join(base_dir, "kb", "major_overview.md")
    output_path = os.path.join(base_dir, "parsed_majors.json")

    print(f"Reading file from: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Split by Major using Regex (Lookahead)
    # Instead of split("---"), we split by the "#### MAJOR:" header
    # to ensure we capture the full block (Introduction -> Admission) as one unit.
    major_blocks = re.split(r'(?=#### MAJOR:)', content)
    
    structured_data = []
    
    for block in major_blocks:
        block = block.strip()
        if not block or "#### MAJOR:" not in block:
            continue

        # 3. Extract Major Name & Metadata
        major_match = re.search(r"#### MAJOR:\s*(.+)", block)
        if not major_match:
            continue
            
        major_name = major_match.group(1).strip()
        
        # Regex to capture the key-value pairs
        meta_school = re.search(r"\*\*School:\*\*\s*(.+)", block)
        meta_hours = re.search(r"\*\*Credit Hours:\*\*\s*(.+)", block)
        meta_price = re.search(r"\*\*Price per Hour:\*\*\s*(.+)", block)
        
        # Helper to safely get group text
        school_val = meta_school.group(1).strip() if meta_school else 'N/A'
        hours_val = meta_hours.group(1).strip() if meta_hours else 'N/A'
        price_val = meta_price.group(1).strip() if meta_price else 'N/A'

       # fix this it doesnt work
        body_content = block
        
        # Clean up the header/metadata lines from the body text 
        # so they don't appear twice (once in summary, once in raw text).
        
        
        summary_header = (
            f"Overview for {major_name}. "
            f"School: {school_val}. "
            f"Credit Hours: {hours_val}. "
            f"Price: {price_val}.\n\n"
        )
        
        # Combine Summary + Full Original Block Content
        # We use the full 'block' variable which contains Introduction, Details, and Admission text.
        full_text = summary_header + block

        entry = {
            "text": body_content,
            "metadata": {
                "source": "major_overview.md",
                "type": "major_full_content",
                "major": major_name,
                "school": school_val,
                "credit_hours": hours_val,
                "price_per_hour": price_val
            }
        }
        structured_data.append(entry)

    # 5. Save Output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)

    print(f" Parsed {len(structured_data)} major blocks.")
    print(f" Output saved to: {output_path}")

if __name__ == "__main__":
    parse_majors()