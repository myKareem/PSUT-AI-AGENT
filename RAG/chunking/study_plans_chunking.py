import os
import json
import re

def parse_study_plans():
    # 1. Define Paths
    base_dir = os.getcwd()
    input_path = os.path.join(base_dir, "kb", "study_plans.md")
    output_path = os.path.join(base_dir, "parsed_plans.json")

    print(f"Reading file from: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Split by Major Header
    # We treat each Major's study plan as a distinct block
    major_blocks = re.split(r'(?=#### MAJOR:)', content)
    
    structured_data = []
    
    for block in major_blocks:
        block = block.strip()
        if not block or "#### MAJOR:" not in block:
            continue

        # 3. Extract Context (Major Name)
        major_match = re.search(r"#### MAJOR:\s*(.+)", block)
        major_name = major_match.group(1).strip() if major_match else "General"
        
        # 4. Find and Parse the Table
        # We look for lines starting with '|'
        lines = block.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines or non-table lines
            if not line.startswith("|"):
                continue
                
            # Skip the Header Row (contains "Major" or "اسم المادة")
            if "اسم المادة" in line or "Course Name" in line:
                continue
                
            # Skip the Markdown Separator Row (contains "---")
            if "---" in line:
                continue
            
            # Extract Cells
            # Split by '|' and remove the first/last empty elements caused by leading/trailing pipes
            cells = [c.strip() for c in line.split('|') if c.strip()]
            
            # Safety Check: Ensure we have the expected 5 columns
            # Structure: | Major | Category | Course Name | Credits | Prerequisite |
            if len(cells) < 5:
                continue
                
            # Map columns to variables (Adjust index if your table structure varies slightly)
            # Assuming strictly: | Major | Category | Course | Credits | Prereq |
            col_major = cells[0]
            col_category = cells[1]
            col_course = cells[2]
            col_credits = cells[3]
            col_prereq = cells[4]

            # 5. Intelligent Contextualization (The Magic Step)
            # Instead of just embedding the raw row, we rewrite it into a clear Arabic sentence.
            # This makes the chunk "stand alone" without needing the table headers.
            
            contextual_text = (
                f"في تخصص {major_name}، مادة '{col_course}' تندرج تحت فئة {col_category}. "
                f"عدد الساعات المعتمدة: {col_credits}. "
                f"المتطلب السابق: {col_prereq}."
            )

            # 6. Create the RAG Entry
            entry = {
                "text": contextual_text,
                "metadata": {
                    "source": "study_plans.md",
                    "type": "course_requirement",
                    "major": major_name,
                    "course_name": col_course,
                    "credits": col_credits,
                    "prerequisite": col_prereq,
                    "category": col_category,
                    # We store the raw row too, in case the LLM wants to see the original table format later
                    "raw_row": line 
                }
            }
            structured_data.append(entry)

    # 7. Save Output
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        print("-" * 30)
        print(f"Success! Parsed {len(structured_data)} course chunks.")
        print(f"Output saved to: {output_path}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    parse_study_plans()