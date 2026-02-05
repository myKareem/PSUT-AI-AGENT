import os
import json
import glob

# =================CONFIGURATION=================
# Define your source directories and files
DATA_ROOT = "data_source"
OUTPUT_FILE = "PSUT_Master_Knowledge_Book_2026.md"

# Define the Intent Mapping
# Format: (Folder/File Path, Intent Label, Human Readable Description)
SOURCES = [
    {
        "type": "folder",
        "path": os.path.join(DATA_ROOT, "majors_descriptions_data"),
        "intent": "[INTENT: MAJOR_OVERVIEW]",
        "desc": "General descriptions and career info for all majors"
    },
    {
        "type": "folder",
        "path": os.path.join(DATA_ROOT, "majors_study_plans"),
        "intent": "[INTENT: STUDY_PLANS]",
        "desc": "Detailed credit hours and course requirements"
    },
    {
        "type": "json_faq",
        "path": os.path.join(DATA_ROOT, "psut_faq_data.json"),
        "intent": "[INTENT: GENERAL_FAQ]",
        "desc": "Common questions about registration, ID cards, and campus life"
    },
    {
        "type": "json_staff",
        "path": os.path.join(DATA_ROOT, "psut_professors_173.json"),
        "intent": "[INTENT: STAFF_DIRECTORY]",
        "desc": "Contact info, emails, and titles for university professors"
    },
    {
        "type": "markdown_file",
        "path": os.path.join(DATA_ROOT, "student_guide_clean.md"),
        "intent": "[INTENT: STUDENT_GUIDE]",
        "desc": "University regulations, code of conduct, grading, and library rules"
    }
]
# ===============================================

def clean_text(text):
    """Removes extra whitespace and ensures clean markdown."""
    if not text: return ""
    return " ".join(text.split())

def build_index(sources):
    """Creates the Top-Level Metadata Index."""
    index_content = "# PSUT UNIVERSITY KNOWLEDGE BASE 2026\n"
    index_content += "## SYSTEM INDEX\n"
    index_content += "> Use these anchors to navigate the KV Cache.\n\n"
    
    for idx, source in enumerate(sources):
        index_content += f"- **{source['intent']}** -> Section {idx+1}: {source['desc']}\n"
    
    index_content += "\n---\n\n"
    return index_content

def process_folder(source_config):
    """Merges all markdown files in a folder into one Intent Section."""
    content = f"### {source_config['intent']}\n"
    
    # Get all .md files
    files = glob.glob(os.path.join(source_config['path'], "*.md"))
    
    for file_path in files:
        filename = os.path.basename(file_path).replace(".md", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        # Add a sub-anchor for the specific major
        content += f"#### MAJOR: {filename}\n"
        content += f"{file_content}\n\n"
        content += "---\n\n"
        
    return content

def process_json_faq(source_config):
    """Converts FAQ JSON to Token-Efficient Markdown."""
    content = f"### {source_config['intent']}\n"
    
    try:
        with open(source_config['path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Iterate through categories
        for category in data.get("categories", []):
            cat_name = category.get("category_name", "General")
            content += f"#### CATEGORY: {cat_name}\n"
            
            for q in category.get("questions", []):
                question = clean_text(q.get("question", ""))
                answer = clean_text(q.get("answer", ""))
                content += f"- **Q:** {question}\n"
                content += f"  **A:** {answer}\n\n"
                
            content += "---\n"
            
    except Exception as e:
        print(f"Error processing FAQ JSON: {e}")
        content += f"Error loading FAQ data.\n"
        
    return content

def process_json_staff(source_config):
    """Converts Professors JSON to Token-Efficient Markdown."""
    content = f"### {source_config['intent']}\n"
    
    try:
        with open(source_config['path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        professors = data.get("professors", [])
        
        for prof in professors:
            name = clean_text(prof.get("name", "Unknown"))
            email = clean_text(prof.get("email", "N/A"))
            phone = clean_text(prof.get("phone", "N/A"))
            title = clean_text(prof.get("title", ""))
            
            content += f"**PROF:** {name}\n"
            if title: content += f"- Title: {title}\n"
            content += f"- Email: {email}\n"
            content += f"- Phone: {phone}\n"
            content += "\n"
            
    except Exception as e:
        print(f"Error processing Professors JSON: {e}")
        content += f"Error loading Staff data.\n"
        
    return content

def process_markdown_file(source_config):
    """Ingests a single large Markdown file (Student Guide)."""
    content = f"### {source_config['intent']}\n"
    
    try:
        with open(source_config['path'], 'r', encoding='utf-8') as f:
            file_content = f.read()
            # Simple spacing cleanup
            content += f"{file_content}\n"
    except Exception as e:
        print(f"Error processing Student Guide: {e}")
        
    return content

def main():
    print("Building Master Knowledge Book...")
    full_book_content = ""
    
    # 1. Build Index
    full_book_content += build_index(SOURCES)
    
    # 2. Process contents
    for source in SOURCES:
        print(f"Processing: {source['intent']}...")
        
        if source['type'] == 'folder':
            full_book_content += process_folder(source)
        elif source['type'] == 'json_faq':
            full_book_content += process_json_faq(source)
        elif source['type'] == 'json_staff':
            full_book_content += process_json_staff(source)
        elif source['type'] == 'markdown_file':
            full_book_content += process_markdown_file(source)
            
        # Add a major separator between sections
        full_book_content += "\n========================================\n\n"

    # 3. Save File
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_book_content)
        
    print(f"Success! Master Book saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()