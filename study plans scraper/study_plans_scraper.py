import requests
from bs4 import BeautifulSoup
import os
import re
import time
import sys

# --- Configuration & Constants ---
MAIN_URL = "https://psut.edu.jo/ar/study-plans"
BASE_URL = "https://www.psut.edu.jo"

# Combined output directories
OUTPUT_DIR_MD = "psut_combined_data"
OUTPUT_DIR_PDF = "psut_combined_pdfs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
}

# --- Shared Helper Functions ---

def sanitize_filename(name):
    """
    Cleans up Arabic names to be safe for Windows/Linux filenames.
    """
    name = re.sub(r'[^\w\s\-\u0600-\u06FF]', '', name)
    name = re.sub(r'\s+', '_', name).strip()
    return name

def clean_url(url):
    """
    Fixes malformed URLs found on the PSUT site.
    """
    if not url: return ""
    if '<' in url or '>' in url:
        url = re.sub(r'<[^>]+>', '', url)
    if 'font-color' in url:
        url = url.split('font-color')[0]
    return url.strip()

def get_soup(url):
    """
    Helper to fetch a URL and return a BeautifulSoup object.
    """
    try:
        url = clean_url(url)
        if not url.startswith('http'):
            url = BASE_URL + url
            
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"   [!] Error fetching {url}: {e}")
        return None

def download_file(url, folder, filename):
    """
    Downloads a file from a URL.
    """
    try:
        url = clean_url(url)
        if not url.startswith('http'):
            url = BASE_URL + url
            
        response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        response.raise_for_status()
        
        if not filename.lower().endswith('.pdf'):
            filename += ".pdf"
            
        file_path = os.path.join(folder, filename)
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"   -> [PDF] Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"   -> [PDF] Download Failed: {e}")
        return False

# --- Logic from scrape_details.py ---

def extract_details_content(soup):
    """
    Extracts Intro and Body content from the program home page.
    """
    content_parts = []
    
    # --- PART A: Intro Paragraph ---
    intro_section = soup.find('div', class_='school_name_sec_main_left')
    if intro_section:
        intro_paragraph = intro_section.find('p')
        if intro_paragraph:
            text = intro_paragraph.get_text(strip=True)
            content_parts.append("## مقدمة (Introduction)")
            content_parts.append(text)

    # --- PART B: Body Content ---
    content_section = soup.find('div', class_='page-content')
    if content_section:
        content_parts.append("## تفاصيل البرنامج (Program Details)")
        paragraphs = content_section.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if not text:
                continue
            
            # Check for Header style
            if p.find('strong'):
                content_parts.append(f"### {text}")
            else:
                content_parts.append(text)
                
    return "\n\n".join(content_parts)

# --- Logic from scrape_req.py ---

def extract_requirements_text(soup_content):
    """
    Parses text from the requirements page content div.
    """
    content_lines = []
    content_div = soup_content.find('div', class_='page-content')
    
    if content_div:
        for element in content_div.find_all(['p', 'h4', 'ul', 'ol', 'div']):
            if element.name == 'h4' or (element.name == 'p' and element.find('strong')):
                text = element.get_text(strip=True)
                if text:
                    content_lines.append(f"\n### {text}\n")
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    text = li.get_text(strip=True)
                    if text:
                        content_lines.append(f"- {text}")
            elif element.name == 'p':
                text = element.get_text(strip=True)
                if text and len(text) > 2 and not element.find_parent('li'):
                    content_lines.append(text)
    return "\n".join(content_lines)

def fetch_requirements(soup_home):
    """
    Finds the admission link on home page, goes there, extracts text.
    """
    admission_link = None
    
    # 1. Search for link
    target_headers = soup_home.find_all('h4', string=re.compile("شروط القبول"))
    for h4 in target_headers:
        parent_a = h4.find_parent('a')
        if parent_a and parent_a.get('href'):
            admission_link = parent_a.get('href')
            break
            
    if not admission_link:
        link_obj = soup_home.find('a', string=re.compile("شروط القبول"))
        if link_obj:
            admission_link = link_obj.get('href')

    if not admission_link:
        return "Not Found: Could not find 'Admission Requirements' link."

    # 2. Fetch Page
    soup_req = get_soup(admission_link)
    if not soup_req:
        return "Error: Could not load Admission Requirements page."
        
    # 3. Extract
    return extract_requirements_text(soup_req)

# --- Logic from get_pdfs.py ---

def process_pdf_download(soup_home, program_name):
    """
    Finds the 'Study Plan' link, navigates if necessary, and downloads PDF.
    """
    # 1. Find Hub Link
    keywords_hub = [
        "الخطة الدراسية", "وصف المساقات", "Study Plan", 
        "الخطة الاسترشادية", "Curriculum", "الخطة",
        "Plan", "Structure", "Study", "Guidance", "دليل البرنامج"
    ]
    
    hub_link = None
    
    # Search method A: Direct <a> text
    all_links = soup_home.find_all('a')
    for link in all_links:
        text = link.get_text(strip=True)
        href = link.get('href')
        if href and len(href) > 2 and any(kw in text for kw in keywords_hub):
            hub_link = href
            break
            
    # Search method B: Headers inside <a>
    if not hub_link:
        headers = soup_home.find_all(['h4', 'h5', 'h3', 'span', 'div'])
        for h in headers:
            text = h.get_text(strip=True)
            if any(kw in text for kw in keywords_hub):
                parent = h.find_parent('a')
                if parent and parent.get('href'):
                    hub_link = parent.get('href')
                    break

    if not hub_link:
        print("   -> [PDF] Skipped: No Study Plan link found.")
        return

    # Check if direct PDF
    if hub_link.lower().endswith('.pdf') or "uploads" in hub_link:
        download_file(hub_link, OUTPUT_DIR_PDF, sanitize_filename(program_name))
        return

    # 2. Fetch Hub Page
    soup_hub = get_soup(hub_link)
    if not soup_hub:
        return

    # 3. Find PDF on Hub Page
    pdf_url = None
    keywords_pdf = ["الخطة الدراسية", "Plan", "Curriculum", "الاسترشادية", "Guidance", "دليل البرنامج"]
    
    all_hub_links = soup_hub.find_all('a')
    for link in all_hub_links:
        text = link.get_text(strip=True)
        href = link.get('href')
        if not href: continue
        
        if any(kw in text for kw in keywords_pdf):
             if href not in hub_link and "#" not in href:
                 pdf_url = href
                 break
                 
    # Fallback to any PDF
    if not pdf_url:
        for link in all_hub_links:
            if link.get('href') and link.get('href').lower().endswith('.pdf'):
                pdf_url = link.get('href')
                break

    if pdf_url:
        # Handle Google Drive links if present (logic from original script)
        if "drive.google.com" in pdf_url:
            file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', pdf_url)
            if file_id_match:
                dl_url = f"https://drive.google.com/uc?export=download&id={file_id_match.group(1)}"
                download_file(dl_url, OUTPUT_DIR_PDF, sanitize_filename(program_name))
            else:
                print("   -> [PDF] Drive link found but ID extraction failed.")
        else:
            download_file(pdf_url, OUTPUT_DIR_PDF, sanitize_filename(program_name))
    else:
        print("   -> [PDF] No PDF file found on hub page.")

# --- Main Master Loop ---

def main():
    # Setup directories
    if not os.path.exists(OUTPUT_DIR_MD):
        os.makedirs(OUTPUT_DIR_MD)
    if not os.path.exists(OUTPUT_DIR_PDF):
        os.makedirs(OUTPUT_DIR_PDF)

    print(f"Connecting to Main List: {MAIN_URL}...")
    try:
        response = requests.get(MAIN_URL, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all("table")
    print(f"Found {len(tables)} tables. Starting Master Scrape...")

    current_school = "Unknown School"

    # Iterate through the main tables (Logic from scrape_plans.py)
    for table in tables:
        rows = table.find_all("tr")
        
        for row in rows:
            cells = row.find_all("td")
            
            # CASE 1: School Header Row
            if len(cells) == 1 and cells[0].has_attr("colspan"):
                text = cells[0].get_text(strip=True)
                if text:
                    current_school = text
                continue

            # CASE 2: Program Data Row
            if len(cells) == 4:
                spec_name = cells[0].get_text(strip=True)
                hours = cells[1].get_text(strip=True)
                # Cell 2 is the link - we need to extract it
                link_tag = cells[2].find('a') # Usually the link is here or in cell 0
                
                # Robust link finding (combining logic from scrape_plans and scrape_details)
                program_url = None
                if link_tag and link_tag.get('href'):
                    program_url = link_tag.get('href')
                else:
                    # Try finding "click here" or any link in the row
                    click_link = row.find('a', href=True)
                    if click_link:
                        program_url = click_link['href']

                price = cells[3].get_text(strip=True)

                # Skip header row
                if "التخصص" in spec_name or "ساعات" in hours:
                    continue
                
                print(f"\nProcessing: {spec_name}...")
                
                # --- Start Building the Markdown Content ---
                md_content = []
                md_content.append(f"# Program: {spec_name}")
                md_content.append(f"**School:** {current_school}")
                md_content.append(f"**Credit Hours:** {hours}")
                md_content.append(f"**Price per Hour:** {price}")
                
                if program_url:
                    full_url = clean_url(program_url)
                    if not full_url.startswith('http'):
                        full_url = BASE_URL + full_url
                    
                    md_content.append(f"**URL:** {full_url}")
                    md_content.append("\n---")

                    # Load Program Home Page once
                    soup_program = get_soup(full_url)
                    
                    if soup_program:
                        # 1. Get Details (Intro/Body) - from scrape_details.py
                        print("   -> Extracting Details...")
                        details_text = extract_details_content(soup_program)
                        md_content.append(details_text)
                        
                        md_content.append("\n---\n")
                        
                        # 2. Get Requirements - from scrape_req.py
                        print("   -> Extracting Requirements...")
                        md_content.append("## شروط القبول (Admission Requirements)")
                        req_text = fetch_requirements(soup_program)
                        md_content.append(req_text)
                        
                        # 3. Download PDF - from get_pdfs.py
                        print("   -> Checking for PDF...")
                        process_pdf_download(soup_program, spec_name)
                    else:
                        md_content.append("Error: Could not load program page content.")
                else:
                    md_content.append("\n**Error:** No valid URL found for this program in the main table.")

                # --- Save the Combined Markdown File ---
                safe_name = sanitize_filename(spec_name)
                file_path = os.path.join(OUTPUT_DIR_MD, f"{safe_name}.md")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(md_content))
                
                print(f"   -> Saved MD: {file_path}")
                
                # Sleep briefly to be polite
                time.sleep(1)

    print("\n" + "="*50)
    print("Master Scrape Completed.")
    print(f"Markdown files: {OUTPUT_DIR_MD}/")
    print(f"PDF files:      {OUTPUT_DIR_PDF}/")

if __name__ == "__main__":
    main()