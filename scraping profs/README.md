# PSUT Professor Contact Scraper

A Python web scraper for extracting professor contact information from the Princess Sumaya University for Technology (PSUT) website.

##  Features

-  Scrapes all 9 pages of professor listings
-  Extracts: Name, Title, Email, Phone Number, Profile URL
-  Supports both Arabic and English content
-  Robust error handling with retry logic
-  Progress bars for real-time monitoring
-  Exports data to clean JSON format
-  Detailed statistics and data completeness reports
-  Respectful scraping with configurable delays
-  Works in VS Code, Terminal, and Google Colab

##  Quick Start

### Option 1: Run in VS Code / Terminal

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt 
   ```

2. **Run the scraper:**
   ```bash
   python psut_professor_scraper.py
   ```

3. **Find your output:**
   - Output file: filename
     
### Option 2: Run in Google Colab (was not tested)

1. **Upload `psut_scraper_colab.py` to Google Colab**

2. **Run all cells sequentially**

3. **Download automatically** - The JSON file will download to your browser

##  Output Format

The scraper generates a JSON file with the following structure:

```json
{
  "metadata": {
    "source_url": "https://www.psut.edu.jo/ar/staff/professor",
    "scrape_date": "2026-02-01T10:30:00",
    "total_professors": 85,
    "successful_scrapes": 85,
    "failed_scrapes": 0,
    "pages_processed": 9
  },
  "professors": [
    {
      "name": "د. طارق بديز",
      "title": "أستاذ مساعد",
      "email": "t.badees@psut.edu.jo",
      "phone": "+962 6 535 9949",
      "profile_url": "https://www.psut.edu.jo/ar/staff/..."
    }
  ]
}
```


##  What Gets Scraped

### Phase 1: URL Collection
- Iterates through pages 1-9 of professor listings
- Extracts unique profile URLs for each professor
- Displays progress with real-time updates

### Phase 2: Detail Extraction
- Visits each professor's profile page
- Extracts contact information using multiple strategies
- Handles missing data 
- Provides detailed success/failure statistics


##  Dependencies

- `requests` - HTTP library for making web requests
- `beautifulsoup4` - HTML parsing library
- `tqdm` - Progress bar library

**Note:** GPU acceleration is NOT required. This scraping is I/O-bound, not compute-intensive.

##  Target Website

https://www.psut.edu.jo/ar/staff/professor

---

**Version:** 1.0.0  
**Python Version:** 3.8+
