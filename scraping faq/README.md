# PSUT FAQ Scraper

A Python script to scrape Frequently Asked Questions (FAQs) from the Princess Sumaya University for Technology (PSUT) website and save them in JSON format.

## Features

-  Scrapes all FAQ categories from PSUT Arabic website
-  Extracts questions and answers with embedded links preserved
-  Generates two JSON output formats:
  - Standard hierarchical format
  - RAG-optimized format for vector databases (not tested yet)
-  Respects server with configurable delays between requests
-  Automatic retry logic for failed requests
-  Comprehensive logging to file and console
-  Firefox WebDriver with automatic driver management

## Requirements

- **Python**: 3.8 or higher
- **Firefox Browser**: Must be installed on your system
- **Windows 11**: Tested and compatible
- **VS Code**: Recommended for development

## Installation

### Step 1: Clone or Download the Script

Save the following files to a folder:
- `psut_faq_scraper.py` (main script)
- `requirements.txt` (dependencies)
- `README.md` (this file)

### Step 2: Install Python Dependencies

Open Command Prompt or PowerShell in the script folder and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install requests beautifulsoup4 lxml selenium webdriver-manager python-dateutil
```

### Step 3: Install Firefox

If you don't have Firefox installed:
1. Download from: https://www.mozilla.org/firefox/
2. Install with default settings
3. The script will automatically download the required GeckoDriver

## Usage

### Basic Usage

Run the script from Command Prompt, PowerShell, or VS Code terminal:

```bash
python psut_faq_scraper.py
```

### What Happens During Execution

1. **Initialization**: Script sets up Firefox WebDriver in headless mode
2. **Category Extraction**: Discovers all FAQ categories on the main page
3. **Question Scraping**: Iterates through each category and extracts Q&A pairs
4. **Data Processing**: Structures data in two JSON formats
5. **File Generation**: Creates output files in the same directory

### Expected Output Files

After successful execution, you'll find:

1. **psut_faq_data.json** - Standard hierarchical format with categories
2. **psut_faq_rag_optimized.json** - RAG-optimized format for vector databases
3. **psut_scraper.log** - Detailed execution log

## Output Format Examples

### 1. Standard Format (psut_faq_data.json)

```json
{
  "scrape_metadata": {
    "scrape_date": "2026-02-01T10:30:00",
    "source_url": "https://www.psut.edu.jo/ar/faq",
    "total_categories": 4,
    "total_questions": 30,
    "language": "Arabic"
  },
  "categories": [
    {
      "category_id": 1,
      "category_name": "التسجيل",
      "category_url": "https://www.psut.edu.jo/ar/faq/...",
      "question_count": 8,
      "questions": [
        {
          "question_id": 1,
          "question": "كيف أعرف مرشدي الأكاديمي؟",
          "answer": "عندما يقوم القسم الأكاديمي بتحديد المرشد الأكاديمي، يمكنك معرفته من خلال الدخول لبوابة الطالب الإلكترونية عبر الرابط التالي (https://portal.psut.edu.jo)...",
          "category": "التسجيل"
        }
      ]
    }
  ]
}
```

### 2. RAG-Optimized Format (psut_faq_rag_optimized.json)

```json
{
  "documents": [
    {
      "id": "psut_faq_1_1",
      "text": "السؤال: كيف أعرف مرشدي الأكاديمي؟\n\nالإجابة: عندما يقوم القسم الأكاديمي بتحديد المرشد الأكاديمي...",
      "metadata": {
        "source": "PSUT FAQ",
        "category": "التسجيل",
        "category_id": 1,
        "question_id": 1,
        "question": "كيف أعرف مرشدي الأكاديمي؟",
        "answer": "عندما يقوم القسم الأكاديمي...",
        "url": "https://www.psut.edu.jo/ar/faq/...",
        "language": "ar",
        "scrape_date": "2026-02-01T10:30:00"
      }
    }
  ],
  "metadata": {
    "total_documents": 30,
    "source": "PSUT FAQ Website",
    "language": "Arabic",
    "scrape_date": "2026-02-01T10:30:00"
  }
}
```


