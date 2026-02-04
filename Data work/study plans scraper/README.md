PSUT Study Plans Scraper

This tool automates the process of collecting academic program data from the PSUT website. It scrapes program details, admission requirements, metadata (prices, hours), and downloads the official Study Plan PDFs for each specialization.

Features

Comprehensive Data Collection: Extracts program names, schools, credit hours, and prices.

Detailed Markdown Generation: Creates a structured .md file for each program containing introduction text, detailed descriptions, and admission requirements.

PDF Downloader: Automatically locates and downloads the "Study Plan" (Curriculum) PDF for each program.

Robust Handling: Handles Arabic text, sanitizes filenames for Windows/Linux compatibility, and includes logic to bypass common scraping blockers.

Prerequisites

Python 3.7 or higher

An active internet connection

Installation

Clone or Download this repository to your local machine.

Create a Virtual Environment (Recommended):

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate



Install Dependencies:
Run the following command to install the required libraries:

pip install -r requirements.txt



Usage

Run the general scraper script:

python study_plans_scraper.py



Output

The script will create two folders in the project directory:

psut_combined_data/: Contains markdown (.md) files for each program.

psut_combined_pdfs/: Contains the downloaded PDF study plans.

Troubleshooting

Permission Errors: Ensure you have write permissions in the folder where the script is running. Close any files (like PDFs) if you have them open while the script tries to overwrite them.

Connection Errors: If the script stops due to a connection error, simply run it again; it will overwrite existing files or create missing ones.
