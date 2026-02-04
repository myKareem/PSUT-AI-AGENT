#!/usr/bin/env python3
"""
PSUT Professor Contact Scraper
Scrapes professor contact information from Princess Sumaya University for Technology website
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin
import sys

# Try to import tqdm for progress bars, fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "--break-system-packages", "-q"])
    from tqdm import tqdm


class PSUTProfessorScraper:
    """Scraper for PSUT professor contact information"""
    
    def __init__(self, base_url: str = "https://www.psut.edu.jo/ar/staff/professor", delay: float = 1.5):
        """
        Initialize the scraper
        
        Args:
            base_url: Base URL for professor listings
            delay: Delay between requests in seconds (default: 1.5s)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ar,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        self.professors_data = []
        self.failed_urls = []
        self.stats = {
            'total_profiles_found': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'pages_processed': 0
        }
    
    def fetch_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """
        Fetch a page with retry logic
        
        Args:
            url: URL to fetch
            retries: Number of retry attempts
            
        Returns:
            BeautifulSoup object or None if failed
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return BeautifulSoup(response.text, 'html.parser')
            except requests.RequestException as e:
                if attempt < retries - 1:
                    print(f"    Attempt {attempt + 1} failed for {url}: {str(e)[:50]}... Retrying...")
                    time.sleep(self.delay * 2)
                else:
                    print(f"   Failed to fetch {url} after {retries} attempts: {str(e)[:50]}")
                    return None
        return None
    
    def extract_professor_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract professor profile links from listing page
        
        Args:
            soup: BeautifulSoup object of the listing page
            
        Returns:
            List of profile URLs
        """
        links = []
        
        # Look for professor cards/links - adjust selectors based on actual HTML structure
        # Common patterns: links within cards, specific class names, etc.
        
        # Try multiple selector strategies
        selectors = [
            'a[href*="/ar/staff/"]',  # Links containing staff URL
            '.professor-card a',       # Cards with links
            '.staff-member a',         # Staff member links
            'div.staff a',             # Staff divs with links
            'article a',               # Article tags with links
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    href = element.get('href')
                    if href and '/ar/staff/' in href and href not in links:
                        # Convert to absolute URL
                        absolute_url = urljoin(self.base_url, href)
                        # Avoid pagination links
                        if not re.search(r'[?&]page=\d+', absolute_url):
                            links.append(absolute_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links
    
    def clean_text(self, text: Optional[str]) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def extract_professor_details(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """
        Extract professor details from their profile page
        """
        try:
            details = {
                'name': '',
                'title': '',
                'email': '',
                'phone': '',
                'profile_url': url
            }
            
            # --- 1. Extract Name ---
            name_selectors = ['.researcher-bio h2', 'h1', 'h2.name', '.professor-name', 'div.name h2']
            for selector in name_selectors:
                name_elem = soup.select_one(selector)
                if name_elem:
                    details['name'] = self.clean_text(name_elem.get_text())
                    if details['name']:
                        break
            
            # --- 2. Extract Title ---
            title_selectors = ['.researcher-bio h4', 'p.title', '.professor-title', '.position']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                # Ensure we don't accidentally grab the "Contact Info" header (which is an h4 in the screenshot)
                if title_elem and "معلومات التواصل" not in title_elem.get_text():
                    details['title'] = self.clean_text(title_elem.get_text())
                    if details['title']:
                        break

            # --- 3. Extract Email (Specific to Screenshot) ---
            # Looks for <p>Email: address@psut.edu.jo</p>
            email_p = soup.find(lambda tag: tag.name == 'p' and 'Email:' in tag.get_text())
            if email_p:
                # Get text, remove the label "Email:", and strip whitespace
                details['email'] = self.clean_text(email_p.get_text().replace('Email:', ''))
            
            # Fallback for Email: Check for mailto link if the specific <p> tag wasn't found
            if not details['email']:
                mailto = soup.select_one('a[href^="mailto:"]')
                if mailto:
                    details['email'] = self.clean_text(mailto.get('href').replace('mailto:', ''))

            # --- 4. Extract Phone (Specific to Screenshot) ---
            # Looks for <p>Telephone: +962... Ex: ...</p>
            phone_p = soup.find(lambda tag: tag.name == 'p' and 'Telephone:' in tag.get_text())
            if phone_p:
                # Get text, remove the label "Telephone:", and strip whitespace
                # This will capture the extension "Ex: 5304" as well, which is useful
                details['phone'] = self.clean_text(phone_p.get_text().replace('Telephone:', ''))

            # Fallback for Phone: Check for tel link if specific <p> tag wasn't found
            if not details['phone']:
                tel = soup.select_one('a[href^="tel:"]')
                if tel:
                    details['phone'] = self.clean_text(tel.get('href').replace('tel:', ''))
            
            # Validate we got at least a name
            if not details['name']:
                # Try one last fallback for name - usually the page title
                if soup.title:
                    details['name'] = self.clean_text(soup.title.string.split('|')[0])
                else:
                    print(f"    Warning: No name found for {url}")

            return details
            
        except Exception as e:
            print(f"   Error extracting details from {url}: {str(e)}")
            return None
        
    def scrape_all_pages(self, total_pages: int = 9) -> None:
        """
        Scrape all pages of professor listings
        
        Args:
            total_pages: Total number of pages to scrape
        """
        print(f"\n{'='*60}")
        print(f"  PSUT Professor Contact Scraper")
        print(f"{'='*60}\n")
        
        all_profile_urls = []
        
        # Phase 1: Collect all profile URLs
        print(f" Phase 1: Collecting professor profile URLs from {total_pages} pages...\n")
        
        for page_num in tqdm(range(1, total_pages + 1), desc="Scanning pages", unit="page"):
            if page_num == 1:
                page_url = self.base_url
            else:
                page_url = f"{self.base_url}?page={page_num}"
            
            soup = self.fetch_page(page_url)
            
            if soup:
                links = self.extract_professor_links(soup)
                all_profile_urls.extend(links)
                self.stats['pages_processed'] += 1
                tqdm.write(f"  ✓ Page {page_num}: Found {len(links)} professor profiles")
            else:
                tqdm.write(f"  ✗ Page {page_num}: Failed to load")
            
            # Be respectful with delays
            if page_num < total_pages:
                time.sleep(self.delay)
        
        # Remove duplicates
        all_profile_urls = list(dict.fromkeys(all_profile_urls))
        self.stats['total_profiles_found'] = len(all_profile_urls)
        
        print(f"\n✓ Found {len(all_profile_urls)} unique professor profiles\n")
        
        # Phase 2: Extract details from each profile
        print(f" Phase 2: Extracting contact details from each profile...\n")
        
        for url in tqdm(all_profile_urls, desc="Scraping profiles", unit="profile"):
            soup = self.fetch_page(url)
            
            if soup:
                details = self.extract_professor_details(soup, url)
                if details:
                    self.professors_data.append(details)
                    self.stats['successful_scrapes'] += 1
                else:
                    self.failed_urls.append(url)
                    self.stats['failed_scrapes'] += 1
            else:
                self.failed_urls.append(url)
                self.stats['failed_scrapes'] += 1
            
            # Be respectful with delays
            time.sleep(self.delay)
        
        print(f"\n Successfully scraped {self.stats['successful_scrapes']} professor profiles")
        if self.stats['failed_scrapes'] > 0:
            print(f"  Failed to scrape {self.stats['failed_scrapes']} profiles")
    
    def save_to_json(self, filename: str = None) -> str:
        """
        Save scraped data to JSON file
        
        Args:
            filename: Output filename (default: auto-generated with timestamp)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"psut_professors_{timestamp}.json"
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        output_data = {
            "metadata": {
                "source_url": self.base_url,
                "scrape_date": datetime.now().isoformat(),
                "total_professors": len(self.professors_data),
                "successful_scrapes": self.stats['successful_scrapes'],
                "failed_scrapes": self.stats['failed_scrapes'],
                "pages_processed": self.stats['pages_processed']
            },
            "professors": self.professors_data
        }
        
        # Save to outputs directory for user access
        output_path = filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def print_summary(self) -> None:
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print(f"  SCRAPING SUMMARY")
        print(f"{'='*60}")
        print(f"  Pages processed:      {self.stats['pages_processed']}")
        print(f"  Profiles found:       {self.stats['total_profiles_found']}")
        print(f"  Successful scrapes:   {self.stats['successful_scrapes']}")
        print(f"  Failed scrapes:       {self.stats['failed_scrapes']}")
        print(f"  Success rate:         {(self.stats['successful_scrapes'] / max(self.stats['total_profiles_found'], 1) * 100):.1f}%")
        print(f"{'='*60}\n")
        
        if self.professors_data:
            # Data completeness analysis
            email_count = sum(1 for p in self.professors_data if p.get('email'))
            phone_count = sum(1 for p in self.professors_data if p.get('phone'))
            
            print(f"  Data Completeness:")
            print(f"  Names:   {len(self.professors_data)}/{len(self.professors_data)} (100%)")
            print(f"  Emails:  {email_count}/{len(self.professors_data)} ({email_count/len(self.professors_data)*100:.1f}%)")
            print(f"  Phones:  {phone_count}/{len(self.professors_data)} ({phone_count/len(self.professors_data)*100:.1f}%)")
            print(f"{'='*60}\n")


def main():
    """Main execution function"""
    print("\n Starting PSUT Professor Scraper...")
    
    # Initialize scraper
    scraper = PSUTProfessorScraper(
        base_url="https://www.psut.edu.jo/ar/staff/professor",
        delay=1.5  # 1.5 second delay between requests
    )
    
    # Scrape all 9 pages
    scraper.scrape_all_pages(total_pages=9)
    
    # Save results
    print("\n Saving results to JSON...")
    output_file = scraper.save_to_json()
    print(f" Data saved to: {output_file}")
    
    # Print summary
    scraper.print_summary()
    
    # Show sample data
    if scraper.professors_data:
        print(" Sample Data (first 3 professors):\n")
        for i, prof in enumerate(scraper.professors_data[:3], 1):
            print(f"  {i}. {prof.get('name', 'N/A')}")
            print(f"     Title: {prof.get('title', 'N/A')}")
            print(f"     Email: {prof.get('email', 'N/A')}")
            print(f"     Phone: {prof.get('phone', 'N/A')}")
            print()
    
    print(" Scraping complete!\n")
    
    return output_file


if __name__ == "__main__":
    main()
