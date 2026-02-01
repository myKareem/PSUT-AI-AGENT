#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSUT FAQ Web Scraper - Fixed Version
Extracts Arabic FAQs from Princess Sumaya University website
Based on actual HTML structure analysis
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
import re

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psut_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set selenium logger to WARNING to reduce noise
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class PSUTFAQScraper:
    """Scraper for PSUT FAQ pages"""
    
    def __init__(self, base_url: str = "https://www.psut.edu.jo/ar/faq", delay: float = 2.0):
        """
        Initialize the scraper
        
        Args:
            base_url: Base URL for FAQ pages
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.delay = delay
        self.driver = None
        
        # Known category URLs based on actual HTML
        self.category_urls = {
            "التسجيل": "https://www.psut.edu.jo/ar/faq/faq-category2",
            "البحث العلمي": "https://www.psut.edu.jo/ar/faq/scientific-research",
            "الدراسات العليا": "https://www.psut.edu.jo/ar/faq/graduate-studies",
            "القبول": "https://www.psut.edu.jo/ar/faq/admission-2"
        }
        
    def setup_driver(self):
        """Initialize Firefox WebDriver"""
        logger.info("Setting up Firefox WebDriver...")
        
        firefox_options = Options()
        firefox_options.add_argument('--headless')
        firefox_options.set_preference('intl.accept_languages', 'ar')
        
        service = Service(GeckoDriverManager().install())
        self.driver = webdriver.Firefox(service=service, options=firefox_options)
        
        logger.info("WebDriver initialized successfully")
        
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
    
    def get_page(self, url: str, max_retries: int = 3) -> bool:
        """
        Load a page with retry logic
        
        Args:
            url: URL to load
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Loading page: {url} (Attempt {attempt}/{max_retries})")
                self.driver.get(url)
                
                # Wait for body to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                time.sleep(self.delay)
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(self.delay * attempt)
                    
        logger.error(f"Failed to load page after {max_retries} attempts: {url}")
        return False
    
    def extract_text_with_links(self, element) -> str:
        """
        Extract text from element, preserving links
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Extracted text with links
        """
        if not element:
            return ""
        
        text_parts = []
        
        for child in element.descendants:
            if child.name == 'a' and child.get('href'):
                link_text = child.get_text(strip=True)
                href = child.get('href')
                
                # Make relative URLs absolute
                if href.startswith('/'):
                    href = f"https://www.psut.edu.jo{href}"
                
                if link_text:
                    text_parts.append(f"{link_text} ({href})")
            elif isinstance(child, str):
                text = child.strip()
                if text and child.parent.name not in ['a', 'script', 'style']:
                    text_parts.append(text)
        
        # Join and clean up
        result = ' '.join(text_parts)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def extract_questions_from_category(self, category_name: str, category_url: str) -> List[Dict[str, str]]:
        """
        Extract all questions from a category page
        
        Args:
            category_name: Name of the category
            category_url: URL of the category page
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Extracting questions from: {category_name}")
        logger.info(f"URL: {category_url}")
        
        if not self.get_page(category_url):
            return []
        
        questions = []
        
        try:
            # Parse the page
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            
            # Find the accordion containing FAQ items
            accordion = soup.find('div', id='accordion1')
            
            if not accordion:
                logger.warning(f"No accordion found for {category_name}")
                return []
            
            # Find all card elements within accordion
            cards = accordion.find_all('div', class_='card')
            logger.info(f"Found {len(cards)} FAQ cards")
            
            question_id = 1
            
            for card_idx, card in enumerate(cards, 1):
                try:
                    # Find the card-header
                    card_header = card.find('div', class_='card-header')
                    if not card_header:
                        logger.warning(f"Card {card_idx}: No card-header")
                        continue
                    
                    # Find the h5 with faq-title class
                    h5_elem = card_header.find('h5', class_='faq-title')
                    if not h5_elem:
                        logger.warning(f"Card {card_idx}: No h5.faq-title")
                        continue
                    
                    question_text = h5_elem.get_text(strip=True)
                    
                    # Find the collapse div containing the answer
                    # Use the data-target attribute from h5 to find the correct collapse div
                    data_target = h5_elem.get('data-target', '')
                    if data_target:
                        # Remove the # from the target ID
                        collapse_id = data_target.replace('#', '')
                        collapse_div = card.find('div', id=collapse_id)
                    else:
                        # Fallback: find any collapse div in the card
                        collapse_div = card.find('div', class_='collapse')
                    
                    if not collapse_div:
                        logger.warning(f"Card {card_idx}: No collapse div")
                        continue
                    
                    # Find card-body inside collapse div
                    card_body = collapse_div.find('div', class_='card-body')
                    if not card_body:
                        logger.warning(f"Card {card_idx}: No card-body")
                        continue
                    
                    answer_text = self.extract_text_with_links(card_body)
                    
                    if question_text and answer_text:
                        questions.append({
                            "question_id": question_id,
                            "question": question_text,
                            "answer": answer_text,
                            "category": category_name
                        })
                        logger.info(f"✓ Q{question_id}: {question_text[:60]}...")
                        question_id += 1
                    else:
                        logger.warning(f"Card {card_idx}: Empty question or answer")
                        
                except Exception as e:
                    logger.error(f"Error processing card {card_idx}: {e}")
                    continue
            
            logger.info(f"Extracted {len(questions)} questions from {category_name}")
            
        except Exception as e:
            logger.error(f"Error extracting from {category_name}: {e}")
        
        return questions
    
    def scrape_all_faqs(self) -> Dict:
        """
        Scrape all FAQ categories
        
        Returns:
            Dictionary with all scraped data
        """
        logger.info("="*50)
        logger.info("Starting PSUT FAQ scraping...")
        logger.info("="*50)
        
        all_data = {
            "scrape_metadata": {
                "scrape_date": datetime.now().isoformat(),
                "source_url": self.base_url,
                "total_categories": len(self.category_urls),
                "total_questions": 0,
                "language": "Arabic"
            },
            "categories": []
        }
        
        try:
            self.setup_driver()
            
            for cat_idx, (cat_name, cat_url) in enumerate(self.category_urls.items(), 1):
                logger.info(f"\nProcessing category {cat_idx}/{len(self.category_urls)}: {cat_name}")
                
                questions = self.extract_questions_from_category(cat_name, cat_url)
                
                category_data = {
                    "category_id": cat_idx,
                    "category_name": cat_name,
                    "category_url": cat_url,
                    "question_count": len(questions),
                    "questions": questions
                }
                
                all_data["categories"].append(category_data)
                all_data["scrape_metadata"]["total_questions"] += len(questions)
                
                # Delay between categories
                if cat_idx < len(self.category_urls):
                    time.sleep(self.delay)
            
            logger.info("\n" + "="*50)
            logger.info("Scraping completed successfully!")
            logger.info(f"Total categories: {len(self.category_urls)}")
            logger.info(f"Total questions: {all_data['scrape_metadata']['total_questions']}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise
        finally:
            self.close_driver()
        
        return all_data
    
    def create_rag_optimized_format(self, standard_data: Dict) -> Dict:
        """
        Convert standard format to RAG-optimized format
        
        Args:
            standard_data: Data in standard format
            
        Returns:
            Data in RAG-optimized format
        """
        logger.info("Creating RAG-optimized format...")
        
        rag_data = {
            "documents": [],
            "metadata": {
                "total_documents": 0,
                "source": "PSUT FAQ Website",
                "language": "Arabic",
                "scrape_date": standard_data["scrape_metadata"]["scrape_date"]
            }
        }
        
        for category in standard_data["categories"]:
            for question in category["questions"]:
                # Combine question and answer for better RAG embedding
                combined_text = f"السؤال: {question['question']}\n\nالإجابة: {question['answer']}"
                
                doc = {
                    "id": f"psut_faq_{category['category_id']}_{question['question_id']}",
                    "text": combined_text,
                    "metadata": {
                        "source": "PSUT FAQ",
                        "category": question["category"],
                        "category_id": category["category_id"],
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "answer": question["answer"],
                        "url": category["category_url"],
                        "language": "ar",
                        "scrape_date": standard_data["scrape_metadata"]["scrape_date"]
                    }
                }
                
                rag_data["documents"].append(doc)
        
        rag_data["metadata"]["total_documents"] = len(rag_data["documents"])
        
        logger.info(f"Created {len(rag_data['documents'])} RAG documents")
        
        return rag_data
    
    def save_json(self, data: Dict, filename: str):
        """Save data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved data to: {filename}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")


def main():
    """Main execution function"""
    try:
        # Initialize scraper
        scraper = PSUTFAQScraper()
        
        # Scrape all FAQs
        standard_data = scraper.scrape_all_faqs()
        
        if standard_data["scrape_metadata"]["total_questions"] == 0:
            print("\n✗ No data was scraped. Please check the logs for errors.")
            return
        
        # Save standard format
        scraper.save_json(standard_data, "psut_faq_data.json")
        
        # Create and save RAG-optimized format
        rag_data = scraper.create_rag_optimized_format(standard_data)
        scraper.save_json(rag_data, "psut_faq_rag_optimized.json")
        
        print(f"\n✓ Successfully scraped {standard_data['scrape_metadata']['total_questions']} questions!")
        print(f"✓ Standard format: psut_faq_data.json")
        print(f"✓ RAG-optimized format: psut_faq_rag_optimized.json")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n✗ Scraping failed. Check psut_scraper.log for details.")
        raise


if __name__ == "__main__":
    main()
