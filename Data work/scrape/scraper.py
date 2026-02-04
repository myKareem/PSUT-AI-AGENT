import asyncio
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler
from urllib.parse import urljoin, urlparse
import re

async def scrape_psut_all_links():
    base_url = "https://www.psut.edu.jo/ar"
    visited_urls = set()
    scraped_data = []
    max_pages = 3000  # Adjust this based on your needs
    
    async def crawl_page(crawler, url, depth=0):
        if depth > 10 or len(visited_urls) >= max_pages:
            return
        
        if url in visited_urls:
            return
        
        # Only crawl Arabic pages from PSUT domain
        if "psut.edu.jo" not in url or "/ar" not in url:
            return
        
        visited_urls.add(url)
        print(f"Crawling ({len(visited_urls)}/{max_pages}): {url}")
        
        try:
            result = await crawler.arun(
                url=url,
                bypass_cache=True,
                word_count_threshold=10,
                #css_selector=".page-content"
            )
            
            if result.success and result.markdown:
                cleaned_content = clean_content(result.markdown)
                
                if len(cleaned_content.strip()) > 100 :
                    page_data = {
                        "url": url,
                        "content": cleaned_content,
                        "depth": depth
                    }
                    
                    scraped_data.append(page_data)
                    print(f"  -> Saved {len(cleaned_content)} characters")
                    
                    # Extract and crawl ALL internal links
                    if hasattr(result, 'links') and result.links:
                        internal_links = result.links.get("internal", [])
                        
                        for link_data in internal_links:
                            if isinstance(link_data, dict):
                                link_url = link_data.get("href", "")
                            else:
                                link_url = str(link_data)
                            
                            if link_url:
                                full_url = urljoin(base_url, link_url)
                                if full_url not in visited_urls:
                                    await asyncio.sleep(0.5)  # Rate limiting
                                    await crawl_page(crawler, full_url, depth + 1)
            else:
                print(f"  -> No content or failed")
                
        except Exception as e:
            print(f"  -> Error: {str(e)}")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        await crawl_page(crawler, base_url)
    
    # Save all scraped data
    output_dir = Path("scraped_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    with open(output_dir / "psut_all_content.json", "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
    
    # Save as individual text files
    for i, page in enumerate(scraped_data):
        filename = f"page_{i:03d}.txt"
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {page['url']}\n")
            f.write(f"Depth: {page['depth']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(page['content'])
    
    # Create summary
    summary = {
        "total_pages": len(scraped_data),
        "urls": [page['url'] for page in scraped_data],
        "total_characters": sum(len(page['content']) for page in scraped_data)
    }
    
    with open(output_dir / "scraping_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Scraping Complete!")
    print(f"Total pages scraped: {len(scraped_data)}")
    print(f"Total content: {summary['total_characters']:,} characters")
    print(f"Data saved to {output_dir}")
    print(f"{'='*80}")
    
    return scraped_data

def clean_content(markdown_text):
    """Clean and format the markdown content"""
    # Remove excessive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    # Remove navigation patterns
    cleaned = re.sub(r'\* \[.*?\]\(javascript:void.*?\)', '', cleaned)
    
    # Remove image markdown but keep alt text
    cleaned = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', cleaned)
    
    # Remove empty links
    cleaned = re.sub(r'\[\s*\]\(.*?\)', '', cleaned)
    
    # Clean up extra spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    
    return cleaned.strip()

if __name__ == "__main__":
    asyncio.run(scrape_psut_all_links())