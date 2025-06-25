"""Web scraper for wasmCloud documentation."""

import hashlib
import logging
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import re
import time

logger = logging.getLogger(__name__)


class WasmCloudDocsScraper:
    """Scraper for wasmCloud documentation."""
    
    def __init__(self, base_url: str = "https://wasmcloud.com/docs/"):
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'wasmCloud-RAG-Bot/1.0'})
        self.session.timeout = 30
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
    
    def _is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is a valid documentation URL."""
        parsed = urlparse(url)
        return (
            parsed.netloc == "wasmcloud.com" and
            parsed.path.startswith("/docs/") and
            not parsed.fragment and  # Ignore anchor links
            url not in self.visited_urls
        )
    
    def _extract_content(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract title and main content from HTML."""
        # Extract title
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Untitled"
        
        # Remove navigation, header, footer, and other non-content elements
        for elem in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            elem.decompose()
        
        # Find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if not main_content:
            # Fallback to body content
            main_content = soup.find('body')
        
        if main_content:
            # Clean up the content
            content = self._clean_content(main_content.get_text())
        else:
            content = ""
        
        return {
            'title': title,
            'content': content
        }
    
    def _clean_content(self, text: str) -> str:
        """Clean and normalize extracted text content."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove common navigation text
        text = re.sub(r'Skip to main content.*?(?=\w)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Edit this page.*?$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text.strip()
    
    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all documentation links from the page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)
            
            if self._is_valid_doc_url(absolute_url):
                links.append(absolute_url)
        
        return links
    
    def _scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """Scrape a single page."""
        try:
            logger.info(f"Scraping: {url}")
            
            response = self.session.get(url)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content_data = self._extract_content(soup)
            
            # Skip if no meaningful content
            if len(content_data['content']) < 100:
                logger.info(f"Skipping {url}: insufficient content")
                return None
            
            # Add metadata
            content_data.update({
                'url': url,
                'content_hash': hashlib.md5(content_data['content'].encode()).hexdigest()
            })
            
            # Small delay to be respectful
            time.sleep(0.5)
            
            return content_data
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _discover_urls(self, start_url: str, max_depth: int = 3) -> List[str]:
        """Discover all documentation URLs starting from a base URL."""
        urls_to_visit = [start_url]
        discovered_urls = set([start_url])
        current_depth = 0
        
        while urls_to_visit and current_depth < max_depth:
            current_batch = urls_to_visit.copy()
            urls_to_visit.clear()
            
            for url in current_batch:
                if url in self.visited_urls:
                    continue
                
                try:
                    response = self.session.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        links = self._extract_links(soup, url)
                        
                        for link in links:
                            if link not in discovered_urls:
                                discovered_urls.add(link)
                                urls_to_visit.append(link)
                
                except Exception as e:
                    logger.error(f"Error discovering URLs from {url}: {e}")
                
                self.visited_urls.add(url)
                time.sleep(0.2)  # Small delay between requests
            
            current_depth += 1
            logger.info(f"Discovered {len(discovered_urls)} URLs at depth {current_depth}")
        
        return list(discovered_urls)
    
    def scrape_all_docs(self, max_pages: int = 100) -> List[Dict[str, str]]:
        """Scrape all wasmCloud documentation pages."""
        logger.info("Starting wasmCloud documentation scraping...")
        
        # Discover all documentation URLs
        all_urls = self._discover_urls(self.base_url)
        
        # Limit the number of pages
        urls_to_scrape = all_urls[:max_pages]
        logger.info(f"Found {len(all_urls)} URLs, scraping first {len(urls_to_scrape)}")
        
        # Scrape pages sequentially (to be respectful to the server)
        documents = []
        for i, url in enumerate(urls_to_scrape, 1):
            logger.info(f"Scraping page {i}/{len(urls_to_scrape)}")
            result = self._scrape_page(url)
            
            if result:
                documents.append(result)
        
        logger.info(f"Successfully scraped {len(documents)} documents")
        return documents


def scrape_wasmcloud_docs() -> List[Dict[str, str]]:
    """Convenience function to scrape wasmCloud docs."""
    with WasmCloudDocsScraper() as scraper:
        return scraper.scrape_all_docs() 