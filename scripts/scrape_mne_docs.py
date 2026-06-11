#!/usr/bin/env python3
"""
MNE-Python Documentation Scraper for EEG-ADK Multi-Agent RAG
This script crawls the stable MNE-Python documentation (https://mne.tools/stable/),
extracts clean text (removing sidebar, headers, footers), and saves them
as .txt files in the `rag_docs` directory for ChromaDB ingestion.
"""

import os
import re
import sys
import argparse
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.logging_config import setup_logging

logger = logging.getLogger("eeg_agent.scrape")

# Seed URLs for crawling
API_REF_URL = "https://mne.tools/stable/api/python_reference.html"
TUTORIALS_URL = "https://mne.tools/stable/auto_tutorials/index.html"

# Output directory relative to the project root
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_docs", "mne_python_docs"))

def get_links(url, pattern_func):
    """
    Fetches the URL, parses it, and extracts all links that match pattern_func.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error("Error fetching seed URL %s: %s", url, e)
        return set()

    soup = BeautifulSoup(response.content, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        # Resolve relative links to absolute URLs
        abs_url = urljoin(url, a['href']).split('#')[0]
        if pattern_func(abs_url):
            links.add(abs_url)
    return links

def clean_and_extract_text(html_content):
    """
    Parses HTML content, targets the main PyData Sphinx content container,
    removes nav/sidebar/footer elements, and returns clean text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to find the PyData Sphinx theme content div
    content_div = soup.find('div', class_='bd-content')
    if not content_div:
        # Fallbacks
        content_div = soup.find('main', id='main-content') or soup.find('article') or soup.find('body')
        
    if not content_div:
        return ""

    # Remove navigation, headers, footers, scripts, and sidebar models to keep only core content
    unwanted_selectors = [
        '.bd-header-article',
        '.prev-next-footer',
        '.bd-footer-content',
        'dialog',
        '.bd-sidebar-secondary',
        'script',
        'style',
        'footer',
        'nav'
    ]
    for selector in unwanted_selectors:
        for el in content_div.select(selector):
            el.decompose()

    # Extract text with space separators to prevent token merging
    text = content_div.get_text(' ')
    
    # Clean up whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def scrape_page(url, output_dir):
    """
    Scrapes a single documentation page and writes it to a file.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return url, False, f"HTTP Error: {e}"

    text = clean_and_extract_text(response.content)
    if not text:
        return url, False, "No text content extracted."

    # Generate a descriptive filename
    # E.g. https://mne.tools/stable/generated/mne.io.read_raw_fif.html -> mne.io.read_raw_fif.txt
    # E.g. https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html -> tutorial_epochs_10_epochs_overview.txt
    filename = ""
    if "/generated/" in url:
        # API Page
        filename = url.split('/')[-1].replace('.html', '.txt')
    elif "/auto_tutorials/" in url:
        # Tutorial Page
        parts = url.split('/auto_tutorials/')[-1].split('/')
        filename = "tutorial_" + "_".join(parts).replace('.html', '.txt')
    else:
        # General Page
        filename = "doc_" + url.split('/')[-1].replace('.html', '.txt')

    # Fallback for empty filenames
    if not filename or filename == ".txt":
        filename = f"scraped_{hash(url)}.txt"

    file_path = os.path.join(output_dir, filename)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Prepend metadata at the top for RAG context
            f.write(f"Title: MNE-Python Documentation Page\n")
            f.write(f"Source URL: {url}\n")
            f.write("-" * 80 + "\n\n")
            f.write(text)
        return url, True, filename
    except Exception as e:
        return url, False, f"File Write Error: {e}"

def main():
    load_dotenv(override=True)
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Scrape stable MNE-Python documentation for RAG database.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save scraped text files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pages to scrape (for testing).")
    parser.add_argument("--threads", type=int, default=15, help="Number of concurrent scraper threads.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory initialized at: %s", args.output_dir)

    # Gather API Reference Links
    logger.info("Collecting API Reference pages...")
    is_api_link = lambda u: u.startswith("https://mne.tools/stable/generated/mne.") and u.endswith(".html")
    api_links = get_links(API_REF_URL, is_api_link)
    logger.info("Found %d API reference pages.", len(api_links))

    # Gather Tutorial Links
    logger.info("Collecting Tutorial pages...")
    is_tutorial_link = lambda u: u.startswith("https://mne.tools/stable/auto_tutorials/") and u.endswith(".html") and "/index.html" not in u
    tutorial_links = get_links(TUTORIALS_URL, is_tutorial_link)
    logger.info("Found %d tutorial pages.", len(tutorial_links))

    # Combine all pages
    all_urls = sorted(list(api_links.union(tutorial_links)))
    total_found = len(all_urls)
    
    if args.limit:
        all_urls = all_urls[:args.limit]
        logger.info("Limit option set. Scrape list limited to first %d pages out of %d.", len(all_urls), total_found)
    else:
        logger.info("Scraping all %d pages.", len(all_urls))

    if not all_urls:
        logger.error("No URLs collected. Exiting.")
        sys.exit(1)

    logger.info("Starting crawl with %d threads...", args.threads)
    
    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(scrape_page, url, args.output_dir): url for url in all_urls}
        
        for i, future in enumerate(as_completed(futures), 1):
            url, success, msg = future.result()
            if success:
                success_count += 1
                # Show simple periodic progress
                if i % 25 == 0 or i == len(all_urls):
                    logger.info("Progress: [%d/%d] Scraped %s", i, len(all_urls), msg)
            else:
                failure_count += 1
                logger.error("Failed to scrape %s: %s", url, msg)

    logger.info("--- Scraping Complete ---")
    logger.info("Successfully scraped: %d pages", success_count)
    logger.info("Failed to scrape:     %d pages", failure_count)
    logger.info("Documents saved in:   %s", args.output_dir)
    logger.info("You can now populate your vector database by running:")
    logger.info("  python scripts/ingest_rag_data.py --category api")

if __name__ == "__main__":
    from dotenv import load_dotenv
    main()
