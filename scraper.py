#!/usr/bin/env python3
"""
OpenNews Source Article Scraper

Scrapes articles from https://source.opennews.org/articles/ using pagination
and extracts full text using newspaper4k.
"""

import json
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from newspaper import Article


BASE_URL = "https://source.opennews.org"
ARTICLES_URL = f"{BASE_URL}/articles/"


def get_article_links(page_url: str) -> tuple[List[str], Optional[str]]:
    """
    Extract article links from a listing page.

    Returns:
        tuple: (list of article URLs, next page URL or None)
    """
    print(f"Fetching listing page: {page_url}")
    response = requests.get(page_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all article links from the listing
    article_links = []

    # Look for all links that point to /articles/[slug]/
    # These are the individual article pages
    import re
    for link in soup.find_all('a', href=re.compile(r'^/articles/[^/]+/$')):
        article_url = urljoin(BASE_URL, link['href'])
        # Deduplicate by using a set, but preserve order
        if article_url not in article_links:
            article_links.append(article_url)

    # Find the "Next page" link
    next_page = None
    # Try multiple patterns for pagination
    next_link = soup.find('a', string=lambda text: text and 'next' in text.lower())
    if not next_link:
        next_link = soup.find('a', rel='next')

    if next_link and next_link.get('href'):
        next_page = urljoin(page_url, next_link['href'])

    return article_links, next_page


def extract_article_metadata(soup: BeautifulSoup) -> Dict[str, any]:
    """
    Extract metadata (authors, title, date) from article page.
    """
    import re

    metadata = {
        'authors': [],
        'title': '',
        'date': ''
    }

    # Extract title
    title_elem = soup.find('h1')
    if title_elem:
        metadata['title'] = title_elem.get_text(strip=True)

    # Extract authors from <p class="article-byline"> tag
    # Authors are comma-separated with "and" before the last one
    byline = soup.find('p', class_='article-byline')

    if byline:
        # Get the text content, preserving spaces between elements
        byline_text = byline.get_text(separator=' ', strip=True)

        # Remove "By " prefix if present
        if byline_text.startswith('By '):
            byline_text = byline_text[3:]

        # Split by commas and "and" to get individual authors
        # Use regex to replace " and " with a comma for uniform splitting
        import re
        byline_text = re.sub(r'\s+and\s+', ', ', byline_text)

        # Split by comma and clean up whitespace
        authors = [name.strip() for name in byline_text.split(',') if name.strip()]

        # Remove "By" or "and" prefixes from individual author names
        authors = [name.removeprefix('By').removeprefix('and').strip() for name in authors]

        metadata['authors'] = authors

    # Extract date - look for time element or date class
    date_elem = soup.find('time')
    if date_elem:
        metadata['date'] = date_elem.get('datetime', '') or date_elem.get_text(strip=True)
    else:
        # Fallback: look for "Posted on:" text followed by date
        text = soup.get_text()
        date_match = re.search(r'Posted on:\s*([A-Z][a-z]+ \d{1,2}, \d{4})', text)
        if date_match:
            metadata['date'] = date_match.group(1)

    return metadata


def scrape_article(url: str) -> Optional[Dict[str, any]]:
    """
    Scrape a single article using newspaper4k and extract metadata.
    """
    print(f"Scraping article: {url}")

    try:
        # Use newspaper4k for article extraction
        article = Article(url)
        article.download()
        article.parse()

        # Also get the raw HTML to extract additional metadata
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract metadata from page structure
        metadata = extract_article_metadata(soup)

        # Use newspaper4k's extracted data, but prefer our metadata extraction
        article_data = {
            'url': url,
            'title': metadata['title'] or article.title,
            'authors': metadata['authors'] if metadata['authors'] else article.authors,
            'date': metadata['date'] or (article.publish_date.isoformat() if article.publish_date else ''),
            'text': article.text
        }

        return article_data

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def scrape_all_articles(max_pages: Optional[int] = None, delay: float = 1.0) -> List[Dict[str, any]]:
    """
    Scrape all articles from Source with pagination.

    Args:
        max_pages: Maximum number of listing pages to scrape (None for all)
        delay: Delay in seconds between requests

    Returns:
        List of article dictionaries
    """
    all_articles = []
    current_page = ARTICLES_URL
    page_count = 0

    while current_page:
        # Check if we've reached max pages
        if max_pages and page_count >= max_pages:
            break

        # Get article links from current page
        try:
            article_links, next_page = get_article_links(current_page)
            print(f"Found {len(article_links)} articles on page {page_count + 1}")

            # Scrape each article
            for link in article_links:
                article_data = scrape_article(link)
                if article_data:
                    all_articles.append(article_data)

                # Be polite with delays
                time.sleep(delay)

            # Move to next page
            current_page = next_page
            page_count += 1

            if current_page:
                time.sleep(delay)

        except Exception as e:
            print(f"Error processing page: {e}")
            break

    return all_articles


def save_articles(articles: List[Dict[str, any]], filename: str = "articles.json"):
    """
    Save articles to JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(articles)} articles to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape articles from OpenNews Source")
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Maximum number of listing pages to scrape')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds')
    parser.add_argument('--output', type=str, default='articles.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("Starting OpenNews Source scraper...")
    articles = scrape_all_articles(max_pages=args.max_pages, delay=args.delay)

    if articles:
        save_articles(articles, args.output)
        print(f"Successfully scraped {len(articles)} articles")
    else:
        print("No articles were scraped")
