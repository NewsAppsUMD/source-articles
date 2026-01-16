# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python web scraper that extracts full-text articles from OpenNews Source (https://source.opennews.org/articles/). The scraper handles pagination, extracts article metadata (title, authors, date), and uses newspaper4k for full-text extraction.

## Development Commands

### Setup
```bash
# Install dependencies using uv
uv sync
```

### Running the Scraper
```bash
# Basic usage (scrapes all articles)
uv run python scraper.py

# Limit to first N pages
uv run python scraper.py --max-pages 2

# Adjust request delay (default: 1 second)
uv run python scraper.py --delay 2.0

# Specify output file
uv run python scraper.py --output my_articles.json
```

## Architecture

### Single-File Design
The entire scraper is contained in `scraper.py` with clear functional separation:

**Pagination Layer** (`get_article_links`):
- Fetches article listing pages
- Extracts article URLs using regex pattern `/articles/[slug]/`
- Finds next page links for pagination
- Returns tuple of (article_links, next_page_url)

**Metadata Extraction** (`extract_article_metadata`):
- Parses HTML structure using BeautifulSoup
- Extracts title from `<h1>` tag
- Parses authors from `<p class="article-byline">` (handles "By Author1, Author2 and Author3" format)
- Extracts date from `<time>` element or "Posted on:" text

**Article Scraping** (`scrape_article`):
- Uses newspaper4k's `Article` class for main text extraction
- Falls back to manual metadata extraction for better accuracy
- Combines both approaches: newspaper4k for text, BeautifulSoup for metadata

**Orchestration** (`scrape_all_articles`):
- Manages pagination loop
- Enforces configurable delays between requests
- Respects `--max-pages` limit
- Handles errors gracefully without stopping entire scrape

### Data Flow
1. Start at ARTICLES_URL listing page
2. Extract all article URLs from current page
3. For each article: download, parse metadata, extract text
4. Find next page link and repeat
5. Save all articles to JSON

### Output Format
JSON array with structure:
```json
{
  "url": "string",
  "title": "string",
  "authors": ["string"],
  "date": "string",
  "text": "string"
}
```

## Key Implementation Details

**Metadata Priority**: Custom BeautifulSoup extraction is preferred over newspaper4k's metadata because Source's HTML structure provides more reliable author/date information.

**Deduplication**: Article links are deduplicated while preserving order (using list membership check, not set).

**Politeness**: Default 1-second delay between requests; configurable via `--delay` flag.

**Error Handling**: Individual article scraping errors don't stop the entire process; errors are logged and skipped.

## Dependencies

- `beautifulsoup4`: HTML parsing for metadata extraction
- `newspaper4k`: Full-text article extraction
- `requests`: HTTP client for fetching pages
