# OpenNews Source Article Scraper

A Python scraper that extracts full-text articles from [OpenNews Source](https://source.opennews.org/articles/) using pagination and the newspaper4k library.

## Features

- Automatic pagination through all article listing pages
- Extracts article metadata: title, authors (as array), publication date
- Full article text extraction using newspaper4k
- Configurable page limits and request delays for politeness
- Exports to JSON format

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

## Usage

### Basic usage (scrape all articles)

```bash
uv run python scraper.py
```

### With options

```bash
# Scrape only the first 2 pages
uv run python scraper.py --max-pages 2

# Adjust delay between requests (default: 1 second)
uv run python scraper.py --delay 2.0

# Specify output file
uv run python scraper.py --output my_articles.json
```

### All options

```bash
uv run python scraper.py --max-pages <number> --delay <seconds> --output <filename>
```

## Output Format

Articles are saved as a JSON array with the following structure:

```json
[
  {
    "url": "https://source.opennews.org/articles/article-slug/",
    "title": "Article Title",
    "authors": ["Author Name 1", "Author Name 2"],
    "date": "2024-01-15",
    "text": "Full article text..."
  }
]
```
