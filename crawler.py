"""
Website crawler for the Neural Rays RAG chatbot.

This script crawls internal pages from the Neural Rays website, extracts readable
text content, and saves the results to data/pages.json for the RAG ingestion step.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


# Main website URL to start crawling from
BASE_URL = "https://neuralrays.ai/"

# File where extracted website content will be saved
OUTPUT_FILE = Path("data/pages.json")

# Crawler settings
MAX_PAGES = 50
REQUEST_TIMEOUT_SECONDS = 10
CRAWL_DELAY_SECONDS = 0.5

# Only crawl pages from these domains
ALLOWED_DOMAINS = {"neuralrays.ai", "www.neuralrays.ai"}

# Skip files that are not useful for text-based chatbot knowledge
SKIPPED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".css",
    ".js",
}

# User agent helps identify the crawler when making website requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 NeuralRaysRAGBot/1.0"
}

IGNORED_TEXT_LINES = {
    "Skip to content",
    "Home",
    "About Us",
    "AI Services",
    "Digital Services",
    "Success Stories",
    "Contact Us",
    "Follow Us",
    "Request a Quote",
    "×",
}


@dataclass
class CrawledPage:
    """
    Stores the useful information extracted from one website page.
    """

    url: str
    title: str
    text: str


def setup_logging() -> None:
    """
    Set up simple logging so we can see crawler progress in the terminal.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def normalise_url(url: str) -> str:
    """
    Clean a URL so duplicate versions of the same page are treated as one.

    Example:
    https://neuralrays.ai/about-us/#section
    becomes:
    https://neuralrays.ai/about-us
    """

    parsed_url = urlparse(url)

    # Remove fragments and query parameters
    cleaned_url = parsed_url._replace(
        fragment="",
        query="",
    )

    normalised_url = urlunparse(cleaned_url)

    # Remove trailing slash to avoid duplicate URLs
    if normalised_url.endswith("/"):
        normalised_url = normalised_url[:-1]

    return normalised_url


def is_internal_url(url: str) -> bool:
    """
    Check whether a URL belongs to the Neural Rays website.
    """

    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        return False

    return parsed_url.netloc in ALLOWED_DOMAINS


def has_skipped_extension(url: str) -> bool:
    """
    Check whether the URL points to a file type we do not want to crawl.
    """

    path = urlparse(url).path.lower()

    return any(path.endswith(extension) for extension in SKIPPED_EXTENSIONS)


def should_crawl_url(url: str) -> bool:
    """
    Decide whether a URL should be added to the crawl queue.
    """

    return is_internal_url(url) and not has_skipped_extension(url)


def extract_page_title(soup: BeautifulSoup) -> str:
    """
    Extract the page title from the HTML.

    It first tries the <title> tag, then falls back to the first <h1>.
    """

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    heading = soup.find("h1")

    if heading:
        return heading.get_text(strip=True)

    return "Untitled Page"


def extract_clean_text(soup: BeautifulSoup) -> str:
    """
    Extract clean readable text from the page.

    This removes scripts, styles, forms, navigation-like repeated text,
    and other elements that are not useful for the chatbot knowledge base.
    """

    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "iframe",
        "canvas",
        "nav",
    ]

    # Remove unwanted HTML elements before extracting text
    for tag in soup(unwanted_tags):
        tag.decompose()

    text = soup.get_text(separator="\n")

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line in IGNORED_TEXT_LINES:
            continue

        if line.startswith("Copyright ©"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_internal_links(soup: BeautifulSoup, current_url: str) -> set[str]:
    """
    Find all internal links on the current page.
    """

    links: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        # Ignore links that do not point to normal web pages
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        # Convert relative links into full URLs
        absolute_url = urljoin(current_url, href)
        normalised_url = normalise_url(absolute_url)

        if should_crawl_url(normalised_url):
            links.add(normalised_url)

    return links


def fetch_page(session: requests.Session, url: str) -> BeautifulSoup | None:
    """
    Download a page and return a BeautifulSoup object.

    If the request fails or the response is not HTML, return None.
    """

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        logging.warning("Failed to fetch %s: %s", url, error)
        return None

    content_type = response.headers.get("Content-Type", "")

    # Only process HTML pages
    if "text/html" not in content_type:
        logging.info("Skipping non-HTML page: %s", url)
        return None

    return BeautifulSoup(response.text, "html.parser")


def add_new_links_to_queue(
    links: Iterable[str],
    urls_to_visit: list[str],
    visited_urls: set[str],
) -> None:
    """
    Add newly discovered links to the crawl queue.
    """

    for link in sorted(links):
        if link not in visited_urls and link not in urls_to_visit:
            urls_to_visit.append(link)


def crawl_site(start_url: str, max_pages: int = MAX_PAGES) -> list[CrawledPage]:
    """
    Main crawling function.

    It visits pages one by one, extracts content, finds more internal links,
    and stops when there are no more links or the max page limit is reached.
    """

    start_url = normalise_url(start_url)

    pages: list[CrawledPage] = []
    visited_urls: set[str] = set()
    urls_to_visit: list[str] = [start_url]

    # Reuse one session for all requests
    with requests.Session() as session:
        session.headers.update(HEADERS)

        while urls_to_visit and len(visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)

            if current_url in visited_urls:
                continue

            logging.info("Crawling: %s", current_url)

            soup = fetch_page(session, current_url)
            visited_urls.add(current_url)

            if soup is None:
                continue

            title = extract_page_title(soup)
            text = extract_clean_text(soup)

            # Only save pages with enough useful text
            if len(text) > 200:
                pages.append(
                    CrawledPage(
                        url=current_url,
                        title=title,
                        text=text,
                    )
                )

            # Find more internal links to crawl
            new_links = extract_internal_links(soup, current_url)

            add_new_links_to_queue(
                links=new_links,
                urls_to_visit=urls_to_visit,
                visited_urls=visited_urls,
            )

            # Small delay to avoid sending too many requests too quickly
            time.sleep(CRAWL_DELAY_SECONDS)

    return pages

def create_text_fingerprint(text: str) -> str:
    """
    Create a simple fingerprint of page text to identify duplicate pages.
    """

    return " ".join(text.lower().split())


def deduplicate_pages(pages: list[CrawledPage]) -> list[CrawledPage]:
    """
    Remove pages that have identical or near-identical text content.
    """

    seen_fingerprints: set[str] = set()
    unique_pages: list[CrawledPage] = []

    for page in pages:
        fingerprint = create_text_fingerprint(page.text)

        if fingerprint in seen_fingerprints:
            logging.info("Skipping duplicate page: %s", page.url)
            continue

        seen_fingerprints.add(fingerprint)
        unique_pages.append(page)

    return unique_pages

def save_pages(pages: list[CrawledPage], output_file: Path = OUTPUT_FILE) -> None:
    """
    Save all crawled pages to a JSON file.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    page_records = [asdict(page) for page in pages]

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(page_records, file, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Run the website crawler.
    """

    setup_logging()

    pages = crawl_site(BASE_URL)
    unique_pages = deduplicate_pages(pages)

    save_pages(unique_pages)

    logging.info(
        "Saved %s unique pages to %s",
        len(unique_pages),
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()