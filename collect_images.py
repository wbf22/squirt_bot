"""
Downloads the first N images from the top Wikipedia article for a keyword.

Usage:
    python collect_images.py <keyword> [output_dir] [count]
"""

import os
import sys
import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "squirt_bot/1.0 (data collection script)"}


def search_top_page(keyword: str) -> str | None:
    """Return the title of the top Wikipedia search result."""
    resp = requests.get(WIKI_API, headers=HEADERS, params={
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "format": "json",
    }, timeout=10)
    resp.raise_for_status()
    results = resp.json()["query"]["search"]
    return results[0]["title"] if results else None


def get_image_filenames(page_title: str) -> list[str]:
    """Return all File: names embedded in a Wikipedia page."""
    resp = requests.get(WIKI_API, headers=HEADERS, params={
        "action": "query",
        "titles": page_title,
        "prop": "images",
        "imlimit": 50,
        "format": "json",
    }, timeout=10)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return [img["title"] for img in page.get("images", [])]


def resolve_image_urls(filenames: list[str]) -> list[str]:
    """Resolve File: titles to direct download URLs, skipping SVGs and icons."""
    urls = []
    for filename in filenames:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "gif", "webp"):
            continue
        resp = requests.get(WIKI_API, headers=HEADERS, params={
            "action": "query",
            "titles": filename,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }, timeout=10)
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        info = page.get("imageinfo", [])
        if info:
            urls.append(info[0]["url"])
    return urls


def fetch_images(keyword: str, output_dir: str, count: int = 3):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching Wikipedia for '{keyword}'...")
    page_title = search_top_page(keyword)
    if not page_title:
        print("No Wikipedia article found.")
        return
    print(f"  Top result: {page_title}")

    filenames = get_image_filenames(page_title)
    urls = resolve_image_urls(filenames)[:count]
    if not urls:
        print("No downloadable images found on that page.")
        return

    for i, url in enumerate(urls, start=1):
        ext = url.rsplit(".", 1)[-1].lower()
        path = os.path.join(output_dir, f"{i:06d}.{ext}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            print(f"  [{i}/{count}] saved -> {path}")
        except Exception as e:
            print(f"  [{i}/{count}] failed ({url}): {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_images.py <keyword> [output_dir] [count]")
        sys.exit(1)

    keyword = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"images/{keyword.replace(' ', '_')}"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"Fetching {count} images for '{keyword}' -> {output_dir}/")
    fetch_images(keyword, output_dir, count)
