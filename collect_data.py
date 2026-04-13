"""
Interactive data collection for TargetDetector training.

Picks a random keyword, fetches 3 images from Wikipedia, then prompts you
to draw a mask on each:

  Image 1 — target:      mask drawn → everything outside zeroed → saved as target.jpg
  Image 2 — anti-target: mask drawn → everything outside zeroed → saved as anti_target.jpg
  Image 3 — result:      mask drawn → image saved as-is (result.jpg) + mask (result_mask.png)

Controls:
  Click         add polygon vertex
  Click 1st pt  close the polygon
  Enter         confirm and move to next image

Usage:
    python collect_data.py [output_dir]
"""

import io
import os
import sys
import random
import datetime
import time

import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path


# --- Colors ------------------------------------------------------------------

R = "\033[0m"           # reset
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

def info(msg):    print(f"{CYAN}{msg}{R}")
def dim(msg):     print(f"{DIM}{msg}{R}")
def success(msg): print(f"{GREEN}{msg}{R}")
def warn(msg):    print(f"{YELLOW}{msg}{R}")
def error(msg):   print(f"{RED}{msg}{R}")
def header(msg):  print(f"\n{BOLD}{CYAN}{msg}{R}")


# --- Wikipedia fetch (same as collect_images.py) ----------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "squirt_bot/1.0 (data collection script)"}

KEYWORDS = [
    "cat", "dog", "bird", "rabbit", "squirrel", "raccoon",
    "crow", "pigeon", "sparrow", "duck", "chicken", "goose",
]

def _search_top_page(keyword: str) -> str | None:
    resp = requests.get(WIKI_API, headers=HEADERS, params={
        "action": "query", "list": "search",
        "srsearch": keyword, "format": "json",
    }, timeout=10)
    resp.raise_for_status()
    results = resp.json()["query"]["search"]
    return results[0]["title"] if results else None


def _get_image_filenames(page_title: str) -> list[str]:
    resp = requests.get(WIKI_API, headers=HEADERS, params={
        "action": "query", "titles": page_title,
        "prop": "images", "imlimit": 50, "format": "json",
    }, timeout=10)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return [img["title"] for img in page.get("images", [])]


def _resolve_image_urls(filenames: list[str]) -> list[str]:
    urls = []
    for filename in filenames:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "gif", "webp"):
            continue
        resp = requests.get(WIKI_API, headers=HEADERS, params={
            "action": "query", "titles": filename,
            "prop": "imageinfo", "iiprop": "url", "format": "json",
        }, timeout=10)
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        info = page.get("imageinfo", [])
        if info:
            urls.append(info[0]["url"])
    return urls


def get_image_urls(keyword: str) -> list[str]:
    """Resolve all image URLs from the top Wikipedia article (no downloads)."""
    page_title = _search_top_page(keyword)
    if not page_title:
        return []
    info(f"  Wikipedia article: {page_title}")
    filenames = _get_image_filenames(page_title)
    urls = _resolve_image_urls(filenames)
    dim(f"  Found {len(urls)} images")
    return urls


def download_image(url: str) -> np.ndarray | None:
    """Download a single image and return it as a numpy array."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    except Exception as e:
        warn(f"  Failed ({e})\n trying again after a 15 second sleep ...")
        time.sleep(15)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
        except Exception as e:
            print(f"  Skipped ({e})")
            return None


# --- Mask drawing ------------------------------------------------------------

def draw_mask(
    image_array: np.ndarray,
    title: str,
    get_replacement=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Display image_array and let the user draw a polygon mask.

    Left-click:   add vertex (polygon draws live)
    Middle-click: swap in the next image from the pool (calls get_replacement)
    Right-click:  confirm mask and move on

    Returns (final_image_array, mask) where mask is (H x W, uint8) with 1 inside polygon.
    """
    current = [image_array]
    vertices = []

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(current[0])
    line, = ax.plot([], [], "r-o", linewidth=2, markersize=6)
    fig.tight_layout()

    def _set_title():
        extra = "  |  Middle-click: next image" if get_replacement else ""
        ax.set_title(f"{title}\nLeft-click: add vertex  |  Right-click: confirm{extra}")

    _set_title()

    def _redraw_poly():
        if not vertices:
            return
        xs = [v[0] for v in vertices] + [vertices[0][0]]
        ys = [v[1] for v in vertices] + [vertices[0][1]]
        line.set_data(xs, ys)
        fig.canvas.draw_idle()

    def _onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:       # left-click → add vertex
            vertices.append((event.xdata, event.ydata))
            _redraw_poly()
        elif event.button == 2:     # middle-click → swap image
            if get_replacement is None:
                return
            new_img = get_replacement()
            if new_img is not None:
                current[0] = new_img
                vertices.clear()
                im.set_data(new_img)
                im.set_extent([-0.5, new_img.shape[1] - 0.5, new_img.shape[0] - 0.5, -0.5])
                line.set_data([], [])
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
        elif event.button == 3:     # right-click → done
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _onclick)
    plt.show()

    h, w = current[0].shape[:2]
    if len(vertices) < 3:
        warn("  (fewer than 3 vertices — using full image as mask)")
        return current[0], np.ones((h, w), dtype=np.uint8)

    path = Path(vertices)
    y_idx, x_idx = np.mgrid[:h, :w]
    points = np.column_stack([x_idx.ravel(), y_idx.ravel()])
    mask = path.contains_points(points).reshape(h, w).astype(np.uint8)
    return current[0], mask


# --- Main data collection loop -----------------------------------------------

def collect_sample(keyword: str, output_dir: str):
    header(f"Looking up '{keyword}' on Wikipedia...")
    urls = get_image_urls(keyword)

    # Shared cursor — each call to next_image() advances it, whether it's the
    # initial load or a middle-click swap
    cursor = [0]

    def next_image() -> np.ndarray | None:
        while cursor[0] < len(urls):
            url = urls[cursor[0]]
            cursor[0] += 1
            img = download_image(url)
            if img is not None:
                return img
        warn("  No more images available.")
        return None

    if not urls:
        error("No images found. Try a different keyword.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = os.path.join(output_dir, f"{keyword.replace(' ', '_')}_{timestamp}")
    os.makedirs(sample_dir, exist_ok=True)

    with open(os.path.join(sample_dir, "keyword.txt"), "w") as f:
        f.write(keyword)

    steps = [
        ("target",      "Image 1 — Target example",      True),
        ("anti_target", "Image 2 — Anti-target example", True),
        ("result",      "Image 3 — Result image",        False),
    ]

    for stem, label, zero_outside in steps:
        img = next_image()
        if img is None:
            error("Ran out of images. Try a different keyword.")
            return
        final_arr, mask = draw_mask(img, label, get_replacement=next_image)

        if zero_outside:
            out = final_arr.copy()
            out[mask == 0] = 0
            Image.fromarray(out).save(os.path.join(sample_dir, f"{stem}.jpg"))
            success(f"  Saved {stem}.jpg")
        else:
            Image.fromarray(final_arr).save(os.path.join(sample_dir, f"{stem}.jpg"))
            Image.fromarray(mask * 255).save(os.path.join(sample_dir, f"{stem}_mask.png"))
            success(f"  Saved {stem}.jpg + {stem}_mask.png")

    print(f"\n{BOLD}{GREEN}Sample saved to: {sample_dir}/{R}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    
    keyword = ""
    while keyword != "done":
        keyword = input("Keyword: ")
        collect_sample(keyword, output_dir)
