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


# --- Wikipedia fetch (same as collect_images.py) ----------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "squirt_bot/1.0 (data collection script)"}

KEYWORDS = [
    "cat", "dog", "bird", "rabbit", "squirrel", "raccoon",
    "crow", "pigeon", "sparrow", "duck", "chicken", "goose",
]

slot_pool = None
index = 0
keyword = ''
count = 1

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


def fetch_images(keyword: str, index: int = 0, count: int = 3) -> list[Image.Image]:
    page_title = _search_top_page(keyword)
    if not page_title:
        return []
    print(f"  Wikipedia article: {page_title}")
    filenames = _get_image_filenames(page_title)
    urls = _resolve_image_urls(filenames)[index:index+count]
    images = []
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(0.5)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            images.append(img)
            print(f"  Fetched {len(images)}/{len(urls)}")
        except Exception as e:
            print(f"  Skipped image {i + 1}: {e}")
    return images


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
        print("  (fewer than 3 vertices — using full image as mask)")
        return current[0], np.ones((h, w), dtype=np.uint8)

    path = Path(vertices)
    y_idx, x_idx = np.mgrid[:h, :w]
    points = np.column_stack([x_idx.ravel(), y_idx.ravel()])
    mask = path.contains_points(points).reshape(h, w).astype(np.uint8)
    return current[0], mask


# --- Main data collection loop -----------------------------------------------

def collect_sample(keyword: str, output_dir: str):
    global slot_pool, index

    print(f"\nFetching images for '{keyword}'...")
    pool = fetch_images(keyword, count=count)
    index += count
    if len(pool) < count:
        print(f"Only found {len(pool)} images (need at least {count}). Try a different keyword.")
        return
    print(f"  Pool: {len(pool)} images available")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = os.path.join(output_dir, f"{keyword.replace(' ', '_')}_{timestamp}")
    os.makedirs(sample_dir, exist_ok=True)

    with open(os.path.join(sample_dir, "keyword.txt"), "w") as f:
        f.write(keyword)

    steps = [
        # (filename_stem, display_label, zero_outside_mask)
        ("target",      "Image 1 — Target example",      True),
        ("anti_target", "Image 2 — Anti-target example", True),
        ("result",      "Image 3 — Result image",        False),
    ]

    for slot_idx, (stem, label, zero_outside) in enumerate(steps):
        # Each slot starts at its assigned image and can cycle forward through the pool
        slot_pool = [np.array(img) for img in pool[slot_idx:]]
        cursor = [0]

        def get_replacement(c=cursor):
            global slot_pool, index
            c[0] += 1
            # get more images if needed
            if c[0] >= len(slot_pool):
                new_images = fetch_images(keyword, index=index, count=count)
                index += count
                slot_pool.extend([np.array(img) for img in new_images])
                if c[0] >= len(slot_pool):
                    c[0] = 0

            print(f"  Swapped to image {slot_idx + 1 + c[0]} from pool")
            return slot_pool[c[0]]

        final_arr, mask = draw_mask(slot_pool[0], label, get_replacement=get_replacement)

        if zero_outside:
            out = final_arr.copy()
            out[mask == 0] = 0
            Image.fromarray(out).save(os.path.join(sample_dir, f"{stem}.jpg"))
            print(f"  Saved {stem}.jpg")
        else:
            Image.fromarray(final_arr).save(os.path.join(sample_dir, f"{stem}.jpg"))
            Image.fromarray(mask * 255).save(os.path.join(sample_dir, f"{stem}_mask.png"))
            print(f"  Saved {stem}.jpg + {stem}_mask.png")

    print(f"\nSample saved to: {sample_dir}/")
    slot_pool.clear()
    index = 0


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    
    while keyword != "done":
        keyword = input("Keyword: ")
        collect_sample(keyword, output_dir)
