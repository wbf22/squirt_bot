"""
Interactive data collection for TargetDetector training.

This version crawls images from arbitrary web pages starting from a URL. For each
sample the script pulls images from the current page (and replacement images
while drawing), allows you to draw masks, then offers links found on that page
as choices for the next page to crawl.

Controls:
  Left-click    add polygon vertex
  Right-click   confirm and move on
  Middle-click  swap in next image from the pool (if available)

Usage:
    python collect_data.py [output_dir]

Requirements:
    pip install requests beautifulsoup4 pillow matplotlib numpy
"""

import io
import os
import pathlib
import random
import sys
import datetime
import time
from typing import List, Union

import matplotlib
from matplotlib import patches
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

print(matplotlib.get_backend())
matplotlib.use("Qt5Agg")

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

# --- File Scraping utilities -----------------------------
def get_subfolders(directory: Union[str, Path],
                   absolute: bool = True,
                   include_hidden: bool = False) -> List[str]:
    """
    Return the paths to immediate subfolders of `directory`.

    Args:
        directory: path to the directory to inspect.
        absolute: if True, return absolute paths; if False, return paths relative to `directory`.
        include_hidden: if False, skip hidden folders (those starting with '.').

    Returns:
        List of folder paths as strings.

    Raises:
        FileNotFoundError if directory doesn't exist.
        NotADirectoryError if path is not a directory.
    """
    p = pathlib.Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"{directory} does not exist")
    if not p.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    result = []
    for child in p.iterdir():
        if child.is_dir():
            if not include_hidden and child.name.startswith('.'):
                continue
            result.append(str(child.resolve()) if absolute else str(child.relative_to(p)))
    return result

# --- HTTP / scraping utilities -----------------------------------------------

HEADERS = {"User-Agent": "collect-data-bot/1.0"}
SLEEP_BETWEEN_REQUESTS = 0.8

def fetch_page(url: str, timeout: int = 15) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        warn(f"Failed to fetch {url}: {e}")
        return None


def find_images_on_page(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    imgs = set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        absolute = urljoin(base_url, src)
        if absolute.startswith("data:"):
            continue
        if not absolute.lower().startswith(("http://", "https://")):
            continue
        # filter by common image extensions if present
        path = urlparse(absolute).path
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            continue
        imgs.add(absolute)
    return sorted(imgs)


def find_links_on_page(html: str, base_url: str, same_domain: bool = True) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    base_netloc = urlparse(base_url).netloc
    links = []
    seen = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_domain and parsed.netloc != base_netloc:
            continue
        norm = parsed.scheme + "://" + parsed.netloc + parsed.path + (("?" + parsed.query) if parsed.query else "")
        if norm in seen:
            continue
        seen.add(norm)
        text = (a.get_text() or "").strip()
        links.append((text or "-", norm))
    return links


def filename_from_url(url: str) -> str:
    p = urlparse(url).path
    name = os.path.basename(p)
    if not name:
        name = "image"
    if not os.path.splitext(name)[1]:
        name = name + ".jpg"
    return name


def unique_filename(directory: str, basename: str) -> str:
    base, ext = os.path.splitext(basename)
    candidate = basename
    i = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate


def download_image_to_array(url: str) -> np.ndarray | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
    except Exception as e:
        warn(f"  Failed to download {url}: {e}")
        return None


def download_image_to_file(url: str, out_dir: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        r.raise_for_status()
        fname = filename_from_url(url)
        fname = unique_filename(out_dir, fname)
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return fpath
    except Exception as e:
        warn(f"  Failed to download {url}: {e}")
        return None


# --- Mask drawing ------------------------------------------------------------

def draw_mask(
    image_array: np.ndarray,
    title: str,
    get_replacement=None,
) -> tuple[np.ndarray, np.ndarray]:
    current = [image_array]
    vertices = []
    vertex_groups = []

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(current[0])
    line, = ax.plot([], [], "r-o", linewidth=2, markersize=6)
    lines = []
    extra = "  |  Middle-click: next image" if get_replacement else ""
    ax.set_title(f"{title}\nLeft-click: add vertex  |  Right-click: confirm |  Middle-click: next image | Ctrl + Right-Click: new polygon")
    fig.tight_layout()


    def _redraw_poly(fig, ax):
        if not vertices:
            line.set_data([], [])
        else:
            xs = [v[0] for v in vertices] + [vertices[0][0]]
            ys = [v[1] for v in vertices] + [vertices[0][1]]
            line.set_data(xs, ys)

        for i, vertex_group in enumerate(vertex_groups):
            curr_line = None
            if i < len(lines):
                curr_line = lines[i]
            else:
                curr_line, = ax.plot([], [], "r-o", linewidth=2, markersize=6)
                lines.append(curr_line)
            xs = [v[0] for v in vertex_group] + [vertex_group[0][0]]
            ys = [v[1] for v in vertex_group] + [vertex_group[0][1]]
            curr_line.set_data(xs, ys)

        fig.canvas.draw_idle()

    def _onclick(event):
        if event.inaxes != ax:
            return
        

        key = (event.key or "").lower()
        ctrl = ("ctrl" in key) or ("control" in key)
        cmd = ("cmd" in key) or ("meta" in key)

        # matplotlib button: 1 left, 2 middle, 3 right
        if event.button == 1:       # left-click → add vertex or new poly
            if ctrl or cmd:
                if len(vertices) > 0:
                    new_group = vertices.copy()
                    vertex_groups.append(new_group)
                vertices.clear()
            vertices.append((event.xdata, event.ydata))
            _redraw_poly(fig, ax)
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
    plt.show(block=True)

    h, w = current[0].shape[:2]
    if len(vertices) < 3 and len(vertex_groups) == 0:
        warn("  (fewer than 3 vertices — using full image as mask)")
        return current[0], np.zeros((h, w), dtype=np.uint8)
    else:
        new_group = vertices.copy()
        vertex_groups.append(new_group)

    y_idx, x_idx = np.mgrid[:h, :w]
    points = np.column_stack([x_idx.ravel(), y_idx.ravel()])   # (x,y) points

    mask = np.zeros((h, w), dtype=np.uint8)
    for group in vertex_groups:
        path = Path(group)   # group must be sequence of (x,y)
        inside = path.contains_points(points).reshape(h, w)
        mask |= inside

    return current[0], mask


# --- Fit / pad utilities ----------------------------------------------------

TARGET_W = 1920
TARGET_H = 1080


def _fit_and_pad_to_target(img_arr: np.ndarray, target_w: int = TARGET_W, target_h: int = TARGET_H) -> np.ndarray:
    assert img_arr.ndim == 3 and img_arr.shape[2] == 3
    h0, w0 = img_arr.shape[:2]
    scale_orig = min(target_w / w0, target_h / h0)
    scale_rot = min(target_w / h0, target_h / w0)
    rotate = scale_rot > scale_orig
    if rotate:
        arr = np.rot90(img_arr, k=1)
        h, w = arr.shape[:2]
    else:
        arr = img_arr
        h, w = h0, w0
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(arr)
    pil_resized = pil.resize((new_w, new_h), resample=Image.LANCZOS)
    resized = np.array(pil_resized)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    paste_x = target_w - new_w
    paste_y = 0
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized
    return canvas


def _fit_and_pad_mask(mask_arr: np.ndarray, target_w: int = TARGET_W, target_h: int = TARGET_H) -> np.ndarray:
    assert mask_arr.ndim == 2
    h0, w0 = mask_arr.shape[:2]
    scale_orig = min(target_w / w0, target_h / h0)
    scale_rot = min(target_w / h0, target_h / w0)
    rotate = scale_rot > scale_orig
    if rotate:
        arr = np.rot90(mask_arr, k=1)
        h, w = arr.shape[:2]
    else:
        arr = mask_arr
        h, w = h0, w0
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil_resized = pil.resize((new_w, new_h), resample=Image.NEAREST)
    resized = (np.array(pil_resized) // 255).astype(np.uint8)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    paste_x = target_w - new_w
    paste_y = 0
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized
    return canvas


# --- Main collection routine -------------------------------------------------


def collect_from_url(start_url: str, output_dir: str):
    header(f"Starting at: {start_url}")
    r = fetch_page(start_url)
    if not r:
        error("Failed to fetch starting URL.")
        return None
    base_html = r.text
    image_urls = find_images_on_page(base_html, start_url)
    if not image_urls:
        warn("No images found on the starting page.")
        return None
    dim(f"Found {len(image_urls)} images on page.")

    cursor = [0]
    def next_image():
        if cursor[0] >= len(image_urls):
            cursor[0] = 0
            warn("No more images available on this page.")

        url = image_urls[cursor[0]]
        cursor[0] += 1
        img = download_image_to_array(url)
        if img is not None:
            return img
        
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parsed = urlparse(start_url)
    safe_netloc = parsed.netloc.replace(":", "_")
    sample_dir = os.path.join(output_dir, f"{safe_netloc}_{timestamp}")
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, "source_url.txt"), "w") as f:
        f.write(start_url)

    steps = [
        ("target", True, "Image 1 — Target example"),
        ("anti_target", True, "Image 2 — Anti-target example"),
        ("result", False, "Image 3 — Result image"),
    ]

    for stem, zero_outside, title in steps:
        img = next_image()
        if img is None:
            error("Ran out of images while collecting a sample.")
            return sample_dir
        final_arr, mask = draw_mask(img, title, get_replacement=next_image)
        final_fixed = _fit_and_pad_to_target(final_arr)
        mask_fixed = _fit_and_pad_mask(mask)
        if zero_outside:
            out = final_fixed.copy()
            out[mask_fixed == 0] = 0
            Image.fromarray(out).save(os.path.join(sample_dir, f"{stem}.jpg"))
            success(f"  Saved {stem}.jpg")
        else:
            Image.fromarray(final_fixed).save(os.path.join(sample_dir, f"{stem}.jpg"))
            Image.fromarray((mask_fixed * 255).astype(np.uint8)).save(os.path.join(sample_dir, f"{stem}_mask.png"))
            success(f"  Saved {stem}.jpg + {stem}_mask.png")

    # After saving, parse links from the base page and offer choices
    # Gather all links (including cross-domain) then sort so that links from
    # different domains or with a different first path segment appear first.
    links = find_links_on_page(base_html, start_url, same_domain=False)

    def _is_different(link_url: str) -> bool:
        p_base = urlparse(start_url)
        p = urlparse(link_url)
        # different domain -> consider different
        if p.netloc != p_base.netloc:
            return True
        def first_seg(parsed_obj):
            seg = parsed_obj.path.lstrip('/').split('/', 1)[0]
            return seg
        if first_seg(p_base) != first_seg(p):
            return True
        return False

    # stable sort: different links first, then the rest in original order
    # links_sorted = sorted(links, key=lambda t: 0 if _is_different(t[1]) else 1)
    links_sorted = sorted(links, key=lambda t: random.randint(0,1))
    return sample_dir, links_sorted


def collect_from_folder(folder: str, output_dir: str):
    header(f"Starting from folder: {folder}")
    if not os.path.exists(folder) or not os.path.isdir(folder):
        error(f"Folder not found: {folder}")
        return None

    # gather image files
    exts = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in exts]
    if not files:
        warn("No image files found in the folder.")
        return None
    random.shuffle(files)
    dim(f"Found {len(files)} images in folder.")

    cursor = [0]
    def next_image():
        # behave similarly to collect_from_url: wrap to start and warn when exhausted
        if cursor[0] >= len(files):
            cursor[0] = 0
            warn("No more images available in this folder.")

        while cursor[0] < len(files):
            path = files[cursor[0]]
            cursor[0] += 1
            try:
                img = np.array(Image.open(path).convert("RGB"))
                return img
            except Exception as e:
                warn(f"  Failed to load {path}: {e}")
                continue
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.basename(os.path.normpath(folder)).replace(".", "_")
    sample_dir = os.path.join(output_dir, f"{safe_name}_{timestamp}")
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, "source_folder.txt"), "w") as f:
        f.write(folder)

    steps = [
        ("result", False, "Image 1 — Result image"),
        ("target", True, "Image 2 — Target example"),
        ("anti_target", True, "Image 3 — Anti-target example"),
    ]

    for stem, zero_outside, title in steps:
        img = next_image()
        if img is None:
            error("Ran out of images while collecting a sample.")
            return sample_dir
        final_arr, mask = draw_mask(img, title, get_replacement=next_image)
        final_fixed = _fit_and_pad_to_target(final_arr)
        mask_fixed = _fit_and_pad_mask(mask)
        if zero_outside:
            out = final_fixed.copy()
            out[mask_fixed == 0] = 0
            Image.fromarray(out).save(os.path.join(sample_dir, f"{stem}.jpg"))
            success(f"  Saved {stem}.jpg")
        else:
            Image.fromarray(final_fixed).save(os.path.join(sample_dir, f"{stem}.jpg"))
            Image.fromarray((mask_fixed * 255).astype(np.uint8)).save(os.path.join(sample_dir, f"{stem}_mask.png"))
            success(f"  Saved {stem}.jpg + {stem}_mask.png")

    # No links to return when collecting from a folder; return empty list
    return sample_dir, []

def choose_from_list(prompt: str, max_items: int):
    while True:
        choice = input(prompt).strip().lower()
        if choice == "q":
            return "q"
        if choice == "c":
            return "c"
        if choice == "f":
            return "f"
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= max_items:
                return n - 1
        print("Invalid choice. Enter number, 'c' to enter a custom URL, 'f' to enter a folder, or 'q' to quit.")

def get_source(links):
    max_show = 0
    if links:
        print("\nLinks found on this site:")
        max_show = min(30, len(links))
        for idx, (text, url) in enumerate(links[:max_show], start=1):
            short = text if len(text) <= 60 else text[:57] + "..."
            print(f"  {idx}. {short} -> {url}")
        if len(links) > max_show:
            print(f"  ... and {len(links) - max_show} more")

    print("\nChoose the next link to visit by number, 'c' to enter a custom URL, 'f' to enter a folder, or 'q' to quit.")
    sel = choose_from_list("Your choice: ", max_show)
    is_url = True
    next_target = None
    if sel == "q":
        print("Quitting.")
        sys.exit(0)
    if sel == "c":
        next_target = input("Enter next URL: ").strip()
    elif sel == "f":
        next_target = input("Enter folder path: ").strip()
        is_url = False
    else:
        next_target = links[sel][1]
        is_url = False


    if not next_target:
        print("Nothing entered. Exiting.")
        sys.exit(0)

    return next_target, is_url

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    os.makedirs(output_dir, exist_ok=True)

    links = [['', x] for x in get_subfolders("data_source")]
    current_target, is_url = get_source(links)

    while True:
        if is_url:
            res = collect_from_url(current_target, output_dir)
            if res is not None:
                sample_dir, links = res
                print(f"Sample saved to: {sample_dir}")
        else:
            res = collect_from_folder(current_target, output_dir)
            if res is not None:
                sample_dir, links = res
                print(f"Sample saved to: {sample_dir}")



        links = [['', x] for x in get_subfolders("data_source")]
        current_target, is_url = get_source(links)

        time.sleep(SLEEP_BETWEEN_REQUESTS)
