"""
gate_detection.py
=================
Detect a gate by:
  1. Finding two parallel, similar blue blobs (candidate gate pair).
  2. Computing a candidate midpoint between them.
  3. CONFIRMING the gate by checking that each blue blob has a
     white checker pattern on its inner face (the side facing the
     gate interior, i.e. facing the other blob).

Pipeline
--------
  blue_mask()        — YUV threshold → binary mask
  extract_blobs()    — connected components → filtered blob list
  find_gate_pair()   — best parallel, size-matched pair → candidate midpoint
  confirm_gate()     — check checker pattern on each blob's inner side
                       using a contrast-range score on a search band

The checker confirmation uses a separate, simpler mask: for each blue
blob, a band on its INNER side (facing the partner blob) is analysed.
A checker is declared present if a large fraction of columns in that band
have high min-to-max contrast (dark checker squares next to bright ones
create a consistently high column range).

Usage
-----
Set IMAGE_FOLDER at the bottom and run:
    python gate_detection.py
"""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np
from glob import glob
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── YUV blue thresholds ───────────────────────────────────────────────────
    blue_y_min: int   = 0
    blue_y_max: int   = 220
    blue_u_min: int   = 122
    blue_u_max: int   = 255
    blue_v_min: int   = 0
    blue_v_max: int   = 123
    blue_morph_k: int = 5

    # ── Blob filtering ────────────────────────────────────────────────────────
    blob_min_area_ratio: float = 0.003
    blob_min_dimension:  int   = 10
    blob_max_aspect:     float = 5.0
    blob_min_aspect:     float = 0.2

    # ── Parallelism criteria ──────────────────────────────────────────────────
    max_aspect_diff: float = 0.5
    max_area_diff:   float = 0.6
    max_skew_deg:    float = 10.0

    # ── Checker confirmation ──────────────────────────────────────────────────
    # The search band width on the inner side of each blob, as a fraction
    # of that blob's own width (in the direction perpendicular to the gate).
    checker_band_ratio: float = 0.8
    # Minimum fraction of columns in the band that must have a
    # min-to-max range above checker_contrast_thresh to confirm a checker.
    checker_min_col_frac:    float = 0.35
    checker_contrast_thresh: int   = 120

    # ── Display ───────────────────────────────────────────────────────────────
    display_scale: float = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Blue mask
# ─────────────────────────────────────────────────────────────────────────────

def blue_mask(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Binary mask of blue pixels via YUV decomposition.

    BGR→YUV separates luma (Y) from chroma (U=Cb, V=Cr).
    TU Delft blue: U well above 128, V well below 128.
    """
    yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    mask = (
        (Y >= cfg.blue_y_min) & (Y <= cfg.blue_y_max) &
        (U >= cfg.blue_u_min) & (U <= cfg.blue_u_max) &
        (V >= cfg.blue_v_min) & (V <= cfg.blue_v_max)
    ).astype(np.uint8) * 255

    k      = cfg.blue_morph_k
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Blob extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_blobs(mask: np.ndarray, image_shape: tuple, cfg: Config) -> list[dict]:
    """
    Connected components on the blue mask, filtered to keep only
    significant blobs of the right size and shape.
    """
    H, W     = image_shape[:2]
    min_area = int(H * W * cfg.blob_min_area_ratio)

    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if bw < cfg.blob_min_dimension or bh < cfg.blob_min_dimension:
            continue

        aspect = bw / max(bh, 1)
        if aspect > cfg.blob_max_aspect or aspect < cfg.blob_min_aspect:
            continue

        blobs.append({
            "x_min":  x,
            "y_min":  y,
            "x_max":  x + bw,
            "y_max":  y + bh,
            "cx":     int(cents[i, 0]),
            "cy":     int(cents[i, 1]),
            "area":   area,
            "aspect": aspect,
        })
    return blobs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Parallel pair detection → candidate midpoint
# ─────────────────────────────────────────────────────────────────────────────

def _are_parallel(b1: dict, b2: dict, cfg: Config) -> bool:
    """
    Return True if two blobs look like the two bars of a gate.

    Checks:
      0. Blobs are separated vertically in the image (|dy| > |dx|),
         corresponding to real-world side-by-side vertical pillars in a
         90°-rotated image where the floor is on the left.
      1. Similar aspect ratio.
      2. Similar area.
      3. Centroid join is roughly perpendicular to the blobs' long axis.
    """
    # 0. Vertical separation in image = horizontal in real world
    dx0 = b2["cx"] - b1["cx"]
    dy0 = b2["cy"] - b1["cy"]
    if abs(dy0) <= abs(dx0):
        return False

    # 1. Aspect ratio
    ar1, ar2 = b1["aspect"], b2["aspect"]
    larger   = max(ar1, ar2)
    if larger == 0:
        return False
    if (larger - min(ar1, ar2)) / larger > cfg.max_aspect_diff:
        return False

    # 2. Area
    a1, a2   = b1["area"], b2["area"]
    larger_a = max(a1, a2)
    if (larger_a - min(a1, a2)) / larger_a > cfg.max_area_diff:
        return False

    # 3. Centroid join ⊥ long axis
    avg_w   = (b1["x_max"] - b1["x_min"] + b2["x_max"] - b2["x_min"]) / 2
    avg_h   = (b1["y_max"] - b1["y_min"] + b2["y_max"] - b2["y_min"]) / 2
    long_ax = np.array([1.0, 0.0]) if avg_w >= avg_h else np.array([0.0, 1.0])

    join    = np.array([dx0, dy0], dtype=float)
    join   /= np.linalg.norm(join)
    dot     = float(np.clip(np.dot(join, long_ax), -1.0, 1.0))
    angle   = np.degrees(np.arccos(abs(dot)))
    skew    = abs(90.0 - angle)

    return skew <= cfg.max_skew_deg


def find_gate_pair(blobs: list[dict], cfg: Config) -> Optional[tuple[dict, dict]]:
    """
    Find the best (largest combined area) pair of parallel similar blobs.
    Returns (blob1, blob2) sorted top-to-bottom by cy, or None.
    """
    best      = None
    best_area = 0

    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            b1, b2 = blobs[i], blobs[j]
            if not _are_parallel(b1, b2, cfg):
                continue
            combined = b1["area"] + b2["area"]
            if combined > best_area:
                best_area = combined
                # sort so b1 is the one closer to the top of the image
                best = (b1, b2) if b1["cy"] < b2["cy"] else (b2, b1)

    return best


def gate_midpoint(b1: dict, b2: dict) -> tuple[int, int]:
    """Pixel midpoint between the two blob centroids."""
    return (
        (b1["cx"] + b2["cx"]) // 2,
        (b1["cy"] + b2["cy"]) // 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Checker confirmation
# ─────────────────────────────────────────────────────────────────────────────

def _checker_band_score(
    gray:  np.ndarray,
    blob:  dict,
    inner_side: str,
    cfg:   Config,
) -> tuple[float, tuple[int,int,int,int]]:
    """
    Score the search band on the INNER side of a blue blob for a checker.

    Returns
    -------
    (score, (rx0, ry0, rx1, ry1))
      score          : fraction of columns with range > checker_contrast_thresh
      (rx0,ry0,rx1,ry1) : band rectangle in image coordinates
    """
    H, W = gray.shape
    x0, y0, x1, y1 = blob["x_min"], blob["y_min"], blob["x_max"], blob["y_max"]
    bw = x1 - x0
    bh = y1 - y0

    band_w = max(8, int(bw * cfg.checker_band_ratio))
    band_h = max(8, int(bh * cfg.checker_band_ratio))

    if inner_side == "bottom":
        rx0, ry0, rx1, ry1 = x0, y1, x1, min(H, y1 + band_h)
    elif inner_side == "top":
        rx0, ry0, rx1, ry1 = x0, max(0, y0 - band_h), x1, y0
    elif inner_side == "left":
        rx0, ry0, rx1, ry1 = max(0, x0 - band_w), y0, x0, y1
    else:  # "right"
        rx0, ry0, rx1, ry1 = x1, y0, min(W, x1 + band_w), y1

    strip = gray[ry0:ry1, rx0:rx1]
    if strip.size == 0:
        return 0.0, (rx0, ry0, rx1, ry1)

    col_ranges = strip.max(axis=0).astype(float) - strip.min(axis=0).astype(float)
    score = float((col_ranges > cfg.checker_contrast_thresh).mean())
    return score, (rx0, ry0, rx1, ry1)


def confirm_gate(
    gray:  np.ndarray,
    b_top: dict,
    b_bot: dict,
    cfg:   Config,
) -> tuple[bool, float, float,
           tuple[int,int,int,int], tuple[int,int,int,int]]:
    """
    Confirm a candidate gate pair by checking for a checker pattern on the
    inner face of each blue blob.

    Primary search: LEFT face of each bar (floor side, lower x values).
    Fallback: inner vertical face (bottom of top blob / top of bottom blob).

    Returns
    -------
    (confirmed, score_top, score_bot, rect_top, rect_bot)
      confirmed          : True if both blobs have a checker
      score_top/bot      : checker score for each blob
      rect_top/bot       : (rx0,ry0,rx1,ry1) of the best band for each blob
    """
    score_top, rect_top = _checker_band_score(gray, b_top, "left", cfg)
    score_bot, rect_bot = _checker_band_score(gray, b_bot, "left", cfg)

    if score_top < cfg.checker_min_col_frac:
        s2, r2 = _checker_band_score(gray, b_top, "bottom", cfg)
        if s2 > score_top:
            score_top, rect_top = s2, r2

    if score_bot < cfg.checker_min_col_frac:
        s2, r2 = _checker_band_score(gray, b_bot, "top", cfg)
        if s2 > score_bot:
            score_bot, rect_bot = s2, r2

    confirmed = (score_top >= cfg.checker_min_col_frac and
                 score_bot >= cfg.checker_min_col_frac)
    return confirmed, score_top, score_bot, rect_top, rect_bot


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _rot(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def build_display(
    image_bgr:  np.ndarray,
    mask:       np.ndarray,
    blobs:      list[dict],
    pair:       Optional[tuple[dict, dict]],
    mid:        Optional[tuple[int, int]],
    confirmed:  bool,
    rect_top:   Optional[tuple[int,int,int,int]],
    rect_bot:   Optional[tuple[int,int,int,int]],
    score_top:  float,
    score_bot:  float,
    cfg:        Config,
) -> np.ndarray:
    """
    Two panels side by side, rotated 90° CCW, upscaled:
      [ blue mask ]  |  [ annotated ]
    """
    anno = image_bgr.copy()

    # Semi-transparent blue overlay
    overlay = anno.copy()
    overlay[mask > 0] = [200, 100, 0]
    cv2.addWeighted(overlay, 0.35, anno, 0.65, 0, anno)

    # All blobs (grey outline)
    for b in blobs:
        cv2.rectangle(anno, (b["x_min"], b["y_min"]),
                      (b["x_max"], b["y_max"]), (120, 120, 120), 1)

    # Candidate pair (orange = parallel match, not yet confirmed)
    if pair is not None:
        pair_colour = (0, 255, 0) if confirmed else (0, 165, 255)
        for b in pair:
            cv2.rectangle(anno, (b["x_min"], b["y_min"]),
                          (b["x_max"], b["y_max"]), pair_colour, 2)

    # Checker pattern boxes + corner markers
    CHECKER_COLOUR = (0, 255, 255)   # cyan
    for rect, score, label in [
        (rect_top, score_top, "checker1"),
        (rect_bot, score_bot, "checker2"),
    ]:
        if rect is None:
            continue
        rx0, ry0, rx1, ry1 = rect
        has_checker = score >= cfg.checker_min_col_frac

        # Filled semi-transparent overlay on the checker band
        if has_checker:
            overlay2 = anno.copy()
            cv2.rectangle(overlay2, (rx0, ry0), (rx1, ry1), CHECKER_COLOUR, -1)
            cv2.addWeighted(overlay2, 0.25, anno, 0.75, 0, anno)

        # Box outline: cyan if checker found, red if not
        box_col = CHECKER_COLOUR if has_checker else (0, 0, 200)
        cv2.rectangle(anno, (rx0, ry0), (rx1, ry1), box_col, 2)

        # Four corner L-marks to highlight the corners of the checker region
        clen = max(6, (rx1 - rx0) // 5)   # corner arm length
        for (cx, cy) in [(rx0,ry0),(rx1,ry0),(rx0,ry1),(rx1,ry1)]:
            sx = 1 if cx == rx0 else -1   # sign: draw inward
            sy = 1 if cy == ry0 else -1
            cv2.line(anno, (cx, cy), (cx + sx*clen, cy), box_col, 2, cv2.LINE_AA)
            cv2.line(anno, (cx, cy), (cx, cy + sy*clen), box_col, 2, cv2.LINE_AA)

        # Score label
        cv2.putText(anno, f"{label} {score:.2f}",
                    (rx0, max(12, ry0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1, cv2.LINE_AA)

    # Midpoint
    if mid is not None:
        colour = (0, 255, 0) if confirmed else (0, 165, 255)
        cv2.drawMarker(anno, mid, colour, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(anno, mid, 8, colour, -1, cv2.LINE_AA)
        label = f"GATE ({mid[0]},{mid[1]})" if confirmed else f"candidate ({mid[0]},{mid[1]})"
        cv2.putText(anno, label,
                    (max(0, mid[0] - 55), max(16, mid[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2, cv2.LINE_AA)

    # Blue mask panel
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    p_mask = _rot(mask_bgr)
    p_anno = _rot(anno)

    cv2.putText(p_mask, "blue mask",  (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(p_anno, "detections", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    combined = np.hstack([p_mask, p_anno])

    s = cfg.display_scale
    if s != 1.0:
        combined = cv2.resize(combined,
                              (int(combined.shape[1] * s),
                               int(combined.shape[0] * s)),
                              interpolation=cv2.INTER_LINEAR)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Main sequence runner
# ─────────────────────────────────────────────────────────────────────────────

def process_sequence(image_folder: str, cfg: Optional[Config] = None, image_paths: Optional[list] = None):
    """
    Process every image in image_folder sequentially.
    Displays result live. Press 'q' to quit.
    """
    if cfg is None:
        cfg = Config()

    if image_paths is not None:
        paths = image_paths
    else:
        paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
        if not paths:
            paths = sorted(glob(os.path.join(image_folder, "*.png")))
    if not paths:
        sys.exit(f"No images found in: {image_folder}")

    print(f"Processing {len(paths)} images from '{image_folder}'")

    WIN = "Gate detector  |  q = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for idx, path in enumerate(paths):
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  [skip] {path}")
            continue

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1 — blue mask
        mask = blue_mask(image_bgr, cfg)

        # 2 — blobs
        blobs = extract_blobs(mask, image_bgr.shape, cfg)

        # 3 — candidate pair + midpoint
        pair = find_gate_pair(blobs, cfg)
        mid  = gate_midpoint(*pair) if pair is not None else None

        # 4 — checker confirmation
        confirmed   = False
        score_top   = 0.0
        score_bot   = 0.0
        rect_top    = None
        rect_bot    = None
        if pair is not None:
            confirmed, score_top, score_bot, rect_top, rect_bot = (
                confirm_gate(gray, pair[0], pair[1], cfg)
            )

        # Console
        fname = os.path.basename(path)
        if confirmed:
            print(f"  [{idx:04d}] {fname}  GATE CONFIRMED  midpoint={mid}"
                  f"  checker=({score_top:.2f},{score_bot:.2f})")
        elif mid is not None:
            print(f"  [{idx:04d}] {fname}  candidate (no checker)"
                  f"  midpoint={mid}  checker=({score_top:.2f},{score_bot:.2f})")
        else:
            print(f"  [{idx:04d}] {fname}  no gate  (blobs={len(blobs)})")

        # Display
        vis = build_display(image_bgr, mask, blobs, pair, mid, confirmed,
                            rect_top, rect_bot, score_top, score_bot, cfg)
        cv2.imshow(WIN, vis)
        cv2.resizeWindow(WIN, vis.shape[1], vis.shape[0])
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _images_in_order(folder: str) -> list[str]:
    """
    Return image paths sorted numerically by filename stem.
    Drone images are named by timestamp (e.g. 1352900896.jpg); plain
    lexicographic order would mis-sort them.
    """
    paths = glob(os.path.join(folder, "*.jpg"))
    if not paths:
        paths = glob(os.path.join(folder, "*.png"))
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return paths


if __name__ == "__main__":

    # ── Set your image folder here ────────────────────────────────────────────
    IMAGE_FOLDER = "../paparazzi10/DEVELOPMENT/downloads from drone/20260320"
    # ─────────────────────────────────────────────────────────────────────────

    process_sequence(IMAGE_FOLDER, image_paths=_images_in_order(IMAGE_FOLDER))