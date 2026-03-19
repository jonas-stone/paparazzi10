"""
gate_detection.py
=================
Detect a gate by finding two parallel, similar blue blobs and placing
a midpoint between them.

Pipeline
--------
1. Blue mask via YUV thresholding
2. Extract connected components, keep significant blobs
3. For every pair of blobs check:
     - similar aspect ratio
     - similar size (area)
     - centroids aligned perpendicular to their long axis (parallel bars)
4. Best valid pair → midpoint computed and drawn

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
    blue_u_min: int   = 122     # blue drives Cb above 128
    blue_u_max: int   = 255
    blue_v_min: int   = 0
    blue_v_max: int   = 123     # blue keeps Cr below 128
    blue_morph_k: int = 5

    # ── Blob filtering ────────────────────────────────────────────────────────
    # Minimum blob area as fraction of total image area
    blob_min_area_ratio: float = 0.003

    # ── Parallelism criteria ──────────────────────────────────────────────────
    # Max relative difference in aspect ratio (w/h) between the two blobs
    max_aspect_diff: float = 0.5
    # Max relative difference in area between the two blobs
    max_area_diff: float = 0.6
    # The vector joining the two centroids should be perpendicular to the
    # blobs' long axis. This is the max allowed angular deviation from 90°.
    max_skew_deg: float = 10

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
    Thresholding on U and V isolates the blue reliably regardless of
    brightness, making it robust to different viewing angles and lighting.
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
    Find connected components in the blue mask and return significant blobs.

    Each blob dict contains:
        x_min, y_min, x_max, y_max  — bounding box
        cx, cy                       — centroid
        area                         — pixel count
        aspect                       — width / height
    """
    H, W     = image_shape[:2]
    min_area = int(H * W * cfg.blob_min_area_ratio)

    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs = []
    for i in range(1, n):          # skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append({
            "x_min":  x,
            "y_min":  y,
            "x_max":  x + bw,
            "y_max":  y + bh,
            "cx":     int(cents[i, 0]),
            "cy":     int(cents[i, 1]),
            "area":   area,
            "aspect": bw / max(bh, 1),
        })
    return blobs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Parallel pair detection
# ─────────────────────────────────────────────────────────────────────────────

def _are_parallel(b1: dict, b2: dict, cfg: Config) -> bool:
    """
    Return True if two blobs look like the two bars of a gate.

    Three conditions must all hold:

    1. Similar aspect ratio  — both bars have roughly the same shape (w/h).
       Rejects pairs where one is a tall thin pole and the other a wide bar.

    2. Similar area          — bars at the same depth appear the same size.
       Rejects pairs of very different-sized blobs.

    3. Centroid join ⊥ long axis — the line connecting the two bar centres
       must run roughly perpendicular to the direction the bars point.
       A real gate has both bars pointing the same way (parallel) and the
       gap between them in the perpendicular direction.
    """
    # 0. The image is rotated 90° CW (floor on the left).
    #    Two real-world vertical pillars appear as blobs separated along
    #    the image Y-axis (one near top of image, one near bottom).
    #    Require |dy| > |dx| so we only match blobs stacked vertically
    #    in the image, which corresponds to side-by-side pillars in reality.
    dx0 = b2["cx"] - b1["cx"]
    dy0 = b2["cy"] - b1["cy"]
    if abs(dy0) <= abs(dx0):
        return False

    # 1. Aspect ratio
    ar1, ar2  = b1["aspect"], b2["aspect"]
    larger    = max(ar1, ar2)
    if larger == 0:
        return False
    if (larger - min(ar1, ar2)) / larger > cfg.max_aspect_diff:
        return False

    # 2. Area
    a1, a2   = b1["area"], b2["area"]
    larger_a = max(a1, a2)
    if (larger_a - min(a1, a2)) / larger_a > cfg.max_area_diff:
        return False

    # 3. Centroid join perpendicular to long axis
    dx = b2["cx"] - b1["cx"]
    dy = b2["cy"] - b1["cy"]
    if dx == 0 and dy == 0:
        return False

    # Average long-axis direction across both blobs
    avg_w = (b1["x_max"] - b1["x_min"] + b2["x_max"] - b2["x_min"]) / 2
    avg_h = (b1["y_max"] - b1["y_min"] + b2["y_max"] - b2["y_min"]) / 2
    # Long axis unit vector: horizontal if wider, vertical if taller
    long_ax = np.array([1.0, 0.0]) if avg_w >= avg_h else np.array([0.0, 1.0])

    join    = np.array([dx, dy], dtype=float)
    join   /= np.linalg.norm(join)
    dot     = float(np.clip(np.dot(join, long_ax), -1.0, 1.0))
    angle   = np.degrees(np.arccos(abs(dot)))   # 0° = parallel, 90° = perpendicular
    skew    = abs(90.0 - angle)                 # how far from perfectly perpendicular

    return skew <= cfg.max_skew_deg


def find_gate_pair(blobs: list[dict], cfg: Config) -> Optional[tuple[dict, dict]]:
    """
    Find the best pair of parallel, similar blobs.

    'Best' = largest combined area (most prominent pair in the image).
    Returns (blob1, blob2) or None.
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
                best      = (b1, b2)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Midpoint
# ─────────────────────────────────────────────────────────────────────────────

def gate_midpoint(b1: dict, b2: dict) -> tuple[int, int]:
    """Centre point between the two blob centroids."""
    return (
        (b1["cx"] + b2["cx"]) // 2,
        (b1["cy"] + b2["cy"]) // 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _rot(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def build_display(
    image_bgr: np.ndarray,
    mask:      np.ndarray,
    blobs:     list[dict],
    pair:      Optional[tuple[dict, dict]],
    mid:       Optional[tuple[int, int]],
    cfg:       Config,
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

    # All blobs (thin grey box)
    for b in blobs:
        cv2.rectangle(anno, (b["x_min"], b["y_min"]),
                      (b["x_max"], b["y_max"]), (120, 120, 120), 1)

    # Gate pair (bright orange boxes)
    if pair is not None:
        for b in pair:
            cv2.rectangle(anno, (b["x_min"], b["y_min"]),
                          (b["x_max"], b["y_max"]), (0, 165, 255), 2)

    # Midpoint
    if mid is not None:
        cv2.drawMarker(anno, mid, (0, 255, 0),
                       cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(anno, mid, 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(anno, f"({mid[0]}, {mid[1]})",
                    (max(0, mid[0] - 45), max(16, mid[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(anno, "no gate", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    # Blue mask as BGR
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Rotate CCW → landscape
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

def process_sequence(image_folder: str, cfg: Optional[Config] = None):
    """
    Process every image in image_folder sequentially.
    Displays result live. Press 'q' to quit.
    """
    if cfg is None:
        cfg = Config()

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
        
        # Step 1 — blue mask
        mask = blue_mask(image_bgr, cfg)

        # Step 2 — blobs
        blobs = extract_blobs(mask, image_bgr.shape, cfg)

        # Step 3 — find parallel pair
        pair = find_gate_pair(blobs, cfg)

        # Step 4 — midpoint
        mid = gate_midpoint(*pair) if pair is not None else None

        # Console
        fname = os.path.basename(path)
        if mid is not None:
            print(f"  [{idx:04d}] {fname}  GATE  midpoint={mid}")
        else:
            print(f"  [{idx:04d}] {fname}  no gate  (blobs={len(blobs)})")

        # Display
        vis = build_display(image_bgr, mask, blobs, pair, mid, cfg)
        cv2.imshow(WIN, vis)
        cv2.resizeWindow(WIN, vis.shape[1], vis.shape[0])
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Set your image folder here ────────────────────────────────────────────
    IMAGE_FOLDER = "../paparazzi10/DEVELOPMENT/downloads from drone/20260306-095826"
    # ─────────────────────────────────────────────────────────────────────────

    process_sequence(IMAGE_FOLDER)