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
Set IMAGE_FOLDER at the bottom, pick mode, and run:
    python gate_detection.py
"""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
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

    # ── Parallelism criteria ──────────────────────────────────────────────────
    max_aspect_diff: float = 0.5
    max_area_diff:   float = 0.6
    max_skew_deg:    float = 10

    # ── Display ───────────────────────────────────────────────────────────────
    display_scale: float = 1.5

    # ── Preview grid ──────────────────────────────────────────────────────────
    num_preview:  int  = 16        # must be perfect square: 4, 9, 16, 25 ...
    save_figure:  bool = True
    save_path:    str  = 'gate_detection_results.png'


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Blue mask
# ─────────────────────────────────────────────────────────────────────────────

def blue_mask(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
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
        blobs.append({
            "x_min":  x,        "y_min":  y,
            "x_max":  x + bw,   "y_max":  y + bh,
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
    dx0 = b2["cx"] - b1["cx"]
    dy0 = b2["cy"] - b1["cy"]
    if abs(dy0) <= abs(dx0):
        return False

    ar1, ar2 = b1["aspect"], b2["aspect"]
    larger   = max(ar1, ar2)
    if larger == 0:
        return False
    if (larger - min(ar1, ar2)) / larger > cfg.max_aspect_diff:
        return False

    a1, a2   = b1["area"], b2["area"]
    larger_a = max(a1, a2)
    if (larger_a - min(a1, a2)) / larger_a > cfg.max_area_diff:
        return False

    dx = b2["cx"] - b1["cx"]
    dy = b2["cy"] - b1["cy"]
    if dx == 0 and dy == 0:
        return False

    avg_w   = (b1["x_max"] - b1["x_min"] + b2["x_max"] - b2["x_min"]) / 2
    avg_h   = (b1["y_max"] - b1["y_min"] + b2["y_max"] - b2["y_min"]) / 2
    long_ax = np.array([1.0, 0.0]) if avg_w >= avg_h else np.array([0.0, 1.0])
    join    = np.array([dx, dy], dtype=float)
    join   /= np.linalg.norm(join)
    dot     = float(np.clip(np.dot(join, long_ax), -1.0, 1.0))
    angle   = np.degrees(np.arccos(abs(dot)))
    skew    = abs(90.0 - angle)
    return skew <= cfg.max_skew_deg


def find_gate_pair(blobs: list[dict], cfg: Config) -> Optional[tuple[dict, dict]]:
    best, best_area = None, 0
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
    return ((b1["cx"] + b2["cx"]) // 2, (b1["cy"] + b2["cy"]) // 2)


# ─────────────────────────────────────────────────────────────────────────────
# process_frame — single image in, results out
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    image_bgr: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict], Optional[tuple[dict, dict]], Optional[tuple[int, int]]]:
    """
    Run the full gate detection pipeline on one image.

    Parameters
    ----------
    image_bgr : np.ndarray   — BGR image
    cfg       : Config

    Returns
    -------
    mask  : binary blue mask
    blobs : all significant blue blobs
    pair  : the two gate blobs, or None
    mid   : gate midpoint (x, y), or None
    """
    mask  = blue_mask(image_bgr, cfg)
    blobs = extract_blobs(mask, image_bgr.shape, cfg)
    pair  = find_gate_pair(blobs, cfg)
    mid   = gate_midpoint(*pair) if pair is not None else None
    return mask, blobs, pair, mid


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper
# ─────────────────────────────────────────────────────────────────────────────

def annotate(
    image_bgr: np.ndarray,
    mask:      np.ndarray,
    blobs:     list[dict],
    pair:      Optional[tuple[dict, dict]],
    mid:       Optional[tuple[int, int]],
) -> np.ndarray:
    anno    = image_bgr.copy()
    overlay = anno.copy()
    overlay[mask > 0] = [200, 100, 0]
    cv2.addWeighted(overlay, 0.35, anno, 0.65, 0, anno)

    for b in blobs:
        cv2.rectangle(anno, (b["x_min"], b["y_min"]),
                      (b["x_max"], b["y_max"]), (120, 120, 120), 1)

    if pair is not None:
        for b in pair:
            cv2.rectangle(anno, (b["x_min"], b["y_min"]),
                          (b["x_max"], b["y_max"]), (0, 165, 255), 2)

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

    return anno


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — live sequence (every image in folder, in order)
# ─────────────────────────────────────────────────────────────────────────────

def process_sequence(image_folder: str, cfg: Optional[Config] = None):
    """Process every image in order. Press q to quit."""
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

        mask, blobs, pair, mid = process_frame(image_bgr, cfg)
        anno = annotate(image_bgr, mask, blobs, pair, mid)

        # Rotate for display (camera is 90° CW)
        vis = cv2.rotate(anno, cv2.ROTATE_90_COUNTERCLOCKWISE)
        s   = cfg.display_scale
        if s != 1.0:
            vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))

        fname = os.path.basename(path)
        if mid is not None:
            print(f"  [{idx:04d}] {fname}  GATE  midpoint={mid}")
        else:
            print(f"  [{idx:04d}] {fname}  no gate  (blobs={len(blobs)})")

        cv2.imshow(WIN, vis)
        cv2.resizeWindow(WIN, vis.shape[1], vis.shape[0])
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — random preview grid (n² images, matplotlib figure)
# ─────────────────────────────────────────────────────────────────────────────

def process_random_preview(image_folder: str, cfg: Optional[Config] = None):
    """Pick num_preview random images and show original vs detection grid."""
    if cfg is None:
        cfg = Config()

    all_paths = glob(os.path.join(image_folder, "*.jpg"))
    all_paths = [p for p in all_paths if "_mask" not in p]
    all_paths += glob(os.path.join(image_folder, "*.png"))
    if not all_paths:
        sys.exit(f"No images found in: {image_folder}")

    n        = cfg.num_preview
    selected = random.sample(all_paths, min(n, len(all_paths)))
    print(f"Selected {len(selected)} random images.")

    grid_side      = int(np.sqrt(n))
    cols_per_image = 2                          # original | detection
    grid_cols      = grid_side * cols_per_image
    grid_rows      = grid_side

    fig, axes = plt.subplots(grid_rows, grid_cols,
                             figsize=(grid_cols * 3, grid_rows * 3))
    fig.suptitle("Gate Detection — Original (left) vs Detection (right)", fontsize=12)

    for idx, path in enumerate(selected):
        img = cv2.imread(path)
        if img is None:
            continue

        print(f"[{idx+1}/{len(selected)}] {os.path.basename(path)}")
        mask, blobs, pair, mid = process_frame(img, cfg)
        anno = annotate(img, mask, blobs, pair, mid)

        row      = idx // grid_side
        col_pair = idx %  grid_side
        col_orig = col_pair * cols_per_image
        col_det  = col_pair * cols_per_image + 1

        axes[row, col_orig].imshow(cv2.cvtColor(img,  cv2.COLOR_BGR2RGB))
        axes[row, col_orig].set_title(f"#{idx+1} Original", fontsize=7)
        axes[row, col_orig].axis("off")

        axes[row, col_det].imshow(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB))
        axes[row, col_det].set_title(
            f"#{idx+1} {'GATE' if mid is not None else 'no gate'}",
            fontsize=7, color="green" if mid is not None else "gray",
        )
        axes[row, col_det].axis("off")

    plt.tight_layout()

    if cfg.save_figure:
        plt.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {cfg.save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    IMAGE_FOLDER = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320"

    cfg = Config(
        num_preview = 16,       # change to 4, 9, 25 etc.
        save_figure = True,
        save_path   = "gate_detection_results.png",
    )

    # ── Pick one mode ─────────────────────────────────────────────────────────
    process_random_preview(IMAGE_FOLDER, cfg)
    # process_sequence(IMAGE_FOLDER, cfg)
