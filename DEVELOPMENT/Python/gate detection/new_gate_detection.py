"""
gate_detector.py
================
Gate detection for autonomous drone flight.

Pipeline
--------
1. Call get_obstacle_info() on each frame to obtain obstacle detections
   (each row: [left_x, width, det_type]).
2. For every DET_OBSTACLE detection, reconstruct a bounding box using the
   boundary_rows debug data (the exact same geometry the existing display
   code uses).
3. Crop the obstacle region from the image and decompose it into YUV.
4. Threshold the U and V channels to isolate blue pixels, apply morphological
   cleanup, then look for a tall blue column on the LEFT and RIGHT edges.
5. Declare a gate when both a left-pillar and a right-pillar are found.
6. Draw coloured boxes around each pillar on the original frame.

Usage
-----
    python gate_detector.py                        # synthetic demo
    python gate_detector.py path/to/image.jpg      # single image
    python gate_detector.py path/to/folder/        # image sequence

Dependencies
------------
    numpy, opencv-python, scipy
    get_obstacle_info_Imp.py  (+ colored_blob_separator, solidity_detection)
    → all must be on PYTHONPATH / same directory.
"""

from __future__ import annotations

import sys
import os
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path  # kept for potential caller use
from typing import Optional

# ── import the existing pipeline ──────────────────────────────────────────────
import get_obstacle_info_Imp as oai

DET_OBSTACLE = oai.DET_OBSTACLE
DET_PLANT    = oai.DET_PLANT


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateConfig:
    """All tuneable thresholds in one place."""

    # ── Blue colour bounds in YUV (OpenCV 0-255 range) ──────────────────────
    # Y  (luma)   – keep loose; lighting varies outdoors
    y_min: int = 0
    y_max: int = 220
    # U  (Cb)     – blue pushes U HIGH  (> 128 in 0-255 range)
    u_min: int = 128
    u_max: int = 255
    # V  (Cr)     – blue keeps V LOW    (< 128 in 0-255 range)
    v_min: int = 0
    v_max: int = 118

    # ── Morphological cleanup ────────────────────────────────────────────────
    morph_kernel_size: int = 5          # increase for noisy cameras

    # ── Pillar geometry ──────────────────────────────────────────────────────
    # Fraction of obstacle width that counts as an "edge" region
    edge_fraction: float = 0.30
    # Minimum fraction of obstacle height a pillar must span vertically
    min_pillar_height_ratio: float = 0.20
    # Minimum fraction of the edge-region width occupied by a pillar
    min_pillar_width_ratio: float = 0.04
    # Minimum fraction of a column's pixels that must be blue
    min_col_density: float = 0.08

    # ── Bounding-box bottom padding ──────────────────────────────────────────
    box_bottom_pad: int = 15


# ─────────────────────────────────────────────────────────────────────────────
# YUV blue-mask
# ─────────────────────────────────────────────────────────────────────────────

def _blue_mask_yuv(crop_bgr: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """
    Binary mask of blue pixels using YUV decomposition.

    Why YUV?
    --------
    OpenCV's BGR→YUV conversion separates luminance (Y) from chrominance (U, V).
      U (Cb): blue-yellow axis — pure blue drives U above 128
      V (Cr): red-cyan axis   — pure blue drives V below 128
    Thresholding U and V together captures blue reliably across lighting
    conditions, independently of how dark or bright the pillar appears.
    """
    yuv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    mask = (
        (Y >= cfg.y_min) & (Y <= cfg.y_max) &
        (U >= cfg.u_min) & (U <= cfg.u_max) &
        (V >= cfg.v_min) & (V <= cfg.v_max)
    ).astype(np.uint8) * 255

    k = cfg.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # kill noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Pillar detector
# ─────────────────────────────────────────────────────────────────────────────

def _has_pillar(
    region_mask: np.ndarray,
    cfg: GateConfig,
    min_col_span: int,
    min_row_span: int,
) -> tuple[bool, tuple]:
    """
    Return (found, tight_bbox) where bbox = (x0, y0, x1, y1) in region coords.

    Algorithm
    ---------
    1. Compute per-column blue-pixel density (fraction of rows that are blue).
    2. Find the longest contiguous run of columns above min_col_density.
    3. Check that the run spans at least min_col_span columns.
    4. Within that band, count how many rows contain any blue pixel.
    5. Require at least min_row_span such rows (vertical coverage check).
    """
    h, w = region_mask.shape
    if h == 0 or w == 0:
        return False, (0, 0, 0, 0)

    col_density = region_mask.sum(axis=0) / (255.0 * h)   # (w,)
    dense_cols  = col_density >= cfg.min_col_density

    if not dense_cols.any():
        return False, (0, 0, 0, 0)

    # Longest run
    best_start = best_end = 0
    best_len   = 0
    run_start  = -1
    for i, d in enumerate(dense_cols):
        if d:
            if run_start == -1:
                run_start = i
        else:
            if run_start != -1:
                rlen = i - run_start
                if rlen > best_len:
                    best_len, best_start, best_end = rlen, run_start, i - 1
                run_start = -1
    if run_start != -1:
        rlen = len(dense_cols) - run_start
        if rlen > best_len:
            best_len, best_start, best_end = rlen, run_start, len(dense_cols) - 1

    if best_len < min_col_span:
        return False, (0, 0, 0, 0)

    pillar_band  = region_mask[:, best_start: best_end + 1]
    row_has_blue = pillar_band.sum(axis=1) > 0

    if int(row_has_blue.sum()) < min_row_span:
        return False, (0, 0, 0, 0)

    blue_rows = np.where(row_has_blue)[0]
    return True, (best_start, int(blue_rows[0]), best_end, int(blue_rows[-1]))


def detect_blue_pillars(crop_bgr: np.ndarray, cfg: GateConfig) -> dict:
    """
    Full blue-pillar analysis on an obstacle crop.

    Coordinate note — portrait image, floor on the LEFT
    ----------------------------------------------------
    The Bebop camera feed arrives rotated 90° CW: the image is portrait with
    the floor on the left edge and the sky on the right.  get_obstacle_info
    works in this native orientation, so the obstacle crop we receive here is
    also in portrait orientation.

    In this frame a gate pillar is a HORIZONTAL band: it runs across the full
    width of the crop but is confined to the TOP or BOTTOM edge (which
    correspond to the two sides of the gate opening in the real world).

    Therefore we search for blue pixel bands along the TOP and BOTTOM edges
    of the crop (rows), not the left/right edges (columns).

    Returns
    -------
    dict with keys:
        blue_mask        – binary mask of blue pixels (crop-sized)
        top_pillar       – bool: blue pillar on the top edge of the crop?
        bottom_pillar    – bool: blue pillar on the bottom edge of the crop?
        is_gate          – bool: both pillars present?
        top_bbox         – (x0,y0,x1,y1) in crop coords, or None
        bottom_bbox      – (x0,y0,x1,y1) in crop coords, or None
    """
    h, w = crop_bgr.shape[:2]
    result = dict(blue_mask=None, top_pillar=False, bottom_pillar=False,
                  is_gate=False, top_bbox=None, bottom_bbox=None)
    if h < 4 or w < 4:
        return result

    blue_mask = _blue_mask_yuv(crop_bgr, cfg)
    result["blue_mask"] = blue_mask

    # Edge band height (rows) and minimum spans
    edge_h     = max(1, int(h * cfg.edge_fraction))
    min_row_sp = max(1, int(edge_h * cfg.min_pillar_width_ratio))   # min contiguous dense rows
    min_col_sp = max(1, int(w     * cfg.min_pillar_height_ratio))   # min horizontal coverage

    # Top edge region: blue_mask[:edge_h, :]
    # Transpose so _has_pillar (which scans columns) now scans rows
    found_t, bbox_t = _has_pillar(blue_mask[:edge_h, :].T, cfg, min_row_sp, min_col_sp)
    result["top_pillar"] = found_t
    if found_t:
        # bbox_t is (col0, row0, col1, row1) in the transposed frame
        # → swap back: x = col (=original row), y = row (=original col)
        c0, r0, c1, r1 = bbox_t
        result["top_bbox"] = (r0, c0, r1, c1)   # (x0,y0,x1,y1) in crop coords

    # Bottom edge region: blue_mask[h-edge_h:, :]
    found_b, bbox_b = _has_pillar(blue_mask[h - edge_h:, :].T, cfg, min_row_sp, min_col_sp)
    result["bottom_pillar"] = found_b
    if found_b:
        c0, r0, c1, r1 = bbox_b
        # translate row coords back into full-crop row space
        result["bottom_bbox"] = (r0, h - edge_h + c0, r1, h - edge_h + c1)

    result["is_gate"] = found_t and found_b
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box reconstruction from get_obstacle_info output
# ─────────────────────────────────────────────────────────────────────────────

def _obstacle_bbox(
    det_row: np.ndarray,
    image_w: int,
    image_h: int,
    boundary_rows: np.ndarray,
    box_bottom_pad: int,
) -> tuple[int, int, int, int]:
    """
    Reconstruct pixel bounding box for one obstacle detection.

    get_obstacle_info works in a horizontally-flipped coordinate frame
    (mask_flipped = clean_mask[:, ::-1]).  boundary_rows[col_flipped] gives
    the ground-boundary row in that frame.  det_row[0] = left_x is already
    in original (un-flipped) x coordinates (see the flip-back on line
    "left = W - 1 - e" in get_obstacle_info).

    Returns (x_min, y_min, x_max, y_max) in original image coordinates.
    """
    left_x  = max(0, int(det_row[0]))
    right_x = min(image_w - 1, left_x + int(det_row[1]))

    # Convert original x → flipped column index for boundary lookup
    flipped_start = max(0,              image_w - 1 - right_x)
    flipped_end   = min(len(boundary_rows) - 1, image_w - 1 - left_x)

    local_bnd = boundary_rows[flipped_start: flipped_end + 1]
    valid     = local_bnd < image_w   # sentinels == image_w

    if valid.any():
        y_top = int(local_bnd[valid].min())
        y_bot = int(local_bnd[valid].max()) + box_bottom_pad
    else:
        y_top, y_bot = 0, image_h - 1

    return left_x, max(0, y_top), right_x, min(image_h - 1, y_bot)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text(img, text, pos, color, scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_detection(img, x_min, y_min, x_max, y_max, pillar_info):
    """Annotate one obstacle on img in-place.

    The image is portrait with floor on the LEFT, so pillars appear as
    horizontal bands at the TOP and BOTTOM of the obstacle crop.
    """
    is_gate       = pillar_info["is_gate"]
    top_pillar    = pillar_info["top_pillar"]
    bottom_pillar = pillar_info["bottom_pillar"]
    blue_mask     = pillar_info["blue_mask"]

    # Obstacle box: green = gate, red = no gate
    obs_colour = (0, 255, 0) if is_gate else (0, 0, 255)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), obs_colour, 2)
    label = "GATE" if is_gate else "obstacle"
    _text(img, label, (x_min + 4, max(y_min - 6, 16)), obs_colour)

    # Semi-transparent blue-mask overlay
    if blue_mask is not None:
        full = np.zeros(img.shape[:2], dtype=np.uint8)
        full[y_min:y_max, x_min:x_max] = blue_mask
        overlay = img.copy()
        overlay[full > 0] = [200, 160, 0]
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Top pillar box (bright blue) — top edge of obstacle crop
    if top_pillar and pillar_info["top_bbox"] is not None:
        bx0, by0, bx1, by1 = pillar_info["top_bbox"]
        ax0, ay0 = x_min + bx0, y_min + by0
        ax1, ay1 = x_min + bx1, y_min + by1
        cv2.rectangle(img, (ax0, ay0), (ax1, ay1), (255, 80, 0), 2)
        _text(img, "T-pillar", (ax0, max(ay0 - 4, 12)), (255, 80, 0), 0.45, 1)

    # Bottom pillar box (orange) — bottom edge of obstacle crop
    if bottom_pillar and pillar_info["bottom_bbox"] is not None:
        bx0, by0, bx1, by1 = pillar_info["bottom_bbox"]
        ax0, ay0 = x_min + bx0, y_min + by0
        ax1, ay1 = x_min + bx1, y_min + by1
        cv2.rectangle(img, (ax0, ay0), (ax1, ay1), (0, 140, 255), 2)
        _text(img, "B-pillar", (ax0, max(ay0 - 4, 12)), (0, 140, 255), 0.45, 1)

    # Pillar summary
    parts   = (["T"] if top_pillar else []) + (["B"] if bottom_pillar else [])
    summary = "pillars: " + ("+".join(parts) if parts else "none")
    _text(img, summary, (x_min + 4, min(y_max + 18, img.shape[0] - 4)),
          obs_colour, 0.45, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main per-frame function
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    image_bgr: np.ndarray,
    ground_baseline: Optional[np.ndarray],
    oa_params: dict,
    cfg: GateConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Run the complete pipeline on one downscaled BGR frame.

    Returns
    -------
    annotated        – BGR image with boxes drawn
    new_baseline     – updated ground baseline for the next frame
    gate_results     – list of per-obstacle dicts
    """
    H, W      = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    # ── 1. Existing obstacle detector ─────────────────────────────────────────
    detections, new_baseline, debug = oai.get_obstacle_info(
        image_bgr, ground_baseline, **oa_params
    )

    boundary_rows = debug.get("boundary_rows", np.full(W, W))
    gate_results  = []

    if detections.shape[0] == 0:
        _text(annotated, "No obstacle", (10, 30), (160, 160, 160))
        return annotated, new_baseline, gate_results

    # ── 2. Process each DET_OBSTACLE ─────────────────────────────────────────
    for row in detections:
        if int(row[2]) != DET_OBSTACLE:
            continue

        x_min, y_min, x_max, y_max = _obstacle_bbox(
            row, W, H, boundary_rows, cfg.box_bottom_pad
        )
        if x_max - x_min < 4 or y_max - y_min < 4:
            continue

        crop        = image_bgr[y_min:y_max, x_min:x_max]
        pillar_info = detect_blue_pillars(crop, cfg)

        gate_results.append({
            "bbox"        : (x_min, y_min, x_max, y_max),
            **pillar_info,
        })

        _draw_detection(annotated, x_min, y_min, x_max, y_max, pillar_info)

    if not gate_results:
        _text(annotated, "No obstacle", (10, 30), (160, 160, 160))

    return annotated, new_baseline, gate_results


# ─────────────────────────────────────────────────────────────────────────────
# Sequence runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence(
    image_paths: list[str],
    scale_factor: float = 0.8,
    oa_color_count_frac: float = 0.05,
    plant_scale_factor: float = 0.15 * 0.8,
    cfg: Optional[GateConfig] = None,
):
    """
    Run gate detection over a sorted list of image paths.
    Results are displayed live in an OpenCV window; nothing is saved to disk.
    Press 'q' to quit early.
    """
    if cfg is None:
        cfg = GateConfig()
    if not image_paths:
        print("No images to process.")
        return

    p        = oai.get_scaled_params(scale_factor)
    plant_bk = oai.scale_odd(5, plant_scale_factor)
    cfg.box_bottom_pad = p["box_bottom_pad"]

    oa_params = dict(
        oa_color_count_frac      = oa_color_count_frac,
        median_ksize             = p["median_ksize"],
        min_width                = p["min_width"],
        max_col_gap              = p["max_col_gap"],
        min_ground_pixels        = p["min_ground_pixels"],
        max_gap                  = p["max_gap"],
        smooth_kernel            = p["smooth_kernel"],
        obstacle_threshold       = p["obstacle_threshold"],
        no_ground_baseline       = p["no_ground_baseline"],
        plant_scale_factor       = plant_scale_factor,
        plant_blur_ksize         = plant_bk,
        plant_min_width          = 20,
        plant_min_pixels_per_col = 2,
        plant_max_col_gap        = 5,
    )

    probe = cv2.imread(image_paths[0])
    if probe is None:
        sys.exit(f"Cannot read {image_paths[0]}")
    orig_H, orig_W = probe.shape[:2]
    target_W = max(1, int(orig_W * scale_factor))
    target_H = max(1, int(orig_H * scale_factor))

    ground_baseline = np.full(target_H, p["no_ground_baseline"], dtype=np.float32)

    WIN = "Gate Detector  |  q = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, target_W, target_H)

    for idx, path in enumerate(image_paths):
        raw = cv2.imread(path)
        if raw is None:
            print(f"  [skip] {path}")
            continue

        frame = cv2.resize(raw, (target_W, target_H), interpolation=cv2.INTER_AREA)
        annotated, ground_baseline, results = process_frame(
            frame, ground_baseline, oa_params, cfg
        )

        for r in results:
            tag = "GATE DETECTED" if r["is_gate"] else "obstacle – no gate"
            print(f"  [{idx:04d}] {os.path.basename(path)}  bbox={r['bbox']}  {tag}")

        cv2.imshow(WIN, annotated)
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from glob import glob

    # ── Set your image folder here ────────────────────────────────────────────
    IMAGE_FOLDER = "../paparazzi10/DEVELOPMENT/downloads from drone/20260306-095826"
    # ─────────────────────────────────────────────────────────────────────────

    if not os.path.isdir(IMAGE_FOLDER):
        sys.exit(f"Folder not found: {IMAGE_FOLDER}")

    paths = sorted(glob(os.path.join(IMAGE_FOLDER, "*.jpg")))
    if not paths:
        paths = sorted(glob(os.path.join(IMAGE_FOLDER, "*.png")))
    if not paths:
        sys.exit(f"No .jpg or .png images found in {IMAGE_FOLDER}")

    print(f"Found {len(paths)} images in '{IMAGE_FOLDER}'")
    run_sequence(paths)