"""
mask_viewer.py
==============
Loads an image and overlays its ground, pole, tree masks and gate detection
on top, each in a different colour with adjustable opacity.

The display is rotated 90° CCW so that the ground (originally on the left
side of the raw fisheye frame) appears at the bottom, as expected.

Below the image a two-panel analysis strip is shown:
  Left  : per-column ground-pixel count (line plot), with the 7 bi-exp
           strips shaded in alternating colours and vertical dividers.
  Right : bar chart — total ground pixels per strip (the "score" used to
           pick the best direction).

Controls
--------
  a / d          — previous / next image
  1              — toggle ground mask  (green)
  2              — toggle pole mask    (orange)
  3              — toggle tree mask    (purple)
  4              — toggle gate overlay (cyan)
  +  / -         — increase / decrease opacity of all masks
  q              — quit
"""

import cv2
import numpy as np
import os
import glob
import sys
from dataclasses import dataclass
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_FOLDER      = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\latest_flight_all_generated_masks'

GROUND_MASK_DIR = os.path.join(MASKS_OUTPUT_ROOT, 'ground_masks')
POLE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'pole_masks')
TREE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'tree_masks')

# Overlay colours (BGR)
GROUND_COLOUR = (0,   255,   0)
POLE_COLOUR   = (0,   165, 255)
TREE_COLOUR   = (255,   0, 200)
GATE_COLOUR   = (255, 255,   0)

INITIAL_OPACITY = 0.5
OPACITY_STEP    = 0.05

# Strip analysis
N_STRIPS    = 7
BIEXP_EXP   = 0.7
STRIP_ALPHA = 0.18

# Hole filler defaults
HOLE_CLOSING_RADIUS = 0.04
HOLE_MIN_AREA       = 200

# Tree bounding-box padding tuning
# TREE_PAD_W_INIT      = 0
# TREE_PAD_H_INIT      = 0
TREE_PAD_STEP        = 5
# TREE_CLUSTER_MARGIN_INIT = 10   # initial dilation radius for blob clustering (px)
TREE_CLUSTER_STEP        = 5    # step size for ,/. keys
# TREE_MEAS_MODES      = ['pixels', 'ratio', 'aspect']


TREE_PAD_W_INIT              = 0
TREE_PAD_H_INIT              = 30
TREE_CLUSTER_MARGIN_INIT     = 35
TREE_MEAS_MODES              = ['pixels', 'ratio', 'aspect']
# set initial mode to aspect (index 2)

# Alternating strip shade colours (BGR) for the line-plot background
STRIP_SHADES = [
    (60,  60,  60),
    (90,  90,  90),
]

# Bar chart colour for the best strip
BAR_BEST_COLOUR    = (0, 220, 80)     # bright green
BAR_NORMAL_COLOUR  = (80, 140, 200)   # steel blue
BAR_AXIS_COLOUR    = (200, 200, 200)
PLOT_BG_COLOUR     = (30,  30,  30)

# ══════════════════════════════════════════════════════════════════════════════
# STRIP / PARTITION HELPERS  (your existing logic, embedded here)
# ══════════════════════════════════════════════════════════════════════════════

def get_partition_edges(W, n_partitions=7, spacing="biexp", biexp_exp=0.7):
    """
    Returns a list of (x_start, x_end) pixel ranges for each partition.
    W is the axis being divided (height of the rotated image).
    Strip 1 = bottom of rotated image (drone's left), strip 7 = top (drone's right).
    """
    if spacing == "uniform":
        edges = np.linspace(0, W, n_partitions + 1, dtype=int)

    elif spacing == "biexp":
        t      = np.linspace(-1, 1, n_partitions + 1)
        warped = np.sign(t) * (np.abs(t) ** biexp_exp)
        warped = (warped - warped[0]) / (warped[-1] - warped[0])
        edges  = (warped * W).astype(int)

    else:
        raise ValueError(f"Unknown spacing: '{spacing}'.")

    # Reverse so strip 1 is at the bottom of the rotated image
    pairs = [(int(edges[i]), int(edges[i + 1])) for i in range(n_partitions)]
    return list(reversed(pairs))


def partition_ground_counts(col_counts, partitions, total_rows):
    """
    col_counts  : 1-D array of length W_rot — ground pixels per column
    partitions  : list of (x0, x1) from get_partition_edges
    total_rows  : height of rotated image  (= original image width)
    Returns list of (ground_pixels, percentage) per strip.
    """
    result = []
    for (x0, x1) in partitions:
        total_pixels  = (x1 - x0) * total_rows
        ground_pixels = int(col_counts[x0:x1].sum())
        pct = 100.0 * ground_pixels / total_pixels if total_pixels > 0 else 0.0
        result.append((ground_pixels, pct))
    return result


def get_best_partition(part_data):
    pcts = [pct for (_, pct) in part_data]
    return int(np.argmax(pcts))


# ══════════════════════════════════════════════════════════════════════════════
# GATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    blue_y_min: int   = 0
    blue_y_max: int   = 220
    blue_u_min: int   = 122
    blue_u_max: int   = 255
    blue_v_min: int   = 0
    blue_v_max: int   = 123
    blue_morph_k: int = 5
    blob_min_area_ratio: float = 0.003
    max_aspect_diff: float = 0.5
    max_area_diff:   float = 0.6
    max_skew_deg:    float = 10
    display_scale: float = 1.5
    num_preview:  int  = 16
    save_figure:  bool = True
    save_path:    str  = 'gate_detection_results.png'


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


def extract_blobs(mask: np.ndarray, image_shape: tuple, cfg: Config) -> list:
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
            "x_min": x,      "y_min": y,
            "x_max": x + bw, "y_max": y + bh,
            "cx":    int(cents[i, 0]),
            "cy":    int(cents[i, 1]),
            "area":  area,
            "aspect": bw / max(bh, 1),
        })
    return blobs


def _are_parallel(b1, b2, cfg):
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


def find_gate_pair(blobs, cfg):
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


def gate_midpoint(b1, b2):
    return ((b1["cx"] + b2["cx"]) // 2, (b1["cy"] + b2["cy"]) // 2)


def process_frame(image_bgr, cfg):
    mask  = blue_mask(image_bgr, cfg)
    blobs = extract_blobs(mask, image_bgr.shape, cfg)
    pair  = find_gate_pair(blobs, cfg)
    mid   = gate_midpoint(*pair) if pair is not None else None
    return mask, blobs, pair, mid


def draw_gate_overlay(base, mask, blobs, pair, mid, opacity):
    out      = base.copy()
    coloured = np.zeros_like(out)
    coloured[mask > 0] = GATE_COLOUR
    out = cv2.addWeighted(out, 1.0, coloured, opacity, 0)
    for b in blobs:
        cv2.rectangle(out, (b["x_min"], b["y_min"]),
                      (b["x_max"], b["y_max"]), (120, 120, 120), 1)
    if pair is not None:
        for b in pair:
            cv2.rectangle(out, (b["x_min"], b["y_min"]),
                          (b["x_max"], b["y_max"]), GATE_COLOUR, 2)
    if mid is not None:
        cv2.drawMarker(out, mid, (0, 255, 0),
                       cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(out, mid, 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(out, f'GATE ({mid[0]},{mid[1]})',
                    (max(0, mid[0] - 45), max(16, mid[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MASK HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_mask(mask_dir, stem, suffix):
    path = os.path.join(mask_dir, f'{stem}_mask_{suffix}.png')
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def apply_colour_overlay(base, mask, colour, opacity):
    if mask is None:
        return base
    coloured           = np.zeros_like(base)
    coloured[mask > 0] = colour
    return cv2.addWeighted(base, 1.0, coloured, opacity, 0)


def make_bw_panel(mask, label, W, H):
    if mask is None:
        panel = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(panel, f'{label}: NOT FOUND', (10, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return panel
    resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    panel   = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# GROUND HOLE FILLER
# ══════════════════════════════════════════════════════════════════════════════

def fill_ground_holes(ground_mask, closing_radius=0.04, min_hole_area=200):
    """
    Fills holes and carpet-like interruptions in the ground mask using
    morphological closing, constrained to the region that is plausibly
    ground (below the top of the existing ground region).

    Strategy
    --------
    1. Find the topmost row that contains any ground pixel — everything
       above that row is sky/obstacle and will never be filled.
    2. Apply morphological closing with a large elliptical kernel.
       Closing = dilation followed by erosion: it bridges gaps and fills
       holes without expanding the overall ground boundary much.
    3. The fill is only accepted in pixels that are BELOW the ground
       top-row found in step 1, preventing false fills in the sky.
    4. A minimum-area filter removes tiny isolated specks.

    Parameters
    ----------
    ground_mask    : uint8 grayscale mask (255 = ground, 0 = non-ground)
    closing_radius : kernel radius as fraction of image height (0.02–0.10)
    min_hole_area  : connected non-ground regions smaller than this that
                     sit inside the closed mask are filled regardless
    """
    if ground_mask is None:
        return None

    H, W   = ground_mask.shape
    binary = (ground_mask > 0).astype(np.uint8)

    # ── Step 1: find the top of the ground region ────────────────────────────
    rows_with_ground = np.any(binary > 0, axis=1)
    if not rows_with_ground.any():
        return ground_mask   # no ground at all, nothing to do
    top_ground_row = int(np.argmax(rows_with_ground))   # first True

    # ── Step 2: morphological closing ────────────────────────────────────────
    ksize  = max(3, int(H * closing_radius) | 1)        # odd, scales with H
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Step 3: constrain fill to below ground top-row ───────────────────────
    # Only accept new pixels (closed=1, binary=0) below the ground top
    new_pixels        = ((closed == 1) & (binary == 0)).astype(np.uint8)
    new_pixels[:top_ground_row, :] = 0     # mask out sky region

    # ── Step 4: min-area filter on the new pixels ────────────────────────────
    n, labels, stats, _ = cv2.connectedComponentsWithStats(new_pixels,
                                                           connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
            new_pixels[labels == i] = 0

    result = binary.copy()
    result[new_pixels == 1] = 1
    return (result * 255).astype(np.uint8)


def clean_ground_blobs(ground_mask, bottom_frac=0.025):
    """
    Keeps only ground pixels connected to the "ground root" region.

    The drone camera is rotated 90° CCW for display, so the floor appears
    at the BOTTOM of the rotated view — which corresponds to the LEFT COLUMNS
    of the original (unrotated) mask.  The seed region is therefore the
    leftmost `bottom_frac` of columns in the original mask.

    Everything in the seed band is always kept.
    Everything outside the seed band is kept only if it is connected
    (via a continuous path of ground pixels) to the seed band.

    If cleaning removes all pixels (e.g. no ground reaches the seed band
    at all), the original mask is returned unchanged as a safety fallback.
    """
    if ground_mask is None:
        return None

    H, W   = ground_mask.shape
    binary = (ground_mask > 0).astype(np.uint8)

    # Seed = leftmost columns of original mask (= bottom of rotated view)
    seed_end = max(1, int(W * bottom_frac))

    _, labels = cv2.connectedComponents(binary, connectivity=8)

    seed_labels = set(np.unique(labels[:, :seed_end]))
    seed_labels.discard(0)

    if not seed_labels:
        return ground_mask   # nothing touches seed band — return original

    keep = np.isin(labels, list(seed_labels)).astype(np.uint8)

    if keep.sum() == 0:
        return ground_mask   # safety fallback

    return (keep * 255).astype(np.uint8)





def carve_poles_from_ground(ground_mask, pole_mask):
    """
    Removes ground pixels that are covered by the pole mask.
    Applied last so the blob cleaner and hole filler are unaffected.
    If either mask is None the ground mask is returned unchanged.
    """
    if ground_mask is None or pole_mask is None:
        return ground_mask

    result = ground_mask.copy()
    result[pole_mask > 0] = 0
    return result


def get_tree_bboxes(tree_mask, pad_w=0, pad_h=0, cluster_margin=10):
    """
    Clusters nearby tree blobs by dilating the mask by `cluster_margin` pixels
    before finding connected components — blobs within that distance merge into
    one.  The tight bounding box is then computed over the ORIGINAL (undilated)
    pixels belonging to each cluster, so the margin only affects grouping, not
    the box size.  Padding is then added on top of the tight box.

    Parameters
    ----------
    tree_mask      : uint8 grayscale mask
    pad_w          : extra pixels added to each horizontal side of tight box
    pad_h          : extra pixels added to each vertical side of tight box
    cluster_margin : dilation radius (px) used for merging nearby blobs
    """
    if tree_mask is None:
        return []

    H, W   = tree_mask.shape
    binary = (tree_mask > 0).astype(np.uint8)

    if binary.sum() == 0:
        return []

    # ── Cluster by dilation ───────────────────────────────────────────────────
    if cluster_margin > 0:
        k       = cluster_margin * 2 + 1
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(binary, kernel, iterations=1)
    else:
        dilated = binary

    _, cluster_labels = cv2.connectedComponents(dilated, connectivity=8)

    # ── For each cluster, get tight box over ORIGINAL pixels ─────────────────
    boxes = []
    for cid in np.unique(cluster_labels):
        if cid == 0:
            continue
        # Original pixels that belong to this cluster
        orig_pixels = (binary == 1) & (cluster_labels == cid)
        if not orig_pixels.any():
            continue

        rows = np.where(orig_pixels.any(axis=1))[0]
        cols = np.where(orig_pixels.any(axis=0))[0]
        ty, by_ = int(rows[0]), int(rows[-1])
        tx, bx_ = int(cols[0]), int(cols[-1])
        tw = bx_ - tx + 1
        th = by_ - ty + 1

        # Padded box clamped to image
        px  = max(0,     tx - pad_w)
        py  = max(0,     ty - pad_h)
        px2 = min(W - 1, tx + tw + pad_w)
        py2 = min(H - 1, ty + th + pad_h)
        pw  = px2 - px
        ph  = py2 - py

        boxes.append({
            'tight':  (tx, ty, tw, th),
            'padded': (px, py, pw, ph),
        })
    return boxes


def draw_tree_bboxes(image, tree_mask, pad_w, pad_h, meas_mode, cluster_margin):
    """
    Draws padded tree bounding boxes onto image (already rotated).
    tree_mask must be in the same (rotated) coordinate space as image.

    meas_mode : 'pixels'  → show padded W×H in pixels
                'ratio'   → show padded/tight ratio for W and H
                'aspect'  → show W/H aspect ratio of padded box
    """
    boxes = get_tree_bboxes(tree_mask, pad_w, pad_h, cluster_margin)
    out   = image.copy()

    for b in boxes:
        tx, ty, tw, th = b['tight']
        px, py, pw, ph = b['padded']

        # Tight box (grey)
        cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (160, 160, 160), 1)

        # Padded box (bright magenta)
        cv2.rectangle(out, (px, py), (px + pw, py + ph), (255, 80, 255), 2)

        # Measurement label
        if meas_mode == 'pixels':
            label = f'{pw}x{ph}px'
        elif meas_mode == 'ratio':
            rw = pw / tw if tw > 0 else 0
            rh = ph / th if th > 0 else 0
            label = f'W:{rw:.2f}x H:{rh:.2f}x'
        else:  # aspect
            label = f'AR:{pw/ph:.2f}' if ph > 0 else 'AR:--'

        lx = px
        ly = max(12, py - 4)
        cv2.putText(out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 255), 1, cv2.LINE_AA)

    # HUD
    H_img = out.shape[0]
    cv2.putText(out,
                f'TreeBox  pad_w={pad_w}  pad_h={pad_h}  margin={cluster_margin}  [{meas_mode}]',
                (8, H_img - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(out, '[/]=pad_w   ;/\'=pad_h   ,/.=margin   m=mode   t=toggle',
                (8, H_img - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return out



def build_analysis_panel(ground_mask_rot, W_rot, H_rot, panel_w, panel_h):
    """
    Builds a side-by-side (line plot | bar chart) analysis panel.

    ground_mask_rot : the ground mask already rotated 90° CCW  (H_rot × W_rot)
    W_rot           : width of rotated image  — THIS is the strip axis (left→right)
    H_rot           : height of rotated image
    panel_w, panel_h: pixel size of the whole analysis panel
    """

    # Per-column ground counts: sum down each column (axis=0) → length W_rot
    if ground_mask_rot is not None:
        col_counts = (ground_mask_rot > 0).astype(np.int32).sum(axis=0).astype(float)
    else:
        col_counts = np.zeros(W_rot, dtype=float)

    partitions = get_partition_edges(W_rot, N_STRIPS, spacing="biexp",
                                     biexp_exp=BIEXP_EXP)
    part_data  = partition_ground_counts(col_counts, partitions, H_rot)
    best_idx   = get_best_partition(part_data)

    half_w     = panel_w // 2
    half_w2    = panel_w - half_w   # right panel absorbs any odd pixel
    plot_h     = panel_h

    # ── LEFT: per-column line plot ────────────────────────────────────────────
    line_panel = np.full((plot_h, half_w, 3), PLOT_BG_COLOUR, dtype=np.uint8)

    pad_top    = 24
    pad_bottom = 30
    pad_left   = 40
    pad_right  = 10
    draw_w     = half_w  - pad_left  - pad_right
    draw_h     = plot_h  - pad_top   - pad_bottom

    max_count  = col_counts.max() if col_counts.max() > 0 else 1.0

    # Strip shading
    for si, (x0, x1) in enumerate(partitions):
        shade   = STRIP_SHADES[si % len(STRIP_SHADES)]
        px0 = pad_left + int(x0 / W_rot * draw_w)
        px1 = pad_left + int(x1 / W_rot * draw_w)
        overlay = line_panel.copy()
        cv2.rectangle(overlay, (px0, pad_top), (px1, pad_top + draw_h), shade, -1)
        cv2.addWeighted(overlay, STRIP_ALPHA, line_panel, 1 - STRIP_ALPHA, 0, line_panel)
        # Divider line
        cv2.line(line_panel, (px0, pad_top), (px0, pad_top + draw_h),
                 (100, 100, 100), 1)
        # Strip label
        mid_px = (px0 + px1) // 2
        cv2.putText(line_panel, str(si + 1),
                    (mid_px - 4, pad_top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Highlight best strip
    bx0 = pad_left + int(partitions[best_idx][0] / W_rot * draw_w)
    bx1 = pad_left + int(partitions[best_idx][1] / W_rot * draw_w)
    highlight = line_panel.copy()
    cv2.rectangle(highlight, (bx0, pad_top), (bx1, pad_top + draw_h),
                  BAR_BEST_COLOUR, -1)
    cv2.addWeighted(highlight, 0.22, line_panel, 0.78, 0, line_panel)

    # Y-axis label
    cv2.putText(line_panel, 'ground px',
                (2, pad_top + draw_h // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

    # Axes
    cv2.line(line_panel,
             (pad_left, pad_top), (pad_left, pad_top + draw_h),
             BAR_AXIS_COLOUR, 1)
    cv2.line(line_panel,
             (pad_left, pad_top + draw_h),
             (pad_left + draw_w, pad_top + draw_h),
             BAR_AXIS_COLOUR, 1)

    # Y-axis ticks
    for frac in (0.25, 0.5, 0.75, 1.0):
        ty    = pad_top + draw_h - int(frac * draw_h)
        label = str(int(frac * max_count))
        cv2.line(line_panel, (pad_left - 3, ty), (pad_left, ty),
                 BAR_AXIS_COLOUR, 1)
        cv2.putText(line_panel, label, (2, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)

    # Line plot
    xs = np.linspace(0, len(col_counts) - 1, len(col_counts))
    px_arr = (pad_left + xs / W_rot * draw_w).astype(int)
    py_arr = (pad_top + draw_h - (col_counts / max_count) * draw_h).astype(int)
    pts    = np.stack([px_arr, py_arr], axis=1).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(line_panel, [pts], False, (0, 230, 80), 1, cv2.LINE_AA)

    # Title
    cv2.putText(line_panel, 'Ground pixels per column',
                (pad_left, plot_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    # ── RIGHT: bar chart ──────────────────────────────────────────────────────
    bar_panel  = np.full((plot_h, half_w2, 3), PLOT_BG_COLOUR, dtype=np.uint8)

    pad_top_b  = 24
    pad_bot_b  = 30
    pad_left_b = 44
    pad_right_b= 10
    draw_wb    = half_w2  - pad_left_b  - pad_right_b
    draw_hb    = plot_h   - pad_top_b   - pad_bot_b

    max_pct    = max((pct for (_, pct) in part_data), default=1.0)
    if max_pct == 0:
        max_pct = 1.0

    bar_gap    = max(1, draw_wb // (N_STRIPS * 6))
    bar_w      = (draw_wb - bar_gap * (N_STRIPS + 1)) // N_STRIPS

    for si, (gp, pct) in enumerate(part_data):
        bx     = pad_left_b + bar_gap + si * (bar_w + bar_gap)
        bh     = int(pct / max_pct * draw_hb)
        by     = pad_top_b + draw_hb - bh
        colour = BAR_BEST_COLOUR if si == best_idx else BAR_NORMAL_COLOUR
        cv2.rectangle(bar_panel, (bx, by), (bx + bar_w, pad_top_b + draw_hb),
                      colour, -1)
        # Percentage label above bar
        label  = f'{pct:.1f}%'
        lx     = bx + bar_w // 2 - len(label) * 3
        ly     = max(pad_top_b - 2, by - 3)
        cv2.putText(bar_panel, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1)
        # Strip number below
        cv2.putText(bar_panel, str(si + 1),
                    (bx + bar_w // 2 - 4, pad_top_b + draw_hb + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Axes
    cv2.line(bar_panel,
             (pad_left_b, pad_top_b), (pad_left_b, pad_top_b + draw_hb),
             BAR_AXIS_COLOUR, 1)
    cv2.line(bar_panel,
             (pad_left_b, pad_top_b + draw_hb),
             (pad_left_b + draw_wb, pad_top_b + draw_hb),
             BAR_AXIS_COLOUR, 1)

    # Y-axis ticks
    for frac in (0.25, 0.5, 0.75, 1.0):
        ty    = pad_top_b + draw_hb - int(frac * draw_hb)
        label = f'{frac * max_pct:.0f}%'
        cv2.line(bar_panel, (pad_left_b - 3, ty), (pad_left_b, ty),
                 BAR_AXIS_COLOUR, 1)
        cv2.putText(bar_panel, label, (2, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)

    # "BEST → strip N" annotation
    cv2.putText(bar_panel,
                f'BEST: strip {best_idx + 1}',
                (pad_left_b, plot_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, BAR_BEST_COLOUR, 1)

    # Title
    cv2.putText(bar_panel, 'Ground % per strip',
                (pad_left_b + draw_wb // 2 - 40, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return np.hstack((line_panel, bar_panel)), best_idx, partitions


# ══════════════════════════════════════════════════════════════════════════════
# FRAME BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_frame(img, ground_mask, pole_mask, tree_mask,
                gate_data,
                show_ground, show_pole, show_tree, show_gate, opacity,
                show_tree_bbox=False, tree_pad_w=0, tree_pad_h=0,
                tree_meas_mode='pixels', tree_cluster_margin=10):
    """
    img and all masks are in their ORIGINAL orientation.
    Everything is rotated 90° CCW before display.
    """
    # ── Apply overlays on original image ─────────────────────────────────────
    top = img.copy()
    if show_ground: top = apply_colour_overlay(top, ground_mask, GROUND_COLOUR, opacity)
    if show_pole:   top = apply_colour_overlay(top, pole_mask,   POLE_COLOUR,   opacity)
    if show_tree:   top = apply_colour_overlay(top, tree_mask,   TREE_COLOUR,   opacity)
    if show_gate:
        gate_mask, blobs, pair, mid = gate_data
        top = draw_gate_overlay(top, gate_mask, blobs, pair, mid, opacity)

    # ── Rotate 90° CCW ───────────────────────────────────────────────────────
    top_rot = cv2.rotate(top, cv2.ROTATE_90_COUNTERCLOCKWISE)

    H_rot, W_rot = top_rot.shape[:2]

    # Rotate ground mask for analysis
    ground_rot = None
    if ground_mask is not None:
        ground_rot = cv2.rotate(ground_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ── Strip divider lines on rotated image (vertical strips) ───────────────
    partitions = get_partition_edges(W_rot, N_STRIPS, spacing="biexp",
                                     biexp_exp=BIEXP_EXP)
    for si, (x0, _) in enumerate(partitions):
        if si == 0:
            continue
        cv2.line(top_rot, (x0, 0), (x0, H_rot), (200, 200, 200), 1)

    # Strip numbers along top of image
    for si, (x0, x1) in enumerate(partitions):
        mid_x = (x0 + x1) // 2
        cv2.putText(top_rot, str(si + 1),
                    (mid_x - 5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Tree bounding boxes (drawn on rotated image) ──────────────────────────
    tree_rot = None
    if tree_mask is not None:
        tree_rot = cv2.rotate(tree_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if show_tree_bbox:
        top_rot = draw_tree_bboxes(top_rot, tree_rot, tree_pad_w, tree_pad_h,
                                   tree_meas_mode, tree_cluster_margin)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = []
    if show_ground: legend_items.append(('[1] Ground', GROUND_COLOUR))
    if show_pole:   legend_items.append(('[2] Pole',   POLE_COLOUR))
    if show_tree:   legend_items.append(('[3] Tree',   TREE_COLOUR))
    if show_gate:   legend_items.append(('[4] Gate',   GATE_COLOUR))

    for i, (label, colour) in enumerate(legend_items):
        cv2.putText(top_rot, label, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    cv2.putText(top_rot, f'opacity: {opacity:.2f}  (+/-)',
                (8, H_rot - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Bottom panel: raw B&W masks (also rotated) ────────────────────────────
    panel_h = H_rot // 3
    # Divide W_rot into exactly 3 panels that sum to W_rot (no off-by-one)
    pw0 = W_rot // 3
    pw1 = W_rot // 3
    pw2 = W_rot - pw0 - pw1   # absorbs any remainder

    def rot_mask(m):
        if m is None: return None
        return cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE)

    ground_panel = make_bw_panel(rot_mask(ground_mask), 'Ground', pw0, panel_h)
    pole_panel   = make_bw_panel(rot_mask(pole_mask),   'Pole',   pw1, panel_h)
    tree_panel   = make_bw_panel(rot_mask(tree_mask),   'Tree',   pw2, panel_h)
    bw_row       = np.hstack((ground_panel, pole_panel, tree_panel))

    # ── Analysis panel ────────────────────────────────────────────────────────
    analysis_h    = max(120, H_rot // 3)
    analysis_panel, best_idx, _ = build_analysis_panel(
        ground_rot, W_rot, H_rot, W_rot, analysis_h)

    if analysis_panel.shape[1] != W_rot:
        analysis_panel = cv2.resize(analysis_panel,
                                    (W_rot, analysis_panel.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

    # Highlight best strip on the rotated image (vertical band)
    bx0, bx1 = partitions[best_idx]
    highlight = top_rot.copy()
    cv2.rectangle(highlight, (bx0, 0), (bx1, H_rot), BAR_BEST_COLOUR, -1)
    cv2.addWeighted(highlight, 0.15, top_rot, 0.85, 0, top_rot)
    cv2.line(top_rot, (bx0, 0), (bx0, H_rot), BAR_BEST_COLOUR, 2)
    cv2.line(top_rot, (bx1, 0), (bx1, H_rot), BAR_BEST_COLOUR, 2)

    # ── Force all panels to identical width before stacking ──────────────────
    def _fix_w(arr, w):
        if arr.shape[1] == w:
            return arr
        return cv2.resize(arr, (w, arr.shape[0]), interpolation=cv2.INTER_NEAREST)

    bw_row        = _fix_w(bw_row,        W_rot)
    analysis_panel = _fix_w(analysis_panel, W_rot)

    return np.vstack((top_rot, bw_row, analysis_panel))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) +
                        glob.glob(os.path.join(IMAGE_FOLDER, '*.png')))
    all_images = [p for p in all_images
                  if '_mask' not in os.path.basename(p).lower()]

    def has_any_mask(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return any(os.path.exists(os.path.join(d, f'{stem}_mask_{s}.png'))
                   for d, s in [(GROUND_MASK_DIR, 'ground'),
                                (POLE_MASK_DIR,   'pole'),
                                (TREE_MASK_DIR,   'tree')])

    image_paths = [p for p in all_images if has_any_mask(p)]

    if not image_paths:
        print('No images with masks found. Run generate_masks.py first.')
        sys.exit(1)

    print(f'Found {len(image_paths)} images with masks.')

    gate_cfg = Config()

    WINDOW = ('Mask Viewer  |  a/d=prev/next  |  1/2/3/4=toggle  '
              '|  5=fill  t=treebox  [/]=pad_w  ;/\'=pad_h  ,/.=margin  m=mode  |  +/-=opacity  |  q=quit')
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    idx          = 0
    opacity      = INITIAL_OPACITY
    show_ground  = True
    show_pole    = True
    show_tree    = True
    show_gate    = True
    show_fill      = False
    show_tree_bbox       = False
    tree_pad_w           = TREE_PAD_W_INIT
    tree_pad_h           = TREE_PAD_H_INIT
    tree_cluster_margin  = TREE_CLUSTER_MARGIN_INIT
    tree_meas_idx        = 0
    needs_redraw         = True

    while True:
        if needs_redraw:
            path = image_paths[idx]
            stem = os.path.splitext(os.path.basename(path))[0]

            img         = cv2.imread(path)
            ground_mask = load_mask(GROUND_MASK_DIR, stem, 'ground')
            pole_mask   = load_mask(POLE_MASK_DIR,   stem, 'pole')
            tree_mask   = load_mask(TREE_MASK_DIR,   stem, 'tree')
            gate_data   = process_frame(img, gate_cfg)

            # Step 1 (always): remove all ground blobs not touching the seed band
            display_ground = clean_ground_blobs(ground_mask)
            # Step 2 (toggle 5): fill carpet-like holes in the cleaned mask
            if show_fill:
                display_ground = fill_ground_holes(display_ground,
                                                   HOLE_CLOSING_RADIUS,
                                                   HOLE_MIN_AREA)
            # Step 3 (always): carve pole footprints out of ground last
            display_ground = carve_poles_from_ground(display_ground, pole_mask)

            mid      = gate_data[3]
            gate_str = f'GATE at {mid}' if mid is not None else 'no gate'

            frame = build_frame(img, display_ground, pole_mask, tree_mask,
                                gate_data,
                                show_ground, show_pole, show_tree, show_gate,
                                opacity,
                                show_tree_bbox, tree_pad_w, tree_pad_h,
                                TREE_MEAS_MODES[tree_meas_idx],
                                tree_cluster_margin)

            cv2.setWindowTitle(WINDOW,
                f'[{idx+1}/{len(image_paths)}]  {stem}  |  {gate_str}  |  '
                f'G={"ON" if show_ground else "off"}  '
                f'P={"ON" if show_pole else "off"}  '
                f'T={"ON" if show_tree else "off"}  '
                f'Gate={"ON" if show_gate else "off"}  '
                f'Fill={"ON" if show_fill else "off"}  '
                f'TreeBox={"ON" if show_tree_bbox else "off"}'
                + (f'  pw={tree_pad_w} ph={tree_pad_h} mg={tree_cluster_margin} [{TREE_MEAS_MODES[tree_meas_idx]}]'
                   if show_tree_bbox else '') +
                f'  opacity={opacity:.2f}')
            cv2.imshow(WINDOW, frame)
            needs_redraw = False

        key = cv2.waitKey(15)

        if key == ord('q'):
            break
        elif key in (ord('d'), 83, 65363):
            idx = (idx + 1) % len(image_paths)
            needs_redraw = True
        elif key in (ord('a'), 81, 65361):
            idx = (idx - 1) % len(image_paths)
            needs_redraw = True
        elif key == ord('1'):
            show_ground  = not show_ground
            needs_redraw = True
        elif key == ord('2'):
            show_pole    = not show_pole
            needs_redraw = True
        elif key == ord('3'):
            show_tree    = not show_tree
            needs_redraw = True
        elif key == ord('4'):
            show_gate    = not show_gate
            needs_redraw = True
        elif key == ord('5'):
            show_fill      = not show_fill
            needs_redraw   = True
        elif key == ord('t'):
            show_tree_bbox = not show_tree_bbox
            needs_redraw   = True
        elif key == ord('['):
            tree_pad_w     = max(0, tree_pad_w - TREE_PAD_STEP)
            needs_redraw   = True
        elif key == ord(']'):
            tree_pad_w    += TREE_PAD_STEP
            needs_redraw   = True
        elif key == ord(';'):
            tree_pad_h     = max(0, tree_pad_h - TREE_PAD_STEP)
            needs_redraw   = True
        elif key == ord("'"):
            tree_pad_h    += TREE_PAD_STEP
            needs_redraw   = True
        elif key == ord(','):
            tree_cluster_margin = max(0, tree_cluster_margin - TREE_CLUSTER_STEP)
            needs_redraw   = True
        elif key == ord('.'):
            tree_cluster_margin += TREE_CLUSTER_STEP
            needs_redraw   = True
        elif key == ord('m'):
            tree_meas_idx  = (tree_meas_idx + 1) % len(TREE_MEAS_MODES)
            needs_redraw   = True
        elif key in (ord('+'), ord('=')):
            opacity      = min(1.0, opacity + OPACITY_STEP)
            needs_redraw = True
        elif key == ord('-'):
            opacity      = max(0.0, opacity - OPACITY_STEP)
            needs_redraw = True

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()