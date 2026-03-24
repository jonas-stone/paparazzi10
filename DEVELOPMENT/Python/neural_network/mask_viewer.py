# """
# mask_viewer.py
# ==============
# Loads an image and overlays its ground, pole and tree masks on top,
# each in a different colour with adjustable opacity.
#
# Controls
# --------
#   a / d          — previous / next image
#   1              — toggle ground mask  (green)
#   2              — toggle pole mask    (orange)
#   3              — toggle tree mask    (purple)
#   +  / -         — increase / decrease opacity of all masks
#   q              — quit
#
# Layout
# ------
#   Top panel    : original image + mask overlays
#   Bottom panel : the three raw masks side by side (B&W)
# """
#
# import cv2
# import numpy as np
# import os
# import glob
# import sys
#
# # ══════════════════════════════════════════════════════════════════════════════
# # CONFIG
# # ══════════════════════════════════════════════════════════════════════════════
#
# IMAGE_FOLDER      = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
# MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\testorni_generated_masks_new_test'
#
# GROUND_MASK_DIR = os.path.join(MASKS_OUTPUT_ROOT, 'ground_masks')
# POLE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'pole_masks')
# TREE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'tree_masks')
#
# # Overlay colours (BGR)
# GROUND_COLOUR = (0,   255,   0)    # green
# POLE_COLOUR   = (0,   165, 255)    # orange
# TREE_COLOUR   = (255,   0, 200)    # purple
#
# INITIAL_OPACITY   = 0.5    # 0.0 = invisible, 1.0 = fully opaque
# OPACITY_STEP      = 0.05
#
# # ══════════════════════════════════════════════════════════════════════════════
# # HELPERS
# # ══════════════════════════════════════════════════════════════════════════════
#
# def load_mask(mask_dir, stem, suffix):
#     path = os.path.join(mask_dir, f'{stem}_mask_{suffix}.png')
#     if not os.path.exists(path):
#         return None
#     return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#
#
# def apply_colour_overlay(base, mask, colour, opacity):
#     """Paints 'colour' onto 'base' wherever mask > 0, blended by opacity."""
#     if mask is None:
#         return base
#     coloured        = np.zeros_like(base)
#     coloured[mask > 0] = colour
#     return cv2.addWeighted(base, 1.0, coloured, opacity, 0)
#
#
# def make_bw_panel(mask, label, W, H):
#     """Renders a single B&W mask as a labelled BGR panel of size (H, W)."""
#     if mask is None:
#         panel = np.zeros((H, W, 3), dtype=np.uint8)
#         cv2.putText(panel, f'{label}: NOT FOUND', (10, H // 2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
#         return panel
#
#     resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
#     panel   = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
#     cv2.putText(panel, label, (8, 22),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
#     return panel
#
#
# def build_frame(img, ground_mask, pole_mask, tree_mask,
#                 show_ground, show_pole, show_tree, opacity):
#     H, W = img.shape[:2]
#
#     # ── Top panel: overlays on original ──────────────────────────────────────
#     top = img.copy()
#     if show_ground: top = apply_colour_overlay(top, ground_mask, GROUND_COLOUR, opacity)
#     if show_pole:   top = apply_colour_overlay(top, pole_mask,   POLE_COLOUR,   opacity)
#     if show_tree:   top = apply_colour_overlay(top, tree_mask,   TREE_COLOUR,   opacity)
#
#     # Legend
#     legend_items = []
#     if show_ground: legend_items.append(('[1] Ground', GROUND_COLOUR))
#     if show_pole:   legend_items.append(('[2] Pole',   POLE_COLOUR))
#     if show_tree:   legend_items.append(('[3] Tree',   TREE_COLOUR))
#
#     for i, (label, colour) in enumerate(legend_items):
#         cv2.putText(top, label, (8, 22 + i * 22),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
#
#     cv2.putText(top, f'opacity: {opacity:.2f}  (+/-)',
#                 (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
#
#     # ── Bottom panel: raw B&W masks side by side ──────────────────────────────
#     panel_w = W // 3
#     panel_h = H // 3   # shorter strip below
#
#     ground_panel = make_bw_panel(ground_mask, 'Ground', panel_w, panel_h)
#     pole_panel   = make_bw_panel(pole_mask,   'Pole',   panel_w, panel_h)
#     tree_panel   = make_bw_panel(tree_mask,   'Tree',   panel_w, panel_h)
#
#     bottom = np.hstack((ground_panel, pole_panel, tree_panel))
#
#     return np.vstack((top, bottom))
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════
#
# def main():
#     # Collect images that have at least one mask generated
#     all_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) +
#                         glob.glob(os.path.join(IMAGE_FOLDER, '*.png')))
#     all_images = [p for p in all_images if '_mask' not in os.path.basename(p).lower()]
#
#     # Only keep images that have at least one mask
#     def has_any_mask(path):
#         stem = os.path.splitext(os.path.basename(path))[0]
#         return any(os.path.exists(os.path.join(d, f'{stem}_mask_{s}.png'))
#                    for d, s in [(GROUND_MASK_DIR, 'ground'),
#                                 (POLE_MASK_DIR,   'pole'),
#                                 (TREE_MASK_DIR,   'tree')])
#
#     image_paths = [p for p in all_images if has_any_mask(p)]
#
#     if not image_paths:
#         print('No images with masks found. Run generate_masks.py first.')
#         sys.exit(1)
#
#     print(f'Found {len(image_paths)} images with masks.')
#
#     WINDOW = 'Mask Viewer  |  a/d = prev/next  |  1/2/3 = toggle masks  |  +/- = opacity  |  q = quit'
#     cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
#
#     idx          = 0
#     opacity      = INITIAL_OPACITY
#     show_ground  = True
#     show_pole    = True
#     show_tree    = True
#     needs_redraw = True
#
#     while True:
#         if needs_redraw:
#             path = image_paths[idx]
#             stem = os.path.splitext(os.path.basename(path))[0]
#
#             img         = cv2.imread(path)
#             ground_mask = load_mask(GROUND_MASK_DIR, stem, 'ground')
#             pole_mask   = load_mask(POLE_MASK_DIR,   stem, 'pole')
#             tree_mask   = load_mask(TREE_MASK_DIR,   stem, 'tree')
#
#             frame = build_frame(img, ground_mask, pole_mask, tree_mask,
#                                 show_ground, show_pole, show_tree, opacity)
#
#             cv2.setWindowTitle(WINDOW,
#                 f'[{idx+1}/{len(image_paths)}]  {stem}  |  '
#                 f'ground={"ON" if show_ground else "off"}  '
#                 f'pole={"ON" if show_pole else "off"}  '
#                 f'tree={"ON" if show_tree else "off"}  '
#                 f'opacity={opacity:.2f}')
#             cv2.imshow(WINDOW, frame)
#             needs_redraw = False
#
#         key = cv2.waitKey(15)
#
#         if key == ord('q'):
#             break
#         elif key in (ord('d'), 83, 65363):
#             idx = (idx + 1) % len(image_paths)
#             needs_redraw = True
#         elif key in (ord('a'), 81, 65361):
#             idx = (idx - 1) % len(image_paths)
#             needs_redraw = True
#         elif key == ord('1'):
#             show_ground  = not show_ground
#             needs_redraw = True
#         elif key == ord('2'):
#             show_pole    = not show_pole
#             needs_redraw = True
#         elif key == ord('3'):
#             show_tree    = not show_tree
#             needs_redraw = True
#         elif key in (ord('+'), ord('=')):
#             opacity      = min(1.0, opacity + OPACITY_STEP)
#             needs_redraw = True
#         elif key == ord('-'):
#             opacity      = max(0.0, opacity - OPACITY_STEP)
#             needs_redraw = True
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()

"""
mask_viewer.py
==============
Loads an image and overlays its ground, pole, tree masks and gate detection
on top, each in a different colour with adjustable opacity.

Controls
--------
  a / d          — previous / next image
  1              — toggle ground mask  (green)
  2              — toggle pole mask    (orange)
  3              — toggle tree mask    (purple)
  4              — toggle gate overlay (cyan)
  +  / -         — increase / decrease opacity of all masks
  q              — quit

Layout
------
  Top panel    : original image + mask overlays + gate annotation
  Bottom panel : the three raw masks side by side (B&W)
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
# MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\testorni_generated_masks_new_test'

MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\generated_masks'


GROUND_MASK_DIR = os.path.join(MASKS_OUTPUT_ROOT, 'ground_masks')
POLE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'pole_masks')
TREE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'tree_masks')

# Overlay colours (BGR)
GROUND_COLOUR = (0,   255,   0)    # green
POLE_COLOUR   = (0,   165, 255)    # orange
TREE_COLOUR   = (255,   0, 200)    # purple
GATE_COLOUR   = (255, 255,   0)    # cyan

INITIAL_OPACITY = 0.5
OPACITY_STEP    = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# GATE DETECTION — copied exactly from gate_detection.py
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
            "x_min":  x,        "y_min":  y,
            "x_max":  x + bw,   "y_max":  y + bh,
            "cx":     int(cents[i, 0]),
            "cy":     int(cents[i, 1]),
            "area":   area,
            "aspect": bw / max(bh, 1),
        })
    return blobs


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


def find_gate_pair(blobs: list, cfg: Config) -> Optional[tuple]:
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


def gate_midpoint(b1: dict, b2: dict) -> tuple:
    return ((b1["cx"] + b2["cx"]) // 2, (b1["cy"] + b2["cy"]) // 2)


def process_frame(image_bgr: np.ndarray, cfg: Config) -> tuple:
    mask  = blue_mask(image_bgr, cfg)
    blobs = extract_blobs(mask, image_bgr.shape, cfg)
    pair  = find_gate_pair(blobs, cfg)
    mid   = gate_midpoint(*pair) if pair is not None else None
    return mask, blobs, pair, mid


def draw_gate_overlay(base: np.ndarray, mask: np.ndarray, blobs: list,
                      pair: Optional[tuple], mid: Optional[tuple],
                      opacity: float) -> np.ndarray:
    """Draws the gate blue mask + blob boxes + midpoint onto base."""
    out = base.copy()

    # Blue mask tint
    coloured = np.zeros_like(out)
    coloured[mask > 0] = GATE_COLOUR
    out = cv2.addWeighted(out, 1.0, coloured, opacity, 0)

    # All blob outlines (grey)
    for b in blobs:
        cv2.rectangle(out, (b["x_min"], b["y_min"]),
                      (b["x_max"], b["y_max"]), (120, 120, 120), 1)

    # Gate pair boxes (cyan)
    if pair is not None:
        for b in pair:
            cv2.rectangle(out, (b["x_min"], b["y_min"]),
                          (b["x_max"], b["y_max"]), GATE_COLOUR, 2)

    # Midpoint marker
    if mid is not None:
        cv2.drawMarker(out, mid, (0, 255, 0),
                       cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(out, mid, 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(out, f'GATE ({mid[0]},{mid[1]})',
                    (max(0, mid[0] - 45), max(16, mid[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MASK VIEWER HELPERS
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


def build_frame(img, ground_mask, pole_mask, tree_mask,
                gate_data,
                show_ground, show_pole, show_tree, show_gate, opacity):
    H, W = img.shape[:2]

    # ── Top panel ─────────────────────────────────────────────────────────────
    top = img.copy()
    if show_ground: top = apply_colour_overlay(top, ground_mask, GROUND_COLOUR, opacity)
    if show_pole:   top = apply_colour_overlay(top, pole_mask,   POLE_COLOUR,   opacity)
    if show_tree:   top = apply_colour_overlay(top, tree_mask,   TREE_COLOUR,   opacity)
    if show_gate:
        gate_mask, blobs, pair, mid = gate_data
        top = draw_gate_overlay(top, gate_mask, blobs, pair, mid, opacity)

    # Legend
    legend_items = []
    if show_ground: legend_items.append(('[1] Ground', GROUND_COLOUR))
    if show_pole:   legend_items.append(('[2] Pole',   POLE_COLOUR))
    if show_tree:   legend_items.append(('[3] Tree',   TREE_COLOUR))
    if show_gate:   legend_items.append(('[4] Gate',   GATE_COLOUR))

    for i, (label, colour) in enumerate(legend_items):
        cv2.putText(top, label, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    cv2.putText(top, f'opacity: {opacity:.2f}  (+/-)',
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Bottom panel: raw B&W masks side by side ──────────────────────────────
    panel_w = W // 3
    panel_h = H // 3

    ground_panel = make_bw_panel(ground_mask, 'Ground', panel_w, panel_h)
    pole_panel   = make_bw_panel(pole_mask,   'Pole',   panel_w, panel_h)
    tree_panel   = make_bw_panel(tree_mask,   'Tree',   panel_w, panel_h)

    bottom = np.hstack((ground_panel, pole_panel, tree_panel))
    return np.vstack((top, bottom))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) +
                        glob.glob(os.path.join(IMAGE_FOLDER, '*.png')))
    all_images = [p for p in all_images if '_mask' not in os.path.basename(p).lower()]

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

    WINDOW = 'Mask Viewer  |  a/d=prev/next  |  1/2/3/4=toggle  |  +/-=opacity  |  q=quit'
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    idx          = 0
    opacity      = INITIAL_OPACITY
    show_ground  = True
    show_pole    = True
    show_tree    = True
    show_gate    = True
    needs_redraw = True

    while True:
        if needs_redraw:
            path = image_paths[idx]
            stem = os.path.splitext(os.path.basename(path))[0]

            img         = cv2.imread(path)
            ground_mask = load_mask(GROUND_MASK_DIR, stem, 'ground')
            pole_mask   = load_mask(POLE_MASK_DIR,   stem, 'pole')
            tree_mask   = load_mask(TREE_MASK_DIR,   stem, 'tree')
            gate_data   = process_frame(img, gate_cfg)   # always computed, only drawn if show_gate

            mid = gate_data[3]
            gate_str = f'GATE at {mid}' if mid is not None else 'no gate'

            frame = build_frame(img, ground_mask, pole_mask, tree_mask,
                                gate_data,
                                show_ground, show_pole, show_tree, show_gate, opacity)

            cv2.setWindowTitle(WINDOW,
                f'[{idx+1}/{len(image_paths)}]  {stem}  |  {gate_str}  |  '
                f'G={"ON" if show_ground else "off"}  '
                f'P={"ON" if show_pole else "off"}  '
                f'T={"ON" if show_tree else "off"}  '
                f'Gate={"ON" if show_gate else "off"}  '
                f'opacity={opacity:.2f}')
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
        elif key in (ord('+'), ord('=')):
            opacity      = min(1.0, opacity + OPACITY_STEP)
            needs_redraw = True
        elif key == ord('-'):
            opacity      = max(0.0, opacity - OPACITY_STEP)
            needs_redraw = True

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()