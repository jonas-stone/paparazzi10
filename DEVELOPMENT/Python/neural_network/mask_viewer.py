"""
mask_viewer.py
==============
Loads an image and overlays its ground, pole and tree masks on top,
each in a different colour with adjustable opacity.

Controls
--------
  a / d          — previous / next image
  1              — toggle ground mask  (green)
  2              — toggle pole mask    (orange)
  3              — toggle tree mask    (purple)
  +  / -         — increase / decrease opacity of all masks
  q              — quit

Layout
------
  Top panel    : original image + mask overlays
  Bottom panel : the three raw masks side by side (B&W)
"""

import cv2
import numpy as np
import os
import glob
import sys

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_FOLDER      = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\generated_masks'

GROUND_MASK_DIR = os.path.join(MASKS_OUTPUT_ROOT, 'ground_masks')
POLE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'pole_masks')
TREE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'tree_masks')

# Overlay colours (BGR)
GROUND_COLOUR = (0,   255,   0)    # green
POLE_COLOUR   = (0,   165, 255)    # orange
TREE_COLOUR   = (255,   0, 200)    # purple

INITIAL_OPACITY   = 0.5    # 0.0 = invisible, 1.0 = fully opaque
OPACITY_STEP      = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_mask(mask_dir, stem, suffix):
    path = os.path.join(mask_dir, f'{stem}_mask_{suffix}.png')
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def apply_colour_overlay(base, mask, colour, opacity):
    """Paints 'colour' onto 'base' wherever mask > 0, blended by opacity."""
    if mask is None:
        return base
    coloured        = np.zeros_like(base)
    coloured[mask > 0] = colour
    return cv2.addWeighted(base, 1.0, coloured, opacity, 0)


def make_bw_panel(mask, label, W, H):
    """Renders a single B&W mask as a labelled BGR panel of size (H, W)."""
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
                show_ground, show_pole, show_tree, opacity):
    H, W = img.shape[:2]

    # ── Top panel: overlays on original ──────────────────────────────────────
    top = img.copy()
    if show_ground: top = apply_colour_overlay(top, ground_mask, GROUND_COLOUR, opacity)
    if show_pole:   top = apply_colour_overlay(top, pole_mask,   POLE_COLOUR,   opacity)
    if show_tree:   top = apply_colour_overlay(top, tree_mask,   TREE_COLOUR,   opacity)

    # Legend
    legend_items = []
    if show_ground: legend_items.append(('[1] Ground', GROUND_COLOUR))
    if show_pole:   legend_items.append(('[2] Pole',   POLE_COLOUR))
    if show_tree:   legend_items.append(('[3] Tree',   TREE_COLOUR))

    for i, (label, colour) in enumerate(legend_items):
        cv2.putText(top, label, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    cv2.putText(top, f'opacity: {opacity:.2f}  (+/-)',
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Bottom panel: raw B&W masks side by side ──────────────────────────────
    panel_w = W // 3
    panel_h = H // 3   # shorter strip below

    ground_panel = make_bw_panel(ground_mask, 'Ground', panel_w, panel_h)
    pole_panel   = make_bw_panel(pole_mask,   'Pole',   panel_w, panel_h)
    tree_panel   = make_bw_panel(tree_mask,   'Tree',   panel_w, panel_h)

    bottom = np.hstack((ground_panel, pole_panel, tree_panel))

    return np.vstack((top, bottom))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Collect images that have at least one mask generated
    all_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) +
                        glob.glob(os.path.join(IMAGE_FOLDER, '*.png')))
    all_images = [p for p in all_images if '_mask' not in os.path.basename(p).lower()]

    # Only keep images that have at least one mask
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

    WINDOW = 'Mask Viewer  |  a/d = prev/next  |  1/2/3 = toggle masks  |  +/- = opacity  |  q = quit'
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    idx          = 0
    opacity      = INITIAL_OPACITY
    show_ground  = True
    show_pole    = True
    show_tree    = True
    needs_redraw = True

    while True:
        if needs_redraw:
            path = image_paths[idx]
            stem = os.path.splitext(os.path.basename(path))[0]

            img         = cv2.imread(path)
            ground_mask = load_mask(GROUND_MASK_DIR, stem, 'ground')
            pole_mask   = load_mask(POLE_MASK_DIR,   stem, 'pole')
            tree_mask   = load_mask(TREE_MASK_DIR,   stem, 'tree')

            frame = build_frame(img, ground_mask, pole_mask, tree_mask,
                                show_ground, show_pole, show_tree, opacity)

            cv2.setWindowTitle(WINDOW,
                f'[{idx+1}/{len(image_paths)}]  {stem}  |  '
                f'ground={"ON" if show_ground else "off"}  '
                f'pole={"ON" if show_pole else "off"}  '
                f'tree={"ON" if show_tree else "off"}  '
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
        elif key in (ord('+'), ord('=')):
            opacity      = min(1.0, opacity + OPACITY_STEP)
            needs_redraw = True
        elif key == ord('-'):
            opacity      = max(0.0, opacity - OPACITY_STEP)
            needs_redraw = True

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()