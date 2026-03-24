"""
generate_masks.py
=================
Generates binary masks (white = detected object, black = background) for every
image in a specified image folder and saves them into a structured output folder.

Output folder structure
-----------------------
<MASKS_OUTPUT_ROOT>/
    ground_masks/
        <stem>_mask_ground.png
        ...
    pole_masks/
        <stem>_mask_pole.png
        ...
    tree_masks/
        <stem>_mask_tree.png
        ...

The hand-made training folders inside Python/masks_data/ are NOT touched.

Usage
-----
Set the paths in the CONFIG section below, then run:
    python generate_masks.py
"""

import cv2
import numpy as np
import joblib
import glob
import os

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths to match your setup
# ══════════════════════════════════════════════════════════════════════════════

# Folder that contains the numbered image folders (20260313-100130, 20260320, sim_images …)
# Point this to whichever dated folder you want to process.
IMAGE_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'

# Where the three mask sub-folders will be created.
# Sits alongside the numbered image folders, completely separate from masks_data/.
MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\generated_masks'

# .pkl model files — adjust paths if your Python/ folder is somewhere else
GROUND_MODEL_PATH = 'ground_detector.pkl'
POLE_MODEL_PATH   = 'pole_detector.pkl'
TREE_MODEL_PATH   = 'tree_detector.pkl'
# """
# generate_masks.py
# =================
# Generates binary masks (white = detected object, black = background) for every
# image in a specified image folder and saves them into a structured output folder.
#
# Output folder structure
# -----------------------
# <MASKS_OUTPUT_ROOT>/
#     ground_masks/
#         <stem>_mask_ground.png
#         ...
#     pole_masks/
#         <stem>_mask_pole.png
#         ...
#     tree_masks/
#         <stem>_mask_tree.png
#         ...
#
# The hand-made training folders inside Python/masks_data/ are NOT touched.
#
# Usage
# -----
# Set the paths in the CONFIG section below, then run:
#     python generate_masks.py
# """
#
# import cv2
# import numpy as np
# import joblib
# import glob
# import os
#
# # ══════════════════════════════════════════════════════════════════════════════
# # CONFIG — edit these paths to match your setup
# # ══════════════════════════════════════════════════════════════════════════════
#
# # Folder that contains the numbered image folders (20260313-100130, 20260320, sim_images …)
# # Point this to whichever dated folder you want to process.
# IMAGE_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
#
# # Where the three mask sub-folders will be created.
# # Sits alongside the numbered image folders, completely separate from masks_data/.
# MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\generated_masks'
#
# # .pkl model files — adjust paths if your Python/ folder is somewhere else
# GROUND_MODEL_PATH = r'../ground_detector.pkl'
# POLE_MODEL_PATH   = r'../pole_detector.pkl'
# TREE_MODEL_PATH   = r'../tree_detector.pkl'

# ── Test mode — set to True to only process a few images ─────────────────────
TEST_MODE       = True    # ← flip to False when you want to run on everything
TEST_NUM_IMAGES = 10       # how many random images to pick when TEST_MODE is True

# ── Tree edge-filter settings (mirrors your existing tree script) ─────────────
EDGE_FILTER_ON         = True
MIN_BLOB_AREA          = 35
SMALL_BLOB_THRESHOLD   = 150
EDGE_DENSITY_THRESHOLD = 0.04
CANNY_LOW              = 50
CANNY_HIGH             = 150
CANNY_BLUR             = 5
CANNY_SIGMA            = 1e-18   # effectively 0 = auto

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(yuv, hsv, lab, y, x):
    """
    Identical to the feature extractor used during training.
    Do NOT modify — any change silently breaks predictions.
    """
    p_yuv  = yuv[y, x]
    p_hsv  = hsv[y, x]
    p_lab  = lab[y, x]
    patch  = yuv[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3)
    mean   = patch.mean(axis=0)
    std    = patch.std(axis=0)
    grad   = np.abs(yuv[y, x + 1].astype(int) - yuv[y, x - 1].astype(int))
    return [
        int(p_yuv[0]), int(p_yuv[1]), int(p_yuv[2]),
        int(p_hsv[0]), int(p_hsv[1]), int(p_hsv[2]),
        int(p_lab[0]), int(p_lab[1]), int(p_lab[2]),
        *mean.tolist(), *std.tolist(), *grad.tolist()
    ]


def run_classifier(clf, img):
    """
    Runs a pkl classifier over every valid pixel in the image.
    Returns a binary uint8 mask (0 or 1) the same size as the image.
    The 4-pixel border is left as 0 because the feature extractor needs neighbours.
    """
    h, w, _ = img.shape
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    pixels = np.array([
        extract_features(yuv, hsv, lab, y, x)
        for y in range(4, h - 2)
        for x in range(2, w - 3)
    ])

    pred_flat   = clf.predict(pixels)
    msk_cropped = pred_flat.reshape(h - 6, w - 5)
    msk         = np.zeros((h, w), dtype=np.uint8)
    msk[4:h - 2, 2:w - 3] = msk_cropped
    return msk


def binary_to_bw(mask):
    """
    Converts a classifier output mask to a black-and-white image:
        any non-zero value → 255 (white)
        zero              → 0   (black)
    Using np.where so it works regardless of what the positive class label is.
    """
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def filter_tree_by_edge_density(raw_mask, img):
    """
    Keeps tree blobs that have enough internal edge texture (leaves / branches).
    Rejects flat false-positive regions (floor, walls, panels).
    Mirrors the logic in your existing tree detection script exactly.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (CANNY_BLUR, CANNY_BLUR), CANNY_SIGMA)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN,  kernel)
    cleaned = cv2.morphologyEx(cleaned,                   cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_mask = np.zeros_like(raw_mask, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue

        # Small blobs — skip edge check, keep directly
        if area < SMALL_BLOB_THRESHOLD:
            cv2.drawContours(output_mask, [cnt], -1, 255, -1)
            continue

        # Larger blobs — check internal edge density
        blob_mask        = np.zeros_like(raw_mask, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        edges_inside     = cv2.bitwise_and(edges, blob_mask)
        edge_pixel_count = cv2.countNonZero(edges_inside)
        blob_pixel_count = cv2.countNonZero(blob_mask)
        edge_density     = edge_pixel_count / blob_pixel_count if blob_pixel_count > 0 else 0

        if edge_density >= EDGE_DENSITY_THRESHOLD:
            cv2.drawContours(output_mask, [cnt], -1, 255, -1)

    return output_mask


def make_tree_mask(tree_clf, img):
    """
    Full tree mask pipeline:
      1. Run pixel classifier
      2. Apply edge-density filter to remove flat false positives
      3. Return clustered blob regions as white, everything else black
      No bounding boxes — just the raw clustered pixel regions.
    """
    raw_mask = run_classifier(tree_clf, img)

    if EDGE_FILTER_ON:
        filtered = filter_tree_by_edge_density(raw_mask, img)
    else:
        filtered = raw_mask

    return binary_to_bw(filtered)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Create output sub-folders ─────────────────────────────────────────────
    ground_dir = os.path.join(MASKS_OUTPUT_ROOT, 'ground_masks')
    pole_dir   = os.path.join(MASKS_OUTPUT_ROOT, 'pole_masks')
    tree_dir   = os.path.join(MASKS_OUTPUT_ROOT, 'tree_masks')

    for d in (ground_dir, pole_dir, tree_dir):
        os.makedirs(d, exist_ok=True)
        print(f'Output folder ready: {d}')

    # ── Load models ───────────────────────────────────────────────────────────
    print('\nLoading models...')
    ground_clf = joblib.load(GROUND_MODEL_PATH)
    pole_clf   = joblib.load(POLE_MODEL_PATH)
    tree_clf   = joblib.load(TREE_MODEL_PATH)
    print('All models loaded.')

    # ── Collect images — skip any existing mask files ─────────────────────────
    all_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) +
                        glob.glob(os.path.join(IMAGE_FOLDER, '*.png')))
    all_images = [p for p in all_images if '_mask' not in os.path.basename(p).lower()]

    if not all_images:
        print(f'\nNo images found in: {IMAGE_FOLDER}')
        return

    if TEST_MODE:
        import random
        all_images = random.sample(all_images, min(TEST_NUM_IMAGES, len(all_images)))
        print(f'TEST MODE — processing {len(all_images)} random images.\n')
    else:
        print(f'\nFound {len(all_images)} images to process.\n')

    # ── Process each image ────────────────────────────────────────────────────
    for idx, path in enumerate(all_images):
        stem = os.path.splitext(os.path.basename(path))[0]   # e.g. "956"
        print(f'[{idx + 1}/{len(all_images)}]  {stem}')

        img = cv2.imread(path)
        if img is None:
            print(f'  WARNING: could not read {path}, skipping.')
            continue

        # ── Ground mask ───────────────────────────────────────────────────────
        ground_raw  = run_classifier(ground_clf, img)
        ground_bw   = binary_to_bw(ground_raw)
        ground_name = f'{stem}_mask_ground.png'
        cv2.imwrite(os.path.join(ground_dir, ground_name), ground_bw)
        print(f'  ground  → {ground_name}')

        # ── Pole mask ─────────────────────────────────────────────────────────
        pole_raw  = run_classifier(pole_clf, img)
        pole_bw   = binary_to_bw(pole_raw)
        pole_name = f'{stem}_mask_pole.png'
        cv2.imwrite(os.path.join(pole_dir, pole_name), pole_bw)
        print(f'  pole    → {pole_name}')

        # ── Tree mask (with edge-density filter, no bounding boxes) ───────────
        tree_bw   = make_tree_mask(tree_clf, img)
        tree_name = f'{stem}_mask_tree.png'
        cv2.imwrite(os.path.join(tree_dir, tree_name), tree_bw)
        print(f'  tree    → {tree_name}')


if __name__ == '__main__':
    main()