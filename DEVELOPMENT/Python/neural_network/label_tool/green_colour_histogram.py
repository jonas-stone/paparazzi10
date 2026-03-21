"""
SCRIPT FOR SELECTING ZONE WHERE DRONE SHOULD GO - LABELLING TOOL

USES SOME FUNCTIONS PREVIOUSLY MADE BY JONAS AND RICCARDO BUT HAS COPIED THEM HERE FOR EASE OF USE

"""
########################################################################################################################
# Imports ##############################################################################################################
########################################################################################################################
import os
import sys
import random
from glob import glob

import cv2
import numpy as np


########################################################################################################################
# Inputs ###############################################################################################################
########################################################################################################################

MODE = 1          # 1 = folder mode,  2 = single image mode

_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))

SINGLE_IMAGE_PATH = os.path.join(_project_root, "DEVELOPMENT", "downloads from drone", "20260320", "475.jpg")
FOLDER_PATH       = os.path.join(_project_root, "DEVELOPMENT", "downloads from drone", "20260320")

FRAME_DELAY = 1   # ms between cv2.waitKey polls

N_PARTITIONS  = 7         # number of vertical partitions / Zones
SPACING       = "biexp"   # "uniform" or "biexp"
BIEXP_EXP     = 0.925     # lower = more centre-narrow, range (0.0, 1.0]
MIN_BLOB_AREA  = 3000     # blobs smaller than this are discarded — tune to filter carpet/leakage
DILATE_KERNEL  = 50      # pixels to close gaps — increase to bridge wider obstacles like pillars (0 = off)

########################################################################################################################
# Functions ############################################################################################################
########################################################################################################################

def is_ground(Y, U, V):
    """
    Ground Classifier Function (copied from get_obstacle_info_lmp.py)
    """
    if U <= 115.50:
        if V <= 145.00:
            if Y <= 85.50:
                if Y <= 78.50:
                    return 0
                else:
                    return 0
            else:
                if U <= 92.50:
                    return 0
                else:
                    return 255
        else:
            if V <= 152.50:
                if Y <= 177.00:
                    return 255
                else:
                    return 0
            else:
                return 0
    else:
        if U <= 121.50:
            if V <= 137.50:
                if Y <= 87.50:
                    return 0
                else:
                    return 255
            else:
                if U <= 116.50:
                    return 0
                else:
                    return 0
        else:
            if U <= 122.50:
                if V <= 126.00:
                    return 0
                else:
                    return 0
            else:
                if Y <= 62.50:
                    return 0
                else:
                    return 0


def isolate_ground_blob(binary_img: np.ndarray):
    """
    1. Morphological CLOSE to bridge gaps (pillars, obstacles) between ground chunks
    2. Remove blobs smaller than MIN_BLOB_AREA
    Returns the closed mask (not masked back to original) so fill_holes
    can see the full bridged region.
    Tune DILATE_KERNEL to bridge wider gaps, MIN_BLOB_AREA to drop spurious blobs.
    """
    img = binary_img.astype(np.uint8)

    if DILATE_KERNEL > 0:
        k          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))
        img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    else:
        img_closed = img

    result = cv2.connectedComponentsWithStats(img_closed, connectivity=8)
    totalLabels, labeled_img, values, _ = result

    for blob_label in range(1, totalLabels):
        if values[blob_label, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
            labeled_img[labeled_img == blob_label] = 0

    labeled_img[labeled_img != 0] = 255
    return labeled_img.astype(np.uint8)


def fill_holes(binary_img: np.ndarray):
    """
    Fill regions fully enclosed by ground on all sides.
    Image edges act as walls — regions touching the border are NOT filled,
    preventing leakage out of the image boundary.
    """
    binary_img = binary_img.astype(np.uint8)
    H, W = binary_img.shape

    # pad with 1px border of zeros so flood fill can reach all edge-connected regions
    padded = cv2.copyMakeBorder(binary_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # flood fill the non-ground regions reachable from outside
    inv  = cv2.bitwise_not(padded)
    mask = np.zeros((H + 4, W + 4), dtype=np.uint8)
    cv2.floodFill(inv, mask, (0, 0), 255)
    inv_flooded = cv2.bitwise_not(inv)

    # remove padding
    inv_flooded = inv_flooded[1:H+1, 1:W+1]

    # enclosed holes → fill as ground
    filled = cv2.bitwise_or(binary_img, inv_flooded)
    return filled


def detect_ground(image_bgr, median_ksize=5):
    """
    Applies the strict decision-tree ground classifier per pixel.
    Returns (mask, holes_mask) both uint8, full resolution.
    """
    _apply = np.vectorize(is_ground)

    yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    mask = _apply(Y, U, V).astype(np.uint8)

    if median_ksize >= 3:
        mask = cv2.medianBlur(mask, median_ksize)

    mask_before_fill = isolate_ground_blob(mask)
    mask_filled      = fill_holes(mask_before_fill)
    holes_mask       = cv2.subtract(mask_filled.astype(np.uint8), mask_before_fill.astype(np.uint8))

    return mask_filled, holes_mask


def ground_column_counts(mask):
    """Returns array of length W: number of ground pixels per column."""
    return (mask > 0).sum(axis=0).astype(np.float32)


def get_partition_edges(W, n_partitions=7, spacing="uniform", biexp_exp=0.7):
    """
    Returns a list of (x_start, x_end) pixel ranges for each partition.

    Parameters
    ----------
    W            : image width in pixels
    n_partitions : number of vertical partitions
    spacing      : "uniform" — equal width partitions
                   "biexp"   — centre-narrow, wider towards the edges
    biexp_exp    : exponent for biexp spacing (0.0, 1.0] — lower = more aggressive
    """
    if spacing == "uniform":
        edges = np.linspace(0, W, n_partitions + 1, dtype=int)

    elif spacing == "biexp":
        t      = np.linspace(-1, 1, n_partitions + 1)
        warped = np.sign(t) * (np.abs(t) ** biexp_exp)
        warped = (warped - warped[0]) / (warped[-1] - warped[0])
        edges  = (warped * W).astype(int)

    else:
        raise ValueError(f"Unknown spacing: '{spacing}'. Use 'uniform' or 'biexp'.")

    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_partitions)]


def partition_ground_counts(counts, partitions, H):
    """
    Returns a list of (ground_pixels, percentage) per partition.
    Percentage = ground pixels / total pixels in that partition.
    Instantaneous — no temporal smoothing, just the current frame's mask.
    """
    result = []
    for (x0, x1) in partitions:
        total_pixels  = (x1 - x0) * H
        ground_pixels = int(counts[x0:x1].sum())
        pct           = 100.0 * ground_pixels / total_pixels if total_pixels > 0 else 0.0
        result.append((ground_pixels, pct))
    return result


def get_best_partition(part_data):
    """
    Given the output of partition_ground_counts, returns the index of the
    partition with the highest ground percentage.
    This is the zone the drone should fly towards.
    """
    pcts = [pct for (_, pct) in part_data]
    return int(np.argmax(pcts))


def build_display(image_bgr, mask, holes_mask, partitions, part_data, best_idx):
    """
    Pure visualisation — takes analysis results as input, draws nothing itself.

    Parameters
    ----------
    image_bgr  : original BGR image (unrotated)
    mask       : ground mask (unrotated)
    holes_mask : pixels filled by fill_holes (shown in blue)
    partitions : output of get_partition_edges
    part_data  : output of partition_ground_counts
    best_idx   : output of get_best_partition
    """
    image_bgr  = cv2.rotate(image_bgr,  cv2.ROTATE_90_COUNTERCLOCKWISE)
    mask       = cv2.rotate(mask,       cv2.ROTATE_90_COUNTERCLOCKWISE)
    holes_mask = cv2.rotate(holes_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    H, W = image_bgr.shape[:2]

    counts    = ground_column_counts(mask)
    part_pcts = [pct for (_, pct) in part_data]
    max_pct   = max(part_pcts) if max(part_pcts) > 0 else 1

    # panel 1: original with ground overlay + partition lines
    # green = detected ground, blue = filled holes
    overlay = image_bgr.copy()
    overlay[mask > 0]       = [0, 255, 0]   # green: detected ground
    overlay[holes_mask > 0] = [255, 100, 0] # blue: filled holes
    panel1 = cv2.addWeighted(image_bgr, 0.5, overlay, 0.5, 0)

    for i, (x0, x1) in enumerate(partitions):
        colour = (0, 255, 255) if i == best_idx else (255, 255, 0)
        cv2.line(panel1, (x0, 0), (x0, H - 1), colour, 1)
    cv2.line(panel1, (W - 1, 0), (W - 1, H - 1), (255, 255, 0), 1)

    # panel 2: per-column histogram + partition bars
    hist_img  = np.zeros((H, W, 3), dtype=np.uint8)
    max_count = counts.max() if counts.max() > 0 else 1
    norm      = (counts / max_count * (H - 1)).astype(int)

    hole_counts = (holes_mask > 0).sum(axis=0).astype(np.float32)
    hole_norm   = (hole_counts / max_count * (H - 1)).astype(int)

    for x, bar_h in enumerate(norm):
        if bar_h > 0:
            cv2.line(hist_img, (x, H - 1), (x, H - 1 - bar_h), (0, 200, 80), 1)
    for x, bar_h in enumerate(hole_norm):
        if bar_h > 0:
            cv2.line(hist_img, (x, H - 1), (x, H - 1 - bar_h), (255, 100, 0), 1)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.4

    for i, ((x0, x1), (gpx, pct)) in enumerate(zip(partitions, part_data)):
        bar_h  = int(pct / max_pct * (H - 1))
        colour = (0, 255, 255) if i == best_idx else (255, 200, 0)

        if i == best_idx:
            cv2.rectangle(hist_img, (x0, H - 1), (x1 - 1, H - 1 - bar_h), colour, -1)
            for x in range(x0, x1):
                if norm[x] > 0:
                    cv2.line(hist_img, (x, H - 1), (x, H - 1 - norm[x]), (0, 255, 0), 1)
        else:
            cv2.rectangle(hist_img, (x0, H - 1), (x1 - 1, H - 1 - bar_h), colour, 1)

        cv2.putText(hist_img, f"{pct:.0f}%",
                    (x0 + 2, H - 1 - bar_h - 4),
                    font, fscale, colour, 1)
        cv2.line(hist_img, (x0, 0), (x0, H - 1), (100, 100, 100), 1)

    cv2.putText(hist_img,
                f"ground % / partition  (instantaneous)  best={best_idx}",
                (4, 20), font, 0.45, (200, 200, 200), 1)
    cv2.line(hist_img, (0, H // 2), (W - 1, H // 2), (80, 80, 80), 1)

    return np.vstack((panel1, hist_img))


########################################################################################################################
# Main #################################################################################################################
########################################################################################################################

def main():
    if MODE == 2:
        image_paths = [SINGLE_IMAGE_PATH]
        start_idx   = 0
    else:
        image_paths = sorted(glob(os.path.join(FOLDER_PATH, "*.jpg")))
        if not image_paths:
            print(f"No .jpg images found in:\n  {FOLDER_PATH}")
            sys.exit(1)
        start_idx = random.randint(0, len(image_paths) - 1)

    WINDOW = "Ground column histogram  |  a/d = prev/next  |  q = quit"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    idx              = start_idx
    needs_processing = True

    while True:
        try:
            if needs_processing:
                image_path = image_paths[idx]
                raw = cv2.imread(image_path)
                if raw is None:
                    print(f"Could not read: {image_path}")
                    idx = (idx + 1) % len(image_paths)
                    continue

                # ── analysis ──────────────────────────────────────────────────
                mask, holes_mask = detect_ground(raw)
                counts     = ground_column_counts(
                                 cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
                H, W       = (cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)).shape[:2]
                partitions = get_partition_edges(W, N_PARTITIONS, SPACING, BIEXP_EXP)
                part_data  = partition_ground_counts(counts, partitions, H)
                best_idx   = get_best_partition(part_data)

                print(f"[{os.path.basename(image_path)}]  best partition: {best_idx}")

                # ── visualisation ─────────────────────────────────────────────
                frame = build_display(raw, mask, holes_mask, partitions, part_data, best_idx)

                cv2.setWindowTitle(WINDOW,
                    f"[{idx+1}/{len(image_paths)}]  {os.path.basename(image_path)}"
                    f"  →  best zone: {best_idx}")
                cv2.imshow(WINDOW, frame)
                needs_processing = False

            key = cv2.waitKey(FRAME_DELAY)
            if key == ord('q'):
                break
            elif key in (ord('d'), 83, 65363) and not needs_processing:
                idx = (idx + 1) % len(image_paths)
                needs_processing = True
            elif key in (ord('a'), 81, 65361) and not needs_processing:
                idx = (idx - 1) % len(image_paths)
                needs_processing = True

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()