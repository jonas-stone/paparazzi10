"""
gate_detection.py
=================
Detect a gate by finding two parallel blue blobs with similar geometric characteristics and placing
a midpoint between them.

The gate we're looking for consists of two vertical blue poles. Because the camera is mounted
sideways (rotated 90° CW), those poles appear as two horizontally-oriented blobs stacked
vertically in the image. The whole pipeline is built around that assumption.

Pipeline
--------
1. Downscale      — resize the image to 80% of its original size for speed
2. Blue mask      — isolate blue pixels using YUV colour space thresholding
3. Blob extraction — find connected regions, discard noise
4. Pair matching   — check every blob pair for the geometric signature of two gate posts
5. Midpoint        — average the two centroids to get the gate centre, scaled back to
                     native resolution

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

# Configuration

@dataclass
class Config:
    # Downscale factor applied to the image before any processing.
    # 0.8 means the pipeline works on 80% of the original width and height,
    # which reduces the number of pixels processed by 36% and makes every
    # subsequent step proportionally cheaper. The detected midpoint is scaled
    # back to native resolution before being returned so the output coordinates
    # are always in the original image's pixel space.
    downscale: float = 0.8

    # YUV blue thresholds
    # These were tuned for TU Delft blue specifically. The YUV colour space
    # separates brightness (Y) from colour (U, V), which makes the thresholds
    # much more stable across different lighting conditions than raw BGR would be.
    # The parameters were chosen by trial and error using images obtained using the drone 
    # at the CyberZoo
    blue_y_min: int   = 0
    blue_y_max: int   = 220       
    blue_u_min: int   = 122       
    blue_u_max: int   = 255
    blue_v_min: int   = 0
    blue_v_max: int   = 123       
    blue_morph_k: int = 3        

    # Blob filtering
    # Blobs smaller than this fraction of the total image are almost certainly
    # noise 
    blob_min_area_ratio: float = 0.003

    # Parallelism criteria
    # These three thresholds define what "looks like a gate" geometrically.
    # They're intentionally fairly loose 

    # The two posts should have a similar aspect ratio (width/height).
    # A difference of 0.5 means one can be up to 50% more elongated than the other.
    max_aspect_diff: float = 0.5

    # The two posts should cover a similar number of pixels. A difference of 0.6
    # gives some room for one post being partially cut off at the image edge.
    max_area_diff: float = 0.6

    # The line connecting the two centroids should be roughly perpendicular to
    # whichever direction the posts point. If the posts are horizontal bars, the
    # centroids should be separated vertically, and vice versa. This catches
    # false positives where two unrelated blue blobs happen to pass the size tests.
    max_skew_deg: float = 10

    # Display
    display_scale: float = 1.5

# Step 1 — Downscale

def downscale_image(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Resize the image to cfg.downscale × its original dimensions.

    All subsequent pipeline stages (masking, blob extraction, pair matching)
    run on this smaller image, which reduces their cost proportionally to the
    square of the scale factor. The returned array is a new image — the original
    is never modified.
    """
    h, w = image_bgr.shape[:2]
    new_w = int(w * cfg.downscale)
    new_h = int(h * cfg.downscale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Step 2 — Blue mask

def blue_mask(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Produce a binary mask that is white wherever the image is blue.

    After thresholding we run a morphological open (removes isolated specks) followed
    by a close (fills small holes inside a blob). The kernel size blue_morph_k trades
    off smoothness against detail — 5 pixels works well for gate posts that cover
    at least a few percent of the image. A smaller kernel introduces more artefacts in 
    the resulting mask but is more computationally efficient.
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

# Step 3 — Blob extraction

def extract_blobs(mask: np.ndarray, image_shape: tuple, cfg: Config) -> list[dict]:
    """
    Label connected regions in the mask and return the ones large enough to matter.

    connectedComponentsWithStats assigns every white pixel to a region and returns 
    bounding boxes, centroids, and pixel counts. 

    The minimum area filter (blob_min_area_ratio × image pixels) is the first line
    of defence against noise. Anything smaller than ~0.3% of the image is ignored.
    What survives gets packed into a plain dict.

    Each blob dict contains:
        x_min, y_min, x_max, y_max  — bounding box corners (in downscaled pixels)
        cx, cy                       — centroid (in downscaled pixels)
        area                         — number of white pixels in this region
        aspect                       — bounding box width / height
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

# Step 4 — Parallel pair detection

def _are_parallel(b1: dict, b2: dict, cfg: Config) -> bool:
    """
    Decide whether two blobs are plausibly the left and right posts of a gate.

    The camera is mounted rotated 90° CW, so the world's vertical axis maps to
    the image's horizontal axis. Two real-world side-by-side pillars therefore
    appear stacked one above the other in the image (separated along image-Y).
    That's the first thing we check, if the blobs are separated mostly
    horizontally in the image, they can't be the two posts.

    Beyond that, we apply three further checks:

    Check 1 — Aspect ratio similarity
        Both posts are the same physical object, so their bounding boxes should
        have roughly the same shape. A large aspect difference means they are
        unlikely to be the same kind of object.

    Check 2 — Area similarity
        At the same depth both posts should cover a similar number of pixels.
        A big area difference usually means one blob is partial 
        (edge of frame, occlusion) or is something else entirely.

    Check 3 — Centroid join perpendicular to long axis
        For two parallel bars the line connecting their centres should run at 90°
        to the direction the bars themselves point. We estimate the average
        orientation from the mean bounding-box dimensions: wider than tall →
        horizontal bar; taller than wide → vertical bar. Then we measure how far
        the centroid-join vector deviates from the ideal 90°. More than
        max_skew_deg and we reject the pair.

    """
    # Pre-check: in the (rotated) image the two post blobs should be separated
    # more in Y than in X. If they're side by side horizontally they're not the posts.
    dx0 = b2["cx"] - b1["cx"]
    dy0 = b2["cy"] - b1["cy"]
    if abs(dy0) <= abs(dx0):
        return False

    # Check 1 — aspect ratio
    ar1, ar2  = b1["aspect"], b2["aspect"]
    larger    = max(ar1, ar2)
    if larger == 0:
        return False
    if (larger - min(ar1, ar2)) / larger > cfg.max_aspect_diff:
        return False

    # Check 2 — area
    a1, a2   = b1["area"], b2["area"]
    larger_a = max(a1, a2)
    if (larger_a - min(a1, a2)) / larger_a > cfg.max_area_diff:
        return False

    # Check 3 — centroid join should be perpendicular to the bars' long axis
    dx = b2["cx"] - b1["cx"]
    dy = b2["cy"] - b1["cy"]
    if dx == 0 and dy == 0:
        return False

    # Average bounding box size across both blobs gives us the dominant orientation
    avg_w = (b1["x_max"] - b1["x_min"] + b2["x_max"] - b2["x_min"]) / 2
    avg_h = (b1["y_max"] - b1["y_min"] + b2["y_max"] - b2["y_min"]) / 2
    long_ax = np.array([1.0, 0.0]) if avg_w >= avg_h else np.array([0.0, 1.0])

    join    = np.array([dx, dy], dtype=float)
    join   /= np.linalg.norm(join)
    dot     = float(np.clip(np.dot(join, long_ax), -1.0, 1.0))
    angle   = np.degrees(np.arccos(abs(dot)))  
    skew    = abs(90.0 - angle)                 # deviation from the ideal perpendicular arrangement

    return skew <= cfg.max_skew_deg


def find_gate_pair(blobs: list[dict], cfg: Config) -> Optional[tuple[dict, dict]]:
    """
    Search every blob pair and return the best gate candidate.

    Returns (blob1, blob2) or None if no valid pair is found.
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

# Step 5 — Midpoint

def gate_midpoint(b1: dict, b2: dict, cfg: Config) -> tuple[int, int]:
    """
    Compute the centre of the gate as the midpoint between the two post centroids,
    then scale back to native (full-resolution) pixel coordinates.

    All blob coordinates are in downscaled-image space because the pipeline ran
    on the resized frame. Dividing by cfg.downscale converts them back to the
    coordinate system of the original image so callers always receive native
    pixel positions regardless of what downscale factor was used.
    """
    cx = (b1["cx"] + b2["cx"]) // 2
    cy = (b1["cy"] + b2["cy"]) // 2
    # Scale back to native resolution
    native_cx = int(round(cx / cfg.downscale))
    native_cy = int(round(cy / cfg.downscale))
    return (native_cx, native_cy)

# Display

def _rot(img: np.ndarray) -> np.ndarray:
    # Undo the camera's 90° CW rotation so the visualisation looks natural.
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
    Build a side-by-side debug view: blue mask on the left, annotated image on the right.

    All annotations (blob boxes, midpoint marker) are drawn on the ORIGINAL full-resolution
    image, with blob coordinates scaled back from downscaled space so they line up correctly.
    The midpoint is already in native coordinates (gate_midpoint() handles the conversion).

    The annotated panel shows:
      - a semi-transparent blue tint over all masked pixels 
      - thin grey boxes around every blob that passed the area filter
      - boxes around whichever two blobs were chosen as the gate pair
      - a green crosshair and circle at the computed midpoint, with its coordinates
    If no gate was found, a small "no gate" label is drawn instead.
    """
    anno = image_bgr.copy()
    inv  = 1.0 / cfg.downscale   # factor to convert downscaled coords → native coords

    # The mask was produced on the downscaled image so it needs to be upscaled
    # to match the full-resolution annotation canvas before blending
    h, w = image_bgr.shape[:2]
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Semi-transparent blue overlay so you can see where the mask fires
    # without completely obscuring the underlying image texture.
    overlay = anno.copy()
    overlay[mask_full > 0] = [200, 100, 0]
    cv2.addWeighted(overlay, 0.35, anno, 0.65, 0, anno)

    # All candidate blobs — scale bounding boxes back to native resolution
    for b in blobs:
        pt1 = (int(b["x_min"] * inv), int(b["y_min"] * inv))
        pt2 = (int(b["x_max"] * inv), int(b["y_max"] * inv))
        cv2.rectangle(anno, pt1, pt2, (120, 120, 120), 1)

    # The matched gate pair
    if pair is not None:
        for b in pair:
            pt1 = (int(b["x_min"] * inv), int(b["y_min"] * inv))
            pt2 = (int(b["x_max"] * inv), int(b["y_max"] * inv))
            cv2.rectangle(anno, pt1, pt2, (0, 165, 255), 2)

    # Midpoint marker — already in native coordinates
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

    # Convert the single-channel mask to BGR and upscale it for the left panel
    mask_bgr = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)

    # Rotate both panels into landscape orientation
    p_mask = _rot(mask_bgr)
    p_anno = _rot(anno)

    cv2.putText(p_mask, "blue mask",  (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(p_anno, "detections", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    combined = np.hstack([p_mask, p_anno])

    # Scale up for readability on high-DPI screens or small native resolutions
    s = cfg.display_scale
    if s != 1.0:
        combined = cv2.resize(combined,
                              (int(combined.shape[1] * s),
                               int(combined.shape[0] * s)),
                              interpolation=cv2.INTER_LINEAR)
    return combined


# Main sequence runner

def process_sequence(image_folder: str, cfg: Optional[Config] = None):
    """
    Load every image in image_folder in order, run the full detection
    pipeline on each one, and show the result in a live window.

    Images in the folder need to be in alphabetical order.

    Press 'q' at any point to quit early.
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

        # Step 1 — downscale before any processing
        small = downscale_image(image_bgr, cfg)

        # Steps 2-4 — run entirely on the downscaled image
        mask  = blue_mask(small, cfg)
        blobs = extract_blobs(mask, small.shape, cfg)
        pair  = find_gate_pair(blobs, cfg)

        # Step 5 — midpoint is converted back to native resolution inside gate_midpoint()
        mid = gate_midpoint(*pair, cfg) if pair is not None else None

        fname = os.path.basename(path)
        if mid is not None:
            print(f"  [{idx:04d}] {fname}  GATE  midpoint={mid}")
        else:
            print(f"  [{idx:04d}] {fname}  no gate  (blobs={len(blobs)})")

        # Display is built on the original full-resolution image; build_display
        # handles the coordinate conversion for blob boxes internally
        vis = build_display(image_bgr, mask, blobs, pair, mid, cfg)
        cv2.imshow(WIN, vis)
        cv2.resizeWindow(WIN, vis.shape[1], vis.shape[0])
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Set your image folder here
    # IMAGE_FOLDER = "../paparazzi10/DEVELOPMENT/downloads from drone/20260306-095826"
    IMAGE_FOLDER = "C:\Users\Sin Yuen\Downloads\20260306-095826"

    process_sequence(IMAGE_FOLDER)