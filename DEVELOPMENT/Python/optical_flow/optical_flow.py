"""
Bebop Parrot — Obstacle Detection + Optical Flow TTC Warning
=============================================================
Combines colour-based ground/obstacle detection with sparse Lucas-Kanade
optical flow.  When an obstacle is detected in the central column band of
the image, the optical flow expansion rate is used to compute a
Time-To-Contact (TTC) and overlay a colour-coded warning.

Controls
--------
    a  — step back one frame
    d  — step forward one frame
    q  — quit

Dependencies
------------
    opencv-python, numpy, scipy
    colored_blob_separator  (cbs — from the original obstacle pipeline)
    solidity_detection      (sdd — from the original obstacle pipeline)
"""

import os
import sys
import random
import argparse
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import median_filter

import colored_blob_separator as cds
import solidity_detection as sdd


# ──────────────────────────────────────────────────────────────────────────────
# ★  USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

FOLDER_PATH   = "../paparazzi10/DEVELOPMENT/downloads from drone/20260306-095826"   # path relative to this script
SCALE_FACTOR  = 0.8        # main pipeline downscale
FPS           = 30.0       # source footage frame rate (for TTC in seconds)

# Optical-flow tuning
EXPANSION_THRESHOLD = 1.5   # px/frame — above this → caution
FOCAL_LENGTH_PX     = 400.0 # approximate focal length in pixels
SPEED_SCALE         = 4.0   # arrow length = flow_magnitude × SPEED_SCALE

# Central band for obstacle-triggered TTC check
# Obstacle must overlap the middle CENTRE_BAND_FRAC of the frame width
CENTRE_BAND_FRAC = 0.40     # ±20 % either side of centre


# ──────────────────────────────────────────────────────────────────────────────
# Scale helpers  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def scale_int(value, sf):
    return max(1, int(round(value * sf)))

def scale_odd(value, sf):
    n = max(3, int(round(value * sf)))
    return n if n % 2 == 1 else n + 1

def get_scaled_params(sf):
    return dict(
        min_ground_pixels  = scale_int(5,   sf),
        max_gap            = scale_int(10,  sf),
        smooth_kernel      = scale_odd(5,   sf),
        min_width          = scale_int(20,  sf),
        max_col_gap        = scale_int(5,   sf),
        obstacle_threshold = scale_int(50,  sf),
        no_ground_baseline = scale_int(220, sf),
        median_ksize       = scale_odd(5,   sf),
        blur_ksize         = scale_odd(5,   sf),
        box_bottom_pad     = scale_int(15,  sf),
        box_thickness      = max(1, scale_int(2,  sf)),
        status_font_scale  = max(0.3, sf * 1.0),
        status_thickness   = max(1, scale_int(2,  sf)),
        status_x           = scale_int(20,  sf),
        status_y           = scale_int(40,  sf),
        label_font_scale   = max(0.2, sf * 0.4),
        label_thickness    = max(1, scale_int(1,  sf)),
        label_y_offset     = scale_int(5,   sf),
        label_y_top        = scale_int(20,  sf),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Obstacle-detection pipeline  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

DET_OBSTACLE = 0
DET_PLANT    = 1


def find_ground_boundary(mask_rotated, min_ground_pixels=5, max_gap=10, smooth_kernel=5):
    image_height = mask_rotated.shape[1]
    n_rows       = mask_rotated.shape[0]
    boundary_rows = np.full(n_rows, image_height, dtype=int)

    for idx, row in enumerate(mask_rotated):
        green_pos = np.where(row > 0)[0]
        if green_pos.size == 0:
            continue
        ground_valid = np.sum(row[-min_ground_pixels:] > 0) >= min_ground_pixels
        if not ground_valid:
            diffs = np.diff(green_pos)
            runs  = np.split(green_pos, np.where(diffs > 1)[0] + 1)
            if max(len(r) for r in runs) < max_gap:
                continue
        gp   = green_pos[::-1]
        if gp.size == 1:
            boundary_rows[idx] = gp[0]
            continue
        gaps             = -np.diff(gp) - 1
        big_gap_indices  = np.where(gaps > max_gap)[0]
        if big_gap_indices.size == 0:
            boundary_rows[idx] = gp[-1]
            continue
        fg               = big_gap_indices[0]
        pending_boundary = gp[fg]
        after_gap        = gp[fg + 1:]
        if after_gap.size >= max_gap:
            after_gaps   = -np.diff(after_gap) - 1
            split_points = np.where(after_gaps > 0)[0]
            runs         = np.split(after_gap, split_points + 1)
            if max(len(r) for r in runs) >= max_gap:
                boundary_rows[idx] = gp[-1]
                continue
        boundary_rows[idx] = pending_boundary

    valid_mask = boundary_rows < image_height
    if valid_mask.sum() > smooth_kernel:
        smoothed              = median_filter(boundary_rows.astype(float), size=smooth_kernel)
        boundary_rows[valid_mask] = smoothed[valid_mask].astype(int)

    return boundary_rows


def get_obstacle_regions(obstacle_cols, min_width=20, max_col_gap=5):
    if len(obstacle_cols) == 0:
        return []
    regions = []
    start = end = obstacle_cols[0]
    for col in obstacle_cols[1:]:
        if col - end <= max_col_gap:
            end = col
        else:
            regions.append((start, end, end - start + 1))
            start = end = col
    regions.append((start, end, end - start + 1))
    return [(s, e, w) for s, e, w in regions if w >= min_width]


def update_and_detect(boundary_row, h, ground_baseline,
                      min_width=20, obstacle_threshold=50,
                      no_ground_baseline=220, max_col_gap=5):
    alpha     = 0.6
    valid     = boundary_row < h
    no_ground = boundary_row >= h

    if ground_baseline is None:
        baseline              = boundary_row.astype(float).copy()
        baseline[no_ground]   = no_ground_baseline
        return [], baseline

    deviation          = boundary_row - ground_baseline
    deviation_obstacle = valid & (deviation > obstacle_threshold)
    obstacle_mask      = deviation_obstacle | no_ground
    obstacle_cols      = np.where(obstacle_mask)[0]
    obstacle_regions   = get_obstacle_regions(obstacle_cols, min_width=min_width, max_col_gap=max_col_gap)

    no_obstacle_mask   = valid & ~deviation_obstacle
    ground_baseline    = ground_baseline.copy()
    ground_baseline[no_obstacle_mask] = (
        (1 - alpha) * ground_baseline[no_obstacle_mask]
        + alpha     * boundary_row[no_obstacle_mask]
    )
    last_good = no_ground_baseline
    for i in range(len(ground_baseline)):
        if no_obstacle_mask[i]:
            last_good = ground_baseline[i]
        else:
            ground_baseline[i] = last_good

    return obstacle_regions, ground_baseline


def is_ground(Y, U, V):
    if U <= 115.50:
        if V <= 145.00:
            if Y <= 85.50:
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
                return 0
        else:
            return 0

vectorized_is_ground = np.vectorize(is_ground)


def detect_green_ground_ml(image_bgr, threshold, median_ksize=5):
    yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)
    mask = vectorized_is_ground(Y, U, V).astype(np.uint8)
    if median_ksize and median_ksize >= 3 and median_ksize % 2 == 1:
        mask = cv2.medianBlur(mask, median_ksize)
    green_frac = cv2.countNonZero(mask) / (image_bgr.shape[0] * image_bgr.shape[1])
    status     = "GROUND FOUND" if green_frac > threshold else "NO GROUND"
    return mask, cv2.bitwise_and(image_bgr, image_bgr, mask=mask), green_frac, status


def detect_all_green_lax(image_bgr, scale_factor=0.15, blur_ksize=5):
    H, W       = image_bgr.shape[:2]
    small_w    = max(1, int(W * scale_factor))
    small_h    = max(1, int(H * scale_factor))
    downscaled = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    yuv        = cv2.cvtColor(downscaled, cv2.COLOR_BGR2YUV)
    Y, U, V    = cv2.split(yuv)
    green_mask_small = (
        (U >= 0) & (U <= 116) & (V >= 0) & (V <= 141) & (Y >= 29) & (Y <= 140)
    ).astype(np.uint8) * 255
    green_mask_small[:, :int(small_w / 3)] = 0
    if blur_ksize >= 3:
        green_mask_small = cv2.GaussianBlur(green_mask_small, (blur_ksize, blur_ksize), 0)
        _, green_mask_small = cv2.threshold(green_mask_small, 127, 255, cv2.THRESH_BINARY)
    mask_full = cv2.resize(green_mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    return green_mask_small, mask_full, downscaled


def detect_plant_regions(plant_mask, min_width=10, min_pixels_per_col=2, max_col_gap=3):
    plant_mask_flipped = plant_mask[:, ::-1]
    active_rows        = []
    for row_idx in range(plant_mask_flipped.shape[0]):
        row       = plant_mask_flipped[row_idx, :]
        green_pos = np.where(row > 0)[0]
        if green_pos.size == 0:
            continue
        diffs = np.diff(green_pos)
        runs  = np.split(green_pos, np.where(diffs > 1)[0] + 1)
        if max(len(r) for r in runs) > min_pixels_per_col:
            active_rows.append(row_idx)
    return get_obstacle_regions(active_rows, min_width=min_width, max_col_gap=max_col_gap)


def get_obstacle_info(image_bgr, ground_baseline,
                      oa_color_count_frac=0.05, median_ksize=5,
                      min_width=20, max_col_gap=5, min_ground_pixels=5,
                      max_gap=10, smooth_kernel=5, obstacle_threshold=50,
                      no_ground_baseline=220,
                      plant_scale_factor=0.15, plant_blur_ksize=3,
                      plant_min_width=30, plant_min_pixels_per_col=2,
                      plant_max_col_gap=5):
    mask, _, green_frac, status = detect_green_ground_ml(
        image_bgr, threshold=oa_color_count_frac, median_ksize=median_ksize)

    H, W  = image_bgr.shape[:2]
    rows  = []
    debug = {"status": status, "green_frac": green_frac}

    if status == "GROUND FOUND":
        clean_mask    = cds.fill_holes(cds.isolate_ground_blob(binary_img=mask))
        mask_flipped  = clean_mask[:, ::-1]
        boundary_rows = find_ground_boundary(mask_flipped, min_ground_pixels, max_gap, smooth_kernel)
        obstacle_regions, new_ground_baseline = update_and_detect(
            boundary_rows, W, ground_baseline,
            min_width=min_width, obstacle_threshold=obstacle_threshold,
            no_ground_baseline=no_ground_baseline, max_col_gap=max_col_gap)
        for (s, e, w) in obstacle_regions:
            rows.append([int(W - 1 - e), int(w), DET_OBSTACLE])
        debug.update({"boundary_rows": boundary_rows,
                      "obstacle_regions_raw": obstacle_regions,
                      "clean_mask": clean_mask})
    else:
        new_ground_baseline          = ground_baseline
        clean_mask                   = np.zeros((H, W), dtype=np.uint8)
        debug["boundary_rows"]        = np.full(W, W)
        debug["obstacle_regions_raw"] = []
        debug["clean_mask"]           = clean_mask

    all_green_small, _, _ = detect_all_green_lax(image_bgr, plant_scale_factor, plant_blur_ksize)
    pH, pW                = all_green_small.shape[:2]
    clean_small           = cv2.resize(clean_mask, (pW, pH), interpolation=cv2.INTER_NEAREST)
    plant_mask_small      = cv2.subtract(all_green_small, clean_small)
    plant_mask            = cv2.resize(plant_mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    plant_regions         = detect_plant_regions(plant_mask, plant_min_width,
                                                 plant_min_pixels_per_col, plant_max_col_gap)
    for (s, e, w) in plant_regions:
        rows.append([int(s), int(w), DET_PLANT])

    debug["plant_mask"]        = plant_mask
    debug["plant_regions_raw"] = plant_regions

    detections = np.array(rows, dtype=np.int32) if rows else np.empty((0, 3), dtype=np.int32)
    return detections, new_ground_baseline, debug


# ──────────────────────────────────────────────────────────────────────────────
# Optical-flow helpers
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_PARAMS = dict(maxCorners=60, qualityLevel=0.15, minDistance=20, blockSize=7)
LK_PARAMS      = dict(winSize=(15, 15), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def detect_flow_features(gray):
    """Detect trackable corners in the central 50 % of the frame."""
    h, w  = gray.shape[:2]
    x0, y0 = w // 4, h // 4
    mask  = np.zeros_like(gray)
    mask[y0:y0*3, x0:x0*3] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)


def compute_expansion(cx, cy, pts_prev, pts_next):
    """Mean radial expansion: positive = approaching obstacle."""
    components = []
    for p, q in zip(pts_prev, pts_next):
        px, py = p.ravel();  qx, qy = q.ravel()
        dx, dy = px - cx, py - cy
        n      = np.hypot(dx, dy)
        if n < 1e-3:
            continue
        components.append(((qx - px) * dx + (qy - py) * dy) / n)
    return float(np.mean(components)) if components else 0.0


def time_to_contact(expansion):
    """TTC in frames.  Returns inf when expansion is negligible."""
    return FOCAL_LENGTH_PX / expansion if expansion > 0.01 else float("inf")


def obstacle_in_centre(detections, frame_w):
    """Return True if any DET_OBSTACLE detection overlaps the central band."""
    if detections.shape[0] == 0:
        return False
    band_l = frame_w * (0.5 - CENTRE_BAND_FRAC / 2)
    band_r = frame_w * (0.5 + CENTRE_BAND_FRAC / 2)
    obs    = detections[detections[:, 2] == DET_OBSTACLE]
    for left, width, _ in obs:
        right = left + width
        if right > band_l and left < band_r:   # any overlap
            return True
    return False


def draw_flow_overlay(vis, pts_prev, pts_next, expansion, ttc_sec, in_centre):
    """
    Draw flow arrows and TTC warning onto *vis* (modified in-place).
    Only the warning banner is drawn when there is no central obstacle.
    """
    h, w   = vis.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Arrow colour depends on expansion severity
    if expansion >= EXPANSION_THRESHOLD * 2:
        arrow_col = (0, 0, 255)      # red
    elif expansion >= EXPANSION_THRESHOLD:
        arrow_col = (0, 140, 255)    # orange
    else:
        arrow_col = (0, 220, 0)      # green

    # Flow arrows (always drawn so the user can see motion)
    for p, q in zip(pts_prev, pts_next):
        px, py = int(p[0][0]), int(p[0][1])
        fx, fy = q[0][0] - p[0][0], q[0][1] - p[0][1]
        mag    = np.hypot(fx, fy)
        if mag < 0.3:
            continue
        ex, ey = int(px + fx * SPEED_SCALE), int(py + fy * SPEED_SCALE)
        cv2.arrowedLine(vis, (px, py), (ex, ey), arrow_col, 1,
                        tipLength=0.35, line_type=cv2.LINE_AA)
        cv2.circle(vis, (px, py), 2, arrow_col, -1, cv2.LINE_AA)

    # Mean-flow dominant arrow
    flows = pts_next.reshape(-1, 2) - pts_prev.reshape(-1, 2)
    mfx, mfy = flows.mean(axis=0)
    dom_mag  = np.hypot(mfx, mfy)
    if dom_mag > 0.3:
        scale = min(60.0, dom_mag * SPEED_SCALE * 3)
        ex = int(cx + mfx / dom_mag * scale)
        ey = int(cy + mfy / dom_mag * scale)
        cv2.arrowedLine(vis, (int(cx), int(cy)), (ex, ey),
                        (255, 255, 0), 3, tipLength=0.3, line_type=cv2.LINE_AA)

    # ── TTC warning banner — only when obstacle is in centre ─────────────────
    if in_centre:
        if expansion >= EXPANSION_THRESHOLD * 2:
            label     = "!! STOP !!"
            label_col = (0, 0, 255)
        elif expansion >= EXPANSION_THRESHOLD:
            label     = "SLOW DOWN"
            label_col = (0, 140, 255)
        else:
            label     = "OBSTACLE AHEAD"
            label_col = (0, 220, 0)

        ttc_str = f"TTC: {ttc_sec:.1f} s" if ttc_sec < float("inf") else "TTC: --"
        exp_str = f"Expansion: {expansion:+.2f} px/frame"

        # Semi-transparent banner at bottom of frame
        banner_h = 70
        overlay  = vis.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Big centred warning label
        (tw, _), _ = cv2.getTextSize(label, font, 1.2, 2)
        tx = int(cx - tw / 2)
        cv2.putText(vis, label, (tx + 1, h - 38), font, 1.2, (0, 0, 0),    2, cv2.LINE_AA)
        cv2.putText(vis, label, (tx,     h - 38), font, 1.2, label_col,    2, cv2.LINE_AA)
        # TTC and expansion below
        cv2.putText(vis, f"{ttc_str}   {exp_str}",
                    (14, h - 10), font, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame processing
# ──────────────────────────────────────────────────────────────────────────────

def load_frames(folder):
    paths = sorted([os.path.join(folder, f)
                    for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    if len(paths) < 2:
        sys.exit(f"[ERROR] Need ≥ 2 images in '{folder}', found {len(paths)}.")
    print(f"[INFO] Found {len(paths)} frames in '{folder}'.")
    return paths


def process_frame(paths, idx, ground_baseline, p, plant_scale_factor,
                  plant_blur_ksize, target_w, target_h, fps):
    """
    Load frames idx-1 and idx, run obstacle detection + optical flow,
    return (annotated_display_image, updated_ground_baseline).
    """
    raw_prev = cv2.rotate(cv2.imread(paths[idx - 1]), cv2.ROTATE_90_COUNTERCLOCKWISE)
    raw_curr = cv2.rotate(cv2.imread(paths[idx]),     cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Downscale for the pipeline
    prev_bgr = cv2.resize(raw_prev, (target_w, target_h), interpolation=cv2.INTER_AREA)
    curr_bgr = cv2.resize(raw_curr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    h, w = curr_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # ── Obstacle detection ────────────────────────────────────────────────────
    detections, new_ground_baseline, debug = get_obstacle_info(
        curr_bgr, ground_baseline,
        median_ksize           = p['median_ksize'],
        min_width              = p['min_width'],
        max_col_gap            = p['max_col_gap'],
        min_ground_pixels      = p['min_ground_pixels'],
        max_gap                = p['max_gap'],
        smooth_kernel          = p['smooth_kernel'],
        obstacle_threshold     = p['obstacle_threshold'],
        no_ground_baseline     = p['no_ground_baseline'],
        plant_scale_factor     = plant_scale_factor,
        plant_blur_ksize       = plant_blur_ksize,
        plant_min_width        = 20,
        plant_min_pixels_per_col = 2,
        plant_max_col_gap      = 5,
    )

    status               = debug["status"]
    green_frac           = debug["green_frac"]
    boundary_rows        = debug["boundary_rows"]
    obstacle_regions_raw = debug["obstacle_regions_raw"]
    clean_mask           = debug["clean_mask"]
    plant_mask           = debug["plant_mask"]
    plant_regions_raw    = debug["plant_regions_raw"]

    obs_dets   = detections[detections[:, 2] == DET_OBSTACLE] if detections.shape[0] > 0 else np.empty((0,3), dtype=np.int32)
    plant_dets = detections[detections[:, 2] == DET_PLANT]    if detections.shape[0] > 0 else np.empty((0,3), dtype=np.int32)

    print(f"  [{idx:>4d}] {status} ({green_frac:.1%})  "
          f"obs={obs_dets.shape[0]}  plants={plant_dets.shape[0]}")

    # ── Optical flow ─────────────────────────────────────────────────────────
    prev_gray = to_gray(prev_bgr)
    curr_gray = to_gray(curr_bgr)

    pts = detect_flow_features(prev_gray)
    expansion  = 0.0
    ttc_frames = float("inf")
    gp = gn    = None

    if pts is not None and len(pts) >= 4:
        pts_next, status_lk, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **LK_PARAMS)
        good_prev = pts[status_lk.ravel() == 1]
        good_next = pts_next[status_lk.ravel() == 1]
        if len(good_prev) >= 4:
            gp        = good_prev.reshape(-1, 1, 2)
            gn        = good_next.reshape(-1, 1, 2)
            expansion = compute_expansion(cx, cy, gp, gn)
            ttc_frames = time_to_contact(expansion)

    ttc_sec    = ttc_frames / fps
    in_centre  = obstacle_in_centre(detections, w)

    print(f"         flow: expansion={expansion:+.3f}  TTC={ttc_sec:.1f}s  "
          f"centre_obstacle={in_centre}")

    # ── Build display ─────────────────────────────────────────────────────────
    display = curr_bgr.copy()

    # Ground mask panel (greyscale tint)
    result = np.zeros_like(curr_bgr)
    result[..., 0] = clean_mask;  result[..., 1] = clean_mask;  result[..., 2] = clean_mask

    # Plant panel
    plant_panel = np.zeros_like(curr_bgr)
    plant_panel[plant_mask > 0] = [0, 255, 0]
    edges = sdd.get_blob_edge(plant_mask)
    plant_panel[edges > 0] = [0, 0, 255]

    combined = np.vstack((display, result, plant_panel))
    panel_h  = display.shape[0]   # height of one panel

    # Status text
    cv2.putText(combined, f"{status} | {green_frac:.2%}  frame {idx}",
                (p['status_x'], p['status_y']),
                cv2.FONT_HERSHEY_SIMPLEX, p['status_font_scale'],
                (0, 0, 255), p['status_thickness'])

    # Ground baseline curve
    if status == "GROUND FOUND" and new_ground_baseline is not None:
        valid_r = np.where(new_ground_baseline < combined.shape[1])[0]
        if len(valid_r) > 1:
            pts_curve = np.array(
                [[int(r), int(new_ground_baseline[r]) + panel_h] for r in valid_r],
                dtype=np.int32)
            cv2.polylines(combined, [pts_curve], False, (255, 0, 0), 1)

    # Obstacle boxes (red)
    if status == "GROUND FOUND":
        for (sc, ec, width) in obstacle_regions_raw:
            y_top = int(min(boundary_rows[sc:ec + 1]))
            y_bot = int(max(boundary_rows[sc:ec + 1])) + p['box_bottom_pad']
            cv2.rectangle(combined, (sc, y_top + panel_h), (ec, y_bot + panel_h),
                          (0, 0, 255), p['box_thickness'])
            cv2.putText(combined, f"w={width}", (sc, y_top + panel_h - p['label_y_offset']),
                        cv2.FONT_HERSHEY_SIMPLEX, p['label_font_scale'],
                        (0, 0, 255), p['label_thickness'])
            cv2.rectangle(combined, (sc, 0), (ec, panel_h - 1),
                          (0, 0, 255), p['box_thickness'])
            cv2.putText(combined, f"w={width}", (sc, p['label_y_top']),
                        cv2.FONT_HERSHEY_SIMPLEX, p['label_font_scale'],
                        (0, 0, 255), p['label_thickness'])

    # Plant boxes (green)
    for (sc, ec, width) in plant_regions_raw:
        cv2.rectangle(combined, (sc, 0), (ec, panel_h - 1), (0, 255, 0), p['box_thickness'])
        cv2.putText(combined, f"p={width}", (sc, p['label_y_top']),
                    cv2.FONT_HERSHEY_SIMPLEX, p['label_font_scale'],
                    (0, 255, 0), p['label_thickness'])

    # Centre-band indicator (thin vertical lines)
    band_l = int(w * (0.5 - CENTRE_BAND_FRAC / 2))
    band_r = int(w * (0.5 + CENTRE_BAND_FRAC / 2))
    cv2.line(combined, (band_l, 0), (band_l, panel_h - 1), (200, 200, 0), 1, cv2.LINE_AA)
    cv2.line(combined, (band_r, 0), (band_r, panel_h - 1), (200, 200, 0), 1, cv2.LINE_AA)

    # Optical-flow overlay on the top panel only
    if gp is not None:
        top_panel = combined[:panel_h, :, :]
        draw_flow_overlay(top_panel, gp, gn, expansion, ttc_sec, in_centre)
        combined[:panel_h, :, :] = top_panel

    return combined, new_ground_baseline


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run(folder, fps):
    paths = load_frames(folder)
    n     = len(paths)

    sf                 = SCALE_FACTOR
    plant_sf           = 0.15 * sf
    p                  = get_scaled_params(sf)
    plant_blur_ksize   = scale_odd(5, plant_sf)

    probe    = cv2.rotate(cv2.imread(paths[0]), cv2.ROTATE_90_COUNTERCLOCKWISE)
    orig_H, orig_W = probe.shape[:2]   # after rotation H/W are swapped vs raw
    target_w = max(1, int(orig_W * sf))
    target_h = max(1, int(orig_H * sf))

    ground_baseline = np.full(target_h, p['no_ground_baseline'], dtype=np.float32)

    WIN = "Obstacle + Flow  [a] back  [d] forward  [q] quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    idx              = 1
    needs_processing = True
    combined         = None

    print(f"[INFO] {n} frames  |  a = back   d = forward   q = quit")

    while True:
        if needs_processing:
            combined, ground_baseline = process_frame(
                paths, idx, ground_baseline, p,
                plant_sf, plant_blur_ksize,
                target_w, target_h, fps)
            # Fit the 3-panel stack to the screen height (max 900 px tall)
            max_h   = 900
            comb_h, comb_w = combined.shape[:2]
            if comb_h > max_h:
                scale_disp = max_h / comb_h
                disp = cv2.resize(combined,
                                  (int(comb_w * scale_disp), max_h),
                                  interpolation=cv2.INTER_AREA)
            else:
                disp = combined
            cv2.imshow(WIN, disp)
            needs_processing = False

        key = cv2.waitKey(10) & 0xFF   # short wait so Ctrl-C still works

        if key == ord("q"):
            break
        elif key == ord("d"):
            if idx < n - 1:
                idx += 1;  needs_processing = True
            else:
                print("[INFO] Already at last frame.")
        elif key == ord("a"):
            if idx > 1:
                idx -= 1;  needs_processing = True
            else:
                print("[INFO] Already at first frame.")

    cv2.destroyAllWindows()
    print("\n[DONE]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Bebop obstacle detection + optical-flow TTC warning.")
    ap.add_argument("--folder", default=FOLDER_PATH,
                    help="Folder of sequential .jpg frames.")
    ap.add_argument("--fps",    type=float, default=FPS,
                    help="Source footage frame rate (default 30).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(folder=args.folder, fps=args.fps)