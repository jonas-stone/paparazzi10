
import cv2
import numpy as np
from scipy.ndimage import median_filter

def find_ground_boundary(mask_rotated, min_ground_pixels=5, max_gap=10, smooth_kernel=5):
    image_height = mask_rotated.shape[1]
    n_rows = mask_rotated.shape[0]
    boundary_rows = np.full(n_rows, image_height, dtype=int)

    for idx, row in enumerate(mask_rotated):
        green_pos = np.where(row > 0)[0] 

        if green_pos.size == 0:
            continue

        ground_valid = np.sum(row[-min_ground_pixels:] > 0) >= min_ground_pixels

        if not ground_valid:
            diffs = np.diff(green_pos)
            runs = np.split(green_pos, np.where(diffs > 1)[0] + 1)
            if max(len(r) for r in runs) < max_gap:
                continue

        gp = green_pos[::-1]

        if gp.size == 1:
            boundary_rows[idx] = gp[0]
            continue

        gaps = -np.diff(gp) - 1
        big_gap_indices = np.where(gaps > max_gap)[0]

        if big_gap_indices.size == 0:
            boundary_rows[idx] = gp[-1]
            continue

        fg = big_gap_indices[0]
        pending_boundary = gp[fg]

        after_gap = gp[fg + 1:]
        if after_gap.size >= max_gap:
            after_gaps = -np.diff(after_gap) - 1
            split_points = np.where(after_gaps > 0)[0]
            runs = np.split(after_gap, split_points + 1)
            if max(len(r) for r in runs) >= max_gap:
                boundary_rows[idx] = gp[-1] 
                continue

        boundary_rows[idx] = pending_boundary

    valid_mask = boundary_rows < image_height
    if valid_mask.sum() > smooth_kernel:
        smoothed = median_filter(boundary_rows.astype(float), size=smooth_kernel)
        boundary_rows[valid_mask] = smoothed[valid_mask].astype(int)

    return boundary_rows


def get_obstacle_regions(obstacle_cols, min_width=20, max_col_gap=5):
    if len(obstacle_cols) == 0:
        return []

    regions = []
    start = obstacle_cols[0]
    end = obstacle_cols[0]

    for col in obstacle_cols[1:]:
        if col - end <= max_col_gap:
            end = col 
        else:
            regions.append((start, end, end - start + 1))
            start = col
            end = col

    regions.append((start, end, end - start + 1)) 

    return [(s, e, w) for s, e, w in regions if w >= min_width]


def update_and_detect(boundary_row, h, ground_baseline, min_width=20):
    alpha = 0.6
    obstacle_threshold = 50
    NO_GROUND_BASELINE = 220

    valid = boundary_row < h
    no_ground = boundary_row >= h

    if ground_baseline is None:
        baseline = boundary_row.astype(float).copy()
        baseline[no_ground] = NO_GROUND_BASELINE 
        return [], baseline

    deviation = boundary_row - ground_baseline
    deviation_obstacle = valid & (deviation > obstacle_threshold)
    no_ground_obstacle = no_ground

    obstacle_mask = deviation_obstacle | no_ground_obstacle
    obstacle_cols = np.where(obstacle_mask)[0]

    obstacle_regions = get_obstacle_regions(obstacle_cols, min_width=min_width)

    no_obstacle_mask = valid & ~deviation_obstacle
    ground_baseline = ground_baseline.copy()
    ground_baseline[no_obstacle_mask] = (
        (1 - alpha) * ground_baseline[no_obstacle_mask]
        + alpha * boundary_row[no_obstacle_mask]
    )

    last_good = NO_GROUND_BASELINE
    for i in range(len(ground_baseline)):
        if no_obstacle_mask[i]:
            last_good = ground_baseline[i]
        else:
            ground_baseline[i] = last_good

    return obstacle_regions, ground_baseline


# def is_ground(Y, U, V):
#     if U <= 116.50:
#         if V <= 149.50:
#             if Y <= 81.50:
#                 return 0
#             else:
#                 if U <= 96.50:
#                     return 0
#                 else:
#                     return 255
#         else:
#             if Y <= 113.50:
#                 if Y <= 101.50:
#                     return 0
#                 else:
#                     return 255
#             else:
#                 return 0
#     else:
#         if U <= 122.50:
#             if Y <= 87.00:
#                 if V <= 119.50:
#                     return 0
#                 else:
#                     return 0
#             else:
#                 if V <= 143.50:
#                     return 255
#                 else:
#                     return 0
#         else:
#             if U <= 145.50:
#                 return 0
#             else:
#                 if U <= 148.50:
#                     return 255
#                 else:
#                     return 0

def is_ground(Y, U, V):
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



vectorized_is_ground = np.vectorize(is_ground)

def detect_green_ground_ml(image_bgr, threshold, median_ksize=3):
    """
    Applies the hardcoded Python decision tree logic to the image.
    """
    # Convert BGR to YUV
    image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    
    # Split the image into its 3 separate color channels
    Y_channel, U_channel, V_channel = cv2.split(image_yuv)
    
    # Pass the entire channels into our vectorized logic tree at once
    mask = vectorized_is_ground(Y_channel, U_channel, V_channel)
    
    # Convert back to an 8-bit unsigned integer array for OpenCV
    mask = mask.astype(np.uint8)

    # Simple median blur to reduce speckles
    if median_ksize is not None and median_ksize >= 3 and median_ksize % 2 == 1:
        mask = cv2.medianBlur(mask, median_ksize)

    # Pixel stats
    green_pixel_count = cv2.countNonZero(mask)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    green_fraction = green_pixel_count / total_pixels if total_pixels > 0 else 0.0

    status = "GROUND FOUND" if green_fraction > threshold else "NO GROUND"

    # Visualization
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    return mask, result, green_fraction, status

def get_obstacle_info(image_bgr,
                                ground_baseline,
                                oa_color_count_frac=0.05,
                                median_ksize=5,
                                min_width=20):
    """
    Detect ground vs obstacle, compute obstacle start (left) and width.
    Inputs:
      - image_bgr: OpenCV BGR image (H x W x 3)
      - ground_baseline: 1D numpy array or None (previous baseline)
      - oa_color_count_frac, median_ksize, min_width: tuning params
    Returns:
      - obstacles: list of (start_col, width) tuples (columns counted left->right)
      - new_ground_baseline: updated baseline (same format as input)
      - debug: dict with mask, boundary_rows, status (optional)
    Notes:
      - This function does only image->obstacle conversion. Keep CIC (single responsibility).
    """
    # 1) detect green ground mask (use your ML or decision-tree)
    mask, _, green_frac, status = detect_green_ground_ml(image_bgr,
                                                         threshold=oa_color_count_frac,
                                                         median_ksize=median_ksize)
    H, W = image_bgr.shape[:2]

    # If no ground found, keep baseline but mark no-ground columns
    obstacles = []
    debug = {"status": status, "green_frac": green_frac}

    if status == "GROUND FOUND":
        # your pipeline flips/mirrors so columns correspond to left/right correctly
        mask_flipped = mask[:, ::-1]  # keep same convention as your pipeline
        boundary_rows = find_ground_boundary(mask_rotated=mask_flipped)
        obstacle_regions, new_ground_baseline = update_and_detect(
            boundary_rows, W, ground_baseline, min_width=min_width)
        # obstacle_regions are (start_col, end_col, width) in *rotated* coords (matching mask_flipped)
        # convert to left->right image coordinates (undo flip)
        obstacles = []
        for (s, e, w) in obstacle_regions:
            # after flipping columns idx c maps to (W-1 - c)
            left = W - 1 - e
            width = w
            obstacles.append((int(left), int(width)))
        debug.update({"boundary_rows": boundary_rows, "obstacle_regions_raw": obstacle_regions})
    else:
        new_ground_baseline = ground_baseline  # unchanged

    return obstacles, new_ground_baseline, debug