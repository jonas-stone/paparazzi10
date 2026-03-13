import numpy as np
import cv2


def detect_obstacles_from_ground(boundary_row, h, ground_baseline):
    """
    Returns: obstacle_cols, updated ground_baseline
    """

    alpha = 0.1
    obstacle_threshold = 20

    valid = boundary_row < h

    if ground_baseline is None:
        return np.array([]), boundary_row.astype(float).copy()

    deviation = boundary_row - ground_baseline

    obstacle_cols = np.where(valid & (deviation > obstacle_threshold))[0]

    no_obstacle_mask = valid & (deviation <= obstacle_threshold)
    ground_baseline = ground_baseline.copy()  # avoid mutating in-place
    ground_baseline[no_obstacle_mask] = (
        (1 - alpha) * ground_baseline[no_obstacle_mask]
            + alpha * boundary_row[no_obstacle_mask]
    )

    if obstacle_cols.size != 0: print("OBSTACLES FOUND!!")

    return obstacle_cols, ground_baseline


def detect_obstacles_from_parabola(green_mask, 
                                    coeffs,
                                    gap_threshold=15,
                                    plant_threshold=15,
                                    min_obstacle_width=10):
    """
    Same logic as before, but adapted for a -90° rotated image:
    - Ground is on the LEFT side of the image
    - Rows    → x axis (horizontal position in the scene)
    - Columns → y axis (depth into the scene)
    
    The parabola is now: x_expected = a*y^2 + b*y + c
    where y is the column index (depth) and x is the row index (lateral position)
    
    green_mask : H x W binary mask (1 = green pixel)
    coeffs     : [a, b, c] fitted on (y, x) data  ← note the swap!
    """
    height, width = green_mask.shape[:2]
    # obstacle_map now has one entry per ROW (since rows = x = lateral position)
    obstacle_map = np.zeros(height, dtype=int)

    for y in range(height):
        row = green_mask[y, :]                              # horizontal slice
        x_expected = coeffs[0]*y**2 + coeffs[1]*y + coeffs[2]
        x_expected = int(np.clip(x_expected, 0, width - 1))

        green_cols = np.where(row > 0)[0]                  # columns with green pixels

        # ── CASE 1: row fully blocked ─────────────────────────────────────────
        # No green at all in this row → close obstacle covering full row
        if len(green_cols) == 0:
            obstacle_map[y] = 1
            continue

        x_leftmost  = green_cols[0]    # leftmost green pixel (closest to ground)
        x_rightmost = green_cols[-1]

        # ── CASE 2: ground contour too far right of expected ──────────────────
        # The leftmost green pixel is much further right than the parabola predicts.
        # Something non-green is sitting between the camera and the ground.
        # Remember: larger x = further right in image = further from ground
        if x_leftmost > x_expected + gap_threshold:
            obstacle_map[y] = 1
            continue

        # ── CASE 3: green pixels significantly left of expected line ──────────
        # Green vegetation sticking out to the left of the ground boundary.
        green_left_of_line = np.sum(green_cols < x_expected - plant_threshold)
        if green_left_of_line > 5:
            obstacle_map[y] = 2
            continue

    # Suppress narrow detections (noise)
    obstacle_map = suppress_narrow_detections(obstacle_map, min_obstacle_width)

    return obstacle_map


def suppress_narrow_detections(obstacle_map, min_width):
    """
    Remove obstacle detections narrower than min_width rows.
    Logic is identical to before — just operating on rows instead of columns.
    """
    cleaned = obstacle_map.copy()
    n = len(obstacle_map)
    y = 0
    while y < n:
        if obstacle_map[y] != 0:
            start = y
            val = obstacle_map[y]
            while y < n and obstacle_map[y] == val:
                y += 1
            run_length = y - start
            if run_length < min_width:
                cleaned[start:y] = 0
        else:
            y += 1
    return cleaned


def draw_obstacle_boxes(mask, obstacle_map, coeffs, color=(255, 0, 0), thickness=2):
    """
    Draw boxes around contiguous obstacle runs directly on the green mask.

    mask         : H x W binary mask (1 channel)
    obstacle_map : 1D array of length H (0=clear, 1=obstacle, 2=plant)
    coeffs       : [a, b, c] parabola coefficients
    color        : pixel value to draw with (default 255 = white)
    thickness    : box border thickness in pixels
    """
    height, width = mask.shape[:2]
    vis = mask.copy()

    # ── STEP 1: find contiguous runs of obstacle rows (value == 1) ────────────
    y = 0
    while y < height:
        if obstacle_map[y] == 1:
            # Found the start of an obstacle run
            start_y = y

            # Walk forward until the run ends
            while y < height and obstacle_map[y] == 1:
                y += 1
            end_y = y - 1  # last row of this obstacle run

            # ── STEP 2: compute box horizontal bounds ─────────────────────────
            # Left edge:  the minimum parabola x across all rows in this run
            # Right edge: the right edge of the image
            # This way the box covers exactly the "should be ground" region
            x_parabola_values = [
                int(np.clip(coeffs[0]*row**2 + coeffs[1]*row + coeffs[2], 0, width-1))
                for row in range(start_y, end_y + 1)
            ]
            x_left  = min(x_parabola_values)
            x_right = width - 1

            # ── STEP 3: draw the box border ───────────────────────────────────
            # Top edge
            vis[start_y : start_y + thickness, x_left:x_right] = color
            # Bottom edge
            vis[end_y - thickness : end_y, x_left:x_right] = color
            # Left edge
            vis[start_y:end_y, x_left : x_left + thickness] = color
            # Right edge
            vis[start_y:end_y, x_right - thickness : x_right] = color

        else:
            y += 1

    return vis
