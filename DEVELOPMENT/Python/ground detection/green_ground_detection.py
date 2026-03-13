import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import random
from glob import glob


# def find_ground_boundary(mask_rotated):

#     """ 
#     The image has to be rotated for this to work. 
#     The ground has to be on the right side. 
#     """
    
#     image_height = mask_rotated.shape[1]
#     boundary_rows = []
#     for row in mask_rotated:

#         # get the index of the first nonzero element of this row
#         nonzero_indices = np.where(row > 0)
#         nonzero = nonzero_indices[0] - 1
#         if nonzero.size == 0:
#             nonzero = np.asarray([image_height])
#         boundary_rows.append(int(nonzero[0]))

#     return np.asarray(boundary_rows)

def find_ground_boundary(mask_rotated, min_ground_pixels=5, max_gap=10, smooth_kernel=5):
    image_height = mask_rotated.shape[1]
    n_rows = mask_rotated.shape[0]
    boundary_rows = np.full(n_rows, image_height, dtype=int)

    for idx, row in enumerate(mask_rotated):
        green_pos = np.where(row > 0)[0]  # positions of all green pixels

        if green_pos.size == 0:
            continue  # sentinel already set

        ground_valid = np.sum(row[-min_ground_pixels:] > 0) >= min_ground_pixels

        if not ground_valid:
            # Check if any consecutive green run is thick enough
            diffs = np.diff(green_pos)
            runs = np.split(green_pos, np.where(diffs > 1)[0] + 1)
            if max(len(r) for r in runs) < max_gap:
                continue  # truly nothing → sentinel

        # Work rightmost first (ground side)
        gp = green_pos[::-1]

        if gp.size == 1:
            boundary_rows[idx] = gp[0]
            continue

        # Gap sizes between consecutive green pixels scanning right to left
        gaps = -np.diff(gp) - 1

        big_gap_indices = np.where(gaps > max_gap)[0]

        if big_gap_indices.size == 0:
            boundary_rows[idx] = gp[-1]  # no big gap → leftmost green
            continue

        # First big gap found
        fg = big_gap_indices[0]
        pending_boundary = gp[fg]

        # Check if green run after the gap is substantial enough to override
        after_gap = gp[fg + 1:]
        if after_gap.size >= max_gap:
            after_gaps = -np.diff(after_gap) - 1
            split_points = np.where(after_gaps > 0)[0]
            runs = np.split(after_gap, split_points + 1)
            if max(len(r) for r in runs) >= max_gap:
                boundary_rows[idx] = gp[-1]  # override → use leftmost green
                continue

        boundary_rows[idx] = pending_boundary

    # Median filter on valid columns only
    valid_mask = boundary_rows < image_height
    if valid_mask.sum() > smooth_kernel:
        smoothed = median_filter(boundary_rows.astype(float), size=smooth_kernel)
        boundary_rows[valid_mask] = smoothed[valid_mask].astype(int)

    return boundary_rows


def get_obstacle_regions(obstacle_cols, min_width=20, max_col_gap=5):
    """
    Groups obstacle columns into contiguous regions and filters by width.
    
    - min_width:    minimum number of columns to count as a real obstacle
    - max_col_gap:  allow small gaps between flagged columns (noise tolerance)
    
    Returns: list of (start_col, end_col, width) for each valid obstacle region
    """
    if len(obstacle_cols) == 0:
        return []

    regions = []
    start = obstacle_cols[0]
    end = obstacle_cols[0]

    for col in obstacle_cols[1:]:
        if col - end <= max_col_gap:
            end = col  # extend current region
        else:
            regions.append((start, end, end - start + 1))
            start = col
            end = col

    regions.append((start, end, end - start + 1))  # final region

    # Filter by minimum width
    return [(s, e, w) for s, e, w in regions if w >= min_width]

def update_and_detect(boundary_row, h, ground_baseline, min_width=20):
    alpha = 0.6
    obstacle_threshold = 50
    NO_GROUND_BASELINE = 220

    # Columns where ground was actually detected
    valid = boundary_row < h
    # Columns where ground side was fully black (sentinel)
    no_ground = boundary_row >= h

    if ground_baseline is None:
        baseline = boundary_row.astype(float).copy()
        baseline[no_ground] = NO_GROUND_BASELINE  # flat line from the start
        return [], baseline

    deviation = boundary_row - ground_baseline

    # Condition 1: ground visible but too far from baseline
    deviation_obstacle = valid & (deviation > obstacle_threshold)

    # Condition 2: no green at ground side at all → possibly right up against obstacle
    no_ground_obstacle = no_ground

    # Combine both
    obstacle_mask = deviation_obstacle | no_ground_obstacle
    obstacle_cols = np.where(obstacle_mask)[0]

    # Group into regions and filter by width
    obstacle_regions = get_obstacle_regions(obstacle_cols, min_width=min_width)

    # Only update baseline on clean columns (valid, no deviation obstacle, no sentinel)
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


def detect_green_ground_simple(image_bgr, lower_green, upper_green, threshold, median_ksize=3):
    """
    Very simple green detector with a single median blur to remove salt-and-pepper noise.
    - median_ksize should be odd and >=3 (3 is the fastest / mildest).
    Returns: mask, result, green_fraction, status
    """
    # HSV mask
    width = image_bgr.shape[1]
    #image_bgr[:, width//2:] = 0 # Crop to width
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_green, upper_green)  # 0/255

    # Simple median blur to reduce speckles
    if median_ksize is not None and median_ksize >= 3 and median_ksize % 2 == 1:
        mask = cv2.medianBlur(mask, median_ksize)

    # Pixel stats
    green_pixel_count = cv2.countNonZero(mask)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    green_fraction = green_pixel_count / total_pixels if total_pixels > 0 else 0.0

    status = "GROUND FOUND" if green_fraction > threshold else "NO GROUND"

    # Visualization (keep only detected regions)
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    return mask, result, green_fraction, status


if __name__ == "__main__":

    # python files
    import solidity_detection as sdd
    import colored_blob_separator as cds

    # libraries
    import matplotlib.pyplot as plt
    from glob import glob
    import os
    cv2.destroyAllWindows()

    folder_path = "TEAM-10-PROTOTYPING/downloads from drone/20260306-095826"
    all_image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))
    start_idx = random.randint(0, len(all_image_paths) - 1)
    image_paths = all_image_paths[start_idx:] + all_image_paths[:start_idx]

    # same HSV bounds as before
    lower_green = np.array([10, 50, 40])
    upper_green = np.array([35, 255, 200])

    oa_color_count_frac = 0.05

    # state initialization
    ground_baseline = None
    obstacle_regions = []
    image_height = np.asarray(cv2.imread(image_paths[0])).shape[1]
    boundary_rows = np.full(image_height, image_height)  # sentinel default

    for image_path in image_paths:

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Skipping: {image_path}")
            continue
        
        image_display = image_bgr.copy()
        # reduce quality of image (TEMPORARY)
        # factor = 0.1
        # image_bgr = cv2.resize(image_bgr, (0,0), fx=factor, fy=factor)

        # Use the simple detector with a 3x3 median blur (fast, mild)
        mask, result, actual_frac, status = detect_green_ground_simple(
            image_bgr,
            lower_green,
            upper_green,
            oa_color_count_frac,
            median_ksize=5
        )

        initial_mask = mask.copy()

        # set detected pixels to white.
        result[result > 0] = 255

        # isolate the ground white blob
        mask = cds.isolate_ground_blob(binary_img=mask, blob_area_threshold=1000)
        mask = cds.fill_holes(mask)
        
        # Subtract ground mask from raw green mask → only plants remain
        plant_mask = cds.isolate_ground_blob(cv2.subtract(initial_mask, mask), blob_area_threshold=100)
        
        result[..., 0] = mask
        result[..., 1] = mask
        result[..., 2] = mask
        

        print(f"{os.path.basename(image_path)} -> {status} ({actual_frac:.2%})")

        # try find the obstacle
        if status == "GROUND FOUND":
            mask_flipped = mask[:, ::-1]
            # plt.imshow(mask_flipped)
            boundary_rows = find_ground_boundary(mask_rotated=mask_flipped)
            obstacle_regions, ground_baseline = update_and_detect(boundary_rows, image_height, ground_baseline)

        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        plant_result = np.zeros_like(image_display)
        plant_result[plant_mask > 0] = [0, 255, 0]
        edges = sdd.get_blob_edge(plant_mask)
        plant_result[edges > 0] = [0, 0, 255]  # Highlight edges in red
        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------

        # Rotate images 90° CCW and stack vertically
        image_rot = cv2.rotate(image_display, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result_rot = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plant_rot    = cv2.rotate(plant_result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined_view = np.vstack((image_rot, result_rot, plant_rot))

        # Annotate
        cv2.putText(combined_view,
                    f"{status} | {actual_frac:.2%}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)
        
        # # put a red rot where the obstacle columns are
        if status == "GROUND FOUND":
            h_rot = image_rot.shape[0]

            if ground_baseline is not None:
                valid_r = np.where(ground_baseline < combined_view.shape[0])[0]
                if len(valid_r) > 1:
                    pts = np.array([[int(r), int(ground_baseline[r]) + h_rot] 
                                    for r in valid_r], dtype=np.int32)
                    cv2.polylines(combined_view, [pts], False, (255, 0, 0), 1)

            FULL_HEIGHT_BOX = 0
            for (start_col, end_col, width) in obstacle_regions:
                if FULL_HEIGHT_BOX:
                    y_top = 0
                    y_bot = combined_view.shape[0] - h_rot  # full height of bottom panel
                else:
                    y_top = int(min(boundary_rows[start_col:end_col+1]))
                    y_bot = int(max(boundary_rows[start_col:end_col+1])) + 15
                cv2.rectangle(combined_view,
                            (start_col, y_top + h_rot),
                            (end_col,   y_bot + h_rot),
                            (0, 0, 255), 2)
                cv2.putText(combined_view, f"w={width}",
                            (start_col, y_top + h_rot - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Top panel (original image) — same x coords, no h_rot offset
                cv2.rectangle(combined_view,
                            (start_col, 0),
                            (end_col,   h_rot - 1),
                            (0, 0, 255), 2)
                cv2.putText(combined_view, f"w={width}",
                            (start_col, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow('Original (top) vs Green Mask (bottom)', combined_view)
        
        key = cv2.waitKey(40)  # short delay; press 'q' to quit
        if key == ord('q'):
            break

    cv2.destroyAllWindows()