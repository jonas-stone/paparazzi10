import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import random
from glob import glob
import joblib # Added to load the ML model

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


if __name__ == "__main__":

    # python files
    import solidity_detection as sdd
    import colored_blob_separator as cds

    cv2.destroyAllWindows()

    # Load the trained model here (so it only loads once)
    try:
        model = joblib.load('yuv_decision_tree.joblib')
        print("Successfully loaded ML model.")
    except FileNotFoundError:
        print("Error: 'yuv_decision_tree.joblib' not found. Please run the training script first.")
        exit()

    folder_path = "TEAM-10-PROTOTYPING/downloads from drone/20260306-095826"
    all_image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))
    start_idx = random.randint(0, len(all_image_paths) - 1)
    image_paths = all_image_paths[start_idx:] + all_image_paths[:start_idx]

    oa_color_count_frac = 0.05

    # state initialization
    ground_baseline = None
    obstacle_regions = []
    
    if len(image_paths) == 0:
        print("No images found in the specified path.")
        exit()
        
    image_height = np.asarray(cv2.imread(image_paths[0])).shape[1]
    boundary_rows = np.full(image_height, image_height)

    for image_path in image_paths:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Skipping: {image_path}")
            continue
        
        image_display = image_bgr.copy()

        # Call the new ML-based detection function instead of the simple one
        mask, result, actual_frac, status = detect_green_ground_ml(
            image_bgr,
            oa_color_count_frac,
            median_ksize=5
        )

        initial_mask = mask.copy()

        result[result > 0] = 255

        mask = cds.isolate_ground_blob(binary_img=mask)
        mask = cds.fill_holes(mask)
        
        plant_mask = cv2.subtract(initial_mask, mask)

        result[..., 0] = mask
        result[..., 1] = mask
        result[..., 2] = mask
        
        print(f"{os.path.basename(image_path)} -> {status} ({actual_frac:.2%})")

        if status == "GROUND FOUND":
            mask_flipped = mask[:, ::-1]
            boundary_rows = find_ground_boundary(mask_rotated=mask_flipped)
            obstacle_regions, ground_baseline = update_and_detect(boundary_rows, image_height, ground_baseline)

        plant_result = np.zeros_like(image_display)
        plant_result[plant_mask > 0] = [0, 255, 0]
        edges = sdd.get_blob_edge(plant_mask)
        plant_result[edges > 0] = [0, 0, 255]

        image_rot = cv2.rotate(image_display, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result_rot = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plant_rot  = cv2.rotate(plant_result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined_view = np.vstack((image_rot, result_rot, plant_rot))

        cv2.putText(combined_view,
                    f"{status} | {actual_frac:.2%}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)
        
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
                    y_bot = combined_view.shape[0] - h_rot
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
                
                cv2.rectangle(combined_view,
                            (start_col, 0),
                            (end_col,   h_rot - 1),
                            (0, 0, 255), 2)
                cv2.putText(combined_view, f"w={width}",
                            (start_col, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow('Original (top) vs Green Mask (bottom)', combined_view)
        
        key = cv2.waitKey(40)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()