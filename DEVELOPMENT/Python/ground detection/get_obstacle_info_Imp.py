import cv2
import numpy as np
from scipy.ndimage import median_filter
import os
import random
from glob import glob
import colored_blob_separator as cds


# ── scale helpers ─────────────────────────────────────────────────────────────

def scale_int(value, scale_factor):
    """Scale a pixel-count value, minimum 1."""
    return max(1, int(round(value * scale_factor)))

def scale_odd(value, scale_factor):
    """Scale a kernel size (must be odd, minimum 3)."""
    n = max(3, int(round(value * scale_factor)))
    return n if n % 2 == 1 else n + 1

def get_scaled_params(scale_factor):
    """
    All pixel-space thresholds and display parameters in one place.
    Calibrated at scale_factor=1.0 (full resolution).
    """
    return dict(
        # ── pipeline ──────────────────────────────────────────────────────────
        min_ground_pixels  = scale_int(5,   scale_factor),
        max_gap            = scale_int(10,  scale_factor),
        smooth_kernel      = scale_odd(5,   scale_factor),
        min_width          = scale_int(20,  scale_factor),
        max_col_gap        = scale_int(5,   scale_factor),
        obstacle_threshold = scale_int(50,  scale_factor),
        no_ground_baseline = scale_int(220, scale_factor),
        median_ksize       = scale_odd(5,   scale_factor),
        blur_ksize         = scale_odd(5,   scale_factor),

        # ── display ───────────────────────────────────────────────────────────
        box_bottom_pad     = scale_int(15,  scale_factor),
        box_thickness      = max(1, scale_int(2,   scale_factor)),
        status_font_scale  = max(0.3, scale_factor * 1.0),
        status_thickness   = max(1, scale_int(2,   scale_factor)),
        status_x           = scale_int(20,  scale_factor),
        status_y           = scale_int(40,  scale_factor),
        label_font_scale   = max(0.2, scale_factor * 0.4),
        label_thickness    = max(1, scale_int(1,   scale_factor)),
        label_y_offset     = scale_int(5,   scale_factor),
        label_y_top        = scale_int(20,  scale_factor),
    )


# ── Detection type flag ───────────────────────────────────────────────────────
# Each row in the detections matrix: [left_x, width, det_type]
#   det_type: 0 = obstacle, 1 = plant
DET_OBSTACLE = 0
DET_PLANT    = 1


# ── core pipeline functions ───────────────────────────────────────────────────

def find_ground_boundary(mask_rotated,
                         min_ground_pixels=5,
                         max_gap=10,
                         smooth_kernel=5):
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
    end   = obstacle_cols[0]

    for col in obstacle_cols[1:]:
        if col - end <= max_col_gap:
            end = col
        else:
            regions.append((start, end, end - start + 1))
            start = col
            end   = col

    regions.append((start, end, end - start + 1))

    return [(s, e, w) for s, e, w in regions if w >= min_width]


def update_and_detect(boundary_row, h, ground_baseline,
                      min_width=20,
                      obstacle_threshold=50,
                      no_ground_baseline=220,
                      max_col_gap=5):
    alpha = 0.6

    valid     = boundary_row < h
    no_ground = boundary_row >= h

    if ground_baseline is None:
        baseline = boundary_row.astype(float).copy()
        baseline[no_ground] = no_ground_baseline
        return [], baseline

    deviation          = boundary_row - ground_baseline
    deviation_obstacle = valid & (deviation > obstacle_threshold)
    obstacle_mask      = deviation_obstacle | no_ground

    obstacle_cols    = np.where(obstacle_mask)[0]
    obstacle_regions = get_obstacle_regions(obstacle_cols,
                                            min_width=min_width,
                                            max_col_gap=max_col_gap)

    no_obstacle_mask = valid & ~deviation_obstacle
    ground_baseline  = ground_baseline.copy()
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


# ── colour classifiers ────────────────────────────────────────────────────────

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


def is_ground_sim(Y, U, V):
    if U <= 96.50:
        if Y <= 102.50:
            return 255
        else:
            return 0
    else:
        if U <= 97.50:
            if V <= 126.00:
                return 255
            else:
                return 0
        else:
            return 0


vectorized_is_ground     = np.vectorize(is_ground)
# vectorized_is_ground_sim = np.vectorize(is_ground_sim)


# ── detection helper functions ────────────────────────────────────────────────

def detect_green_ground_ml(image_bgr, threshold, median_ksize=5):
    image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    Y_channel, U_channel, V_channel = cv2.split(image_yuv)

    mask = vectorized_is_ground(Y_channel, U_channel, V_channel).astype(np.uint8)

    if median_ksize is not None and median_ksize >= 3 and median_ksize % 2 == 1:
        mask = cv2.medianBlur(mask, median_ksize)

    green_pixel_count = cv2.countNonZero(mask)
    total_pixels      = image_bgr.shape[0] * image_bgr.shape[1]
    green_fraction    = green_pixel_count / total_pixels if total_pixels > 0 else 0.0

    status = "GROUND FOUND" if green_fraction > threshold else "NO GROUND"
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    return mask, result, green_fraction, status


def detect_all_green_lax(image_bgr, scale_factor=1.0, blur_ksize=5):
    """
    Lax YUV green filter at a (further) downscaled resolution.
    Returns: (small_mask, full_res_mask, downscaled_bgr)
    """
    H, W = image_bgr.shape[:2]

    small_w    = max(1, int(W * scale_factor))
    small_h    = max(1, int(H * scale_factor))
    downscaled = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    yuv = cv2.cvtColor(downscaled, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    # YUV colour thresholds — colour-space values (0-255), do NOT scale
    green_mask_small = (
        (U >= 0)  & (U <= 116) &
        (V >= 0)  & (V <= 141) &
        (Y >= 29) & (Y <= 140)
    ).astype(np.uint8) * 255

    green_mask_small[:, :int(small_w / 3)] = 0

    if blur_ksize >= 3:
        green_mask_small = cv2.GaussianBlur(
            green_mask_small, (blur_ksize, blur_ksize), 0)
        _, green_mask_small = cv2.threshold(
            green_mask_small, 127, 255, cv2.THRESH_BINARY)

    mask_full = cv2.resize(green_mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    return green_mask_small, mask_full, downscaled


def detect_plant_regions(plant_mask, min_width=10, min_pixels_per_col=2, max_col_gap=3):
    plant_mask_flipped = plant_mask[:, ::-1]   # match obstacle pipeline convention
    n_rows = plant_mask_flipped.shape[0]
    active_rows = []

    for row_idx in range(n_rows):
        row = plant_mask_flipped[row_idx, :]
        green_pos = np.where(row > 0)[0]
        if green_pos.size == 0:
            continue
        diffs = np.diff(green_pos)
        runs = np.split(green_pos, np.where(diffs > 1)[0] + 1)
        if max(len(r) for r in runs) > min_pixels_per_col:
            active_rows.append(row_idx)

    return get_obstacle_regions(active_rows, min_width=min_width, max_col_gap=max_col_gap)


# ── unified detection function ────────────────────────────────────────────────

def get_obstacle_info(image_bgr,
                      ground_baseline,
                      # obstacle params
                      oa_color_count_frac=0.05,
                      median_ksize=5,
                      min_width=20,
                      max_col_gap=5,
                      min_ground_pixels=5,
                      max_gap=10,
                      smooth_kernel=5,
                      obstacle_threshold=50,
                      no_ground_baseline=220,
                      # plant params
                      plant_scale_factor=0.15,
                      plant_blur_ksize=3,
                      plant_min_width=30,
                      plant_min_pixels_per_col=2,
                      plant_max_col_gap=5):
    """
    Unified obstacle + plant detection.

    Returns
    -------
    detections : np.ndarray, shape (N, 3), dtype int32
        Each row is [left_x, width, det_type].
        det_type: 0 = obstacle, 1 = plant.
        Empty detections → shape (0, 3).
    new_ground_baseline : np.ndarray or None
    debug : dict
        Intermediate data for visualisation.
    """
    mask, _, green_frac, status = detect_green_ground_ml(
        image_bgr, threshold=oa_color_count_frac, median_ksize=median_ksize)

    H, W  = image_bgr.shape[:2]
    rows  = []     # accumulate [left_x, width, det_type] rows
    debug = {"status": status, "green_frac": green_frac}

    # ── obstacle detection ─────────────────────────────────────────────────
    if status == "GROUND FOUND":
        clean_mask = cds.isolate_ground_blob(binary_img=mask)
        clean_mask = cds.fill_holes(clean_mask)

        mask_flipped  = clean_mask[:, ::-1]
        boundary_rows = find_ground_boundary(
            mask_rotated      = mask_flipped,
            min_ground_pixels = min_ground_pixels,
            max_gap           = max_gap,
            smooth_kernel     = smooth_kernel,
        )
        obstacle_regions, new_ground_baseline = update_and_detect(
            boundary_rows,
            W,
            ground_baseline,
            min_width          = min_width,
            obstacle_threshold = obstacle_threshold,
            no_ground_baseline = no_ground_baseline,
            max_col_gap        = max_col_gap,
        )

        for (s, e, w) in obstacle_regions:
            left = W - 1 - e
            rows.append([int(left), int(w), DET_OBSTACLE])

        debug.update({
            "boundary_rows"       : boundary_rows,
            "obstacle_regions_raw": obstacle_regions,
            "clean_mask"          : clean_mask,
        })
    else:
        new_ground_baseline = ground_baseline
        clean_mask = np.zeros((H, W), dtype=np.uint8)
        debug["boundary_rows"]        = np.full(W, W)
        debug["obstacle_regions_raw"] = []
        debug["clean_mask"]           = clean_mask

    # ── plant detection ────────────────────────────────────────────────────
    all_green_small, _, _ = detect_all_green_lax(
        image_bgr,
        scale_factor = plant_scale_factor,
        blur_ksize   = plant_blur_ksize,
    )

    plant_H, plant_W = all_green_small.shape[:2]
    clean_mask_small = cv2.resize(
        clean_mask, (plant_W, plant_H), interpolation=cv2.INTER_NEAREST)

    plant_mask_small = cv2.subtract(all_green_small, clean_mask_small)
    plant_mask = cv2.resize(
        plant_mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    plant_regions = detect_plant_regions(
        plant_mask,
        min_width         = plant_min_width,
        min_pixels_per_col = plant_min_pixels_per_col,
        max_col_gap       = plant_max_col_gap,
    )

    for (s, e, w) in plant_regions:
        rows.append([int(s), int(w), DET_PLANT])

    debug["plant_mask"]           = plant_mask
    debug["plant_regions_raw"]    = plant_regions

    # ── build output matrix ────────────────────────────────────────────────
    if rows:
        detections = np.array(rows, dtype=np.int32)   # shape (N, 3)
    else:
        detections = np.empty((0, 3), dtype=np.int32)

    return detections, new_ground_baseline, debug


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import solidity_detection as sdd

    # ── run mode ─────────────────────────────────────────────
    MODE = 2
    # 1 = interactive navigation
    # 2 = single image

    SINGLE_IMAGE_PATH = "DEVELOPMENT/downloads from drone/20260306-095826/1352900896.jpg"
    FRAME_DELAY = 50

    cv2.destroyAllWindows()


    if MODE == 2:
        image_paths = [SINGLE_IMAGE_PATH]
        start_idx   = 0
    else:
        #folder_path = "DEVELOPMENT/downloads from drone/20260306-095826/"
        folder_path = "DEVELOPMENT/downloads from drone/20260313-100130/"
        #folder_path = "DEVELOPMENT/downloads from drone/sim_images/"
        image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))

        start_idx   = random.randint(0, len(image_paths) - 1)

    oa_color_count_frac = 0.05

    # ── scale factors ──────────────────────────────────────────────────────
    SCALE_FACTOR       = 0.8
    PLANT_SCALE_FACTOR = 0.15*0.8  

    p = get_scaled_params(SCALE_FACTOR)
    plant_blur_ksize = scale_odd(5, PLANT_SCALE_FACTOR)

    print(f"Main scale:  {SCALE_FACTOR}  →  pipeline params: {p}")
    print(f"Plant scale: {PLANT_SCALE_FACTOR} = "
          f"{PLANT_SCALE_FACTOR:.2f}  →  plant blur_ksize: {plant_blur_ksize}")

    # state
    #ground_baseline = None

    if len(image_paths) == 0:
        print("No images found in the specified path.")
        exit()

    _probe         = cv2.imread(image_paths[0])
    orig_H, orig_W = _probe.shape[:2]

    target_W = max(1, int(orig_W * SCALE_FACTOR))
    target_H = max(1, int(orig_H * SCALE_FACTOR))

    boundary_rows = np.full(target_W, target_W)

    WINDOW = 'Original (top) | Ground mask (mid) | Plants (bot)'
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, orig_W * 3, orig_H * 3)

    
    needs_processing = True
    idx = start_idx
    ground_baseline = np.full(target_H, p['no_ground_baseline'], dtype=np.float32)

    while True:
        try:
            if needs_processing:
                image_path = image_paths[idx]
                raw = cv2.imread(image_path)
                if raw is None:
                    print(f"Skipping: {image_path}")
                    idx = (idx + 1) % len(image_paths) 
                    continue

                # ── downscale to main resolution ───────────────────────────────────
                image_bgr     = cv2.resize(raw, (target_W, target_H), interpolation=cv2.INTER_AREA)
                image_display = image_bgr.copy()
                
                # If we are in single image mode, ALWAYS reset the baseline before processing
                if MODE == 2:
                    ground_baseline = np.full(target_H, p['no_ground_baseline'], dtype=np.float32)

                # ── single unified call ────────────────────────────────────────────
                detections, ground_baseline, debug = get_obstacle_info(
                    image_bgr,
                    ground_baseline,
                    oa_color_count_frac    = oa_color_count_frac,
                    median_ksize           = p['median_ksize'],
                    min_width              = p['min_width'],
                    max_col_gap            = p['max_col_gap'],
                    min_ground_pixels      = p['min_ground_pixels'],
                    max_gap                = p['max_gap'],
                    smooth_kernel          = p['smooth_kernel'],
                    obstacle_threshold     = p['obstacle_threshold'],
                    no_ground_baseline     = p['no_ground_baseline'],
                    plant_scale_factor     = PLANT_SCALE_FACTOR,
                    plant_blur_ksize       = plant_blur_ksize,
                    plant_min_width        = 20,
                    plant_min_pixels_per_col = 2,
                    plant_max_col_gap      = 5,
                )

                # ── unpack debug info ──────────────────────────────────────────────
                status               = debug["status"]
                actual_frac          = debug["green_frac"]
                boundary_rows        = debug["boundary_rows"]
                obstacle_regions_raw = debug["obstacle_regions_raw"]
                clean_mask           = debug["clean_mask"]
                plant_mask           = debug["plant_mask"]
                plant_regions_raw    = debug["plant_regions_raw"]

                # ── split detections by type for display ───────────────────────────
                if detections.shape[0] > 0:
                    obs_dets   = detections[detections[:, 2] == DET_OBSTACLE]
                    plant_dets = detections[detections[:, 2] == DET_PLANT]
                    print(f"obstacles: {obs_dets} ")
                else:
                    obs_dets   = np.empty((0, 3), dtype=np.int32)
                    plant_dets = np.empty((0, 3), dtype=np.int32)

                n_obs   = obs_dets.shape[0]
                n_plant = plant_dets.shape[0]
                print(f"{os.path.basename(image_path)} -> {status} ({actual_frac:.2%}) "
                      f"| obstacles: {n_obs}  plants: {n_plant}")

                # ── build display panels ───────────────────────────────────────────
                result = np.zeros_like(image_bgr)
                result[..., 0] = clean_mask
                result[..., 1] = clean_mask
                result[..., 2] = clean_mask

                plant_result = np.zeros_like(image_display)
                plant_result[plant_mask > 0] = [0, 255, 0]
                edges = sdd.get_blob_edge(plant_mask)
                plant_result[edges > 0] = [0, 0, 255]

                image_rot  = cv2.rotate(image_display, cv2.ROTATE_90_COUNTERCLOCKWISE)
                result_rot = cv2.rotate(result,        cv2.ROTATE_90_COUNTERCLOCKWISE)
                plant_rot  = cv2.rotate(plant_result,  cv2.ROTATE_90_COUNTERCLOCKWISE)
                combined_view = np.vstack((image_rot, result_rot, plant_rot))

                cv2.putText(combined_view,
                            f"{status} | {actual_frac:.2%}",
                            (p['status_x'], p['status_y']),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            p['status_font_scale'],
                            (0, 0, 255),
                            p['status_thickness'])

                h_rot = image_rot.shape[0]

                # ── draw obstacle boxes (red) ──────────────────────────────────────
                if status == "GROUND FOUND":
                    if ground_baseline is not None:
                        valid_r = np.where(ground_baseline < combined_view.shape[0])[0]
                        if len(valid_r) > 1:
                            pts = np.array(
                                [[int(r), int(ground_baseline[r]) + h_rot] for r in valid_r],
                                dtype=np.int32)
                            cv2.polylines(combined_view, [pts], False, (255, 0, 0), 1)

                    for (start_col, end_col, width) in obstacle_regions_raw:
                        y_top = int(min(boundary_rows[start_col:end_col + 1]))
                        y_bot = int(max(boundary_rows[start_col:end_col + 1])) + p['box_bottom_pad']

                        cv2.rectangle(combined_view,
                                      (start_col, y_top + h_rot),
                                      (end_col,   y_bot + h_rot),
                                      (0, 0, 255), p['box_thickness'])
                        cv2.putText(combined_view, f"w={width}",
                                    (start_col, y_top + h_rot - p['label_y_offset']),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    p['label_font_scale'],
                                    (0, 0, 255),
                                    p['label_thickness'])

                        cv2.rectangle(combined_view,
                                      (start_col, 0),
                                      (end_col,   h_rot - 1),
                                      (0, 0, 255), p['box_thickness'])
                        cv2.putText(combined_view, f"w={width}",
                                    (start_col, p['label_y_top']),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    p['label_font_scale'],
                                    (0, 0, 255),
                                    p['label_thickness'])

                # ── draw plant boxes (green) ───────────────────────────────────────
                for (start_col, end_col, width) in plant_regions_raw:
                    cv2.rectangle(combined_view,
                                  (start_col, 0),
                                  (end_col,   h_rot - 1),
                                  (0, 255, 0), p['box_thickness'])
                    cv2.putText(combined_view, f"p={width}",
                                (start_col, p['label_y_top']),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                p['label_font_scale'],
                                (0, 255, 0),
                                p['label_thickness'])

                cv2.imshow(WINDOW, combined_view)
                
                # Processing complete. Wait for user to trigger the next image.
                needs_processing = False 

            # ── Handle Input / Delay ───────────────────────────────────────────────
            # Using FRAME_DELAY instead of 0 prevents OpenCV from blocking KeyboardInterrupt
            key = cv2.waitKey(FRAME_DELAY) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d'):      # next image
                idx = (idx + 1) % len(image_paths)
                needs_processing = True
            elif key == ord('a'):      # previous image
                idx = (idx - 1) % len(image_paths)
                needs_processing = True

        except KeyboardInterrupt:
            print("\nProcess interrupted by user (Ctrl+C). Exiting...")
            break

    cv2.destroyAllWindows()