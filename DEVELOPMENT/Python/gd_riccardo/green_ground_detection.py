import cv2
import numpy as np

def find_ground_boundary(mask_rotated):

    """ 
    The image has to be rotated for this to work. 
    The ground has to be on the right side. 
    """
    
    image_height = mask_rotated.shape[1]
    boundary_rows = []
    for row in mask_rotated:

        # get the index of the first nonzero element of this row
        nonzero_indices = np.where(row > 0)
        nonzero = nonzero_indices[0] - 1
        if nonzero.size == 0:
            nonzero = np.asarray([image_height])
        boundary_rows.append(int(nonzero[0]))

    return np.asarray(boundary_rows)


def detect_green_ground_simple(image_bgr, lower_green, upper_green, threshold, median_ksize=3):
    """
    Very simple green detector with a single median blur to remove salt-and-pepper noise.
    - median_ksize should be odd and >=3 (3 is the fastest / mildest).
    Returns: mask, result, green_fraction, status
    """
    # HSV mask
    width = image_bgr.shape[1]
    # image_bgr[:, width * 3//5:] = 0 # set upper portion to zero (most likely not ground)
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
    import obstacle_detection_ground as odg

    # libraries
    import matplotlib.pyplot as plt
    from glob import glob
    import os

    cv2.destroyAllWindows()

    folder_path = "TEAM-10-PROTOTYPING/downloads from drone/20260306-095826"
    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))[103:]

    # same HSV bounds as before
    lower_green = np.array([10, 50, 40])
    upper_green = np.array([35, 255, 200])

    oa_color_count_frac = 0.07

    # state initialization
    ground_baseline = None
    image_height = np.asarray(cv2.imread(image_paths[0])).shape[1]

    i = 0
    for image_path in image_paths:

        print(i := i + 1)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Skipping: {image_path}")
            continue

        # reduce quality of image
        factor = 1
        image_bgr = cv2.resize(image_bgr, (0,0), fx=factor, fy=factor)

        # Use the simple detector with a 3x3 median blur (fast, mild)
        mask, result, actual_frac, status = detect_green_ground_simple(
            image_bgr,
            lower_green,
            upper_green,
            oa_color_count_frac,
            median_ksize=5
        )

        # set detected pixels to white.
        result[result > 0] = 255

        # isolate the ground white blob
        mask = cds.isolate_ground_blob(binary_img=mask)
        mask = cds.fill_holes(mask)
        
        result[..., 0] = mask
        result[..., 1] = mask
        result[..., 2] = mask

        print(f"{os.path.basename(image_path)} -> {status} ({actual_frac:.2%})")

        # try find the obstacle
        if status == "GROUND FOUND":
            mask_flipped = mask[:, ::-1]
            # plt.imshow(mask_flipped)
            boundary_rows = find_ground_boundary(mask_rotated=mask_flipped)
            obstacle_cols, ground_baseline = odg.detect_obstacles_from_ground(boundary_rows, image_height, ground_baseline)

        # Rotate images 90° CCW and stack vertically
        image_rot = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result_rot = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined_view = np.vstack((image_rot, result_rot))

        # Annotate
        cv2.putText(combined_view,
                    f"{status} | {actual_frac:.2%}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)
        
        # put a red rot where the obstacle columns are
        if status == "GROUND FOUND":
            for col in obstacle_cols: # type: ignore
                cv2.drawMarker(combined_view, (col, 350), (0,0,255), 1, 5)

        cv2.namedWindow("thing")
        cv2.resizeWindow("thing", 600, 600)
        cv2.imshow('thing', combined_view)
        
        key = cv2.waitKey(100)  # short delay; press 'q' to quit
        if key == ord('q'):
            break
