# python files
import solidity_detection as sdd
import colored_blob_separator as cds
import green_ground_detection as ggd
import ransac_fit as rsf
import obstacle_detection_ground as obs

# libraries
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import cv2
import os

cv2.destroyAllWindows()

folder_path = "TEAM-10-PROTOTYPING/downloads from drone/20260306-095826"
image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))[50:]

# same HSV bounds as before
lower_green = np.array([10, 50, 40])
upper_green = np.array([35, 255, 200])

oa_color_count_frac = 0.07

# ======= INITIALIZE RANSAC APPLICATION ======= #
#                                               #
norm2   = lambda x: np.sqrt(np.sum(x ** 2))     #
coeffs  = np.array([0, 0, 0])                   #
x_all   = np.array([0])                         #
y_fit   = np.array([0])                         #
alpha   = 0.5                                   #
#                                               #
# ======= INITIALIZE RANSAC APPLICATION ======= #

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
    mask, result, actual_frac, status = ggd.detect_green_ground_simple(
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

    # ================= RANSAC APPLICATION =================== #

    # isolate upper edge of ground
    contours = sdd.get_blob_upper_edge(mask) * 255
    span = mask.shape[0]
    contour_pixels = np.count_nonzero(contours)
    visible_fraction = contour_pixels / span
    print(visible_fraction)

    # get (x, y) coords of contour line
    (x, y)   = np.where(contours != 0)

    # use RANSAC
    new_coeffs, new_inlier_mask = rsf.ransac_parabola(x, y, min_inliers=30)

    # weighted temporal averaging of coefficients
    if new_coeffs is  None: continue

    new_coeffs = new_coeffs * alpha + coeffs * (1 - alpha) 
    coeffs = new_coeffs
    
    # get fitted parabola
    x_all = np.arange(mask.shape[0])
    y_fit = np.round(rsf.eval_parabola(coeffs, x_all))

    # ================= FINISH APPLICATION =================== #
    
    result[..., 0] = mask
    result[..., 1] = mask
    result[..., 2] = mask

    # # detect obstacles
    # if status == "GROUND FOUND":
        
    #     obstacle_map = obs.detect_obstacles_from_parabola(
    #         mask,
    #         coeffs,
    #         gap_threshold=50,
    #         plant_threshold=50,
    #         min_obstacle_width=15
    #     )

    #     if (obstacle_map != 0).any(): 
    #         print("OBSTACLE FOUND!")

    # plot RANSAC parabola inside image -> ONLY FOR RANSAC
    contours = rsf.draw_parabola_on_image(result, coeffs, x_all, y_fit)

    print(f"{os.path.basename(image_path)} -> {status} ({actual_frac:.2%})")

    # Rotate images 90° CCW and stack vertically
    image_rot       = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result_rot      = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    combined_view   = np.vstack((image_rot, result_rot))

    # Annotate
    cv2.putText(combined_view,
                f"{status} | {actual_frac:.2%}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)
    
    cv2.imshow('thing', combined_view)
    
    key = cv2.waitKey(100)  # short delay; press 'q' to quit
    if key == ord('q'):
        break
