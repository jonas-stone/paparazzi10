# python files
import solidity_detection as sdd
import colored_blob_separator as cds
import green_ground_detection as ggd
import ransac_fit as rsf
import obstacle_detection_ground as obs
import get_obstacle_info_Imp as imp

# libraries
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import cv2
import os

cv2.destroyAllWindows()

folder_path = "DEVELOPMENT/downloads from drone/20260313-100130"
image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))

oa_color_count_frac = 0.05

i = 0
for image_path in image_paths:

    print(i := i + 1)

    # sample 1 in 5 images
    # if i % 10 != 0: continue

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Skipping: {image_path}")
        continue

    # reduce quality of image
    factor = 0.2
    image_bgr = cv2.resize(image_bgr, (0,0), fx=factor, fy=factor)

    # Use the simple detector with a 3x3 median blur (fast, mild)
    mask, result, green_frac, status = imp.detect_green_ground_ml(image_bgr,
                                                         threshold=oa_color_count_frac,
                                                         median_ksize=5)

    # set detected pixels to white.
    result[result > 0] = 255

    # isolate the ground white blob
    mask = cds.isolate_ground_blob(binary_img=mask)
    mask = cds.fill_holes(mask)
    
    result[..., 0] = mask
    result[..., 1] = mask
    result[..., 2] = mask

    print(f"{os.path.basename(image_path)} -> {status} ({green_frac:.2%})")

    # Rotate images 90° CCW and stack vertically
    image_rotated       = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result_rotated      = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    combined_view       = np.vstack((image_rotated, result_rotated))

    black_and_white_path = f"training_b&w_13_march\\{image_path}.png"
    colored_path         = f"training_color_13_march\\{image_path}.png"

    result = cv2.imwrite(black_and_white_path, result_rotated)
    result = cv2.imwrite(colored_path, image_rotated)
    
    # 1. Create a named window first
    # cv2.WINDOW_NORMAL allows the window to be resized
    cv2.namedWindow("Resized Window", cv2.WINDOW_NORMAL)

    # 2. Set the specific width and height you want
    cv2.resizeWindow("Resized Window", 1400, 800)

    # 3. Show the image using that specific window name
    cv2.imshow("Resized Window", combined_view)

