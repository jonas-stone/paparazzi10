import numpy as np
import cv2


def isolate_ground_blob(binary_img: np.ndarray):
    """
    Having a mask of black and white pixels, determine all the 
    contiguous blobs of white pixels and give each of them a label.
    """

    result = cv2.connectedComponentsWithStats(binary_img.astype(np.uint8), connectivity=8)
    totalLabels, labeled_img, values, _ = result

    # values: rows = 1, 2, 3... for each label
    # row:    [pos left, pos top, box width, box height, area]

    indices = np.asarray(range(totalLabels))
    values = np.block([
        values, indices.reshape([-1, 1])
    ])

    # don't consider the background (index = 0)
    for blob_label in indices[1:]:

        blob_area       = values[blob_label, 4]
        blob_width      = values[blob_label, 3] # width in an absolute sense
        single_blob_img = labeled_img.copy()        

        single_blob_img[single_blob_img != blob_label] = 0

        # filter out blobs smaller than 1000 pixels
        if blob_area < 1:
            labeled_img[labeled_img == blob_label] = 0
            continue
        
        # # filter out "unsmooth" blobs
        # elif not sdd.is_smooth_blob(single_blob_img, blob_area, blob_width):
        #     labeled_img[labeled_img == blob_label] = 0
        #     continue
            
    # set largest blobs to white (max value)
    labeled_img[labeled_img != 0] = 255

    return labeled_img


def fill_holes(binary_img: np.ndarray):
    """
    Fill spots of zeros inside the binary white ground blob
    """
    # Find all contours (external + holes)
    binary_img = binary_img.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    filled = binary_img.copy()
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            # h[3] != -1 means it has a parent → it's a hole (inner contour)
            if h[3] != -1:
                cv2.drawContours(filled, contours, i, 255, thickness=cv2.FILLED)

    return np.asarray(filled)


if __name__ == "__main__":


    import os
    import matplotlib.pyplot as plt
    from glob import glob
    # from scipy.ndimage import binary_fill_holes

    import green_ground_detection as gnd
    import solidity_detection as sdd

    # ============ CODE FROM MAIN IN green_ground_detection.py =============

    cv2.destroyAllWindows()

    folder_path = "TEAM-10-PROTOTYPING/downloads from drone/20260306-095826"
    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))[106:]

    # same HSV bounds as before
    lower_green = np.array([15, 50, 40])
    upper_green = np.array([35, 255, 200])

    oa_color_count_frac = 0.1

    # state initialization
    ground_baseline = None
    image_height = np.asarray(cv2.imread(image_paths[0])).shape[1]

    for image_path in image_paths:

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Skipping: {image_path}")
            continue

        # Use the simple detector with a 3x3 median blur (fast, mild)
        mask, result, actual_frac, status = gnd.detect_green_ground_simple(
            image_bgr,
            lower_green,
            upper_green,
            oa_color_count_frac,
            median_ksize=5
        )

        # set detected pixels to white.
        result[result > 0] = 255

        # try majestic function
        mask = isolate_ground_blob(binary_img=mask)
        # mask = unite_ground_blob(ground_blob_binary_image=mask_largest_blob)
        plt.imshow(fill_holes(mask))
        plt.show()

