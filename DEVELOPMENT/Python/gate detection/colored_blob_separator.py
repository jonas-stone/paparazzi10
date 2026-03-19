import numpy as np
import cv2

import solidity_detection as sdd

def isolate_ground_blob(binary_img: np.ndarray, blob_area_threshold=1000):
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
    remaining_blob_indices = []
    for blob_label in indices[1:]:

        blob_area       = values[blob_label, 4]
        blob_width      = values[blob_label, 3] # width in an absolute sense, which means
        single_blob_img = labeled_img.copy()        

        single_blob_img[single_blob_img != blob_label] = 0

        # filter out blobs smaller than blob_area_threshold pixels
        if blob_area < blob_area_threshold:
            labeled_img[labeled_img == blob_label] = 0
            continue
        
        # filter out "unsmooth" blobs
        elif not sdd.is_smooth_blob(single_blob_img, blob_area, blob_width):
            labeled_img[labeled_img == blob_label] = 0
            continue

        remaining_blob_indices.append(blob_label)
            
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



