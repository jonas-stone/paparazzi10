import numpy as np
import cv2

def is_smooth_blob_old(single_blob_mask: np.ndarray,
                   area: float) -> bool:
    """
    Returns True if the blob in the mask is smooth (likely ground),
    False if it is spiky/irregular (likely plant).
    Uses only basic array operations — easy to port to C.
    
    Args:
        mask: binary image (uint8, 0 or 255) containing *ONE* blob
        area: blob area
    """
    solidity_thresh     = 0.8
    circularity_thresh  = 0.0

    # get a mask to identify the current blob (using the label)
    pixels = (single_blob_mask != 0)
    
    # pad image by 1 pixel
    padded = np.pad(pixels, 1, mode='constant', constant_values=0)

    perimeter = blob_perimeter(padded_binary_img=padded)

    # ys, xs = np.where(pixels)
    # points = np.stack([xs, ys], axis=1).astype(np.float32)
    # hull_area = convex_hull_area_fast(points)
    
    # # ---- SOLIDITY ----
    # if area > 0:
    #     solidity = area / hull_area
    # else:
    #     solidity = 0

    # --- CIRCULARITY ---
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else: 
        circularity = 0

    # check_solidity = solidity > solidity_thresh
    check_circularity = circularity > circularity_thresh
    return check_circularity # and check_solidity


def is_smooth_blob(single_blob_mask: np.ndarray,
                                        area: float,
                                        bounding_box_width: int) -> bool:
    """
    Returns True if the blob in the mask is smooth (likely ground),
    False if it is spiky/irregular (likely plant).
    Uses only basic array operations — easy to port to C.
    
    Args:
        mask: binary image (uint8, 0 or 255) containing *ONE* blob
        area: blob area
        bb_width: bounding box WIDTH (from roof to ground)
        bb_height: bounding box HEIGHT (from left to right)
    
    NOTE: 
        bb width and height are assumed in an absolute sense. 
        If the image is rotated, like the ones produced by the Bebop,
        the width and height will be swapped.
    """

    # get a mask to identify the current blob (using the label)
    pixels = (single_blob_mask != 0)
    
    # pad image by 1 pixel
    padded = np.pad(pixels, 1, mode='constant', constant_values=0)

    perimeter = blob_perimeter(padded_binary_img=padded)

    # compute AVERAGE HEIGHT (per-column) of blob + equivalent rectangle perimeter
    average_height = compute_average_blob_height(area, bounding_box_width)
    equivalent_rectangle_perimeter = 2 * (average_height + bounding_box_width)

    # compute fractal dimension of blob
    fractal_dim = compute_fractal_dimension(padded)

    threshold_perimeter     = 2
    threshold_fractal_dim   = 1.3
    if perimeter / equivalent_rectangle_perimeter < threshold_perimeter \
                                  or fractal_dim < threshold_fractal_dim:
        return True

    return False


def compute_average_blob_height(blob_area: float,
                                bounding_box_width: int):
    """
    Computes the average blob height, considering an image rotated with the groun on the left (-90° rotation).
    Therefore, the blob height corresponds to the column-wise span of the image.
    Moreover, this assumes the image contains ONE single blob.
    """

    # reduce the image size to fit the blob perfectly.
    avg_height = blob_area // bounding_box_width

    return avg_height


def blob_perimeter(padded_binary_img: np.ndarray):#
    """
    Compute perimeter of a binary blob of pixels using vectorized NumPy operations.
    This assumes ONE blob per image.
    """
               
    perimeter = np.sum(padded_binary_img[:,1:] != padded_binary_img[:,:-1]) + \
                np.sum(padded_binary_img[1:,:] != padded_binary_img[:-1,:])

    return perimeter


def convex_hull_area_fast(points: np.ndarray) -> float:
    """
    Convex hull area via Monotone Chain + Shoelace.
    Args:
        points: (N, 2) array of 2D points
    Returns:
        convex hull area
    """
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    pts = np.unique(pts, axis=0)
    n = len(pts)
    if n < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull_area(points: np.ndarray) -> float:
    """
    Computes convex hull area using Graham scan + shoelace formula.
    Pure numpy/python -> portable to C.
    """
    # Find pivot: lowest y, then leftmost x
    idx = np.lexsort((points[:, 0], points[:, 1]))
    points = points[idx]
    pivot = points[0]

    # Sort by polar angle from pivot
    deltas = points[1:] - pivot
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    dists  = deltas[:, 0]**2 + deltas[:, 1]**2
    order  = np.lexsort((dists, angles))
    sorted_pts = np.vstack([pivot, points[1:][order]])

    # Graham scan
    hull = []
    for p in sorted_pts:
        while len(hull) >= 2:
            # Cross product of (hull[-1]-hull[-2]) x (p-hull[-2])
            o = hull[-2]
            a = hull[-1]
            cross = (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0])
            if cross <= 0:  # clockwise or collinear → pop
                hull.pop()
            else:
                break
        hull.append(p)

    hull = np.array(hull)

    # Shoelace formula
    x, y = hull[:, 0], hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def get_blob_edge(padded_binary_img: np.ndarray):

    edge1 = (padded_binary_img[:,1:] != padded_binary_img[:,:-1])
    edge1 = np.hstack([edge1, np.zeros((edge1.shape[0], 1), dtype=edge1.dtype)])

    edge2 = (padded_binary_img[1:,:] != padded_binary_img[:-1,:])
    edge2 = np.vstack([edge2, np.zeros((1, edge2.shape[1]), dtype=edge2.dtype)])
    
    return edge1 | edge2


def compute_fractal_dimension(binary_img: np.ndarray):

    # get only the blob's edge pixels
    img = get_blob_edge((binary_img > 0).astype(np.uint8))

    # box sizes as powers of 2
    min_dim = min(img.shape)
    scales  = [2**i for i in range(1, int(np.log2(min_dim)))]

    eps   = []
    boxes = []

    for s in scales:
        # tile the image with boxes of size s×s, count non-empty ones
        trimmed_h = (img.shape[0] // s) * s
        trimmed_w = (img.shape[1] // s) * s
        trimmed   = img[:trimmed_h, :trimmed_w]

        # reshape into s×s blocks and check if any pixel is set
        blocks = trimmed.reshape(trimmed_h // s, s, trimmed_w // s, s)
        count  = np.any(blocks, axis=(1, 3)).sum()

        eps.append(s)
        boxes.append(count)

    eps_axis  = np.log(eps)
    box_axis  = np.log(boxes)

    coeff = np.polyfit(eps_axis, box_axis, 1)
    return -coeff[0]  # slope is negative, so negate to get positive dimension


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Initialize a 50x50 array of zeros
    blob_50x50 = np.zeros((50, 50), dtype=int)

    # Defined as (row, start_col, end_col)
    # Based on a 50x50 scale where the blob is centered/top-aligned
    rows_to_fill = [
    (6, 4, 34),                             # Top "overhang" row
    (7, 6, 34),
    (8, 5, 34),                             # Small indent
    (9, 6, 34),
    (10, 6, 34),
    (11, 6, 34),
    (12, 6, 34),
    (13, 6, 34),
    (14, 6, 34),
    (15, 6, 35),                            # Right-side bump
    (16, 6, 34),
    (17, 6, 34),
    (18, 6, 34),
    (19, 6, 34),
    (20, 6, 34),
    (21, 6, 34),
    (22, 6, 34),
    (23, 28, 34)                            # Bottom "leg" row
    ]

    # square
    # blob_50x50[15:35, 15:35] = 1

    # rectangle
    # blob_50x50[10:40, 15:35] = 1

    # line
    # blob_50x50[10:40, 15]    = 1

    for r, start, end in rows_to_fill:
        # end + 1 because slicing is exclusive
        blob_50x50[r, start:end+1] = 1

    plt.imshow(blob_50x50)
    plt.show()
    blob_array = np.pad(blob_50x50, 1, mode='constant', constant_values=0)

    hello = get_blob_edge(blob_array)
    plt.imshow(hello)
    plt.show()