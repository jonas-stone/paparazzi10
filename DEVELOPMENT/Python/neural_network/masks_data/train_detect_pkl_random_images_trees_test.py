import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import glob
import random

# ── Load the saved model — instant, no retraining ────────────────────────────
clf = joblib.load('../tree_detector.pkl')
print('Model loaded.')

# ── Folder of NEW images to test on ──────────────────────────────────────────
NEW_IMAGES_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'

# ── Options ───────────────────────────────────────────────────────────────────
NUM_IMAGES = 16  # how many random images to pick, must be a perfect square (4, 9, 16, 25...)
SAVE_FIGURE = True  # set to False to just display without saving
SAVE_PATH = '../tree_filter_detection_results.png'

# ── Edge filter options ───────────────────────────────────────────────────────
EDGE_FILTER_ON         = True
MIN_BLOB_AREA          = 30
SMALL_BLOB_THRESHOLD   = 150
EDGE_DENSITY_THRESHOLD = 0.04
CANNY_LOW              = 50
CANNY_HIGH             = 150
CANNY_BLUR             = 5      # kernel size — must be odd number (3, 5, 7, 9...)
CANNY_SIGMA            = 0.001   # blur strength — higher = more blur = fewer edges detected
                                 # 0 = OpenCV auto-calculates from kernel size
                                 # your original script used 0.01 for real images
                                 # try 0 for maximum edges, 2.0 for heavily smoothed


# ── Same feature extractor as training — do not touch this ───────────────────
def extract_features(yuv, hsv, lab, y, x):
    p_yuv = yuv[y, x]
    p_hsv = hsv[y, x]
    p_lab = lab[y, x]
    patch = yuv[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3)
    mean = patch.mean(axis=0)
    std = patch.std(axis=0)
    grad = np.abs(yuv[y, x + 1].astype(int) - yuv[y, x - 1].astype(int))
    return [
        int(p_yuv[0]), int(p_yuv[1]), int(p_yuv[2]),
        int(p_hsv[0]), int(p_hsv[1]), int(p_hsv[2]),
        int(p_lab[0]), int(p_lab[1]), int(p_lab[2]),
        *mean.tolist(), *std.tolist(), *grad.tolist()
    ]


# ── Edge density filter — reuses your Canny setup from edge_detection script ─
def filter_by_edge_density(raw_mask, img):
    """
    Filters classifier blobs by internal edge density.
    Leafy/textured objects (trees, plants) have high internal edge density.
    Flat surfaces (floor, walls, panels) have low internal edge density.
    Uses the same Canny parameters as your existing edge detection script.
    """

    # Step 1 — greyscale + blur, matching your detect_edges() function exactly
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (CANNY_BLUR, CANNY_BLUR), CANNY_SIGMA)  # same blur as your script
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)  # same thresholds as your script

    # Step 2 — morphological cleanup on raw classifier mask
    # Open removes isolated single-pixel noise before contour analysis
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Step 3 — find individual blobs in cleaned mask
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4 — check each blob individually
    output_mask = np.zeros_like(raw_mask, dtype=np.uint8)  # starts empty, we add passing blobs

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Reject blobs that are too small — pure noise
        if area < MIN_BLOB_AREA:
            continue

        # Build a filled mask for just this blob
        blob_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)  # -1 thickness = filled

        # Count edge pixels that fall inside this blob
        edges_inside = cv2.bitwise_and(edges, blob_mask)
        edge_pixel_count = cv2.countNonZero(edges_inside)
        blob_pixel_count = cv2.countNonZero(blob_mask)

        # Edge density = fraction of blob pixels that are edges
        edge_density = edge_pixel_count / blob_pixel_count if blob_pixel_count > 0 else 0

        # Print per-blob info — useful for tuning the threshold
        print(f'  blob area={int(area):5d}px  edge_density={edge_density:.3f}', end='  →  ')

        if edge_density >= EDGE_DENSITY_THRESHOLD:
            cv2.drawContours(output_mask, [cnt], -1, 255, -1)  # keep this blob
            print('KEPT ✅')
        else:
            print('rejected ❌')

    return output_mask


# ── Pick NUM_IMAGES random images from the folder ────────────────────────────
all_images = glob.glob(NEW_IMAGES_FOLDER + '\\*.jpg')
all_images = [i for i in all_images if '_mask' not in i]

if len(all_images) < NUM_IMAGES:
    print(f'Warning: only {len(all_images)} images found, using all of them.')
    selected = all_images
else:
    selected = random.sample(all_images, NUM_IMAGES)

print(f'Selected {len(selected)} random images.')

# ── Build the figure grid ─────────────────────────────────────────────────────
# Each image gets 3 columns (original | raw detection | filtered), rows = NUM_IMAGES / sqrt
cols_per_image = 3 if EDGE_FILTER_ON else 2  # 3 panels when filter is on, 2 when off
grid_cols = int(np.sqrt(NUM_IMAGES)) * cols_per_image
grid_rows = int(np.sqrt(NUM_IMAGES))

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
fig.suptitle(
    'Tree Detection — Original | Raw Classifier | After Edge Filter' if EDGE_FILTER_ON
    else 'Tree Detection — Original | Raw Classifier',
    fontsize=12
)

# ── Run detection on each selected image and fill the grid ───────────────────
for idx, f in enumerate(selected):
    img = cv2.imread(f)
    h, w, d = img.shape
    print(f'\n[{idx + 1}/{len(selected)}] {f}')

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    pixels = np.array([
        extract_features(yuv, hsv, lab, y, x)
        for y in range(4, h - 2)
        for x in range(2, w - 3)
    ])

    y_pred_flat = clf.predict(pixels)

    msk_cropped = y_pred_flat.reshape(h - 6, w - 5)
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    raw_mask[4:h - 2, 2:w - 3] = msk_cropped

    # ── Apply edge filter if switched on ─────────────────────────────────────
    if EDGE_FILTER_ON:
        filtered_mask = filter_by_edge_density(raw_mask, img)

    # ── Build display images ──────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # original

    raw_overlay = img.copy()
    raw_overlay[:, :, 1] = raw_mask
    raw_rgb = cv2.cvtColor(raw_overlay, cv2.COLOR_BGR2RGB)  # raw classifier

    # Work out grid position for this image
    row = idx // int(np.sqrt(NUM_IMAGES))
    col_pair = idx % int(np.sqrt(NUM_IMAGES))
    col_orig = col_pair * cols_per_image  # original column
    col_raw = col_pair * cols_per_image + 1  # raw detection column
    col_filt = col_pair * cols_per_image + 2  # filtered column (only when EDGE_FILTER_ON)

    axes[row, col_orig].imshow(img_rgb)
    axes[row, col_orig].set_title(f'#{idx + 1} Original', fontsize=7)
    axes[row, col_orig].axis('off')

    axes[row, col_raw].imshow(raw_rgb)
    axes[row, col_raw].set_title(f'#{idx + 1} Raw', fontsize=7)
    axes[row, col_raw].axis('off')

    if EDGE_FILTER_ON:
        filt_overlay = img.copy()
        filt_overlay[:, :, 1] = filtered_mask
        filt_rgb = cv2.cvtColor(filt_overlay, cv2.COLOR_BGR2RGB)

        axes[row, col_filt].imshow(filt_rgb)
        axes[row, col_filt].set_title(f'#{idx + 1} Edge Filtered', fontsize=7)
        axes[row, col_filt].axis('off')

plt.tight_layout()

# ── Save and/or show ──────────────────────────────────────────────────────────
if SAVE_FIGURE:
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved to {SAVE_PATH}')

plt.show()