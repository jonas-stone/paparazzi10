import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import glob
import random

# ── Load models — instant, no retraining ─────────────────────────────────────
tree_clf   = joblib.load('../tree_detector.pkl')
ground_clf = joblib.load('../ground_detector.pkl')
print('Models loaded.')

# ── Folder of NEW images to test on ──────────────────────────────────────────
NEW_IMAGES_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'

# ── Options ───────────────────────────────────────────────────────────────────
NUM_IMAGES  = 3 ** 2    # must be a perfect square (4, 9, 16, 25...)
SAVE_FIGURE = True
SAVE_PATH   = '../tree_filter_detection_extra_results.png'

# ── Edge filter options ───────────────────────────────────────────────────────
EDGE_FILTER_ON         = True
MIN_BLOB_AREA          = 35      # minimum blob size — raised from 50 to kill small false positives
SMALL_BLOB_THRESHOLD   = 150     # blobs smaller than this skip edge density check
EDGE_DENSITY_THRESHOLD = 0.04    # fraction of blob pixels that must be edges
CANNY_LOW              = 50      # Canny lower threshold, tweakable
CANNY_HIGH             = 150     # Canny upper threshold, tweakable
CANNY_BLUR             = 5       # kernel size — must be odd (3, 5, 7, 9...), tweakable
CANNY_SIGMA            = 0.000000000000000001   # blur strength — 0=auto, higher=smoother=fewer edges, tweakable

# ── Bounding box options ──────────────────────────────────────────────────────
# BBOX_ON          = True
# SHOW_GROUND      = True    # switch: set False to hide ground mask panel
# CLUSTER_DILATION = 20      # pixels to expand blobs before merging — reduced from 25
#                            # higher = more aggressive clustering
#                            # lower = only merges very close blobs
# PAD_W_FRAC       = 0.15   # horizontal padding as fraction of box width — reduced from 0.25
# PAD_TOP_FRAC     = 0.05   # top padding as fraction of box height — reduced from 0.20
# PAD_BOT_FRAC     = 0.05   # bottom padding as fraction of box height — halved from 0.40
# MAX_ASPECT_RATIO = 4.0    # max height/width ratio — rejects tall thin false positive boxes
# MIN_ASPECT_RATIO = 0.2    # min height/width ratio — rejects very wide flat boxes
# ── Bounding box options ──────────────────────────────────────────────────────
BBOX_ON              = True
SHOW_GROUND          = True
CLUSTER_DILATION     = 20
PAD_W_FRAC           = 0.15
PAD_TOP_FRAC         = 0.15
PAD_BOT_FRAC         = 0.15
MAX_ASPECT_RATIO     = 4.0
MIN_ASPECT_RATIO     = 0.2
TOO_CLOSE_THRESHOLD  = 0.25   # if detected pixels cover more than this fraction of
                               # the image, declare TOO CLOSE — tweakable
                               # 0.25 = 25% of image covered = tree fills the scene


# ── Feature extractor — do not touch this ────────────────────────────────────
# Must remain identical to what was used during training —
# any change here will silently break predictions
def extract_features(yuv, hsv, lab, y, x):
    p_yuv = yuv[y, x]
    p_hsv = hsv[y, x]
    p_lab = lab[y, x]
    patch = yuv[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3)
    mean  = patch.mean(axis=0)
    std   = patch.std(axis=0)
    grad  = np.abs(yuv[y, x + 1].astype(int) - yuv[y, x - 1].astype(int))
    return [
        int(p_yuv[0]), int(p_yuv[1]), int(p_yuv[2]),
        int(p_hsv[0]), int(p_hsv[1]), int(p_hsv[2]),
        int(p_lab[0]), int(p_lab[1]), int(p_lab[2]),
        *mean.tolist(), *std.tolist(), *grad.tolist()
    ]


# ── Run classifier on full image ──────────────────────────────────────────────
# Shared by both tree and ground detectors — pass the relevant clf
def run_classifier(clf, img):
    h, w, d = img.shape
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = np.array([
        extract_features(yuv, hsv, lab, y, x)
        for y in range(4, h - 2)
        for x in range(2, w - 3)
    ])
    pred_flat   = clf.predict(pixels)
    msk_cropped = pred_flat.reshape(h - 6, w - 5)
    msk         = np.zeros((h, w), dtype=np.uint8)
    msk[4:h - 2, 2:w - 3] = msk_cropped
    return msk


# ── Edge density filter ───────────────────────────────────────────────────────
# Keeps blobs with leafy/textured internal edges (trees)
# Rejects flat surface false positives (floor, walls, panels)
def filter_by_edge_density(raw_mask, img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (CANNY_BLUR, CANNY_BLUR), CANNY_SIGMA)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Morphological cleanup — removes isolated noise pixels before contour analysis
    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN,  kernel)
    cleaned = cv2.morphologyEx(cleaned,                   cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_mask = np.zeros_like(raw_mask, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_BLOB_AREA:
            continue  # too small — pure noise

        # Small but non-trivial blobs skip edge check — too few pixels to measure reliably
        if area < SMALL_BLOB_THRESHOLD:
            cv2.drawContours(output_mask, [cnt], -1, 255, -1)
            print(f'  blob area={int(area):5d}px  → KEPT (small, skipping edge check) ✅')
            continue

        # Larger blobs — check internal edge density
        blob_mask        = np.zeros_like(raw_mask, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        edges_inside     = cv2.bitwise_and(edges, blob_mask)
        edge_pixel_count = cv2.countNonZero(edges_inside)
        blob_pixel_count = cv2.countNonZero(blob_mask)
        edge_density     = edge_pixel_count / blob_pixel_count if blob_pixel_count > 0 else 0

        print(f'  blob area={int(area):5d}px  edge_density={edge_density:.3f}', end='  →  ')

        if edge_density >= EDGE_DENSITY_THRESHOLD:
            cv2.drawContours(output_mask, [cnt], -1, 255, -1)
            print('KEPT ✅')
        else:
            print('rejected ❌')

    return output_mask


# ── Bounding box generator with clustering ────────────────────────────────────
# Dilates tree mask to merge nearby blobs into clusters, then fits one
# padded bounding box per cluster. Aspect ratio filter kills thin false positives.
def generate_tree_bounding_boxes(tree_mask, img_debug=None):
    h, w  = tree_mask.shape
    debug = img_debug.copy() if img_debug is not None else None

    # ── Step 0 — TOO CLOSE check ──────────────────────────────────────────────
    # If detected pixels cover a large fraction of the image the drone is
    # so close to a tree that the whole scene is the tree.
    # Skip normal clustering and return one full-image warning box instead.
    total_pixels    = h * w
    detected_pixels = cv2.countNonZero(tree_mask)
    coverage        = detected_pixels / total_pixels

    if coverage > TOO_CLOSE_THRESHOLD:
        print(f'  ⚠️  TOO CLOSE — {coverage:.1%} of image is tree detections')
        warning_box = {
            'x1'      : 10,
            'y1'      : 10,
            'x2'      : w - 10,
            'y2'      : h - 10,
            'too_close': True
        }
        if debug is not None:
            # Draw red border around entire image
            cv2.rectangle(debug, (10, 10), (w-10, h-10), (0, 0, 255), 3)
            cv2.putText(debug, f'TOO CLOSE ({coverage:.0%})',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return [warning_box], debug

    # ── Step 1 — cluster nearby blobs by dilation ─────────────────────────────
    cluster_kernel = np.ones((CLUSTER_DILATION * 2 + 1,
                              CLUSTER_DILATION * 2 + 1), np.uint8)
    dilated_mask   = cv2.dilate(tree_mask, cluster_kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_BLOB_AREA:
            continue

        # Step 2 — get bounding rect from dilated contour
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        x     = max(0, x)
        y     = max(0, y)
        w_box = min(w - x, w_box)
        h_box = min(h - y, h_box)

        # Step 3 — verify actual tree pixels using original undilated mask
        region        = tree_mask[y:y + h_box, x:x + w_box]
        actual_pixels = cv2.countNonZero(region)
        if actual_pixels < MIN_BLOB_AREA:
            continue

        # Step 4 — apply padding
        pad_w   = int(PAD_W_FRAC   * w_box)
        pad_top = int(PAD_TOP_FRAC * h_box)
        pad_bot = int(PAD_BOT_FRAC * h_box)

        x1 = max(0, x - pad_w)
        x2 = min(w, x + w_box + pad_w)
        y1 = max(0, y - pad_top)
        y2 = min(h, y + h_box + pad_bot)

        # Step 5 — aspect ratio filter
        box_h  = y2 - y1
        box_w  = x2 - x1
        aspect = box_h / max(1, box_w)

        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            print(f'  Cluster at ({x},{y}) aspect={aspect:.2f} → rejected (bad aspect) ❌')
            continue

        print(f'  Cluster at ({x},{y}) size=({w_box}x{h_box}) actual={actual_pixels}px aspect={aspect:.2f} → KEPT ✅')

        boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'too_close': False})

        if debug is not None:
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return boxes, debug


# ── Draw bounding boxes on image ──────────────────────────────────────────────
# Green box + TREE label = normal detection at safe distance
# Red box + TOO CLOSE label = drone is too close, tree fills scene
def draw_boxes(img, boxes):
    visual = img.copy()
    for box in boxes:
        if box.get('too_close', False):
            # Red border — too close warning
            colour = (0, 0, 255)
            label  = 'TOO CLOSE'
        else:
            # Green box — normal tree detection
            colour = (0, 255, 0)
            label  = 'TREE'
        cv2.rectangle(visual, (box['x1'], box['y1']), (box['x2'], box['y2']), colour, 2)
        cv2.putText(visual, label,
                    (box['x1'], max(0, box['y1'] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    return visual


# # ── Draw bounding boxes on image ──────────────────────────────────────────────
# def draw_boxes(img, boxes):
#     visual = img.copy()
#     for box in boxes:
#         cv2.rectangle(visual, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
#         cv2.putText(visual, 'TREE',
#                     (box['x1'], max(0, box['y1'] - 5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#     return visual


# ── Pick NUM_IMAGES random images ─────────────────────────────────────────────
all_images = glob.glob(NEW_IMAGES_FOLDER + '\\*.jpg')
all_images = [i for i in all_images if '_mask' not in i]

if len(all_images) < NUM_IMAGES:
    print(f'Warning: only {len(all_images)} images found, using all of them.')
    selected = all_images
else:
    selected = random.sample(all_images, NUM_IMAGES)

print(f'Selected {len(selected)} random images.')

# ── Build figure grid ─────────────────────────────────────────────────────────
# Panel order: original | raw | edge filtered | ground | bounding boxes
# Each panel is independently controlled by its switch
num_panels = 2                           # always: original + raw
if EDGE_FILTER_ON: num_panels += 1       # + edge filtered
if SHOW_GROUND:    num_panels += 1       # + ground mask
if BBOX_ON:        num_panels += 1       # + bounding boxes

cols_per_image = num_panels
grid_cols      = int(np.sqrt(NUM_IMAGES)) * cols_per_image
grid_rows      = int(np.sqrt(NUM_IMAGES))

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
fig.suptitle('Tree Detection — Original | Raw | Edge Filtered | Ground | Bounding Box', fontsize=12)

# ── Run detection on each image ───────────────────────────────────────────────
for idx, f in enumerate(selected):
    img     = cv2.imread(f)
    h, w, d = img.shape
    print(f'\n[{idx + 1}/{len(selected)}] {f}')

    # Run both classifiers
    raw_mask    = run_classifier(tree_clf,   img)
    ground_mask = run_classifier(ground_clf, img) if (BBOX_ON or SHOW_GROUND) else None

    # Apply edge density filter
    filtered_mask = filter_by_edge_density(raw_mask, img) if EDGE_FILTER_ON else raw_mask

    # Generate clustered bounding boxes
    boxes, _ = generate_tree_bounding_boxes(filtered_mask, img_debug=img) if BBOX_ON else ([], None)

    # ── Build display panels ──────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raw_overlay        = img.copy()
    raw_overlay[:,:,1] = raw_mask
    raw_rgb            = cv2.cvtColor(raw_overlay, cv2.COLOR_BGR2RGB)

    row      = idx // int(np.sqrt(NUM_IMAGES))
    col_pair = idx %  int(np.sqrt(NUM_IMAGES))
    col_base = col_pair * cols_per_image

    # Panel 0 — original image
    axes[row, col_base].imshow(img_rgb)
    axes[row, col_base].set_title(f'#{idx+1} Original', fontsize=7)
    axes[row, col_base].axis('off')

    # Panel 1 — raw tree classifier output
    axes[row, col_base + 1].imshow(raw_rgb)
    axes[row, col_base + 1].set_title(f'#{idx+1} Raw', fontsize=7)
    axes[row, col_base + 1].axis('off')

    panel = 2  # next available panel slot

    # Panel 2 — edge filtered (if on)
    if EDGE_FILTER_ON:
        filt_overlay        = img.copy()
        filt_overlay[:,:,1] = filtered_mask
        filt_rgb            = cv2.cvtColor(filt_overlay, cv2.COLOR_BGR2RGB)
        axes[row, col_base + panel].imshow(filt_rgb)
        axes[row, col_base + panel].set_title(f'#{idx+1} Edge Filtered', fontsize=7)
        axes[row, col_base + panel].axis('off')
        panel += 1

    # Panel 3 — ground mask (if on)
    if SHOW_GROUND:
        ground_overlay        = img.copy()
        ground_overlay[:,:,1] = ground_mask
        ground_rgb            = cv2.cvtColor(ground_overlay, cv2.COLOR_BGR2RGB)
        axes[row, col_base + panel].imshow(ground_rgb)
        axes[row, col_base + panel].set_title(f'#{idx+1} Ground', fontsize=7)
        axes[row, col_base + panel].axis('off')
        panel += 1

    # Panel 4 — bounding boxes overlaid on edge filtered tree mask (if on)
    if BBOX_ON:
        bbox_img        = img.copy()
        bbox_img[:,:,1] = filtered_mask
        bbox_rgb        = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
        bbox_with_boxes = draw_boxes(bbox_rgb, boxes)
        axes[row, col_base + panel].imshow(bbox_with_boxes)
        axes[row, col_base + panel].set_title(f'#{idx+1} Boxes', fontsize=7)
        axes[row, col_base + panel].axis('off')

plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved to {SAVE_PATH}')

plt.show()



# import cv2
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import glob
# import random
#
# # ── Load the saved model — instant, no retraining ────────────────────────────
# clf = joblib.load('../tree_detector.pkl')
# print('Model loaded.')
#
# # ── Folder of NEW images to test on ──────────────────────────────────────────
# NEW_IMAGES_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
#
# # ── Options ───────────────────────────────────────────────────────────────────
# NUM_IMAGES = 16  # how many random images to pick, must be a perfect square (4, 9, 16, 25...)
# SAVE_FIGURE = True  # set to False to just display without saving
# SAVE_PATH = '../tree_filter_detection_results.png'
#
# # ── Edge filter options ───────────────────────────────────────────────────────
# EDGE_FILTER_ON         = True
# MIN_BLOB_AREA          = 30
# SMALL_BLOB_THRESHOLD   = 150
# EDGE_DENSITY_THRESHOLD = 0.04
# CANNY_LOW              = 50
# CANNY_HIGH             = 150
# CANNY_BLUR             = 5      # kernel size — must be odd number (3, 5, 7, 9...)
# CANNY_SIGMA            = 0.001   # blur strength — higher = more blur = fewer edges detected
#                                  # 0 = OpenCV auto-calculates from kernel size
#                                  # your original script used 0.01 for real images
#                                  # try 0 for maximum edges, 2.0 for heavily smoothed
#
#
# # ── Same feature extractor as training — do not touch this ───────────────────
# def extract_features(yuv, hsv, lab, y, x):
#     p_yuv = yuv[y, x]
#     p_hsv = hsv[y, x]
#     p_lab = lab[y, x]
#     patch = yuv[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3)
#     mean = patch.mean(axis=0)
#     std = patch.std(axis=0)
#     grad = np.abs(yuv[y, x + 1].astype(int) - yuv[y, x - 1].astype(int))
#     return [
#         int(p_yuv[0]), int(p_yuv[1]), int(p_yuv[2]),
#         int(p_hsv[0]), int(p_hsv[1]), int(p_hsv[2]),
#         int(p_lab[0]), int(p_lab[1]), int(p_lab[2]),
#         *mean.tolist(), *std.tolist(), *grad.tolist()
#     ]
#
#
# # ── Edge density filter — reuses your Canny setup from edge_detection script ─
# def filter_by_edge_density(raw_mask, img):
#     """
#     Filters classifier blobs by internal edge density.
#     Leafy/textured objects (trees, plants) have high internal edge density.
#     Flat surfaces (floor, walls, panels) have low internal edge density.
#     Uses the same Canny parameters as your existing edge detection script.
#     """
#
#     # Step 1 — greyscale + blur, matching your detect_edges() function exactly
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (CANNY_BLUR, CANNY_BLUR), CANNY_SIGMA)  # same blur as your script
#     edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)  # same thresholds as your script
#
#     # Step 2 — morphological cleanup on raw classifier mask
#     # Open removes isolated single-pixel noise before contour analysis
#     kernel = np.ones((3, 3), np.uint8)
#     cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
#     cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
#
#     # Step 3 — find individual blobs in cleaned mask
#     contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Step 4 — check each blob individually
#     output_mask = np.zeros_like(raw_mask, dtype=np.uint8)  # starts empty, we add passing blobs
#
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#
#         # Reject blobs that are too small — pure noise
#         if area < MIN_BLOB_AREA:
#             continue
#
#         # Build a filled mask for just this blob
#         blob_mask = np.zeros_like(raw_mask, dtype=np.uint8)
#         cv2.drawContours(blob_mask, [cnt], -1, 255, -1)  # -1 thickness = filled
#
#         # Count edge pixels that fall inside this blob
#         edges_inside = cv2.bitwise_and(edges, blob_mask)
#         edge_pixel_count = cv2.countNonZero(edges_inside)
#         blob_pixel_count = cv2.countNonZero(blob_mask)
#
#         # Edge density = fraction of blob pixels that are edges
#         edge_density = edge_pixel_count / blob_pixel_count if blob_pixel_count > 0 else 0
#
#         # Print per-blob info — useful for tuning the threshold
#         print(f'  blob area={int(area):5d}px  edge_density={edge_density:.3f}', end='  →  ')
#
#         if edge_density >= EDGE_DENSITY_THRESHOLD:
#             cv2.drawContours(output_mask, [cnt], -1, 255, -1)  # keep this blob
#             print('KEPT ✅')
#         else:
#             print('rejected ❌')
#
#     return output_mask
#
#
# # ── Pick NUM_IMAGES random images from the folder ────────────────────────────
# all_images = glob.glob(NEW_IMAGES_FOLDER + '\\*.jpg')
# all_images = [i for i in all_images if '_mask' not in i]
#
# if len(all_images) < NUM_IMAGES:
#     print(f'Warning: only {len(all_images)} images found, using all of them.')
#     selected = all_images
# else:
#     selected = random.sample(all_images, NUM_IMAGES)
#
# print(f'Selected {len(selected)} random images.')
#
# # ── Build the figure grid ─────────────────────────────────────────────────────
# # Each image gets 3 columns (original | raw detection | filtered), rows = NUM_IMAGES / sqrt
# cols_per_image = 3 if EDGE_FILTER_ON else 2  # 3 panels when filter is on, 2 when off
# grid_cols = int(np.sqrt(NUM_IMAGES)) * cols_per_image
# grid_rows = int(np.sqrt(NUM_IMAGES))
#
# fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
# fig.suptitle(
#     'Tree Detection — Original | Raw Classifier | After Edge Filter' if EDGE_FILTER_ON
#     else 'Tree Detection — Original | Raw Classifier',
#     fontsize=12
# )
#
# # ── Run detection on each selected image and fill the grid ───────────────────
# for idx, f in enumerate(selected):
#     img = cv2.imread(f)
#     h, w, d = img.shape
#     print(f'\n[{idx + 1}/{len(selected)}] {f}')
#
#     yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
#     pixels = np.array([
#         extract_features(yuv, hsv, lab, y, x)
#         for y in range(4, h - 2)
#         for x in range(2, w - 3)
#     ])
#
#     y_pred_flat = clf.predict(pixels)
#
#     msk_cropped = y_pred_flat.reshape(h - 6, w - 5)
#     raw_mask = np.zeros((h, w), dtype=np.uint8)
#     raw_mask[4:h - 2, 2:w - 3] = msk_cropped
#
#     # ── Apply edge filter if switched on ─────────────────────────────────────
#     if EDGE_FILTER_ON:
#         filtered_mask = filter_by_edge_density(raw_mask, img)
#
#     # ── Build display images ──────────────────────────────────────────────────
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # original
#
#     raw_overlay = img.copy()
#     raw_overlay[:, :, 1] = raw_mask
#     raw_rgb = cv2.cvtColor(raw_overlay, cv2.COLOR_BGR2RGB)  # raw classifier
#
#     # Work out grid position for this image
#     row = idx // int(np.sqrt(NUM_IMAGES))
#     col_pair = idx % int(np.sqrt(NUM_IMAGES))
#     col_orig = col_pair * cols_per_image  # original column
#     col_raw = col_pair * cols_per_image + 1  # raw detection column
#     col_filt = col_pair * cols_per_image + 2  # filtered column (only when EDGE_FILTER_ON)
#
#     axes[row, col_orig].imshow(img_rgb)
#     axes[row, col_orig].set_title(f'#{idx + 1} Original', fontsize=7)
#     axes[row, col_orig].axis('off')
#
#     axes[row, col_raw].imshow(raw_rgb)
#     axes[row, col_raw].set_title(f'#{idx + 1} Raw', fontsize=7)
#     axes[row, col_raw].axis('off')
#
#     if EDGE_FILTER_ON:
#         filt_overlay = img.copy()
#         filt_overlay[:, :, 1] = filtered_mask
#         filt_rgb = cv2.cvtColor(filt_overlay, cv2.COLOR_BGR2RGB)
#
#         axes[row, col_filt].imshow(filt_rgb)
#         axes[row, col_filt].set_title(f'#{idx + 1} Edge Filtered', fontsize=7)
#         axes[row, col_filt].axis('off')
#
# plt.tight_layout()
#
# # ── Save and/or show ──────────────────────────────────────────────────────────
# if SAVE_FIGURE:
#     plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
#     print(f'\nFigure saved to {SAVE_PATH}')
#
# plt.show()