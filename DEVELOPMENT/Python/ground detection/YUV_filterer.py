import cv2
import numpy as np
from glob import glob
import os
import random

# ── folder with your images ───────────────────────────────────────────────────
FOLDER_PATH = "DEVELOPMENT/downloads from drone/20260313-100130/"

# ── initial threshold values (match your current is_ground logic) ─────────────
DEFAULTS = dict(Y_min=40, Y_max=220, U_min=0, U_max=125, V_min=0, V_max=255)

WINDOW_MAIN   = "YUV Tuner  |  original - mask - masked result  |  n=next  q=quit"
WINDOW_CHART  = "YUV Color Chart  (U horizontal, V vertical, Y=current slider)"

# ── helpers ───────────────────────────────────────────────────────────────────

def apply_mask(image_bgr, y_min, y_max, u_min, u_max, v_min, v_max):
    yuv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)
    mask = (
        (Y >= y_min) & (Y <= y_max) &
        (U >= u_min) & (U <= u_max) &
        (V >= v_min) & (V <= v_max)
    ).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return mask, result


def build_uv_chart(y_val, y_min, y_max, u_min, u_max, v_min, v_max,
                   size=512):
    """
    2-D chart: U axis = horizontal (0→255), V axis = vertical (0→255).
    Each pixel is coloured with its actual BGR equivalent at Y=y_val.
    The currently PASSING region is shown at full brightness;
    the failing region is darkened.
    White crosshair lines mark the threshold edges.
    """
    chart_yuv = np.zeros((size, size, 3), dtype=np.uint8)

    u_vals = np.linspace(0, 255, size).astype(np.uint8)
    v_vals = np.linspace(0, 255, size).astype(np.uint8)
    UU, VV = np.meshgrid(u_vals, v_vals)
    YY     = np.full((size, size), y_val, dtype=np.uint8)

    chart_yuv[..., 0] = YY
    chart_yuv[..., 1] = UU
    chart_yuv[..., 2] = VV

    chart_bgr = cv2.cvtColor(chart_yuv, cv2.COLOR_YUV2BGR)

    # darken failing region
    passing = (
        (YY >= y_min) & (YY <= y_max) &
        (UU >= u_min) & (UU <= u_max) &
        (VV >= v_min) & (VV <= v_max)
    )
    dark = chart_bgr.copy()
    dark[~passing] = (dark[~passing] * 0.25).astype(np.uint8)

    # threshold boundary lines
    def u_to_x(u): return int(u / 255 * (size - 1))
    def v_to_y(v): return int(v / 255 * (size - 1))

    for u_thresh in [u_min, u_max]:
        x = u_to_x(u_thresh)
        cv2.line(dark, (x, 0), (x, size - 1), (255, 255, 255), 1)
        cv2.putText(dark, f"U={u_thresh}", (x + 3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for v_thresh in [v_min, v_max]:
        y = v_to_y(v_thresh)
        cv2.line(dark, (0, y), (size - 1, y), (255, 255, 255), 1)
        cv2.putText(dark, f"V={v_thresh}", (4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # axis labels
    cv2.putText(dark, "U  (0 → 255)",  (size // 2 - 50, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    label_img = np.zeros((size, 20, 3), dtype=np.uint8)
    cv2.putText(label_img, "V", (2, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    dark = np.hstack([label_img, dark])

    # Y indicator strip on the right (dark→bright gradient)
    y_strip = np.zeros((size, 30, 3), dtype=np.uint8)
    for row in range(size):
        lum = int(row / (size - 1) * 255)
        y_strip[row, :] = [lum, lum, lum]
    y_lo = int(y_min / 255 * (size - 1))
    y_hi = int(y_max / 255 * (size - 1))
    cv2.rectangle(y_strip, (0, y_lo), (29, y_hi), (0, 255, 0), 1)
    cv2.putText(y_strip, "Y", (6, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    dark = np.hstack([dark, y_strip])

    # current Y value overlay
    cv2.putText(dark, f"Y = {y_val}", (22, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return dark


def print_thresholds(y_min, y_max, u_min, u_max, v_min, v_max):
    print(f"\n--- current thresholds ---")
    print(f"  Y: [{y_min}, {y_max}]  U: [{u_min}, {u_max}]  V: [{v_min}, {v_max}]")
    print(f"  green_mask = (U <= {u_max}) & (U >= {u_min}) & "
          f"(Y >= {y_min}) & (Y <= {y_max}) & "
          f"(V >= {v_min}) & (V <= {v_max})")


# ── load images ───────────────────────────────────────────────────────────────
all_paths = sorted(glob(os.path.join(FOLDER_PATH, "*.jpg")))
if not all_paths:
    print(f"No .jpg images found in {FOLDER_PATH}")
    exit()

random.shuffle(all_paths)
img_idx = [0]   # mutable so callbacks can update it

# ── create windows & trackbars ────────────────────────────────────────────────
cv2.namedWindow(WINDOW_MAIN,  cv2.WINDOW_NORMAL)
cv2.namedWindow(WINDOW_CHART, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_CHART, 562, 512)

def nothing(_): pass

for name, val in DEFAULTS.items():
    cv2.createTrackbar(name, WINDOW_MAIN, val, 255, nothing)

# ── main loop ─────────────────────────────────────────────────────────────────
while True:
    path = all_paths[img_idx[0]]
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        img_idx[0] = (img_idx[0] + 1) % len(all_paths)
        continue

    # read sliders
    y_min = cv2.getTrackbarPos("Y_min", WINDOW_MAIN)
    y_max = cv2.getTrackbarPos("Y_max", WINDOW_MAIN)
    u_min = cv2.getTrackbarPos("U_min", WINDOW_MAIN)
    u_max = cv2.getTrackbarPos("U_max", WINDOW_MAIN)
    v_min = cv2.getTrackbarPos("V_min", WINDOW_MAIN)
    v_max = cv2.getTrackbarPos("V_max", WINDOW_MAIN)
    y_chart = (y_min + y_max) // 2   # chart uses midpoint Y

    mask, result = apply_mask(image_bgr, y_min, y_max, u_min, u_max, v_min, v_max)

    # resize to same height for side-by-side display
    H, W = image_bgr.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    display = np.hstack([image_bgr, mask_bgr, result])
    display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frac = cv2.countNonZero(mask) / (H * W)
    cv2.putText(display, f"{os.path.basename(path)}  |  green={frac:.2%}  "
                         f"Y[{y_min},{y_max}] U[{u_min},{u_max}] V[{v_min},{v_max}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.imshow(WINDOW_MAIN, display)

    # UV chart
    chart = build_uv_chart(y_chart, y_min, y_max, u_min, u_max, v_min, v_max)
    cv2.imshow(WINDOW_CHART, chart)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        img_idx[0] = (img_idx[0] + 1) % len(all_paths)
        print_thresholds(y_min, y_max, u_min, u_max, v_min, v_max)
    elif key == ord('p'):
        img_idx[0] = (img_idx[0] - 1) % len(all_paths)
    elif key == ord('s'):
        print_thresholds(y_min, y_max, u_min, u_max, v_min, v_max)

cv2.destroyAllWindows()
print("\nFinal thresholds:")
print_thresholds(
    cv2.getTrackbarPos("Y_min", WINDOW_MAIN),
    cv2.getTrackbarPos("Y_max", WINDOW_MAIN),
    cv2.getTrackbarPos("U_min", WINDOW_MAIN),
    cv2.getTrackbarPos("U_max", WINDOW_MAIN),
    cv2.getTrackbarPos("V_min", WINDOW_MAIN),
    cv2.getTrackbarPos("V_max", WINDOW_MAIN),
)