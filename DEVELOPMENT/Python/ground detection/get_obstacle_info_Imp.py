"""
optical_flow.py
───────────────
Computes optical flow between consecutive frames in a folder.
Follows the same structure as the obstacle-detection pipeline:
  - scale helpers (scale_int / scale_odd / get_scaled_params)
  - a FlowResult dataclass returned from the core function
  - an interactive __main__ viewer (a / d to navigate, q to quit)

Public API
──────────
    result = get_optical_flow(prev_bgr, curr_bgr,
                              scale_factor=0.5,
                              method="farneback")   # or "lucas_kanade"

    result.magnitude      – np.ndarray (H, W)  per-pixel motion magnitude
    result.angle          – np.ndarray (H, W)  motion direction in degrees
    result.mean_mag       – float              mean magnitude (≈ overall motion)
    result.mean_flow_vec  – (float, float)     (dx, dy) mean flow vector
    result.divergence     – np.ndarray (H, W)  div field (dense only, else zeros)
    result.vectors        – dict               algorithm-specific extras
                             dense  → {"flow": (H,W,2)}
                             sparse → {"prev_pts", "curr_pts", "status"}

    alert = check_obstacle_divergence(result,
                                      div_threshold=0.3,
                                      roi_fraction=0.5)
    alert.triggered    – bool    threshold exceeded
    alert.max_div      – float   peak divergence in ROI
    alert.mean_div     – float   mean divergence in ROI
    alert.roi_mask     – (H, W)  bool array marking the ROI used
"""

import os
import random
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import cv2
import numpy as np


# ── scale helpers (mirrors obstacle_detection.py) ─────────────────────────────

def scale_int(value: int, scale_factor: float) -> int:
    """Scale a pixel-count value; minimum 1."""
    return max(1, int(round(value * scale_factor)))


def scale_odd(value: int, scale_factor: float) -> int:
    """Scale a kernel size; result is always odd, minimum 3."""
    n = max(3, int(round(value * scale_factor)))
    return n if n % 2 == 1 else n + 1


def get_scaled_params(scale_factor: float) -> dict:
    """
    All pixel-space thresholds in one place.
    Calibrated at scale_factor = 1.0 (full resolution).
    """
    return dict(
        # ── preprocessing ─────────────────────────────────────────────────
        blur_ksize          = scale_odd(5,   scale_factor),

        # ── Farnebäck dense flow ───────────────────────────────────────────
        fb_pyr_scale        = 0.5,          # dimensionless, do NOT scale
        fb_levels           = 3,            # pyramid levels
        fb_winsize          = scale_int(15, scale_factor),
        fb_iterations       = 3,
        fb_poly_n           = 5,            # neighbourhood size
        fb_poly_sigma       = 1.2,          # Gaussian std for poly expansion

        # ── Lucas–Kanade sparse flow ───────────────────────────────────────
        lk_max_corners      = 200,          # Shi-Tomasi corner count
        lk_quality_level    = 0.3,
        lk_min_distance     = scale_int(7,  scale_factor),
        lk_block_size       = scale_int(7,  scale_factor),
        lk_win_size         = (scale_odd(15, scale_factor),
                               scale_odd(15, scale_factor)),
        lk_max_level        = 2,            # pyramid levels for LK

        # ── display ───────────────────────────────────────────────────────
        status_font_scale   = max(0.3, scale_factor * 1.0),
        status_thickness    = max(1, scale_int(2, scale_factor)),
        status_x            = scale_int(20, scale_factor),
        status_y            = scale_int(40, scale_factor),
        arrow_step          = scale_int(16, scale_factor),   # grid spacing for vector overlay
        arrow_scale         = scale_factor * 3.0,            # visual amplification
    )


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class FlowResult:
    """
    Returned by get_optical_flow().

    magnitude     : (H, W) float32 — per-pixel motion magnitude in pixels
    angle         : (H, W) float32 — motion direction in degrees [0, 360)
    mean_mag      : float           — mean magnitude across valid pixels
    mean_flow_vec : (float, float)  — (dx, dy) mean flow vector
    divergence    : (H, W) float32 — optical divergence field (dense only)
    vectors       : dict            — method-specific raw data:
                      "farneback"    → {"flow": ndarray (H, W, 2)}
                      "lucas_kanade" → {"prev_pts", "curr_pts", "status"}
    """
    magnitude     : np.ndarray
    angle         : np.ndarray
    mean_mag      : float
    mean_flow_vec : tuple = (0.0, 0.0)
    divergence    : np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    vectors       : dict = field(default_factory=dict)


@dataclass
class DivergenceAlert:
    """
    Returned by check_obstacle_divergence().

    triggered : bool          — True when max_div exceeds the threshold
    max_div   : float         — peak divergence inside the ROI
    mean_div  : float         — mean divergence inside the ROI
    roi_mask  : np.ndarray    — (H, W) bool mask of the region that was checked
    """
    triggered : bool
    max_div   : float
    mean_div  : float
    roi_mask  : np.ndarray


# ── preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(image_bgr: np.ndarray,
                scale_factor: float,
                blur_ksize: int) -> np.ndarray:
    """Downscale → grayscale → optional blur."""
    H, W = image_bgr.shape[:2]
    tW = max(1, int(W * scale_factor))
    tH = max(1, int(H * scale_factor))
    small = cv2.resize(image_bgr, (tW, tH), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


# ── divergence computation ────────────────────────────────────────────────────

def _compute_divergence(flow: np.ndarray) -> np.ndarray:
    """
    Compute the optical divergence of a dense flow field.

    div = dFx/dx + dFy/dy

    Positive divergence at a pixel means flow is expanding outward from that
    point — the signature of an approaching obstacle (looming).
    Negative divergence means flow converging inward (receding object).

    Returns a (H, W) float32 array, same spatial size as `flow`.
    """
    fx = flow[..., 0]
    fy = flow[..., 1]
    # Central-difference gradients; Sobel gives smoother result than np.gradient
    dFx_dx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
    dFy_dy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
    return dFx_dx + dFy_dy



def _farneback_flow(prev_gray: np.ndarray,
                    curr_gray: np.ndarray,
                    p: dict) -> FlowResult:
    """Dense Farnebäck optical flow."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        p["fb_pyr_scale"],
        p["fb_levels"],
        p["fb_winsize"],
        p["fb_iterations"],
        p["fb_poly_n"],
        p["fb_poly_sigma"],
        0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    mean_dx  = float(flow[..., 0].mean())
    mean_dy  = float(flow[..., 1].mean())
    div      = _compute_divergence(flow)
    return FlowResult(
        magnitude     = mag,
        angle         = ang,
        mean_mag      = float(mag.mean()),
        mean_flow_vec = (mean_dx, mean_dy),
        divergence    = div,
        vectors       = {"flow": flow},
    )


def _lucas_kanade_flow(prev_gray: np.ndarray,
                       curr_gray: np.ndarray,
                       p: dict) -> FlowResult:
    """Sparse Lucas–Kanade optical flow on Shi-Tomasi corners."""
    H, W = prev_gray.shape

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners   = p["lk_max_corners"],
        qualityLevel = p["lk_quality_level"],
        minDistance  = p["lk_min_distance"],
        blockSize    = p["lk_block_size"],
    )

    mag = np.zeros((H, W), dtype=np.float32)
    ang = np.zeros((H, W), dtype=np.float32)
    mean_mag = 0.0
    result_prev = result_curr = result_status = None

    if prev_pts is not None and len(prev_pts) > 0:
        lk_params = dict(
            winSize  = p["lk_win_size"],
            maxLevel = p["lk_max_level"],
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params)

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) > 0:
            diff = good_curr - good_prev          # (N, 1, 2)
            dx   = diff[:, 0, 0]
            dy   = diff[:, 0, 1]
            mags = np.sqrt(dx**2 + dy**2)
            angs = np.degrees(np.arctan2(dy, dx)) % 360
            mean_mag = float(mags.mean())

            # Scatter magnitudes / angles onto the pixel grid
            for pt, m, a in zip(good_prev.reshape(-1, 2), mags, angs):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < H and 0 <= x < W:
                    mag[y, x] = m
                    ang[y, x] = a

        result_prev   = good_prev
        result_curr   = good_curr
        result_status = status

    mean_dx = float(np.mean(good_curr[:, 0, 0] - good_prev[:, 0, 0])) if (result_prev is not None and len(result_prev) > 0) else 0.0
    mean_dy = float(np.mean(good_curr[:, 0, 1] - good_prev[:, 0, 1])) if (result_prev is not None and len(result_prev) > 0) else 0.0

    return FlowResult(
        magnitude     = mag,
        angle         = ang,
        mean_mag      = mean_mag,
        mean_flow_vec = (mean_dx, mean_dy),
        divergence    = np.zeros_like(mag),   # sparse flow → no dense div field
        vectors       = {
            "prev_pts": result_prev,
            "curr_pts": result_curr,
            "status"  : result_status,
        },
    )


# ── public API ────────────────────────────────────────────────────────────────

def get_optical_flow(prev_bgr       : np.ndarray,
                     curr_bgr       : np.ndarray,
                     scale_factor   : float = 0.5,
                     method         : str   = "farneback") -> FlowResult:
    """
    Compute optical flow between two BGR frames.

    Parameters
    ----------
    prev_bgr     : Previous frame (BGR, any resolution).
    curr_bgr     : Current frame  (BGR, same resolution as prev_bgr).
    scale_factor : Downscale factor applied before flow computation.
                   Smaller → faster; 0.5 is a good default.
    method       : "farneback"    – dense flow, best for full field analysis
                   "lucas_kanade" – sparse flow, faster, keypoints only

    Returns
    -------
    FlowResult
    """
    p = get_scaled_params(scale_factor)

    prev_gray = _preprocess(prev_bgr, scale_factor, p["blur_ksize"])
    curr_gray = _preprocess(curr_bgr, scale_factor, p["blur_ksize"])

    if method == "farneback":
        return _farneback_flow(prev_gray, curr_gray, p)
    elif method == "lucas_kanade":
        return _lucas_kanade_flow(prev_gray, curr_gray, p)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'farneback' or 'lucas_kanade'.")



# ── obstacle divergence checker ───────────────────────────────────────────────

def check_obstacle_divergence(result        : FlowResult,
                               div_threshold : float = 0.3,
                               roi_fraction  : float = 0.5) -> "DivergenceAlert":
    """
    Check whether the divergence in the central ROI exceeds a threshold.

    An approaching obstacle causes the flow field to *expand* (positive
    divergence) from its centre.  By watching the central region of the frame
    we can detect this looming signature before the obstacle fills the view.

    Parameters
    ----------
    result        : FlowResult from get_optical_flow().
    div_threshold : Alert fires when max divergence inside the ROI ≥ this value.
                    Tune empirically; typical range 0.1 – 1.0.
    roi_fraction  : Fraction of frame width/height used for the central ROI.
                    0.5 = centre 50 % of each axis.

    Returns
    -------
    DivergenceAlert
    """
    H, W  = result.divergence.shape
    pad_y = int(H * (1 - roi_fraction) / 2)
    pad_x = int(W * (1 - roi_fraction) / 2)
    y0, y1 = pad_y, H - pad_y
    x0, x1 = pad_x, W - pad_x

    roi_mask = np.zeros((H, W), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True

    roi_div  = result.divergence[roi_mask]
    max_div  = float(roi_div.max())  if roi_div.size > 0 else 0.0
    mean_div = float(roi_div.mean()) if roi_div.size > 0 else 0.0

    return DivergenceAlert(
        triggered = max_div >= div_threshold,
        max_div   = max_div,
        mean_div  = mean_div,
        roi_mask  = roi_mask,
    )


# ── visualisation helpers ─────────────────────────────────────────────────────

def flow_to_hsv_bgr(result: FlowResult) -> np.ndarray:
    """
    Convert a FlowResult to a colour-coded BGR image.
    Hue = direction, Saturation = 255, Value = normalised magnitude.
    """
    mag = result.magnitude
    ang = result.angle

    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)   # hue: 0–180
    hsv[..., 1] = 255
    max_mag = mag.max()
    if max_mag > 0:
        hsv[..., 2] = (mag / max_mag * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(image_bgr  : np.ndarray,
                     result     : FlowResult,
                     step       : int   = 16,
                     scale      : float = 3.0,
                     color      : tuple = (0, 200, 255),
                     min_mag    : float = 0.5) -> np.ndarray:
    """
    Overlay motion arrows on a BGR image.
    Works for both dense and sparse results.
    """
    out = image_bgr.copy()
    H, W = result.magnitude.shape

    if "flow" in result.vectors:
        # Dense: sample on a regular grid
        flow = result.vectors["flow"]
        ys = range(step // 2, H, step)
        xs = range(step // 2, W, step)
        for y in ys:
            for x in xs:
                mag = result.magnitude[y, x]
                if mag < min_mag:
                    continue
                fx, fy = flow[y, x]
                x2 = int(x + fx * scale)
                y2 = int(y + fy * scale)
                cv2.arrowedLine(out, (x, y), (x2, y2),
                                color, 1, tipLength=0.3)
    else:
        # Sparse: draw arrows between tracked points
        prev_pts = result.vectors.get("prev_pts")
        curr_pts = result.vectors.get("curr_pts")
        if prev_pts is not None and curr_pts is not None:
            for p0, p1 in zip(prev_pts.reshape(-1, 2),
                               curr_pts.reshape(-1, 2)):
                x0, y0 = int(p0[0]), int(p0[1])
                x1, y1 = int(p1[0]), int(p1[1])
                mag = result.magnitude[
                    min(y0, H - 1), min(x0, W - 1)]
                if mag < min_mag:
                    continue
                cv2.arrowedLine(out, (x0, y0), (x1, y1),
                                color, 1, tipLength=0.3)
    return out

def draw_mean_flow_arrow(image_bgr  : np.ndarray,
                         result     : FlowResult,
                         scale      : float = 20.0,
                         color      : tuple = (0, 255, 0),
                         thickness  : int   = 2) -> np.ndarray:
    """
    Draw a single arrow at the image centre representing the mean flow vector.

    The arrow length is proportional to mean_mag; `scale` controls how many
    pixels one flow-pixel maps to on screen.  A circle is drawn at the tail.
    """
    out = image_bgr.copy()
    H, W = out.shape[:2]
    cx, cy = W // 2, H // 2

    dx, dy = result.mean_flow_vec
    ex = int(cx + dx * scale)
    ey = int(cy + dy * scale)

    cv2.circle(out, (cx, cy), thickness + 2, color, -1)
    cv2.arrowedLine(out, (cx, cy), (ex, ey), color,
                    thickness, tipLength=0.25)
    mag_label = f"{result.mean_mag:.1f}px"
    cv2.putText(out, mag_label,
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def draw_divergence_overlay(image_bgr : np.ndarray,
                             result    : FlowResult,
                             alert     : "DivergenceAlert",
                             alpha     : float = 0.45) -> np.ndarray:
    """
    Overlay the divergence heatmap and ROI box on a BGR image.

    Positive divergence (looming) → red tint.
    Negative divergence (receding) → blue tint.
    The ROI rectangle is drawn in yellow (safe) or red (alert triggered).
    """
    out  = image_bgr.copy()
    H, W = out.shape[:2]

    div = result.divergence
    if div.shape != (H, W):
        div = cv2.resize(div, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalise divergence to [0, 255] for display
    max_abs = max(np.abs(div).max(), 1e-6)
    norm    = np.clip(div / max_abs, -1.0, 1.0)   # [-1, 1]

    heatmap = np.zeros((H, W, 3), dtype=np.uint8)
    pos_mask = norm > 0
    neg_mask = norm < 0
    heatmap[pos_mask, 2] = (norm[pos_mask] * 255).astype(np.uint8)   # red channel
    heatmap[neg_mask, 0] = (-norm[neg_mask] * 255).astype(np.uint8)  # blue channel

    cv2.addWeighted(heatmap, alpha, out, 1 - alpha, 0, out)

    # ROI rectangle
    roi_color = (0, 0, 255) if alert.triggered else (0, 220, 255)
    ys, xs = np.where(alert.roi_mask)
    if ys.size > 0:
        cv2.rectangle(out,
                      (int(xs.min()), int(ys.min())),
                      (int(xs.max()), int(ys.max())),
                      roi_color, 2)

    # Status text
    status_txt = (f"DIV ALERT  max={alert.max_div:.2f}"
                  if alert.triggered else
                  f"div ok  max={alert.max_div:.2f}")
    txt_color  = (0, 0, 255) if alert.triggered else (0, 220, 255)
    cv2.putText(out, status_txt, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    return out



if __name__ == "__main__":

    # ── configuration ─────────────────────────────────────────────────────────
    folder_path   = "DEVELOPMENT/downloads from drone/20260320/"
    SCALE_FACTOR  = 0.5
    METHOD        = "farneback"   # "farneback" or "lucas_kanade"
    FRAME_DELAY   = 1             # ms between waitKey polls
    DIV_THRESHOLD = 0.3           # divergence alert threshold (tune per scene)
    ROI_FRACTION  = 0.5           # central ROI size (fraction of frame)

    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))
    if not image_paths:
        print("No images found in:", folder_path)
        exit(1)

    p = get_scaled_params(SCALE_FACTOR)
    print(f"Method: {METHOD}  |  scale: {SCALE_FACTOR}  |  "
          f"blur: {p['blur_ksize']}px  |  "
          f"div_threshold: {DIV_THRESHOLD}  |  "
          f"{len(image_paths)} frames found")
    print("Keys: a/d = prev/next  |  m = toggle method  |  +/- = adjust threshold  |  q = quit")

    idx = random.randint(1, len(image_paths) - 1)

    WINDOW = "Frame | HSV flow | Arrows+mean | Divergence"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    needs_processing = True
    result: Optional[FlowResult] = None

    while True:
        try:
            if needs_processing:
                prev_path = image_paths[idx - 1]
                curr_path = image_paths[idx]

                prev_raw = cv2.imread(prev_path)
                curr_raw = cv2.imread(curr_path)

                if prev_raw is None or curr_raw is None:
                    print(f"Skipping unreadable pair at index {idx}")
                    idx = max(1, (idx + 1) % len(image_paths))
                    continue

                orig_H, orig_W = curr_raw.shape[:2]
                tW = max(1, int(orig_W * SCALE_FACTOR))
                tH = max(1, int(orig_H * SCALE_FACTOR))

                prev_bgr = cv2.resize(prev_raw, (tW, tH), interpolation=cv2.INTER_AREA)
                curr_bgr = cv2.resize(curr_raw, (tW, tH), interpolation=cv2.INTER_AREA)

                result = get_optical_flow(
                    prev_bgr,
                    curr_bgr,
                    scale_factor = 1.0,   # already downscaled above
                    method       = METHOD,
                )

                alert = check_obstacle_divergence(
                    result,
                    div_threshold = DIV_THRESHOLD,
                    roi_fraction  = ROI_FRACTION,
                )

                # ── build display panels ───────────────────────────────────
                hsv_vis = flow_to_hsv_bgr(result)

                # Panel 3: grid arrows + mean flow arrow
                arrows_vis = draw_flow_arrows(
                    curr_bgr, result,
                    step  = p["arrow_step"],
                    scale = p["arrow_scale"],
                )
                arrows_vis = draw_mean_flow_arrow(
                    arrows_vis, result,
                    scale     = 20.0,
                    color     = (0, 255, 0),
                    thickness = max(1, scale_int(2, SCALE_FACTOR)),
                )

                # Panel 4: divergence heatmap + ROI + alert
                div_vis = draw_divergence_overlay(curr_bgr, result, alert)

                # ── rotate all panels 90° clockwise ───────────────────────
                curr_rot   = cv2.rotate(curr_bgr,   cv2.ROTATE_90_COUNTERCLOCKWISE)
                hsv_rot    = cv2.rotate(hsv_vis,    cv2.ROTATE_90_COUNTERCLOCKWISE)
                arrows_rot = cv2.rotate(arrows_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
                div_rot    = cv2.rotate(div_vis,    cv2.ROTATE_90_COUNTERCLOCKWISE)

                combined = np.hstack((curr_rot, hsv_rot, arrows_rot, div_rot))

                # ── status label ──────────────────────────────────────────
                alert_tag = "  !! DIV ALERT !!" if alert.triggered else ""
                label = (f"[{idx}/{len(image_paths)-1}]  "
                         f"{os.path.basename(curr_path)}  |  "
                         f"mean mag: {result.mean_mag:.2f}px  "
                         f"dx={result.mean_flow_vec[0]:.1f} dy={result.mean_flow_vec[1]:.1f}  |  "
                         f"div max={alert.max_div:.2f} thr={DIV_THRESHOLD:.2f}  |  "
                         f"{METHOD}{alert_tag}")
                lbl_color = (0, 0, 255) if alert.triggered else (0, 255, 255)
                cv2.putText(combined, label,
                            (p["status_x"], p["status_y"]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            p["status_font_scale"],
                            lbl_color,
                            p["status_thickness"])

                cv2.imshow(WINDOW, combined)
                # after 90° rotation width↔height swap; 4 panels wide
                cv2.resizeWindow(WINDOW, orig_H * 4, orig_W)
                print(label)
                needs_processing = False

            key = cv2.waitKey(FRAME_DELAY) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                idx = min(len(image_paths) - 1, idx + 1)
                needs_processing = True
            elif key == ord("a"):
                idx = max(1, idx - 1)
                needs_processing = True
            elif key == ord("m"):
                METHOD = ("lucas_kanade"
                          if METHOD == "farneback" else "farneback")
                needs_processing = True
                print(f"Switched to: {METHOD}")
            elif key in (ord("+"), ord("=")):
                DIV_THRESHOLD = round(DIV_THRESHOLD + 0.05, 3)
                needs_processing = True
                print(f"div_threshold → {DIV_THRESHOLD}")
            elif key == ord("-"):
                DIV_THRESHOLD = round(max(0.01, DIV_THRESHOLD - 0.05), 3)
                needs_processing = True
                print(f"div_threshold → {DIV_THRESHOLD}")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break

    cv2.destroyAllWindows()