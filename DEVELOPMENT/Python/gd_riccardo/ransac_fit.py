import numpy as np

# ── STEP 1: HELPER — FIT PARABOLA THROUGH 3 POINTS ───────────────────────────
# RANSAC needs to fit a model to a minimal sample.
# For a degree-2 polynomial (y = ax² + bx + c), the minimal sample is 3 points.
# We solve the system analytically using numpy — no external solver needed.

def fit_parabola_3pts(pts):
    """
    Fit y = ax² + bx + c through exactly 3 (x, y) points.
    Returns (a, b, c) or None if the points are degenerate (e.g. same x).
    
    We build the Vandermonde matrix:
        | x0²  x0  1 |   | a |   | y0 |
        | x1²  x1  1 | × | b | = | y1 |
        | x2²  x2  1 |   | c |   | y2 |
    and solve it with np.linalg.solve.
    """
    (x0, y0), (x1, y1), (x2, y2) = pts

    # Build the 3×3 Vandermonde matrix
    A = np.array([
        [x0**2, x0, 1],
        [x1**2, x1, 1],
        [x2**2, x2, 1],
    ])

    b = np.array([y0, y1, y2])

    try:
        coeffs = np.linalg.solve(A, b)  # returns [a, b, c]
    except np.linalg.LinAlgError:
        return None  # matrix was singular (e.g. two points share the same x)

    return coeffs  # [a, b, c]


# ── STEP 2: HELPER — EVALUATE PARABOLA ────────────────────────────────────────
# Given coefficients and an array of x values, compute predicted y values.

def eval_parabola(coeffs, x):
    """
    Evaluate y = ax² + bx + c at position(s) x.
    coeffs: array [a, b, c]
    x: scalar or numpy array
    """
    a, b, c = coeffs
    return a * x**2 + b * x + c


# ── STEP 3: RANSAC MAIN LOOP ──────────────────────────────────────────────────
# RANSAC works by repeatedly:
#   1. Picking a tiny random sample (3 points for a parabola)
#   2. Fitting the model to that sample
#   3. Counting how many OTHER points are close to that model (inliers)
#   4. Keeping the model that has the most inliers
#
# The key insight: even if 30-40% of your data are outliers,
# there's a good chance that some random draws of 3 points will
# pick 3 clean inliers and produce a good fit.

def ransac_parabola(x, y, iterations=200, threshold=10.0, min_inliers=50):
    """
    Fit y = ax² + bx + c to (x, y) data robustly using RANSAC.

    Parameters:
        x, y        : 1D arrays of contour points (one per image column)
        iterations  : how many random samples to try
        threshold   : max residual (pixels) to count a point as an inlier
        min_inliers : minimum inliers required to accept a fit

    Returns:
        best_coeffs : array [a, b, c] of the best fit found, or None
        inlier_mask : boolean array marking which points are inliers
    """
    n = len(x)
    best_coeffs = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(n, dtype=bool)

    if n < 3:
        return None, np.zeros(n, dtype=bool)

    for _ in range(iterations):

        # ── 3a. RANDOM SAMPLE ─────────────────────────────────────────────────
        # Pick 3 distinct random indices from the data.
        # These 3 points fully determine a parabola.
        sample_idx = np.random.choice(n, size=3, replace=False)
        sample_pts = [(x[idx], y[idx]) for idx in sample_idx]

        # ── 3b. FIT MODEL TO SAMPLE ───────────────────────────────────────────
        # Fit a parabola through just these 3 points.
        coeffs = fit_parabola_3pts(sample_pts)
        if coeffs is None:
            continue  # degenerate sample, skip

        # ── 3c. EVALUATE ALL POINTS AGAINST THIS MODEL ────────────────────────
        # For every point in the dataset, compute how far it is
        # from the candidate parabola (the "residual").
        y_predicted = eval_parabola(coeffs, x)
        residuals = np.abs(y_predicted - y)

        # A point is an "inlier" if its residual is within the threshold.
        # Inliers are consistent with this candidate model.
        # Outliers (obstacles, plants) will have large residuals.
        inlier_mask = residuals < threshold
        inlier_count = inlier_mask.sum()

        # ── 3d. KEEP BEST MODEL ───────────────────────────────────────────────
        # If this candidate has more inliers than any previous one, save it.
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_coeffs = coeffs
            best_inlier_mask = inlier_mask

    # ── 3e. OPTIONAL REFINEMENT ───────────────────────────────────────────────
    # Once we know which points are inliers, we can refit the parabola
    # using ALL inliers (not just the 3 random points), which gives a
    # more accurate final result. This is called the "refinement step".
    if best_coeffs is not None and best_inlier_count >= min_inliers:
        x_inliers = x[best_inlier_mask]
        y_inliers = y[best_inlier_mask]
        # np.polyfit does a least-squares fit — safe here because
        # we've already removed the outliers via RANSAC
        best_coeffs = np.polyfit(x_inliers, y_inliers, deg=2)
    else:
        print("RANSAC failed: not enough inliers found.")
        best_coeffs = None

    return best_coeffs, best_inlier_mask


def draw_parabola_on_image(image_bgr, coeffs, x, y, color=(0, 0, 255), thickness=2):

    width, height = image_bgr.shape[:2]
    x_all = np.arange(width)

    # Compute all y values at once
    y_all = np.round(eval_parabola(coeffs, x_all)).astype(int)

    # x_all = x.astype(bool)
    # y_all = y.astype(bool)

    # For each thickness offset
    for dy in range(-thickness // 2, thickness // 2 + 1):
        y_draw = y_all + dy

        # Only keep columns where y_draw is within the image
        valid = (y_draw >= 0) & (y_draw < height)

        # Index directly: image[row, col] = color
        image_bgr[x_all[valid], y_draw[valid]] = color

    return image_bgr



# code inside main.py
if __name__ == "__main__":

    pass