# import cv2
# import numpy as np
#
# K = np.array([
#     [130.0,   0.0,  120.0],
#     [  0.0, 130.0,  260.0],
#     [  0.0,   0.0,    1.0],
# ], dtype=np.float64)
#
# # k1 negative = corrects barrel distortion (lines bowing outward)
# D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
#
# def correct_fisheye(img):
#     h, w = img.shape[:2]
#     new_k, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0)
#     undistorted = cv2.undistort(img, K, D, None, new_k)
#     x, y, rw, rh = roi
#     return undistorted[y:y + rh, x:x + rw]
#
# img = cv2.imread(r"../../downloads from drone/20240325-133935/359553775.jpg")
# img = correct_fisheye(img)
#
# cv2.imshow("Corrected", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from defisheye import Defisheye
#
# dtype = 'linear'
# format = 'fullframe'
# fov = 180
# pfov = 120
#
# img = "../../downloads from drone/20260313-100130/74130805.jpg"
# img_out = f"./example3_{dtype}_{format}_{pfov}_{fov}.jpg"
#
# obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
#
# # To save image locally
# obj.convert(outfile=img_out)
#
# # To use the converted image in memory
#
# new_image = obj.convert()

import cv2
import numpy as np

# ── Real Parrot Bebop 2 calibration ───────────────────────────────────
# Source: bebop_autonomy ROS driver (bebop2_camera_calib.yaml)
# Calibrated at 856x480. Values scale automatically to your image size.
#
#   fx=537.29  fy=527.00  cx=427.33  cy=240.23
#   distortion model: plumb_bob (standard Brown-Conrady)
#   D = [k1, k2, p1, p2, k3]
# ─────────────────────────────────────────────────────────────────────

BASE_W, BASE_H = 856, 480

K_BASE = np.array([
    [537.292878,   0.0,       427.331854],
    [  0.0,       527.000348, 240.226888],
    [  0.0,         0.0,        1.0     ],
], dtype=np.float64)

# plumb_bob distortion: [k1, k2, p1, p2, k3]
D = np.array([0.004974, -0.000130, -0.001212, 0.002192, 0.0], dtype=np.float64)


def defisheye(img):
    h, w = img.shape[:2]

    # Scale K to match actual image resolution
    sx, sy = w / BASE_W, h / BASE_H
    K = K_BASE.copy()
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy

    # Undistort (standard model — NOT fisheye model)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    undistorted = cv2.undistort(img, K, D, None, new_K)

    # Rotate -90° (camera mounted sideways on the drone)
    rotated = cv2.rotate(undistorted, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rotated


# ── usage ──────────────────────────────────────────────────────────────
img = cv2.imread(r"../../downloads from drone/20240325-133935/359553775.jpg")   # ← change this
out = defisheye(img)
cv2.imshow("Bebop Corrected", out)
cv2.waitKey(0)
cv2.destroyAllWindows()