import cv2
import numpy as np

K = np.array([
    [130.0,   0.0,  120.0],
    [  0.0, 130.0,  260.0],
    [  0.0,   0.0,    1.0],
], dtype=np.float64)

# k1 negative = corrects barrel distortion (lines bowing outward)
D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

def correct_fisheye(img):
    h, w = img.shape[:2]
    new_k, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0)
    undistorted = cv2.undistort(img, K, D, None, new_k)
    x, y, rw, rh = roi
    return undistorted[y:y + rh, x:x + rw]

img = cv2.imread(r"/TEAM-10-PROTOTYPING/downloads from drone/20240325-133935/359553775.jpg")
img = correct_fisheye(img)

cv2.imshow("Corrected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()