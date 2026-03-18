import cv2
import numpy as np
import os
import glob
from corner_detection import detect_gate_corners_template


def gate_present_from_template_output(image_path, corners, debug=False):

    img = cv2.imread(image_path)
    vis = img.copy()

    # Condition 1: need at least two detected corners
    if corners is None or len(corners) < 2:
        if debug:
            cv2.putText(vis, "Gate: NO (fewer than 2 corners)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, vis

    corners = np.array(corners, dtype=np.float32)

    # Draw corners
    if debug:
        for i, pt in enumerate(corners.astype(int)):
            cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"C{i}", (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Blue mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Region defined by detected corners
    x_min = int(np.min(corners[:, 0]))
    x_max = int(np.max(corners[:, 0]))
    y_min = int(np.min(corners[:, 1]))
    y_max = int(np.max(corners[:, 1]))

    # Expand search region a bit
    margin_x = 40
    margin_y = 60

    y0 = max(0, y_min - margin_y)
    y1 = min(img.shape[0], y_max + margin_y)

    lx0 = max(0, x_min - margin_x)
    lx1 = min(img.shape[1], x_min + margin_x)

    rx0 = max(0, x_max - margin_x)
    rx1 = min(img.shape[1], x_max + margin_x)

    left_strip = blue_mask[y0:y1, lx0:lx1]
    right_strip = blue_mask[y0:y1, rx0:rx1]

    def has_vertical_blue_pillar(strip, min_height=40, min_area=150, min_aspect=1.8): # needs to be calibrated better
        contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if h < min_height:
                continue

            # verticality check
            if h / float(max(w, 1)) < min_aspect:
                continue

            return True, (x, y, w, h)

        return False, None

    left_ok, left_rect = has_vertical_blue_pillar(left_strip)
    right_ok, right_rect = has_vertical_blue_pillar(right_strip)

    gate_present = left_ok and right_ok

    if debug:
        # Draw search strips
        cv2.rectangle(vis, (lx0, y0), (lx1, y1), (255, 255, 0), 2)
        cv2.rectangle(vis, (rx0, y0), (rx1, y1), (255, 255, 0), 2)

        # Draw detected pillar rectangles
        if left_rect is not None:
            x, y, w, h = left_rect
            cv2.rectangle(vis, (lx0 + x, y0 + y), (lx0 + x + w, y0 + y + h), (255, 0, 0), 2)

        if right_rect is not None:
            x, y, w, h = right_rect
            cv2.rectangle(vis, (rx0 + x, y0 + y), (rx0 + x + w, y0 + y + h), (255, 0, 0), 2)

        text = "Gate: YES" if gate_present else "Gate: NO"
        color = (0, 255, 0) if gate_present else (0, 0, 255)
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return gate_present, vis

def process_folder_as_video(folder_path, delay=120):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []

    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No images found in folder: {folder_path}")

    for image_path in files:
        corners, vis_temp, white_mask = detect_gate_corners_template(image_path, debug=False)
        gate_present, vis_gate = gate_present_from_template_output(
            image_path,
            corners,
            debug=True
        )

        cv2.putText(
            vis_gate,
            os.path.basename(image_path),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis_gate,
            f"Detected: {len(corners)}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("White mask", white_mask)
        cv2.imshow("Gate corner detection", vis_gate)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "../paparazzi10/DEVELOPMENT/downloads from drone/20260306-095826/1313401090.jpg"
    corners, vis_temp, white_mask = detect_gate_corners_template(image_path, debug=False)
    gate_present, vis_gate = gate_present_from_template_output(
            image_path,
            corners,
            debug=True
        )

    print("Gate present:", gate_present)

    cv2.imshow("Gate decision", vis_gate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    folder_path = "../paparazzi10/DEVELOPMENT/downloads from drone/20260306-095826"
    process_folder_as_video(folder_path)