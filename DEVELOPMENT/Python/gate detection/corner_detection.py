import cv2
import numpy as np
import os
import glob


def make_l_template(size=80, thickness=3, squares=5):
    """
    Create a synthetic L-shaped checker corner template.
    """
    cell = size // squares
    template = np.zeros((size, size), dtype=np.uint8)

    # vertical part of the L
    for r in range(squares):
        if r % 2 == 0:
            template[r * cell:(r + 1) * cell, 0:cell] = 255
        else:
            template[r * cell:(r + 1) * cell, cell:2 * cell] = 255

    # horizontal part of the L
    for c in range(squares):
        if c % 2 == 0:
            template[0:cell, c * cell:(c + 1) * cell] = 255
        else:
            template[cell:2 * cell, c * cell:(c + 1) * cell] = 255

    # small blur makes matching more tolerant
    template = cv2.GaussianBlur(template, (3, 3), 0)
    return template


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return rotated


def non_max_suppression_points(detections, min_dist=40):
    """
    Keep only strong matches that are far enough apart.
    detections = [(x, y, score, w, h), ...]
    """
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []

    for d in detections:
        x, y, score, w, h = d
        cx = x + w / 2
        cy = y + h / 2

        too_close = False
        for kd in kept:
            kx, ky, kscore, kw, kh = kd
            kcx = kx + kw / 2
            kcy = ky + kh / 2
            if np.hypot(cx - kcx, cy - kcy) < min_dist:
                too_close = True
                break

        if not too_close:
            kept.append(d)

    return kept


def detect_gate_corners_template(image_path, debug=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    vis = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White mask
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 110, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    detections = []

    base_template = make_l_template(size=80, squares=5)

    # Try several scales and rotations
    scales = [0.5, 0.7, 0.9, 1.1, 1.3]
    angles = [0, 90, 180, 270]

    for scale in scales:
        temp = cv2.resize(base_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        for angle in angles:
            rotated_temp = rotate_image(temp, angle)
            th, tw = rotated_temp.shape[:2]

            if th >= white_mask.shape[0] or tw >= white_mask.shape[1]:
                continue

            result = cv2.matchTemplate(white_mask, rotated_temp, cv2.TM_CCOEFF_NORMED)

            ys, xs = np.where(result > 0.55)

            for (x, y) in zip(xs, ys):
                score = result[y, x]
                detections.append((x, y, float(score), tw, th))

    if len(detections) == 0:
        return [], vis, white_mask

    detections = non_max_suppression_points(detections, min_dist=50)

    corners = []
    for x, y, score, w, h in detections:
        cx = x + w / 2
        cy = y + h / 2
        corners.append(np.array([cx, cy], dtype=np.float32))

        if debug:
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"{score:.2f}",
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    return corners, vis, white_mask


def process_folder_as_video(folder_path, delay=120):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []

    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No images found in folder: {folder_path}")

    for image_path in files:
        corners, vis, white_mask = detect_gate_corners_template(image_path, debug=True)

        cv2.putText(
            vis,
            os.path.basename(image_path),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f"Detected: {len(corners)}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("White mask", white_mask)
        cv2.imshow("Gate corner detection", vis)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "../MAV_Team_10/TEAM-10-PROTOTYPING/downloads from drone/20260306-095826/1316434405.jpg"
    corners, vis, white_mask = detect_gate_corners_template(image_path, debug=True)

    print("Detected corner markers:")
    for i, c in enumerate(corners):
        print(f"C{i}: {c}")

    cv2.imshow("White mask", white_mask)
    cv2.imshow("Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    folder_path = "../MAV_Team_10/TEAM-10-PROTOTYPING/downloads from drone/20260306-095826"
    process_folder_as_video(folder_path)