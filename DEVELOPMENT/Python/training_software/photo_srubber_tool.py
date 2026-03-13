"""
Photo Scrubber - Mark useless frame ranges and export useful photos

PLAYBACK:
  - Hold LEFT MOUSE on image   : play forward
  - Hold RIGHT MOUSE on image  : play backward
  - LEFT ARROW / A             : step back 1 frame
  - RIGHT ARROW / D            : step forward 1 frame
  - Click/drag on timeline bar : jump to that frame

MARKING:
  - U         : mark current frame as start of USELESS section
  - G         : mark current frame as start of GOOD section again (closes useless range)
  - BACKSPACE : undo last range

ROTATION:
  - R         : rotate all frames 90 degrees clockwise
  - E         : rotate all frames 90 degrees counter-clockwise
  (rotation is applied to the saved output, not the original files)

SAVING:
  - S  : save (dialogs to choose where to save both JSONs)
  - Q  : quit without saving
"""

import os
import json
import cv2
import tkinter as tk
from tkinter import filedialog

# ── Folder Picker ─────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
print("Select the folder containing your photos...")
folder_name = filedialog.askdirectory(title="Select Photo Folder")
root.destroy()

if not folder_name:
    print("No folder selected, exiting.")
    exit()

print(f"Selected folder: {folder_name}")

# ── Settings ──────────────────────────────────────────────────────────────────
playback_fps   = 30
display_width  = 1280
display_height = 720

# ── Load Photos ───────────────────────────────────────────────────────────────
photos = sorted([
    os.path.join(folder_name, f) for f in os.listdir(folder_name)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
])
total = len(photos)

if total == 0:
    print("No photos found in selected folder, exiting.")
    exit()

print(f"Loaded {total} photos")

# ── Load Existing Ranges ──────────────────────────────────────────────────────
auto_ranges_file = os.path.join(os.path.dirname(folder_name), "useless_ranges.json")
if os.path.exists(auto_ranges_file):
    with open(auto_ranges_file, 'r') as f:
        all_ranges_data = json.load(f)
    useless_ranges = [list(r) for r in all_ranges_data.get(folder_name, [])]
    print(f"Loaded {len(useless_ranges)} existing useless ranges")
else:
    all_ranges_data = {}
    useless_ranges = []

# ── Rotation state (multiples of 90, stored as cv2 rotate codes) ─────────────
# rotation_steps: 0=0°, 1=90°CW, 2=180°, 3=270°CW
rotation_steps = 0

def apply_rotation(img, steps):
    steps = steps % 4
    if steps == 0:
        return img
    elif steps == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif steps == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif steps == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# ── Timeline Geometry ─────────────────────────────────────────────────────────
BAR_Y  = display_height - 40
BAR_H  = 16
BAR_X0 = 40
BAR_X1 = display_width - 40
BAR_W  = BAR_X1 - BAR_X0

# ── Mouse State ───────────────────────────────────────────────────────────────
mouse_left  = False
mouse_right = False
scrubbing   = False
current_idx = 0

def is_on_timeline(y):
    return BAR_Y - 6 <= y <= BAR_Y + BAR_H + 6

def x_to_frame(x):
    frac = (x - BAR_X0) / BAR_W
    return int(max(0, min(1, frac)) * (total - 1))

def mouse_callback(event, x, y, flags, param):
    global mouse_left, mouse_right, scrubbing, current_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_on_timeline(y):
            scrubbing = True
            current_idx = x_to_frame(x)
        else:
            mouse_left = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_left = False
        scrubbing  = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        if not is_on_timeline(y):
            mouse_right = True
    elif event == cv2.EVENT_RBUTTONUP:
        mouse_right = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if scrubbing:
            current_idx = x_to_frame(x)

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_useless(idx):
    for (s, e) in useless_ranges:
        if s <= idx <= e:
            return True
    return False

def get_useful_photos(photos, useless_ranges):
    useless_indices = set()
    for (s, e) in useless_ranges:
        useless_indices.update(range(s, e + 1))
    return [p for i, p in enumerate(photos) if i not in useless_indices]

def rotation_label(steps):
    return f"R: {(steps % 4) * 90}°"

def draw_ui(img, idx, useless_mark_start, useless_ranges, rotation_steps):
    overlay = img.copy()
    h, w = img.shape[:2]

    # timeline background
    cv2.rectangle(overlay, (BAR_X0 - 2, BAR_Y - 2), (BAR_X1 + 2, BAR_Y + BAR_H + 2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (BAR_X0, BAR_Y), (BAR_X1, BAR_Y + BAR_H), (70, 70, 70), -1)

    # red zones for saved useless ranges
    for (s, e) in useless_ranges:
        rx0 = BAR_X0 + int(s / max(total - 1, 1) * BAR_W)
        rx1 = BAR_X0 + int(e / max(total - 1, 1) * BAR_W)
        cv2.rectangle(overlay, (rx0, BAR_Y), (max(rx1, rx0+2), BAR_Y + BAR_H), (0, 0, 210), -1)

    # yellow zone for open useless mark
    if useless_mark_start is not None:
        rx0 = BAR_X0 + int(useless_mark_start / max(total - 1, 1) * BAR_W)
        rx1 = BAR_X0 + int(idx / max(total - 1, 1) * BAR_W)
        if rx0 > rx1: rx0, rx1 = rx1, rx0
        cv2.rectangle(overlay, (rx0, BAR_Y), (max(rx1, rx0+2), BAR_Y + BAR_H), (0, 200, 200), -1)

    # playhead
    px = BAR_X0 + int(idx / max(total - 1, 1) * BAR_W)
    cv2.rectangle(overlay, (px - 2, BAR_Y - 5), (px + 2, BAR_Y + BAR_H + 5), (255, 255, 255), -1)

    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    # top status bar
    frame_state  = "[ USELESS ]" if is_useless(idx) else "[  GOOD   ]"
    color        = (0, 60, 200) if is_useless(idx) else (0, 160, 50)
    useful_count = len(get_useful_photos(photos, useless_ranges))
    folder_short = "..." + folder_name[-35:] if len(folder_name) > 38 else folder_name

    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)

    # useless/good badge
    cv2.rectangle(img, (8, 4), (130, 28), color, -1)
    cv2.putText(img, frame_state, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # rotation badge
    rot_label = rotation_label(rotation_steps)
    cv2.rectangle(img, (138, 4), (218, 28), (60, 60, 60), -1)
    cv2.putText(img, rot_label, (143, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 1, cv2.LINE_AA)

    status = f"  Frame {idx+1}/{total}   |   Useful: {useful_count}/{total}   |   {folder_short}"
    if useless_mark_start is not None:
        status = f"  Frame {idx+1}/{total}   |   Useful: {useful_count}/{total}   |   useless from {useless_mark_start+1} ... press G when good again"
    cv2.putText(img, status, (225, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)

    # bottom hint bar
    cv2.rectangle(img, (0, h - 64), (w, h - 46), (0, 0, 0), -1)
    hint = "LMB: fwd  |  RMB: back  |  ARROWS: step  |  DRAG TIMELINE: jump  |  U: useless  |  G: good  |  BKSP: undo  |  R: rotate CW  |  E: rotate CCW  |  S: save  |  Q: quit"
    cv2.putText(img, hint, (8, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (160, 160, 160), 1, cv2.LINE_AA)

    return img

# ── Main Loop ─────────────────────────────────────────────────────────────────
cv2.namedWindow("Photo Scrubber", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Photo Scrubber", display_width, display_height)
cv2.setMouseCallback("Photo Scrubber", mouse_callback)

def letterbox(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = __import__('numpy').zeros((target_h, target_w, 3), dtype=resized.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

useless_mark_start = None
delay = int(1000 / playback_fps)

while True:
    img = cv2.imread(photos[current_idx])
    img = apply_rotation(img, rotation_steps)
    img = letterbox(img, display_width, display_height)
    img = draw_ui(img, current_idx, useless_mark_start, useless_ranges, rotation_steps)
    cv2.imshow("Photo Scrubber", img)

    if scrubbing:
        key = cv2.waitKey(16)
    elif mouse_left:
        current_idx = min(current_idx + 1, total - 1)
        key = cv2.waitKey(delay)
    elif mouse_right:
        current_idx = max(current_idx - 1, 0)
        key = cv2.waitKey(delay)
    else:
        key = cv2.waitKey(30)

    if key == -1:
        continue
    key = key & 0xFF

    # step
    if   key == 83 or key == ord('d'): current_idx = min(current_idx + 1, total - 1)
    elif key == 81 or key == ord('a'): current_idx = max(current_idx - 1, 0)

    # R = rotate clockwise
    elif key == ord('r') or key == ord('R'):
        rotation_steps = (rotation_steps + 1) % 4
        print(f"Rotation: {rotation_steps * 90}° clockwise")

    # E = rotate counter-clockwise
    elif key == ord('e') or key == ord('E'):
        rotation_steps = (rotation_steps - 1) % 4
        print(f"Rotation: {rotation_steps * 90}° clockwise")

    # U = start useless section
    elif key == ord('u') or key == ord('U'):
        if useless_mark_start is None:
            useless_mark_start = current_idx
            print(f"Useless from frame {useless_mark_start+1} ...")
        else:
            print("Already marking — press G to close the range first")

    # G = good again, close range
    elif key == ord('g') or key == ord('G'):
        if useless_mark_start is not None:
            s, e = sorted([useless_mark_start, current_idx])
            useless_ranges.append([s, e])
            print(f"Marked useless: frames {s+1} to {e+1}")
            useless_mark_start = None
        else:
            print("Not currently marking anything")

    # backspace = undo
    elif key == 8:
        if useless_mark_start is not None:
            useless_mark_start = None
            print("Cancelled current mark")
        elif useless_ranges:
            removed = useless_ranges.pop()
            print(f"Undid range: frames {removed[0]+1} to {removed[1]+1}")

    # S = save with file dialogs
    elif key == ord('s') or key == ord('S'):
        cv2.destroyAllWindows()

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # dialog 1: save useful photos
        useful_save_path = filedialog.asksaveasfilename(
            title="Save useful photos JSON as...",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="useful_photos.json",
            initialdir=os.path.dirname(folder_name)
        )

        # dialog 2: save useless ranges (optional)
        ranges_save_path = filedialog.asksaveasfilename(
            title="Save useless ranges JSON as... (cancel to skip)",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="useless_ranges.json",
            initialdir=os.path.dirname(folder_name)
        )

        root.destroy()

        # save useful photos
        if useful_save_path:
            useful = get_useful_photos(photos, useless_ranges)
            if os.path.exists(useful_save_path):
                with open(useful_save_path, 'r') as f:
                    useful_data = json.load(f)
            else:
                useful_data = {}
            useful_data[folder_name] = {
                "photos": useful,
                "rotation": rotation_steps * 90
            }
            with open(useful_save_path, 'w') as f:
                json.dump(useful_data, f, indent=4)
            print(f"Saved {len(useful)} useful photos (rotation: {rotation_steps*90}°) to {useful_save_path}")
        else:
            print("Useful photos save cancelled")

        # save useless ranges (optional)
        if ranges_save_path:
            if os.path.exists(ranges_save_path):
                with open(ranges_save_path, 'r') as f:
                    all_ranges_data = json.load(f)
            else:
                all_ranges_data = {}
            all_ranges_data[folder_name] = useless_ranges
            with open(ranges_save_path, 'w') as f:
                json.dump(all_ranges_data, f, indent=4)
            print(f"Saved {len(useless_ranges)} useless ranges to {ranges_save_path}")
        else:
            print("Useless ranges save skipped")

        break

    # Q = quit
    elif key == ord('q') or key == ord('Q'):
        print("Quit without saving")
        break

cv2.destroyAllWindows()