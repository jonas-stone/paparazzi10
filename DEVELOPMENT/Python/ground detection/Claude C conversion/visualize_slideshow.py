#!/usr/bin/env python3
"""
visualize_slideshow.py

Continuous slideshow: processes every image in a folder through the
COMPILED C PIPELINE and displays detections in real-time.
Includes auto-compilation and VS Code hardcoded arguments.
"""
import os
import sys
import argparse
import ctypes
import subprocess
import numpy as np
import cv2
import re
from glob import glob

# ═══════════════════════════════════════════════════════════════════════════════
#  OVERRIDE FOR VS CODE RUN BUTTON (HARDCODED ARGS)
# ═══════════════════════════════════════════════════════════════════════════════
if len(sys.argv) == 1:
    sys.argv = [
        "visualize_slideshow.py",
        "/home/jonas/paparazzi/DEVELOPMENT/downloads from drone/20260320/",
        "--python-dir", "/home/jonas/paparazzi/DEVELOPMENT/Python/ground detection/",
        "--start", "1",
        "--delay", "50"
    ]
# "/home/jonas/paparazzi/DEVELOPMENT/downloads from drone/20260313-100130/",
# "/home/jonas/paparazzi/DEVELOPMENT/downloads from drone/20260306-095826/",
# ═══════════════════════════════════════════════════════════════════════════════
#  C-TYPES BINDINGS & AUTO-COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

class FloatEulers(ctypes.Structure):
    _fields_ = [("phi", ctypes.c_float), ("theta", ctypes.c_float), ("psi", ctypes.c_float)]

class ImageT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("w", ctypes.c_uint16),
        ("h", ctypes.c_uint16),
        ("ts", ctypes.c_byte * 16),
        ("eulers", FloatEulers),
        ("pprz_ts", ctypes.c_uint32),
        ("buf_idx", ctypes.c_uint8),
        ("buf_size", ctypes.c_uint32),
        ("buf", ctypes.c_void_p)
    ]

class ObstacleRegionT(ctypes.Structure):
    _fields_ = [("start", ctypes.c_uint16), ("width", ctypes.c_uint16), ("baseline_height", ctypes.c_uint16)]

# Setup paths for auto-compilation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(SCRIPT_DIR, 'libobstacle.so')
c_wrapper_path = os.path.join(SCRIPT_DIR, 'c_wrapper.c')

if os.path.exists(lib_path):
    print(f"🗑️  Found old C binary, deleting to force fresh recompile...")
    try:
        os.remove(lib_path)
    except OSError as e:
        print(f"⚠️  Warning: Could not delete {lib_path}. Close any other scripts using it. ({e})")
# Auto-compile if the library doesn't exist
if not os.path.exists(lib_path):
    print(f"⚙️  {lib_path} not found! Compiling automatically...")
    try:
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O3", "-o", lib_path, c_wrapper_path],
            check=True
        )
        print("✅ Compilation successful!")
    except subprocess.CalledProcessError:
        print(f"❌ ERROR: Compilation failed. Please check {c_wrapper_path} for errors.")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ ERROR: 'gcc' not found. Please ensure build-essential is installed.")
        sys.exit(1)

# Load the library
try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"ERROR: Could not load {lib_path}. Details: {e}")
    sys.exit(1)

lib.get_obstacle_info.argtypes = [
    ctypes.POINTER(ImageT), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ObstacleRegionT), ctypes.POINTER(ObstacleRegionT), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
]
lib.get_obstacle_info.restype = ctypes.c_uint8

lib.get_work_mask.restype = ctypes.POINTER(ctypes.c_uint8)
lib.get_plant_mask.restype = ctypes.POINTER(ctypes.c_uint8)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS & DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def bgr_to_uyvy_bytes(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    h, w = bgr.shape[:2]
    uyvy = np.zeros((h, w, 2), dtype=np.uint8)
    uyvy[:, :, 1] = yuv[:, :, 0]       
    uyvy[:, 0::2, 0] = yuv[:, 0::2, 1] 
    uyvy[:, 1::2, 0] = yuv[:, 0::2, 2] 
    return uyvy.tobytes()

def draw_original_style(img_rot, obs_regions_raw, plant_regions_raw,
                        boundary_rows, ground_baseline, label):
    vis = img_rot.copy()
    h_rot = vis.shape[0] # This is 240 (original image row-wise, rotated width)
    w_rot = vis.shape[1] # This is 520 (original image column-wise, rotated height)

    if ground_baseline is not None:
        valid = ground_baseline < h_rot
        rv = np.where(valid)[0]
        if len(rv) > 1:
            # Drawing on the rotated image, so x=column index (r), y=ground_baseline[r]
            pts = np.array([[int(r), int(ground_baseline[r])] for r in rv], np.int32)
            cv2.polylines(vis, [pts], False, (255, 0, 0), 1)

    if boundary_rows is not None:
        valid = boundary_rows < h_rot
        rv = np.where(valid)[0]
        if len(rv) > 1:
            pts = np.array([[int(r), int(boundary_rows[r])] for r in rv], np.int32)
            cv2.polylines(vis, [pts], False, (255, 255, 0), 1)

    # Draw Obstacles exactly as C outputs them
    for region in obs_regions_raw:
        s, e, w, bh = int(region[0]), int(region[1]), int(region[2]), int(region[3])
        if boundary_rows is not None:
            b_slice = boundary_rows[s:e+1]
            vb = b_slice[b_slice < h_rot]
            if len(vb) > 0:
                yt, yb = int(np.min(vb)), int(np.max(vb)) + 15
            else:
                yt, yb = 0, h_rot - 1
        else:
            yt, yb = 0, h_rot - 1
            
        ov = vis.copy()
        cv2.rectangle(vis, (s, 0), (e, h_rot-1), (0, 0, 255), 2)
        cv2.putText(vis, f"w={w}", (s, max(15, yt-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        # Original text display
        # cv2.putText(vis, f"h={bh}", (s, max(30, yt-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,150,255), 1)

        # Updated display for h=bh and the white horizontal line
        y_text = max(30, yt-20)
        cv2.putText(vis, f"h={bh}", (s, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,150,255), 1)
        
        # Calculate the y-coordinate for the line, slightly below the text baseline
        y_line = min(bh, h_rot - 1)
        cv2.line(vis, (s, y_line), (e, y_line), (255, 255, 255), 1)

    for region in plant_regions_raw:
        s, e, w, bh = int(region[0]), int(region[1]), int(region[2]), int(region[3])
        cv2.rectangle(vis, (s, 0), (e, h_rot-1), (0, 255, 0), 2)
        cv2.putText(vis, f"p={w}", (s, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    cv2.putText(vis, label, (5, h_rot-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(vis, label, (5, h_rot-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    return vis

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Continuous slideshow driven by C pipeline")
    p.add_argument("folder")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--delay", type=int, default=50, help="ms between frames (default 50)")
    p.add_argument("--python-dir", type=str, help="Path to Python directory") # Added to prevent crash 
    a = p.parse_args()

    import re
    paths = glob(os.path.join(a.folder, "*.jpg"))
    paths.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    if not paths: 
        print("No images found in", a.folder)
        sys.exit(1)
    paths = paths[a.start:]
    
    # Matching the hardcoded maximums in your C header
    MAX_IMAGE_WIDTH = 240
    MAX_IMAGE_HEIGHT = 520

    # Removed scaling - using raw parameters
    min_width_c = 20
    median_ksize_c = 5

    WIN = "C Pipeline Slideshow (q=quit, SPACE=pause)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    idx = 0
    paused = False
    needs_processing = True
    last_combined = None

    # ── PERSISTENT C STATE ──
    baseline_inited = ctypes.c_int(0)
    
    max_obs = 20
    obs_out = (ObstacleRegionT * max_obs)()
    plants_out = (ObstacleRegionT * max_obs)()
    plant_count_out = ctypes.c_uint8(0)
    ground_found_out = ctypes.c_int(0)
    green_frac_out = ctypes.c_float(0.0)

    # Allocate baseline arrays to exactly 520, guaranteeing we never overflow C
    ground_baseline_arr = (ctypes.c_float * MAX_IMAGE_HEIGHT)()
    boundary_arr = (ctypes.c_int * MAX_IMAGE_HEIGHT)()

    while idx < len(paths):
        ip = paths[idx]
        name = os.path.basename(ip)

        if needs_processing:
            bgr = cv2.imread(ip)
            if bgr is None:
                idx += 1; continue

            H, W = bgr.shape[:2]
            
            # Clamp target dimensions so they never exceed C's buffers
            tW = max(1, int(W))#max(1, int(W * 0.8))
            tH = max(1, int(H))#max(1, int(H * 0.8))
            
            bgr_s = cv2.resize(bgr, (tW, tH), interpolation=cv2.INTER_AREA)
            uyvy_bytes = bgr_to_uyvy_bytes(bgr_s)

            img_t = ImageT()
            img_t.type = 0 
            img_t.w = tW
            img_t.h = tH
            img_t.buf = ctypes.cast(uyvy_bytes, ctypes.c_void_p)

            num_obs = lib.get_obstacle_info(
                ctypes.byref(img_t),
                ground_baseline_arr,
                ctypes.byref(baseline_inited),
                ctypes.c_float(0.05),
                ctypes.c_int(median_ksize_c),
                ctypes.c_int(min_width_c),
                obs_out,
                plants_out,
                ctypes.byref(plant_count_out),
                boundary_arr,
                ctypes.byref(ground_found_out),
                ctypes.byref(green_frac_out)
            )

            num_plants = plant_count_out.value
            is_ground = ground_found_out.value == 1
            st = "GROUND FOUND" if is_ground else "NO GROUND"

            obs_raw = []
            for i in range(num_obs):
                s, w, bh = obs_out[i].start, obs_out[i].width, obs_out[i].baseline_height
                obs_raw.append((s, s + w - 1, w, bh))

            plt_raw = []
            for i in range(num_plants):
                s, w = plants_out[i].start, plants_out[i].width
                plt_raw.append((s, s + w - 1, w, 0))
            
            # Safe read from C buffers
            mask_ptr = lib.get_work_mask()
            clean_mask = np.ctypeslib.as_array(mask_ptr, shape=(tH, tW)).copy()
            
            pmask_ptr = lib.get_plant_mask()
            plant_mask = np.ctypeslib.as_array(pmask_ptr, shape=(tH, tW)).copy()

            # Read up to tH (520) because the arrays represent rows of the rotated image
            boundary = np.array([boundary_arr[i] for i in range(tH)])
            py_baseline = np.array([ground_baseline_arr[i] for i in range(tH)]) if baseline_inited.value else None

            print(f"  [{idx:4d}/{len(paths)}] {name}  green={green_frac_out.value:.3f} {st}  {num_obs} obs  {num_plants} plt")

            img_rot = cv2.rotate(bgr_s, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_rot = cv2.rotate(clean_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            plant_rot = cv2.rotate(plant_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

            det_vis = draw_original_style(img_rot, obs_raw, plt_raw,
                                          boundary, py_baseline,
                                          f"{num_obs} obs, {num_plants} plt")

            mask_bgr = cv2.cvtColor(mask_rot, cv2.COLOR_GRAY2BGR)
            po = img_rot.copy(); po[plant_rot > 0] = [0, 200, 0]

            top = np.hstack([img_rot, mask_bgr, po])

            det_h, det_w = det_vis.shape[:2]
            target_w = top.shape[1]
            info_panel = np.zeros((det_h, target_w - det_w, 3), np.uint8) if det_w < target_w else None

            if info_panel is not None:
                y = 25
                for line in [
                    f"Frame {idx}/{len(paths)}", f"{name}", f"",
                    f"Status: {st}", f"Green:  {green_frac_out.value:.3f}", f"",
                    f"Obstacles: {num_obs}",
                ] + [f"  [{i}] row={int(r[0])}-{int(r[1])} w={int(r[2])} h={int(r[3])}" for i, r in enumerate(obs_raw)] + [
                    f"", f"Plants: {num_plants}",
                ] + [f"  [{i}] row={int(r[0])}-{int(r[1])} w={int(r[2])}" for i, r in enumerate(plt_raw)] + [
                    f"", f"{'PAUSED' if paused else 'PLAYING'}  delay={a.delay}ms",
                ]:
                    cv2.putText(info_panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y += 18
                bot = np.hstack([det_vis, info_panel])
            else:
                bot = det_vis[:, :target_w, :]

            hdr = np.zeros((30, target_w, 3), np.uint8)
            cv2.putText(hdr, f"C PIPELINE | {name} ({tW}x{tH})  {st}  green={green_frac_out.value:.3f}  "
                        f"obs={num_obs} plt={num_plants}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            combined = np.vstack([hdr, top, bot]) 
            last_combined = combined
            needs_processing = False

        if last_combined is not None:
            cv2.imshow(WIN, last_combined)

        key = cv2.waitKey(a.delay if not paused else 30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                idx += 1
                needs_processing = True
        elif key == ord('d'):
            idx += 1
            needs_processing = True
            if not paused: paused = False
        elif key == ord('a'):
            if idx > 0:
                idx -= 1
                needs_processing = True
        elif not paused:
            idx += 1
            needs_processing = True

    cv2.destroyAllWindows()
    print(f"\nDone. Processed {idx} frames using C backend.")

if __name__ == "__main__":
    main()