#!/usr/bin/env python3
"""
save_report_steps.py

Runs the FULL C pipeline from frame 0 through the target frame (default 902)
so the baseline accumulates correctly, then saves one image per processing
step for the report.

Usage:
    python3 save_report_steps.py <image_folder> [--target 902] [--out ./report_figures]

Output files (rotated to the natural "landscape" orientation):
    step0_original.png          — raw camera image (rotated)
    step1_green_mask.png        — after decision-tree + median blur
    step2_blob_isolation.png    — after CC + area + smooth-blob filter
    step3_hole_filling.png      — after border flood-fill hole filling
    step4_obstacle_detect.png   — boundary + baseline + obstacle boxes
    step5_plant_lax_green.png   — lax green mask (upsampled, before subtract)
    step5_plant_after_sub.png   — plant mask after ground subtraction
    step5_plant_detect.png      — plant boxes on original image
"""
import os, sys, argparse, ctypes, subprocess, re
import numpy as np
import cv2
from glob import glob

# ═══════════════════════════════════════════════════════════════════════════════
#  C-TYPES BINDINGS
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


def bgr_to_uyvy_bytes(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    h, w = bgr.shape[:2]
    uyvy = np.zeros((h, w, 2), dtype=np.uint8)
    uyvy[:, :, 1] = yuv[:, :, 0]
    uyvy[:, 0::2, 0] = yuv[:, 0::2, 1]
    uyvy[:, 1::2, 0] = yuv[:, 0::2, 2]
    return uyvy.tobytes()


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPILE & LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def compile_and_load(script_dir):
    """Compile the debug C file and return the loaded library."""
    lib_path = os.path.join(script_dir, 'libobstacle_debug.so')
    c_src    = os.path.join(script_dir, 'team10_get_obstacle_info_standalone_debug.c')
    h_src    = os.path.join(script_dir, 'team10_get_obstacle_info.h')

    # Check that source files exist
    if not os.path.exists(c_src):
        print(f"ERROR: {c_src} not found!")
        sys.exit(1)
    if not os.path.exists(h_src):
        print(f"ERROR: {h_src} not found!")
        sys.exit(1)

    # Always recompile to pick up changes
    if os.path.exists(lib_path):
        os.remove(lib_path)

    print(f"Compiling {c_src} ...")
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O3", "-lm",
         "-I", script_dir,           # so #include "team10_get_obstacle_info.h" works
         "-o", lib_path, c_src],
        check=True
    )
    print("Compilation OK.")

    lib = ctypes.CDLL(lib_path)

    # ── get_obstacle_info ──
    lib.get_obstacle_info.argtypes = [
        ctypes.POINTER(ImageT), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
        ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ObstacleRegionT), ctypes.POINTER(ObstacleRegionT), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    ]
    lib.get_obstacle_info.restype = ctypes.c_uint8

    # ── original getters ──
    lib.get_work_mask.restype   = ctypes.POINTER(ctypes.c_uint8)
    lib.get_plant_mask.restype  = ctypes.POINTER(ctypes.c_uint8)

    # ── debug getters ──
    lib.get_debug_after_green.restype      = ctypes.POINTER(ctypes.c_uint8)
    lib.get_debug_after_blob.restype       = ctypes.POINTER(ctypes.c_uint8)
    lib.get_debug_after_holes.restype      = ctypes.POINTER(ctypes.c_uint8)
    lib.get_debug_flipped.restype          = ctypes.POINTER(ctypes.c_uint8)
    lib.get_debug_plant_green_lax.restype  = ctypes.POINTER(ctypes.c_uint8)
    lib.get_debug_plant_after_sub.restype  = ctypes.POINTER(ctypes.c_uint8)
    lib.get_debug_w.restype  = ctypes.c_int
    lib.get_debug_h.restype  = ctypes.c_int
    lib.get_debug_pw.restype = ctypes.c_int
    lib.get_debug_ph.restype = ctypes.c_int

    return lib


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ONE FRAME THROUGH THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_frame(lib, bgr, ground_baseline_arr, baseline_inited, boundary_arr,
              obs_out, plants_out, plant_count_out, ground_found_out, green_frac_out):
    """Feed one BGR image through get_obstacle_info. Returns (num_obs, num_plants)."""
    tW, tH = bgr.shape[1], bgr.shape[0]
    uyvy_bytes = bgr_to_uyvy_bytes(bgr)

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
        ctypes.c_int(5),
        ctypes.c_int(20),
        obs_out, plants_out,
        ctypes.byref(plant_count_out),
        boundary_arr,
        ctypes.byref(ground_found_out),
        ctypes.byref(green_frac_out)
    )
    return num_obs, plant_count_out.value


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE DEBUG IMAGES FOR THE TARGET FRAME
# ═══════════════════════════════════════════════════════════════════════════════

def save_debug_images(lib, bgr, num_obs, num_plants,
                      obs_out, plants_out, boundary_arr, ground_baseline_arr,
                      baseline_inited, ground_found_out, green_frac_out,
                      out_dir):
    """Read all debug snapshot buffers and save report-ready images."""
    os.makedirs(out_dir, exist_ok=True)

    tW, tH = bgr.shape[1], bgr.shape[0]
    w = lib.get_debug_w()
    h = lib.get_debug_h()
    pw = lib.get_debug_pw()
    ph = lib.get_debug_ph()

    # ── Helper: read a C buffer as numpy, rotate to landscape ──
    def read_mask(getter, width, height):
        ptr = getter()
        return np.ctypeslib.as_array(ptr, shape=(height, width)).copy()

    def rot(img):
        """Rotate 90° CCW — same as the slideshow visualizer."""
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ── Step 0: Original image (rotated) ──
    img_rot = rot(bgr)
    cv2.imwrite(os.path.join(out_dir, "step0_original.png"), img_rot)
    print(f"  Saved step0_original.png  ({img_rot.shape[1]}x{img_rot.shape[0]})")

    # ── Step 1: Green color mask (§2.1.1) ──
    green_mask = read_mask(lib.get_debug_after_green, w, h)
    green_rot  = rot(green_mask)
    cv2.imwrite(os.path.join(out_dir, "step1_green_mask.png"), green_rot)
    # Also save an overlay version
    overlay = img_rot.copy()
    overlay[green_rot > 0] = [0, 255, 0]
    blended = cv2.addWeighted(img_rot, 0.5, overlay, 0.5, 0)
    cv2.imwrite(os.path.join(out_dir, "step1_green_mask_overlay.png"), blended)
    print(f"  Saved step1_green_mask.png + overlay")

    # ── Step 2: After blob isolation (§2.1.2) ──
    blob_mask = read_mask(lib.get_debug_after_blob, w, h)
    blob_rot  = rot(blob_mask)
    cv2.imwrite(os.path.join(out_dir, "step2_blob_isolation.png"), blob_rot)
    overlay2 = img_rot.copy()
    overlay2[blob_rot > 0] = [0, 200, 0]
    blended2 = cv2.addWeighted(img_rot, 0.5, overlay2, 0.5, 0)
    cv2.imwrite(os.path.join(out_dir, "step2_blob_isolation_overlay.png"), blended2)
    print(f"  Saved step2_blob_isolation.png + overlay")

    # ── Step 3: After hole filling (§2.1.3) ──
    holes_mask = read_mask(lib.get_debug_after_holes, w, h)
    holes_rot  = rot(holes_mask)
    cv2.imwrite(os.path.join(out_dir, "step3_hole_filling.png"), holes_rot)
    overlay3 = img_rot.copy()
    overlay3[holes_rot > 0] = [0, 200, 0]
    blended3 = cv2.addWeighted(img_rot, 0.5, overlay3, 0.5, 0)
    cv2.imwrite(os.path.join(out_dir, "step3_hole_filling_overlay.png"), blended3)
    print(f"  Saved step3_hole_filling.png + overlay")

    # ── Step 4: Obstacle detection (§2.1.4) ──
    #    Draw boundary, baseline, and obstacle boxes on the rotated image
    h_rot = img_rot.shape[0]  # 520 after rotation
    w_rot = img_rot.shape[1]  # 240 after rotation
    det_vis = img_rot.copy()

    boundary = np.array([boundary_arr[i] for i in range(tH)])
    py_baseline = np.array([ground_baseline_arr[i] for i in range(tH)]) if baseline_inited.value else None

    # Draw baseline (blue)
    if py_baseline is not None:
        valid = py_baseline < h_rot
        rv = np.where(valid)[0]
        if len(rv) > 1:
            pts = np.array([[int(r), int(py_baseline[r])] for r in rv], np.int32)
            cv2.polylines(det_vis, [pts], False, (255, 0, 0), 1)

    # Draw boundary (yellow)
    valid = boundary < h_rot
    rv = np.where(valid)[0]
    if len(rv) > 1:
        pts = np.array([[int(r), int(boundary[r])] for r in rv], np.int32)
        cv2.polylines(det_vis, [pts], False, (0, 255, 255), 1)

    # Draw obstacle boxes (red)
    for i in range(num_obs):
        s = int(obs_out[i].start)
        wd = int(obs_out[i].width)
        bh = int(obs_out[i].baseline_height)
        e = s + wd - 1
        cv2.rectangle(det_vis, (s, 0), (e, h_rot - 1), (0, 0, 255), 2)
        cv2.putText(det_vis, f"w={wd}", (s, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(det_vis, f"h={bh}", (s, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 255), 1)
        y_line = min(bh, h_rot - 1)
        cv2.line(det_vis, (s, y_line), (e, y_line), (255, 255, 255), 1)

    cv2.imwrite(os.path.join(out_dir, "step4_obstacle_detect.png"), det_vis)
    print(f"  Saved step4_obstacle_detect.png  ({num_obs} obstacles)")

    # Also save boundary-only overlay on the hole-filled mask for clarity
    boundary_vis = cv2.cvtColor(holes_rot, cv2.COLOR_GRAY2BGR)
    if len(rv) > 1:
        pts_b = np.array([[int(r), int(boundary[r])] for r in np.where(boundary < h_rot)[0]], np.int32)
        cv2.polylines(boundary_vis, [pts_b], False, (0, 255, 255), 2)
    if py_baseline is not None:
        valid_bl = py_baseline < h_rot
        rv_bl = np.where(valid_bl)[0]
        if len(rv_bl) > 1:
            pts_bl = np.array([[int(r), int(py_baseline[r])] for r in rv_bl], np.int32)
            cv2.polylines(boundary_vis, [pts_bl], False, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "step4_boundary_on_mask.png"), boundary_vis)
    print(f"  Saved step4_boundary_on_mask.png")

    # ── Step 5: Plant detection (§2.1.5) ──
    # 5a: lax green mask (before ground subtraction), upsampled to full res
    if pw > 0 and ph > 0:
        plant_lax = read_mask(lib.get_debug_plant_green_lax, pw, ph)
        plant_lax_full = cv2.resize(plant_lax, (w, h), interpolation=cv2.INTER_NEAREST)
        plant_lax_rot = rot(plant_lax_full)
        cv2.imwrite(os.path.join(out_dir, "step5_plant_lax_green.png"), plant_lax_rot)
        overlay5a = img_rot.copy()
        overlay5a[plant_lax_rot > 0] = [0, 255, 0]
        blended5a = cv2.addWeighted(img_rot, 0.5, overlay5a, 0.5, 0)
        cv2.imwrite(os.path.join(out_dir, "step5_plant_lax_green_overlay.png"), blended5a)
        print(f"  Saved step5_plant_lax_green.png + overlay  (small={pw}x{ph})")

        # 5b: after ground subtraction
        plant_sub = read_mask(lib.get_debug_plant_after_sub, pw, ph)
        plant_sub_full = cv2.resize(plant_sub, (w, h), interpolation=cv2.INTER_NEAREST)
        plant_sub_rot = rot(plant_sub_full)
        cv2.imwrite(os.path.join(out_dir, "step5_plant_after_sub.png"), plant_sub_rot)
        print(f"  Saved step5_plant_after_sub.png")

    # 5c: plant boxes on original
    plant_vis = img_rot.copy()
    plant_mask_ptr = lib.get_plant_mask()
    plant_mask_full_np = np.ctypeslib.as_array(plant_mask_ptr, shape=(h, w)).copy()
    plant_rot = rot(plant_mask_full_np)
    plant_vis[plant_rot > 0] = [0, 200, 0]
    for i in range(num_plants):
        s = int(plants_out[i].start)
        wd = int(plants_out[i].width)
        e = s + wd - 1
        cv2.rectangle(plant_vis, (s, 0), (e, h_rot - 1), (0, 255, 0), 2)
        cv2.putText(plant_vis, f"p={wd}", (s, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, "step5_plant_detect.png"), plant_vis)
    print(f"  Saved step5_plant_detect.png  ({num_plants} plants)")

    # ── Combined final: all detections ──
    final = img_rot.copy()
    final[plant_rot > 0] = [0, 180, 0]
    if py_baseline is not None:
        valid_bl = py_baseline < h_rot
        rv_bl = np.where(valid_bl)[0]
        if len(rv_bl) > 1:
            pts = np.array([[int(r), int(py_baseline[r])] for r in rv_bl], np.int32)
            cv2.polylines(final, [pts], False, (255, 0, 0), 1)
    valid_bd = boundary < h_rot
    rv_bd = np.where(valid_bd)[0]
    if len(rv_bd) > 1:
        pts = np.array([[int(r), int(boundary[r])] for r in rv_bd], np.int32)
        cv2.polylines(final, [pts], False, (0, 255, 255), 1)
    for i in range(num_obs):
        s = int(obs_out[i].start)
        wd = int(obs_out[i].width)
        bh = int(obs_out[i].baseline_height)
        e = s + wd - 1
        cv2.rectangle(final, (s, 0), (e, h_rot - 1), (0, 0, 255), 2)
        cv2.putText(final, f"w={wd}", (s, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(final, f"h={bh}", (s, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 255), 1)
    for i in range(num_plants):
        s = int(plants_out[i].start)
        wd = int(plants_out[i].width)
        e = s + wd - 1
        cv2.rectangle(final, (s, 0), (e, h_rot - 1), (0, 255, 0), 2)
        cv2.putText(final, f"p={wd}", (s, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, "step_final_combined.png"), final)
    print(f"  Saved step_final_combined.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Run the full C pipeline up to a target frame and save "
                    "one image per processing step for the report.")
    p.add_argument("folder", help="Image folder (same as visualize_slideshow)")
    p.add_argument("--target", type=int, default=902,
                   help="Target image number to save debug steps for (default: 902)")
    p.add_argument("--out", type=str, default="./report_figures",
                   help="Output directory for saved images")
    p.add_argument("--start", type=int, default=0,
                   help="Starting frame index (usually 0)")
    a = p.parse_args()

    # ── Discover images ──
    paths = glob(os.path.join(a.folder, "*.jpg"))
    paths.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    if not paths:
        print("No images found in", a.folder)
        sys.exit(1)

    # ── Find target ──
    target_name = f"{a.target}.jpg"
    target_idx = None
    for i, p_path in enumerate(paths):
        if os.path.basename(p_path) == target_name:
            target_idx = i
            break
    if target_idx is None:
        print(f"ERROR: {target_name} not found in {a.folder}")
        print(f"  Available range: {os.path.basename(paths[0])} .. {os.path.basename(paths[-1])}")
        sys.exit(1)

    print(f"Target: {target_name} (index {target_idx} of {len(paths)} images)")
    print(f"Will process frames 0..{target_idx} to build baseline, then save debug images.\n")

    # ── Compile & load ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib = compile_and_load(script_dir)

    # ── Persistent C state (same as visualize_slideshow) ──
    MAX_IMAGE_HEIGHT = 520
    max_obs = 20
    baseline_inited  = ctypes.c_int(0)
    ground_baseline_arr = (ctypes.c_float * MAX_IMAGE_HEIGHT)()
    boundary_arr     = (ctypes.c_int * MAX_IMAGE_HEIGHT)()
    obs_out          = (ObstacleRegionT * max_obs)()
    plants_out       = (ObstacleRegionT * max_obs)()
    plant_count_out  = ctypes.c_uint8(0)
    ground_found_out = ctypes.c_int(0)
    green_frac_out   = ctypes.c_float(0.0)

    # ── Process every frame up to and including the target ──
    for idx in range(a.start, target_idx + 1):
        bgr = cv2.imread(paths[idx])
        if bgr is None:
            continue

        num_obs, num_plants = run_frame(
            lib, bgr, ground_baseline_arr, baseline_inited, boundary_arr,
            obs_out, plants_out, plant_count_out, ground_found_out, green_frac_out
        )

        is_ground = ground_found_out.value == 1
        st = "GROUND" if is_ground else "NO_GND"

        # Print progress every 100 frames + always the last few
        if idx % 100 == 0 or idx >= target_idx - 2:
            name = os.path.basename(paths[idx])
            print(f"  [{idx:4d}/{target_idx}] {name}  green={green_frac_out.value:.3f} {st}  "
                  f"{num_obs} obs  {num_plants} plt"
                  f"  {'<-- TARGET' if idx == target_idx else ''}")

    # ── Save debug images for the target frame ──
    print(f"\n{'='*60}")
    print(f"  SAVING DEBUG IMAGES FOR {target_name}")
    print(f"{'='*60}")

    target_bgr = cv2.imread(paths[target_idx])
    save_debug_images(
        lib, target_bgr, num_obs, num_plants,
        obs_out, plants_out, boundary_arr, ground_baseline_arr,
        baseline_inited, ground_found_out, green_frac_out,
        a.out
    )

    print(f"\nDone! All report figures saved to: {os.path.abspath(a.out)}/")
    print(f"\nMapping to report sections:")
    print(f"  §2.1.1 Green Color Masking    → step1_green_mask.png / step1_green_mask_overlay.png")
    print(f"  §2.1.2 Isolating Ground Blob  → step2_blob_isolation.png / step2_blob_isolation_overlay.png")
    print(f"  §2.1.3 Filling Holes          → step3_hole_filling.png / step3_hole_filling_overlay.png")
    print(f"  §2.1.4 Obstacle Detection     → step4_obstacle_detect.png / step4_boundary_on_mask.png")
    print(f"  §2.1.5 Plant Detection        → step5_plant_lax_green.png / step5_plant_detect.png")


if __name__ == "__main__":
    main()
