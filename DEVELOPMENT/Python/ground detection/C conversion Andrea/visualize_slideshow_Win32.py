#!/usr/bin/env python3
"""
visualize_slideshow.py

Continuous slideshow: processes every image in a folder through the
COMPILED C PIPELINE and displays detections in real-time.
Cross-platform: works on Windows (.dll) and Linux/macOS (.so).
"""
import os
import sys
import platform
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
        "../paparazzi10/DEVELOPMENT/downloads from drone/20260320/",
        "--python-dir", "../paparazzi10/DEVELOPMENT/Python/ground detection/",
        "--start", "1",
        "--delay", "50"
    ]

# ═══════════════════════════════════════════════════════════════════════════════
#  C BUFFER LIMITS — must match the #defines in team10_get_obstacle_info.h
# ═══════════════════════════════════════════════════════════════════════════════
MAX_IMAGE_WIDTH  = 240
MAX_IMAGE_HEIGHT = 520

# ── Display upscale factor ────────────────────────────────────────────────────
# All panels are upscaled by this factor before display.
# 2.0 = comfortable on 1080p.  Raise to 2.5 on 4K, lower to 1.5 if too big.
DISPLAY_SCALE = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-PLATFORM LIBRARY NAME + COMPILE FLAGS
# ═══════════════════════════════════════════════════════════════════════════════
IS_WINDOWS = platform.system() == "Windows"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set this to the full path of your gcc.exe if auto-detection fails
MINGW64_GCC_OVERRIDE = r"C:\MinGW\bin\gcc.exe"

def _find_mingw64_gcc_exe():
    if MINGW64_GCC_OVERRIDE and os.path.isfile(MINGW64_GCC_OVERRIDE):
        try:
            machine = subprocess.check_output(
                [MINGW64_GCC_OVERRIDE, "-dumpmachine"], stderr=subprocess.STDOUT
            ).decode().strip()
            if "x86_64" in machine:
                print(f"🔧 Using gcc: {MINGW64_GCC_OVERRIDE}  (target: {machine})")
                return MINGW64_GCC_OVERRIDE
            else:
                print(f"   ⚠️  Override gcc is 32-bit ({machine}), searching...")
        except Exception:
            pass

    import glob as _glob
    candidate_dirs = [
        "C:\\MinGW\\bin", "C:\\mingw64\\bin", "C:\\mingw-w64\\bin",
        "C:\\msys64\\ucrt64\\bin", "C:\\msys64\\mingw64\\bin",
        "C:\\tools\\mingw64\\bin",
        "C:\\ProgramData\\chocolatey\\lib\\mingw\\tools\\install\\mingw64\\bin",
    ]
    for drive in ["C:\\", "D:\\"]:
        candidate_dirs += _glob.glob(os.path.join(drive, "mingw*", "bin"))
        candidate_dirs += _glob.glob(os.path.join(drive, "msys*", "ucrt64", "bin"))
        candidate_dirs += _glob.glob(os.path.join(drive, "msys*", "mingw64", "bin"))
    for d in os.environ.get("PATH", "").split(os.pathsep):
        candidate_dirs.append(d)

    seen = set()
    for bindir in candidate_dirs:
        if bindir in seen:
            continue
        seen.add(bindir)
        gcc_exe = os.path.join(bindir, "gcc.exe")
        if not os.path.isfile(gcc_exe):
            continue
        try:
            machine = subprocess.check_output(
                [gcc_exe, "-dumpmachine"], stderr=subprocess.STDOUT
            ).decode().strip()
            if "x86_64" in machine:
                print(f"🔧 Found 64-bit gcc: {gcc_exe}  (target: {machine})")
                return gcc_exe
            else:
                print(f"   Skipping 32-bit: {gcc_exe}")
        except Exception:
            pass
    return None

if IS_WINDOWS:
    lib_name = "libobstacle.dll"
    _gcc_exe = _find_mingw64_gcc_exe()
    if _gcc_exe is None:
        print("❌ No 64-bit gcc found. Download from https://winlibs.com (UCRT x86_64)")
        sys.exit(1)
    compile_cmd = lambda src, dst: [
        _gcc_exe, "-shared", "-fPIC", "-O3", "-m64",
        "-o", dst, src, "-lm", "-static", "-static-libgcc",
    ]
else:
    lib_name = "libobstacle.so"
    compile_cmd = lambda src, dst: [
        "gcc", "-shared", "-fPIC", "-O3", "-o", dst, src, "-lm"
    ]

lib_path       = os.path.join(SCRIPT_DIR, lib_name)
c_wrapper_path = os.path.join(SCRIPT_DIR, "c_wrapper.c")

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(lib_path):
    print(f"🗑️  Deleting old {lib_name} for fresh recompile...")
    try:
        os.remove(lib_path)
    except OSError as e:
        print(f"⚠️  Could not delete: {e}")

if not os.path.exists(lib_path):
    print(f"⚙️  Compiling {lib_name}...")
    cmd = compile_cmd(c_wrapper_path, lib_path)
    print("   Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compilation successful!")
    except subprocess.CalledProcessError:
        print("❌ Compilation failed. Check c_wrapper.c for errors.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ gcc not found at: {compile_cmd('','')[0]}")
        sys.exit(1)

try:
    lib = ctypes.CDLL(lib_path)
    print(f"✅ Loaded {lib_name}")
except OSError as e:
    print(f"❌ Could not load {lib_path}: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  C-TYPES BINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
class FloatEulers(ctypes.Structure):
    _fields_ = [("phi", ctypes.c_float), ("theta", ctypes.c_float), ("psi", ctypes.c_float)]

class ImageT(ctypes.Structure):
    # On Windows, MinGW long = 32-bit → struct timeval = 8 bytes (not 16).
    # Using c_byte*16 caused buf to land at offset 48 in Python vs offset 40
    # in C, so C read 0x0 as the buffer pointer → access violation crash.
    _fields_ = [
        ("type",     ctypes.c_int),
        ("w",        ctypes.c_uint16),
        ("h",        ctypes.c_uint16),
        ("tv_sec",   ctypes.c_int32),   # timeval.tv_sec  (4 bytes on Windows)
        ("tv_usec",  ctypes.c_int32),   # timeval.tv_usec (4 bytes on Windows)
        ("eulers",   FloatEulers),
        ("pprz_ts",  ctypes.c_uint32),
        ("buf_idx",  ctypes.c_uint8),
        ("buf_size", ctypes.c_uint32),
        ("buf",      ctypes.c_uint64),  # c_uint64 avoids c_void_p silent-None truncation
    ]

class ObstacleRegionT(ctypes.Structure):
    _fields_ = [("start",           ctypes.c_uint16),
                ("width",           ctypes.c_uint16),
                ("baseline_height", ctypes.c_uint16),
                ("edge_score",      ctypes.c_float)]  # 0.0–1.0 edge confidence

lib.get_obstacle_info.argtypes = [
    ctypes.POINTER(ImageT), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ObstacleRegionT), ctypes.POINTER(ObstacleRegionT),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
]
lib.get_obstacle_info.restype  = ctypes.c_uint8
lib.get_work_mask.restype      = ctypes.POINTER(ctypes.c_uint8)
lib.get_plant_mask.restype     = ctypes.POINTER(ctypes.c_uint8)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def bgr_to_uyvy_numpy(bgr):
    """
    Returns a contiguous uint8 numpy array in UYVY format.
    Caller must keep this array alive for the entire C call.
    """
    yuv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    h, w = bgr.shape[:2]
    uyvy = np.zeros((h, w, 2), dtype=np.uint8)
    uyvy[:, :,    1] = yuv[:, :,    0]   # Y
    uyvy[:, 0::2, 0] = yuv[:, 0::2, 1]  # U (even cols)
    uyvy[:, 1::2, 0] = yuv[:, 0::2, 2]  # V (odd cols)
    return np.ascontiguousarray(uyvy)    # guaranteed C-contiguous


def draw_original_style(img_rot, obs_regions_raw, plant_regions_raw,
                        boundary_rows, ground_baseline, label, scale=1.0):
    vis   = img_rot.copy()
    h_rot = vis.shape[0]
    lw    = max(1, int(scale))
    fs    = 0.4 * scale
    fs2   = 0.35 * scale

    if ground_baseline is not None:
        valid = ground_baseline < h_rot
        rv    = np.where(valid)[0]
        if len(rv) > 1:
            pts = np.array([[int(r), int(ground_baseline[r])] for r in rv], np.int32)
            cv2.polylines(vis, [pts], False, (255, 0, 0), lw)

    if boundary_rows is not None:
        valid = boundary_rows < h_rot
        rv    = np.where(valid)[0]
        if len(rv) > 1:
            pts = np.array([[int(r), int(boundary_rows[r])] for r in rv], np.int32)
            cv2.polylines(vis, [pts], False, (255, 255, 0), lw)

    for region in obs_regions_raw:
        s, e, w, bh, escore = int(region[0]), int(region[1]), int(region[2]), int(region[3]), float(region[4])
        if boundary_rows is not None:
            b_slice = boundary_rows[s:e+1]
            vb      = b_slice[b_slice < h_rot]
            yt      = int(np.min(vb)) if len(vb) > 0 else 0
        else:
            yt = 0
        # box colour fades red→green with edge score
        score_g = int(escore * 200)
        box_col = (0, score_g, 255 - score_g)
        cv2.rectangle(vis, (s, 0), (e, h_rot - 1), box_col, lw)
        cv2.putText(vis, f"w={w}",          (s, max(int(15*scale), yt - int(5*scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,  box_col,       lw)
        cv2.putText(vis, f"h={bh}",         (s, max(int(30*scale), yt - int(20*scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2, (0, 150, 255), lw)
        cv2.putText(vis, f"e={escore:.2f}", (s, max(int(46*scale), yt - int(36*scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2, (200, 200, 0), lw)
        y_line = min(bh, h_rot - 1)
        cv2.line(vis, (s, y_line), (e, y_line), (255, 255, 255), lw)

    for region in plant_regions_raw:
        s, e, w, bh = int(region[0]), int(region[1]), int(region[2]), int(region[3])
        cv2.rectangle(vis, (s, 0), (e, h_rot - 1), (0, 255, 0), lw)
        cv2.putText(vis, f"p={w}", (s, int(15*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), lw)

    cv2.putText(vis, label, (int(5*scale), h_rot - int(8*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55*scale, (255, 255, 255), lw+1)
    cv2.putText(vis, label, (int(5*scale), h_rot - int(8*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55*scale, (0, 0, 0),       lw)
    return vis

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--start",      type=int, default=0)
    p.add_argument("--delay",      type=int, default=50)
    p.add_argument("--python-dir", type=str)
    a = p.parse_args()

    paths = glob(os.path.join(a.folder, "*.jpg"))
    paths.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    if not paths:
        print("No images found in", a.folder); sys.exit(1)
    paths = paths[a.start:]

    min_width_c    = 20
    median_ksize_c = 5

    WIN = "C Pipeline Slideshow (q=quit, SPACE=pause, a=back, d=forward)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int(1400 * DISPLAY_SCALE), int(560 * DISPLAY_SCALE))

    idx              = 0
    paused           = False
    needs_processing = True
    last_combined    = None

    baseline_inited     = ctypes.c_int(0)
    obs_out             = (ObstacleRegionT * 20)()
    plants_out          = (ObstacleRegionT * 20)()
    plant_count_out     = ctypes.c_uint8(0)
    ground_found_out    = ctypes.c_int(0)
    green_frac_out      = ctypes.c_float(0.0)
    ground_baseline_arr = (ctypes.c_float * MAX_IMAGE_HEIGHT)()
    boundary_arr        = (ctypes.c_int   * MAX_IMAGE_HEIGHT)()

    while idx < len(paths):
        ip   = paths[idx]
        name = os.path.basename(ip)

        if needs_processing:
            bgr = cv2.imread(ip)
            if bgr is None:
                idx += 1; continue

            H, W = bgr.shape[:2]

            # ── Clamp to C buffer limits ──────────────────────────────────────
            tW = min(W, MAX_IMAGE_WIDTH)
            tH = min(H, MAX_IMAGE_HEIGHT)
            if tW != W or tH != H:
                print(f"   ⚠️  Image {W}x{H} clamped to {tW}x{tH} (C buffer limit)")

            bgr_s = cv2.resize(bgr, (tW, tH), interpolation=cv2.INTER_AREA)

            # ── Build UYVY buffer as a numpy array and pin its memory ─────────
            # uyvy_np MUST stay alive until get_obstacle_info() returns.
            # We use .ctypes.data (a plain Python int) to avoid the c_void_p
            # silent-None truncation bug on Windows 64-bit.
            uyvy_np  = bgr_to_uyvy_numpy(bgr_s)   # shape (tH, tW, 2), C-contiguous
            buf_addr = uyvy_np.ctypes.data          # raw address as Python int

            img_t      = ImageT()
            img_t.type = 0
            img_t.w    = tW
            img_t.h    = tH
            img_t.buf  = buf_addr  # assign as int to avoid c_void_p truncation

            print(f"  [{idx:4d}/{len(paths)}] {name}  size={tW}x{tH}  buf=0x{buf_addr:016x}  ", end="", flush=True)
            if buf_addr == 0:
                print("NULL buffer — skipping frame.")
                idx += 1; needs_processing = True; continue

            # ── Debug: verify the struct fields Python-side before calling C ──
            print(f"  img_t: type={img_t.type} w={img_t.w} h={img_t.h} buf=0x{img_t.buf:016x}", flush=True)

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
                ctypes.byref(green_frac_out),
            )
            # uyvy_np is still referenced here — safe to release now
            del uyvy_np  # release numpy array (buf_addr is just an int, already safe)

            num_plants = plant_count_out.value
            is_ground  = ground_found_out.value == 1
            st         = "GROUND FOUND" if is_ground else "NO GROUND"

            print(f"green={green_frac_out.value:.3f} {st}  {num_obs} obs  {num_plants} plt")

            obs_raw = [(obs_out[i].start,
                        obs_out[i].start + obs_out[i].width - 1,
                        obs_out[i].width,
                        obs_out[i].baseline_height,
                        obs_out[i].edge_score)
                       for i in range(num_obs)]
            plt_raw = [(plants_out[i].start,
                        plants_out[i].start + plants_out[i].width - 1,
                        plants_out[i].width, 0)
                       for i in range(num_plants)]

            mask_ptr   = lib.get_work_mask()
            clean_mask = np.ctypeslib.as_array(mask_ptr,  shape=(tH, tW)).copy()
            pmask_ptr  = lib.get_plant_mask()
            plant_mask = np.ctypeslib.as_array(pmask_ptr, shape=(tH, tW)).copy()

            boundary    = np.array([boundary_arr[i]         for i in range(tH)])
            py_baseline = (np.array([ground_baseline_arr[i] for i in range(tH)])
                           if baseline_inited.value else None)

            img_rot   = cv2.rotate(bgr_s,      cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_rot  = cv2.rotate(clean_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            plant_rot = cv2.rotate(plant_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ── Upscale all panels for display ────────────────────────────────
            S = DISPLAY_SCALE
            def upscale(img):
                h, w = img.shape[:2]
                return cv2.resize(img, (int(w*S), int(h*S)), interpolation=cv2.INTER_LINEAR)

            img_up   = upscale(img_rot)
            mask_up  = upscale(cv2.cvtColor(mask_rot, cv2.COLOR_GRAY2BGR))
            po       = img_rot.copy(); po[plant_rot > 0] = [0, 200, 0]
            po_up    = upscale(po)
            top      = np.hstack([img_up, mask_up, po_up])

            # ── Canny + parabola fit overlay ──────────────────────────────────
            # Canny gives clean binary edges for visual reference.
            # For each obstacle, compute the same per-row peak-x as the C scorer
            # and draw the fitted parabola so you can see what the scorer measured.
            grey_rot  = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
            canny     = cv2.Canny(grey_rot, threshold1=40, threshold2=120)
            edge_vis  = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            h_e, w_e  = edge_vis.shape[:2]
            MAG_THRESH = 20   # must match C EDGE_SCORE_MAG_THRESH

            for r in obs_raw:
                s, e_col = int(r[0]), int(r[1])
                score_g  = int(r[4] * 200)
                col      = (0, score_g, 255 - score_g)
                cv2.rectangle(edge_vis, (s, 0), (e_col, h_e - 1), col, 1)
                cv2.putText(edge_vis, f"e={r[4]:.2f}", (s + 2, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

                # compute per-row peak-x in sensor coords, then convert to rotated
                # rotated x = row in original = r_orig
                # rotated y = (tW-1) - col_orig  (90° CCW: x_rot=h-1-y_orig, y_rot=x_orig... )
                # Actually img_rot x-axis = original row axis, so obstacle cols
                # in rotated space are already r[0]..r[1]. The band in sensor
                # coords is tW-1-r[1] .. tW-1-r[0].
                # We work directly in rotated image coords for the peak search.
                grey_band = grey_rot  # already rotated
                rows_r = range(h_e)
                peak_xs_r, peak_rows_r = [], []
                for row_r in rows_r:
                    bx, bm = s, 0
                    for cx in range(s, e_col + 1):
                        # 5-tap on rotated grey image
                        xm2 = max(0, cx-2); xm1 = max(0, cx-1)
                        xp1 = min(w_e-1, cx+1); xp2 = min(w_e-1, cx+2)
                        g = (-2*int(grey_band[row_r, xm2]) - int(grey_band[row_r, xm1])
                              + int(grey_band[row_r, xp1]) + 2*int(grey_band[row_r, xp2]))
                        m = abs(g)
                        if m > bm: bm = m; bx = cx
                    if bm >= MAG_THRESH:
                        peak_xs_r.append(bx)
                        peak_rows_r.append(row_r)

                if len(peak_rows_r) >= 4:
                    # fit parabola x = a*r^2 + b*r + c
                    rs = np.array(peak_rows_r, dtype=np.float64)
                    xs = np.array(peak_xs_r,   dtype=np.float64)
                    A  = np.column_stack([rs**2, rs, np.ones_like(rs)])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
                        pa, pb, pc = coeffs   # pa/pb/pc to avoid shadowing argparse 'a'
                        # draw the parabola
                        pts = []
                        for row_r in range(h_e):
                            px = int(pa*row_r**2 + pb*row_r + pc)
                            if s <= px <= e_col:
                                pts.append((px, row_r))
                        if len(pts) > 1:
                            for i in range(len(pts)-1):
                                cv2.line(edge_vis, pts[i], pts[i+1], (0, 255, 255), 1)
                    except Exception:
                        pass
            edge_up  = upscale(edge_vis)

            # scale detection coordinates + regions to match upscaled image
            boundary_sc  = (boundary   * S).astype(int) if boundary   is not None else None
            baseline_sc  = (py_baseline * S).astype(int) if py_baseline is not None else None
            obs_sc  = [(int(r[0]*S), int(r[1]*S), int(r[2]*S), int(r[3]*S), r[4]) for r in obs_raw]
            plt_sc  = [(int(r[0]*S), int(r[1]*S), int(r[2]*S), int(r[3]*S))       for r in plt_raw]

            det_vis  = draw_original_style(img_up, obs_sc, plt_sc,
                                           boundary_sc, baseline_sc,
                                           f"{num_obs} obs, {num_plants} plt",
                                           scale=S)

            det_h, det_w = det_vis.shape[:2]
            target_w     = top.shape[1]
            # bottom row = detection | canny | info
            # canny panel gets the same width as detection view
            edge_up_r = cv2.resize(edge_up, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            used_w    = det_w + edge_up_r.shape[1]
            extra_w   = target_w - used_w
            info_panel = (np.zeros((det_h, extra_w, 3), np.uint8) if extra_w > 0 else None)

            txt_s = 0.45 * S
            lh    = int(20 * S)

            if info_panel is not None:
                y = int(25 * S)
                lines = (
                    [f"Frame {idx}/{len(paths)}", name, "",
                     f"Status: {st}", f"Green:  {green_frac_out.value:.3f}", "",
                     f"Obstacles: {num_obs}"]
                    + [f"  [{i}] row={int(r[0])}-{int(r[1])} w={int(r[2])} h={int(r[3])} e={r[4]:.2f}"
                       for i, r in enumerate(obs_raw)]
                    + ["", f"Plants: {num_plants}"]
                    + [f"  [{i}] row={int(r[0])}-{int(r[1])} w={int(r[2])}"
                       for i, r in enumerate(plt_raw)]
                    + ["", f"{'PAUSED' if paused else 'PLAYING'}  delay={a.delay}ms",
                       "", "edge score: 0.0=no edge  1.0=clean edge",
                       "box colour: red=low  green=high"]
                )
                for line in lines:
                    cv2.putText(info_panel, line, (int(10*S), y),
                                cv2.FONT_HERSHEY_SIMPLEX, txt_s,
                                (200, 200, 200), max(1, int(S)))
                    y += lh
                bot = np.hstack([det_vis, edge_up_r, info_panel])
            else:
                bot = np.hstack([det_vis, edge_up_r]) if edge_up_r.shape[1] <= target_w - det_w \
                      else det_vis[:, :target_w, :]

            hdr_h = int(40 * S)
            hdr   = np.zeros((hdr_h, target_w, 3), np.uint8)
            cv2.putText(hdr,
                        f"C PIPELINE | {name} ({tW}x{tH})  {st}  "
                        f"green={green_frac_out.value:.3f}  "
                        f"obs={num_obs} plt={num_plants}",
                        (int(8*S), int(28*S)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S,
                        (0, 255, 255), max(1, int(S)))
            # panel labels at top of image panels
            panel_w = img_up.shape[1]
            for pi, label_txt in enumerate(["original", "ground mask", "plants"]):
                cv2.putText(hdr, label_txt,
                            (int(pi * panel_w + 4), int(16 * S)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * S,
                            (180, 180, 180), max(1, int(S)))

            last_combined    = np.vstack([hdr, top, bot])
            needs_processing = False

        if last_combined is not None:
            cv2.imshow(WIN, last_combined)

        key = cv2.waitKey(a.delay if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                idx += 1; needs_processing = True
        elif key == ord('d'):
            idx += 1; needs_processing = True
        elif key == ord('a'):
            if idx > 0:
                idx -= 1; needs_processing = True
        elif not paused:
            idx += 1; needs_processing = True

    cv2.destroyAllWindows()
    print(f"\nDone. Processed {idx} frames using C backend.")

if __name__ == "__main__":
    main()