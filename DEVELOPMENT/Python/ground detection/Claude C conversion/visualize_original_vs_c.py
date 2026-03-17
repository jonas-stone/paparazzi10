#!/usr/bin/env python3
"""
visualize_original_vs_c.py

PROPER comparison between original Python and C:
  1. Python processes the full image sequence → builds baseline naturally
  2. C processes ONLY the target frame → produces clean mask
  3. We compute boundary from C's mask, then apply PYTHON'S BASELINE
  4. Both panels drawn with the same method on the same scaled image

This isolates the actual difference (mask processing: CC, smooth_blob, fill)
from the baseline issue (which requires a sequence C can't run).

USAGE:
    python3 visualize_original_vs_c.py <image_folder> \
        --target 1352900896 --start 300 --show-every 20 \
        --python-dir /path/to/get_obstacle_info_Imp.py \
        --src-dir .
"""
import os, sys, argparse, subprocess, types
import numpy as np
import cv2
from glob import glob

# ═══════════════════════════════════════════════════════════════════════════════
def bgr_to_uyvy(bgr):
    h, w = bgr.shape[:2]
    if w % 2: bgr = bgr[:, :w-1, :]; w -= 1
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = yuv[:,:,0], yuv[:,:,1], yuv[:,:,2]
    Up = ((U[:,0::2].astype(np.int16) + U[:,1::2].astype(np.int16)) // 2).astype(np.uint8)
    Vp = ((V[:,0::2].astype(np.int16) + V[:,1::2].astype(np.int16)) // 2).astype(np.uint8)
    raw = np.zeros((h, w*2), np.uint8)
    raw[:,0::4] = Up; raw[:,1::4] = Y[:,0::2]; raw[:,2::4] = Vp; raw[:,3::4] = Y[:,1::2]
    return raw.tobytes(), w, h

def numpy_median_filter_1d(data, size):
    out = data.copy().astype(float); half = size // 2
    for i in range(len(data)):
        out[i] = np.median(data[max(0, i - half):min(len(data), i + half + 1)])
    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING (matches original Python display exactly)
# ═══════════════════════════════════════════════════════════════════════════════
def draw_original_style(img_rot, obs_regions_raw, plant_regions_raw,
                        boundary_rows, ground_baseline, W_orig, label):
    vis = img_rot.copy()
    h_rot = vis.shape[0]

    # Baseline (blue)
    if ground_baseline is not None:
        valid = ground_baseline < h_rot
        rv = np.where(valid)[0]
        if len(rv) > 1:
            pts = np.array([[int(r), int(ground_baseline[r])] for r in rv], np.int32)
            cv2.polylines(vis, [pts], False, (255, 0, 0), 1)

    # Boundary (cyan)
    if boundary_rows is not None:
        valid = boundary_rows < h_rot
        rv = np.where(valid)[0]
        if len(rv) > 1:
            pts = np.array([[int(r), int(boundary_rows[r])] for r in rv], np.int32)
            cv2.polylines(vis, [pts], False, (255, 255, 0), 1)

    # Obstacles (red)
    for region in obs_regions_raw:
        s, e, w = int(region[0]), int(region[1]), int(region[2])
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
        cv2.rectangle(ov, (s, yt), (e, yb), (0, 0, 255), -1)
        cv2.addWeighted(ov, 0.25, vis, 0.75, 0, vis)
        cv2.rectangle(vis, (s, 0), (e, h_rot-1), (0, 0, 255), 2)
        cv2.rectangle(vis, (s, yt), (e, yb), (0, 0, 255), 2)
        cv2.putText(vis, f"w={w}", (s, max(15, yt-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    # Plants (green)
    for region in plant_regions_raw:
        s, e, w = int(region[0]), int(region[1]), int(region[2])
        cv2.rectangle(vis, (s, 0), (e, h_rot-1), (0, 255, 0), 2)
        cv2.putText(vis, f"p={w}", (s, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    cv2.putText(vis, label, (5, h_rot-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(vis, label, (5, h_rot-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    return vis

def add_title(img, text, color=(255,255,255)):
    h, w = img.shape[:2]; bar = np.zeros((28, w, 3), np.uint8)
    cv2.putText(bar, text, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return np.vstack([bar, img])

# ═══════════════════════════════════════════════════════════════════════════════
#  C: compile + run → get clean mask
# ═══════════════════════════════════════════════════════════════════════════════
def compile_c(sd, bp):
    cmd = ["gcc", "-O2", "-Wall", "-Wno-unused-function", "-o", bp,
           os.path.join(sd, "test_obstacle_detection.c"),
           os.path.join(sd, "team10_get_obstacle_info_standalone.c"), "-lm", f"-I{sd}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: print(f"C error:\n{r.stderr}"); return False
    return True

def get_c_clean_mask(bin_path, bgr_scaled, work_dir):
    """Run C on scaled image (1 frame only — just to get the mask)."""
    raw, w, h = bgr_to_uyvy(bgr_scaled)
    rp = os.path.join(work_dir, "input.yuv422")
    open(rp, "wb").write(raw)
    cout = os.path.join(work_dir, "c_output")
    os.makedirs(cout, exist_ok=True)
    cmd = [bin_path, rp, str(w), str(h), cout, "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout or "").strip().split('\n'):
        if line.strip(): print(f"    C> {line}")
    if r.returncode != 0: return None, w, h

    mask_path = os.path.join(cout, "mask_after_fill.raw")
    if os.path.exists(mask_path):
        return np.fromfile(mask_path, np.uint8).reshape(h, w), w, h
    # fallback: try mask_after_cc
    mask_path = os.path.join(cout, "mask_after_cc.raw")
    if os.path.exists(mask_path):
        return np.fromfile(mask_path, np.uint8).reshape(h, w), w, h
    return None, w, h

# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDARY + OBSTACLE DETECTION (reusable on any mask + any baseline)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_boundary(clean_mask, w, h, min_gp=5, max_gap=10, smooth_k=5):
    flipped = clean_mask[:, ::-1]
    boundary = np.full(h, w, dtype=int)
    for idx in range(h):
        row = flipped[idx]; gp = np.where(row > 0)[0]
        if gp.size == 0: continue
        gv = np.sum(row[-min_gp:] > 0) >= min_gp
        if not gv:
            d = np.diff(gp); runs = np.split(gp, np.where(d > 1)[0] + 1)
            if max(len(r) for r in runs) < max_gap: continue
        gr = gp[::-1]
        if gr.size == 1: boundary[idx] = gr[0]; continue
        gaps = -np.diff(gr) - 1; big = np.where(gaps > max_gap)[0]
        if big.size == 0: boundary[idx] = gr[-1]; continue
        fg = big[0]; pend = gr[fg]; af = gr[fg+1:]
        if af.size >= max_gap:
            ag = -np.diff(af) - 1; sp = np.where(ag > 0)[0]
            runs2 = np.split(af, sp + 1)
            if max(len(r) for r in runs2) >= max_gap: boundary[idx] = gr[-1]; continue
        boundary[idx] = pend
    valid = boundary < w
    if valid.sum() > smooth_k:
        s = numpy_median_filter_1d(boundary.astype(float), size=smooth_k)
        boundary[valid] = s[valid].astype(int)
    return boundary

def detect_obstacles_with_baseline(boundary, h_col_count, baseline,
                                    min_width=20, obs_thr=50,
                                    no_gnd=220, max_col_gap=5):
    """Run update_and_detect logic using a given baseline. Returns (raw_regions, new_baseline)."""
    alpha = 0.6
    valid = boundary < h_col_count; no_ground = ~valid
    if baseline is None:
        bl = boundary.astype(float).copy(); bl[no_ground] = no_gnd
        return [], bl
    dev = boundary.astype(float) - baseline
    dev_obs = valid & (dev > obs_thr); obs_mask = dev_obs | no_ground
    obs_cols = np.where(obs_mask)[0]
    # merge
    regions = []
    if len(obs_cols) > 0:
        s = obs_cols[0]; e = obs_cols[0]
        for c in obs_cols[1:]:
            if c - e <= max_col_gap: e = c
            else:
                rw = e - s + 1
                if rw >= min_width: regions.append((s, e, rw))
                s = c; e = c
        rw = e - s + 1
        if rw >= min_width: regions.append((s, e, rw))
    # update baseline
    no_obs = valid & ~dev_obs; bl = baseline.copy()
    bl[no_obs] = (1-alpha)*bl[no_obs] + alpha*boundary[no_obs].astype(float)
    last = no_gnd
    for i in range(len(bl)):
        if no_obs[i]: last = bl[i]
        else: bl[i] = last
    return regions, bl

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder"); p.add_argument("--target", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--show-every", type=int, default=10)
    p.add_argument("--python-dir", default=None)
    p.add_argument("--src-dir", default=".")
    p.add_argument("--scale", type=float, default=0.8)
    p.add_argument("--save-dir", default="sequence_results")
    p.add_argument("--work-dir", default="/tmp/obstacle_validation")
    a = p.parse_args()
    os.makedirs(a.save_dir, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)

    # ── Import original Python ────────────────────────────────────────────
    search = [a.python_dir, ".", "..", os.path.dirname(os.path.abspath(a.folder))]
    search = [d for d in search if d]
    found = False
    for d in search:
        if os.path.exists(os.path.join(d, "get_obstacle_info_Imp.py")):
            sys.path.insert(0, os.path.abspath(d)); found = True
            print(f"Python source: {os.path.abspath(d)}"); break
    if not found: print("ERROR: get_obstacle_info_Imp.py not found"); sys.exit(1)

    def _mf(data, size):
        out = data.copy().astype(float); half = size // 2
        for i in range(len(data)):
            out[i] = np.median(data[max(0, i-half):min(len(data), i+half+1)])
        return out
    sn = types.ModuleType("scipy.ndimage"); sn.median_filter = _mf
    sm = types.ModuleType("scipy"); sm.ndimage = sn
    sys.modules["scipy"] = sm; sys.modules["scipy.ndimage"] = sn

    import get_obstacle_info_Imp as orig
    print(f"Imported: {orig.__file__}")

    # ── Compile C ─────────────────────────────────────────────────────────
    c_bin = os.path.join(a.work_dir, "test_obstacle_detection")
    c_ok = compile_c(a.src_dir, c_bin)
    print(f"C: {'OK' if c_ok else 'FAILED'}")

    # ── Images + params ───────────────────────────────────────────────────
    paths = sorted(glob(os.path.join(a.folder, "*.jpg")))[a.start:]
    if not paths: print("No images"); sys.exit(1)
    print(f"{len(paths)} images from index {a.start}")

    S = a.scale
    pp = orig.get_scaled_params(S)
    PS = 0.15 * S; pbk = orig.scale_odd(5, PS)
    print(f"Scale={S} min_w={pp['min_width']} obs_thr={pp['obstacle_threshold']}")

    # ── Run sequence ──────────────────────────────────────────────────────
    py_baseline = None

    for fi, ip in enumerate(paths):
        name = os.path.basename(ip)
        bgr = cv2.imread(ip)
        if bgr is None: continue
        H, W = bgr.shape[:2]
        tW, tH = max(1, int(W * S)), max(1, int(H * S))
        bgr_s = cv2.resize(bgr, (tW, tH), interpolation=cv2.INTER_AREA)

        # ── Original Python ───────────────────────────────────────────────
        dets, py_baseline, dbg = orig.get_obstacle_info(
            bgr_s, py_baseline,
            oa_color_count_frac=0.05, median_ksize=pp['median_ksize'],
            min_width=pp['min_width'], max_col_gap=pp['max_col_gap'],
            min_ground_pixels=pp['min_ground_pixels'], max_gap=pp['max_gap'],
            smooth_kernel=pp['smooth_kernel'], obstacle_threshold=pp['obstacle_threshold'],
            no_ground_baseline=pp['no_ground_baseline'],
            plant_scale_factor=PS, plant_blur_ksize=pbk,
            plant_min_width=20, plant_min_pixels_per_col=2, plant_max_col_gap=5,
        )

        st, gf = dbg["status"], dbg["green_frac"]
        obs_raw = dbg.get("obstacle_regions_raw", [])
        plt_raw = dbg.get("plant_regions_raw", [])

        init = " (INIT)" if fi == 0 else ""
        print(f"  [{fi:4d}] {name}  green={gf:.3f} {st:14s}  {len(obs_raw)} obs  {len(plt_raw)} plt{init}")

        is_target = a.target is not None and a.target in name
        if not (fi % a.show_every == 0 or is_target):
            continue

        # ── Visualize Python ──────────────────────────────────────────────
        img_rot = cv2.rotate(bgr_s, cv2.ROTATE_90_COUNTERCLOCKWISE)
        py_boundary = dbg.get("boundary_rows", np.full(tH, tW, dtype=int))
        clean_mask = dbg.get("clean_mask", np.zeros((tH, tW), np.uint8))
        plant_mask = dbg.get("plant_mask", np.zeros((tH, tW), np.uint8))

        py_vis = draw_original_style(img_rot, obs_raw, plt_raw,
                                     py_boundary, py_baseline, tW,
                                     f"PYTHON: {len(obs_raw)} obs, {len(plt_raw)} plt")

        mask_rot = cv2.rotate(clean_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plant_rot = cv2.rotate(plant_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        po = img_rot.copy(); po[plant_rot > 0] = [0, 200, 0]

        # ── C comparison on target ────────────────────────────────────────
        c_vis = np.zeros_like(py_vis)
        c_info = "C: (shows on --target frame)"
        c_mask_panel = np.zeros_like(mask_rot)

        if is_target and c_ok:
            print(f"\n  Running C on scaled {tW}x{tH}...")

            # Get C's clean mask (run 1 frame — we just want the mask)
            c_mask, c_w, c_h = get_c_clean_mask(c_bin, bgr_s, a.work_dir)

            if c_mask is not None:
                # Compute boundary from C's mask
                c_boundary = compute_boundary(c_mask, c_w, c_h,
                    min_gp=pp['min_ground_pixels'], max_gap=pp['max_gap'],
                    smooth_k=pp['smooth_kernel'])

                # Apply PYTHON'S baseline to C's boundary → detect obstacles
                c_obs_raw, _ = detect_obstacles_with_baseline(
                    c_boundary, c_w, py_baseline,
                    min_width=pp['min_width'],
                    obs_thr=pp['obstacle_threshold'],
                    no_gnd=pp['no_ground_baseline'],
                    max_col_gap=pp['max_col_gap'])

                # Plant detection from C mask
                # (reuse Python's plant detection on C's clean mask)
                c_pm_small, _, _ = orig.detect_all_green_lax(
                    bgr_s, scale_factor=PS, blur_ksize=pbk)
                pph_s, ppw_s = c_pm_small.shape[:2]
                c_cm_small = cv2.resize(c_mask, (ppw_s, pph_s), interpolation=cv2.INTER_NEAREST)
                c_pm_diff = cv2.subtract(c_pm_small, c_cm_small)
                c_pm_full = cv2.resize(c_pm_diff, (c_w, c_h), interpolation=cv2.INTER_NEAREST)
                c_plt_raw = orig.detect_plant_regions(c_pm_full, min_width=20,
                                                      min_pixels_per_col=2, max_col_gap=5)

                # Draw C results on same rotated image
                c_vis = draw_original_style(img_rot, c_obs_raw, c_plt_raw,
                                           c_boundary, py_baseline, c_w,
                                           f"C mask + Py baseline: {len(c_obs_raw)} obs, {len(c_plt_raw)} plt")

                c_mask_rot = cv2.rotate(c_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                c_mask_panel = c_mask_rot

                c_info = f"C: {len(c_obs_raw)} obs, {len(c_plt_raw)} plt"
                print(f"  Python obs_raw: {obs_raw}")
                print(f"  C obs_raw:      {c_obs_raw}")
                print(f"  Python plt_raw: {plt_raw}")
                print(f"  C plt_raw:      {c_plt_raw}")

                # Compare masks
                mask_diff = np.abs(clean_mask.astype(int) - c_mask.astype(int))
                n_diff = np.count_nonzero(mask_diff)
                pct = 100.0 * n_diff / (c_w * c_h)
                print(f"  Mask diff: {n_diff}/{c_w*c_h} pixels ({pct:.1f}%)")
            else:
                c_info = "C: mask extraction failed"

        # ── Build composite ───────────────────────────────────────────────
        top = np.hstack([
            add_title(img_rot, f"Frame {fi}: {name}"),
            add_title(cv2.cvtColor(mask_rot, cv2.COLOR_GRAY2BGR), "Python ground mask"),
            add_title(po, "Plant overlay"),
        ])

        c_mask_bgr = cv2.cvtColor(c_mask_panel, cv2.COLOR_GRAY2BGR) if c_mask_panel.any() else np.zeros_like(py_vis)

        bot = np.hstack([
            add_title(py_vis, ""),
            add_title(c_vis, c_info),
            add_title(c_mask_bgr, "C ground mask" if c_mask_panel.any() else ""),
        ])

        tw = max(top.shape[1], bot.shape[1])
        def pad(img, t):
            if img.shape[1] < t: return np.hstack([img, np.zeros((img.shape[0], t-img.shape[1], 3), np.uint8)])
            return img[:, :t, :]
        top, bot = pad(top, tw), pad(bot, tw)

        hdr = np.zeros((35, tw, 3), np.uint8)
        cv2.putText(hdr, f"{name} ({tW}x{tH} @{S}x) {st} green={gf:.3f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        combined = np.vstack([hdr, top, bot])

        sp = os.path.join(a.save_dir, f"frame_{fi:04d}_{name.replace('.jpg','.png')}")
        # cv2.imwrite(sp, combined)

        if is_target:
            tp = os.path.join(a.save_dir, f"TARGET_{name.replace('.jpg','.png')}")
            cv2.imwrite(tp, combined)
            print(f"\n  Saved: {tp}")
            try:
                wn = f"TARGET: {name}"
                cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(wn, min(tw, 1800), min(combined.shape[0], 1000))
                cv2.imshow(wn, combined); cv2.waitKey(0); cv2.destroyAllWindows()
            except cv2.error: pass
            break

    print(f"\nResults: {a.save_dir}/")

if __name__ == "__main__":
    main()
