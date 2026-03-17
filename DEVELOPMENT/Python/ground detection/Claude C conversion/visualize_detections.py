#!/usr/bin/env python3
"""
visualize_sequence.py - Process a folder of drone images in order

Processes images sequentially, building baseline naturally (like the drone).
Shows detections on every Nth frame. This matches your original Python code.

USAGE:
    python3 visualize_sequence.py <image_folder> --src-dir . [--start 0] [--show-every 5]

    --start N       Start from image index N (default: 0)
    --show-every N  Show/save visualization every N frames (default: 10)
    --target IMG    Stop and show detailed output when this specific image is reached
"""
import os, sys, argparse
import numpy as np
import cv2
from glob import glob

# ═══════════════════════════════════════════════════════════════════════════════
def numpy_median_filter_1d(data, size):
    out = data.copy().astype(float); half = size // 2
    for i in range(len(data)):
        out[i] = np.median(data[max(0, i - half):min(len(data), i + half + 1)])
    return out

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

def uyvy_planes(raw, w, h):
    a = np.frombuffer(raw, np.uint8).reshape(h, w*2)
    Y = np.zeros((h,w), np.uint8); U = np.zeros((h,w), np.uint8); V = np.zeros((h,w), np.uint8)
    Y[:,0::2] = a[:,1::4]; Y[:,1::2] = a[:,3::4]
    U[:,0::2] = a[:,0::4]; U[:,1::2] = a[:,0::4]
    V[:,0::2] = a[:,2::4]; V[:,1::2] = a[:,2::4]
    return Y, U, V

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def py_is_ground(Y, U, V):
    if U <= 115:
        if V <= 145: return 0 if Y <= 85 else (0 if U <= 92 else 255)
        else: return (255 if Y <= 177 else 0) if V <= 152 else 0
    else:
        if U <= 121: return (0 if Y <= 87 else 255) if V <= 137 else 0
        else: return 0

py_vec = np.vectorize(py_is_ground)
def py_classify(Y, U, V): return py_vec(Y.astype(int), U.astype(int), V.astype(int)).astype(np.uint8)

def py_median_blur(m, ks):
    h, w = m.shape; half = ks//2; thr = (ks*ks)//2; out = np.zeros_like(m)
    for y in range(h):
        y0, y1 = max(0, y-half), min(h-1, y+half)
        for x in range(w):
            x0, x1 = max(0, x-half), min(w-1, x+half)
            out[y,x] = 255 if np.count_nonzero(m[y0:y1+1, x0:x1+1]) > thr else 0
    return out

def py_isolate(mask, thr=1000):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for l in range(1, n):
        if stats[l, cv2.CC_STAT_AREA] >= thr: out[labels == l] = 255
    return out

def py_fill_holes(mask):
    mask = mask.astype(np.uint8)
    cnt, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    f = mask.copy()
    if hier is not None:
        for i, h in enumerate(hier[0]):
            if h[3] != -1: cv2.drawContours(f, cnt, i, 255, thickness=cv2.FILLED)
    return f

def py_boundary(mf, mgp=5, mg=10, sk=5):
    ih = mf.shape[1]; nr = mf.shape[0]; b = np.full(nr, ih, dtype=int)
    for idx in range(nr):
        row = mf[idx]; gp = np.where(row > 0)[0]
        if gp.size == 0: continue
        gv = np.sum(row[-mgp:] > 0) >= mgp
        if not gv:
            d = np.diff(gp); runs = np.split(gp, np.where(d > 1)[0] + 1)
            if max(len(r) for r in runs) < mg: continue
        gr = gp[::-1]
        if gr.size == 1: b[idx] = gr[0]; continue
        gaps = -np.diff(gr) - 1; big = np.where(gaps > mg)[0]
        if big.size == 0: b[idx] = gr[-1]; continue
        fg = big[0]; pend = gr[fg]; af = gr[fg+1:]
        if af.size >= mg:
            ag = -np.diff(af) - 1; sp = np.where(ag > 0)[0]; runs2 = np.split(af, sp + 1)
            if max(len(r) for r in runs2) >= mg: b[idx] = gr[-1]; continue
        b[idx] = pend
    v = b < ih
    if v.sum() > sk:
        s = numpy_median_filter_1d(b.astype(float), size=sk); b[v] = s[v].astype(int)
    return b

def py_get_regions(cols, mw=20, mcg=5):
    if len(cols) == 0: return []
    regs = []; s = cols[0]; e = cols[0]
    for c in cols[1:]:
        if c - e <= mcg: e = c
        else: regs.append((s, e, e-s+1)); s = c; e = c
    regs.append((s, e, e-s+1))
    return [(s, e, w) for s, e, w in regs if w >= mw]

def py_update_detect(br, h, bl, mw=20, ot=50, ng=220, mcg=5):
    alpha = 0.6; v = br < h; nv = ~v
    if bl is None: b = br.astype(float).copy(); b[nv] = ng; return [], b
    dev = br.astype(float) - bl
    do_ = v & (dev > ot); om = do_ | nv
    oc = np.where(om)[0]; oregs = py_get_regions(oc, mw, mcg)
    no = v & ~do_; b = bl.copy()
    b[no] = (1-alpha)*b[no] + alpha*br[no].astype(float)
    last = ng
    for i in range(len(b)):
        if no[i]: last = b[i]
        else: b[i] = last
    return oregs, b

def py_lax_green(Y, U, V, w, h, sn=3, sd=25, bk=3):
    pw = max(1, (w*sn)//sd); ph = max(1, (h*sn)//sd)
    ms = np.zeros((ph, pw), np.uint8)
    for py_ in range(ph):
        sy = min((py_*sd)//sn, h-1)
        for px in range(pw):
            sx = min((px*sd)//sn, w-1)
            y, u, v = int(Y[sy,sx]), int(U[sy,sx]), int(V[sy,sx])
            if u <= 116 and v <= 141 and 29 <= y <= 140: ms[py_, px] = 255
    ms[:, :pw//3] = 0
    if bk >= 3:
        bl = cv2.GaussianBlur(ms, (bk, bk), 0); _, ms = cv2.threshold(bl, 127, 255, cv2.THRESH_BINARY)
    return ms, pw, ph

def py_plant_regions(pm, mw=20, mp=2, mcg=5):
    fl = pm[:,::-1]; nr = fl.shape[0]; act = []
    for r in range(nr):
        gp = np.where(fl[r] > 0)[0]
        if gp.size == 0: continue
        d = np.diff(gp); runs = np.split(gp, np.where(d > 1)[0] + 1)
        if max(len(r_) for r_ in runs) > mp: act.append(r)
    return py_get_regions(act, mw, mcg)

def process_one_frame(bgr, baseline):
    """Process a single frame. Returns (obs, plants, new_baseline, boundary, clean_mask, green_frac, ground_found, blur_mask)."""
    raw, w, h = bgr_to_uyvy(bgr)
    Yp, Up, Vp = uyvy_planes(raw, w, h)

    cls = py_classify(Yp, Up, Vp)
    blur = py_median_blur(cls, 5)
    gf = np.count_nonzero(blur) / (w*h); gnd = gf > 0.05

    cm = np.zeros((h, w), np.uint8)
    br = np.full(h, w, dtype=int)
    obs_raw = []
    new_bl = baseline

    if gnd:
        cm = py_isolate(blur, 1000); cm = py_fill_holes(cm)
        fl = cm[:, ::-1]; br = py_boundary(fl)
        obs_raw, new_bl = py_update_detect(br, w, baseline)

    obs = [(max(0, w-1-e), rw) for (s, e, rw) in obs_raw]

    ps, ppw, pph = py_lax_green(Yp, Up, Vp, w, h)
    cs = cv2.resize(cm, (ppw, pph), interpolation=cv2.INTER_NEAREST)
    pms = cv2.subtract(ps, cs)
    pmf = cv2.resize(pms, (w, h), interpolation=cv2.INTER_NEAREST)
    pr = py_plant_regions(pmf)
    plants = [(s, rw) for (s, e, rw) in pr]

    return obs, plants, new_bl, br, cm, gf, gnd, blur, w, h

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════
def draw_detections(bgr, obs, plants, boundary, baseline, w, h, label):
    vis = bgr.copy()
    # Boundary (cyan)
    if boundary is not None:
        vld = boundary < w; rv = np.where(vld)[0]
        if len(rv) > 1:
            for i in range(len(rv) - 1):
                r1, r2 = rv[i], rv[i+1]
                c1 = max(0, min(w-1, int(w - 1 - boundary[r1])))
                c2 = max(0, min(w-1, int(w - 1 - boundary[r2])))
                cv2.line(vis, (c1, r1), (c2, r2), (255, 255, 0), 2)
    # Baseline (blue dashed)
    if baseline is not None:
        vb = (baseline < w) & (baseline >= 0)
        rb = np.where(vb)[0]
        if len(rb) > 1:
            for i in range(0, len(rb) - 1, 2):
                r1, r2 = rb[i], rb[min(i+1, len(rb)-1)]
                c1 = max(0, min(w-1, int(w - 1 - baseline[r1])))
                c2 = max(0, min(w-1, int(w - 1 - baseline[r2])))
                cv2.line(vis, (c1, r1), (c2, r2), (255, 100, 100), 1)
    # Obstacles (red)
    for i, (s, rw) in enumerate(obs):
        ov = vis.copy(); cv2.rectangle(ov, (0, s), (w-1, s+rw-1), (0, 0, 255), -1)
        cv2.addWeighted(ov, 0.3, vis, 0.7, 0, vis)
        cv2.rectangle(vis, (0, s), (w-1, s+rw-1), (0, 0, 255), 2)
        cv2.putText(vis, f"OBS {i}: row={s} w={rw}", (5, s+rw//2+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # Plants (green)
    for i, (s, rw) in enumerate(plants):
        ov = vis.copy(); cv2.rectangle(ov, (0, s), (w-1, s+rw-1), (0, 255, 0), -1)
        cv2.addWeighted(ov, 0.2, vis, 0.8, 0, vis)
        cv2.rectangle(vis, (0, s), (w-1, s+rw-1), (0, 255, 0), 2)
        cv2.putText(vis, f"PLT {i}: row={s} w={rw}", (w-155, s+rw//2+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
    cv2.putText(vis, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return vis

def add_title(img, text, color=(255,255,255)):
    h, w = img.shape[:2]; bar = np.zeros((28, w, 3), np.uint8)
    cv2.putText(bar, text, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return np.vstack([bar, img])

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Process image sequence with evolving baseline")
    p.add_argument("folder", help="Folder with .jpg images")
    p.add_argument("--src-dir", default=".")
    p.add_argument("--start", type=int, default=0, help="Start from image index N")
    p.add_argument("--show-every", type=int, default=10, help="Visualize every N frames")
    p.add_argument("--target", default=None, help="Stop when this filename is reached")
    p.add_argument("--save-dir", default="sequence_results", help="Directory for output images")
    a = p.parse_args()

    paths = sorted(glob(os.path.join(a.folder, "*.jpg")))
    if not paths: print("No .jpg images found!"); sys.exit(1)

    paths = paths[a.start:]
    print(f"Found {len(paths)} images starting from index {a.start}")
    os.makedirs(a.save_dir, exist_ok=True)

    baseline = None
    target_found = False

    for frame_idx, img_path in enumerate(paths):
        name = os.path.basename(img_path)
        bgr = cv2.imread(img_path)
        if bgr is None: continue
        if bgr.shape[1] % 2: bgr = bgr[:, :bgr.shape[1]-1, :]

        obs, plants, baseline, br, cm, gf, gnd, blur, w, h = process_one_frame(bgr, baseline)

        # Summary line for every frame
        obs_str = f"{len(obs)} obs" + (f" {obs}" if obs else "")
        plt_str = f"{len(plants)} plt" + (f" {plants}" if plants else "")
        gnd_str = "GND" if gnd else "---"
        init_str = " (BASELINE INIT)" if frame_idx == 0 and gnd else ""
        print(f"  [{frame_idx:4d}] {name}  green={gf:.3f} {gnd_str}  {obs_str}  {plt_str}{init_str}")

        # Check if this is the target image
        is_target = (a.target is not None and a.target in name)
        should_show = (frame_idx % a.show_every == 0) or is_target

        if should_show:
            # Build visualization
            vis = draw_detections(bgr, obs, plants, br, baseline, w, h,
                                  f"Frame {frame_idx}: {name}  {len(obs)} obs, {len(plants)} plt")

            mask_bgr = cv2.cvtColor(cm, cv2.COLOR_GRAY2BGR)
            green_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

            top = np.hstack([
                add_title(bgr.copy(), f"Frame {frame_idx}: {name}"),
                add_title(green_bgr, "Green mask"),
                add_title(mask_bgr, "Clean mask")
            ])
            bot = np.hstack([
                add_title(vis, f"{len(obs)} obstacles, {len(plants)} plants"),
                add_title(np.zeros_like(bgr), ""),  # placeholder
                add_title(np.zeros_like(bgr), "")
            ])

            # Fill bottom-middle with baseline info
            info_img = np.zeros_like(bgr)
            y_pos = 30
            for line in [
                f"Frame: {frame_idx}",
                f"Image: {name}",
                f"Green: {gf:.3f}  Ground: {'YES' if gnd else 'NO'}",
                f"Obstacles: {len(obs)}",
            ] + [f"  obs[{i}]: start={s} width={rw}" for i, (s, rw) in enumerate(obs)] + [
                f"Plants: {len(plants)}",
            ] + [f"  plt[{i}]: start={s} width={rw}" for i, (s, rw) in enumerate(plants)] + [
                f"",
                f"Baseline range: {baseline.min():.0f}-{baseline.max():.0f}" if baseline is not None else "No baseline",
                f"Boundary range: {br[br<w].min()}-{br[br<w].max()}" if (br < w).any() else "No boundary",
                f"no_ground rows: {(br >= w).sum()}",
            ]:
                cv2.putText(info_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_pos += 20

            # Replace placeholders
            bot[:, w+28:2*w+28, :] = add_title(info_img, "Detection info")[:bot.shape[0], :w, :]

            combined = np.vstack([top, bot])
            save_path = os.path.join(a.save_dir, f"frame_{frame_idx:04d}_{name.replace('.jpg', '.png')}")
            cv2.imwrite(save_path, combined)

            if is_target:
                print(f"\n  *** TARGET IMAGE REACHED: {name} ***")
                print(f"  Saved: {save_path}")

                # Also save just the detection overlay
                det_path = os.path.join(a.save_dir, f"TARGET_{name.replace('.jpg', '.png')}")
                cv2.imwrite(det_path, vis)
                print(f"  Detection overlay: {det_path}")

                # Show it
                try:
                    wn = f"TARGET: {name} (press key)"
                    cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(wn, min(combined.shape[1], 1800), min(combined.shape[0], 1000))
                    cv2.imshow(wn, combined); cv2.waitKey(0); cv2.destroyAllWindows()
                except cv2.error:
                    pass
                target_found = True
                break

    if a.target and not target_found:
        print(f"\n  WARNING: target image '{a.target}' not found in sequence")

    print(f"\nDone. Results in {a.save_dir}/")

if __name__ == "__main__":
    main()
