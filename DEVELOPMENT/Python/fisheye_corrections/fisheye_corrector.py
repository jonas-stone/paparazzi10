"""
Fisheye Lens Correction Studio  v4
─────────────────────────────────────────────────────────────
• All settings are GLOBAL — same camera, same correction
• Open a folder → browse with ◀/▶ or ← → keys
• Mouse-wheel on any slider to nudge it
• Click any value label to type an exact number
• Sensitivity toggle: COARSE / NORMAL / FINE / ULTRA
• Save & load settings profiles (JSON)
• Export single image or entire folder
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os, json

# ─────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────
PREVIEW_W    = 500
PREVIEW_H    = 375
PANEL_PAD    = 10
GRID_SPACING = 40
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DARK_BG   = "#0d0d11"
PANEL_BG  = "#13131b"
CARD_BG   = "#181822"
ACCENT    = "#00e5ff"
ACCENT2   = "#ff6b6b"
ACCENT3   = "#a78bfa"
ACCENT4   = "#34d399"
GOLD      = "#f59e0b"
TEXT_MAIN = "#dde0f0"
TEXT_DIM  = "#4a4a62"
TROUGH    = "#23233a"
BORDER    = "#23233a"

# ─────────────────────────────────────────────
#  Parameter table
#  (key, label, lo, hi, default, group, color)
# ─────────────────────────────────────────────
PARAM_DEFS = [
    # ORIENTATION
    ("rotation",    "Rotation (°)",          -180, 180,   0.0,  "orient", ACCENT ),
    ("zoom",        "Zoom / crop",             0.5, 2.5,   1.0,  "orient", ACCENT ),
    # FISHEYE
    ("k1",          "k1  radial 1",           -2.0, 2.0,  -0.30, "fisheye", ACCENT3),
    ("k2",          "k2  radial 2",           -2.0, 2.0,   0.10, "fisheye", ACCENT3),
    ("k3",          "k3  radial 3",           -1.0, 1.0,   0.00, "fisheye", ACCENT3),
    ("k4",          "k4  radial 4",           -1.0, 1.0,   0.00, "fisheye", ACCENT3),
    ("focal_scale", "Focal length scale",      0.2, 4.0,   1.00, "fisheye", ACCENT3),
    ("cx_offset",   "Centre X offset (px)",  -400, 400,    0.0,  "fisheye", ACCENT3),
    ("cy_offset",   "Centre Y offset (px)",  -400, 400,    0.0,  "fisheye", ACCENT3),
    ("output_scale","Output zoom",             0.2, 3.0,   1.00, "fisheye", ACCENT3),
    ("balance",     "Crop balance",            0.0, 1.0,   0.50, "fisheye", ACCENT3),
    # DEFISHEYE-specific
    ("defov",       "FOV (defisheye °)",       1,   220,  180.0, "fisheye", ACCENT3),
    # PERSPECTIVE
    ("tilt_x",      "Keystone H",            -20,  20,    0.0,  "persp", GOLD  ),
    ("tilt_y",      "Keystone V",            -20,  20,    0.0,  "persp", GOLD  ),
    # POST
    ("brightness",  "Brightness",            -80,  80,    0.0,  "post", ACCENT4),
    ("contrast",    "Contrast",               0.5, 2.0,   1.00, "post", ACCENT4),
    ("saturation",  "Saturation",             0.0, 2.5,   1.00, "post", ACCENT4),
    ("sharpness",   "Sharpness",              0.0, 2.0,   0.00, "post", ACCENT4),
    ("vignette",    "Vignette",               0.0, 1.0,   0.00, "post", ACCENT4),
]

GROUP_INFO = {
    "orient":  ("ORIENTATION",   "rotation · zoom · flip — applied to every image", ACCENT ),
    "fisheye": ("FISHEYE MODEL", "lens distortion coefficients",                     ACCENT3),
    "persp":   ("PERSPECTIVE",   "keystone / trapezoid correction",                  GOLD   ),
    "post":    ("POST-PROCESS",  "brightness · contrast · colour · sharpness",       ACCENT4),
}

DEFAULTS = {k: d for k, _, _, _, d, *_ in PARAM_DEFS}

# Sensitivity: fraction of full slider range per mouse-wheel tick
SENSITIVITY = {
    "COARSE": 0.050,
    "NORMAL": 0.010,
    "FINE":   0.002,
    "ULTRA":  0.0004,
}

# ─────────────────────────────────────────────
#  Image processing
# ─────────────────────────────────────────────
def rotate_image(img, angle_deg):
    if abs(angle_deg) < 0.01:
        return img
    h, w   = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def flip_image(img, flip_h, flip_v):
    if flip_h: img = cv2.flip(img, 1)
    if flip_v: img = cv2.flip(img, 0)
    return img


def zoom_crop(img, zoom):
    if abs(zoom - 1.0) < 0.001:
        return img
    h, w = img.shape[:2]
    if zoom > 1.0:
        nw, nh = int(w / zoom), int(h / zoom)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        return cv2.resize(img[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        nw, nh = int(w * zoom), int(h * zoom)
        canvas = np.zeros_like(img)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return canvas


def correct_fisheye(img, k1, k2, k3, k4, focal_scale,
                    cx_offset, cy_offset, output_scale, balance,
                    lens_model, defov, dftype):
    h, w = img.shape[:2]
    f = max(w, h) * focal_scale
    cx = w / 2 + cx_offset
    cy = h / 2 + cy_offset

    if lens_model == "defisheye":
        try:
            from defisheye import Defisheye
            import tempfile, os
            # defisheye works via file paths, so write a temp file
            tmp_in  = os.path.join(tempfile.gettempdir(), "_fe_in.png")
            tmp_out = os.path.join(tempfile.gettempdir(), "_fe_out.png")
            cv2.imwrite(tmp_in, img)
            obj = Defisheye(tmp_in, dtype=dftype, format="png",
                            fov=int(defov), pfov=int(min(defov, 120)))
            obj.convert(tmp_out)
            result = cv2.imread(tmp_out)
            if result is None:
                return img
            # Resize back to original size if defisheye changed it
            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
            return result
        except Exception as e:
            # Fall through to fisheye model on error
            print(f"defisheye error: {e}")
            return img

    elif lens_model == "standard":
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1 ]], dtype=np.float64)
        D = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
        new_K = cv2.getOptimalNewCameraMatrix(
            K, D, (w, h), alpha=1.0 - balance)[0]
        new_K[0,0] *= output_scale
        new_K[1,1] *= output_scale
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2)

    else:  # "fisheye" — equidistant model
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1 ]], dtype=np.float64)
        D = np.array([k1, k2, k3, k4], dtype=np.float64)
        new_K = K.copy()
        new_K[0, 0] *= output_scale
        new_K[1, 1] *= output_scale
        new_K_opt = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3),
            balance=float(np.clip(balance, 0, 1)),
            new_size=(w, h), fov_scale=1.0)
        final_K = new_K_opt * balance + new_K * (1 - balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), final_K, (w, h), cv2.CV_16SC2)

    return cv2.remap(img, map1, map2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(0, 0, 0))


def apply_perspective(img, tilt_x, tilt_y):
    if tilt_x == 0 and tilt_y == 0:
        return img
    h, w = img.shape[:2]
    tx, ty = tilt_x / 100.0, tilt_y / 100.0
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [w*tx,       h*ty      ],
        [w*(1-tx),   h*ty      ],
        [w*(1+tx),   h*(1-ty)  ],
        [w*(-tx),    h*(1-ty)  ],
    ])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def apply_post(img, brightness, contrast, saturation, sharpness, vignette):
    out = img.astype(np.float32)
    out = np.clip(out * contrast + brightness, 0, 255)
    if abs(saturation - 1.0) > 0.01:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if sharpness > 0.01:
        blurred = cv2.GaussianBlur(out, (0,0), 3)
        out = np.clip(cv2.addWeighted(out, 1+sharpness, blurred, -sharpness, 0), 0, 255)
    if vignette > 0.01:
        h, w = out.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt(((X-w/2)/(w/2))**2 + ((Y-h/2)/(h/2))**2)
        out = np.clip(out * (1 - np.clip(dist*vignette, 0, 1))[:,:,np.newaxis], 0, 255)
    return out.astype(np.uint8)


def process_image(img, rotation, zoom, flip_h, flip_v,
                  k1, k2, k3, k4, focal_scale, cx_offset, cy_offset,
                  output_scale, balance, defov, tilt_x, tilt_y,
                  brightness, contrast, saturation, sharpness, vignette,
                  lens_model="fisheye", dftype="equalarea"):
    out = rotate_image(img, rotation)
    out = flip_image(out, flip_h, flip_v)
    out = zoom_crop(out, zoom)
    out = correct_fisheye(out, k1, k2, k3, k4, focal_scale,
                          cx_offset, cy_offset, output_scale, balance,
                          lens_model, defov, dftype)
    out = apply_perspective(out, tilt_x, tilt_y)
    out = apply_post(out, brightness, contrast, saturation, sharpness, vignette)
    return out


def draw_grid(img, spacing=GRID_SPACING):
    ov = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(ov, (x, 0), (x, h), (0, 200, 80), 1)
    for y in range(0, h, spacing):
        cv2.line(ov, (0, y), (w, y), (0, 200, 80), 1)
    cv2.line(ov, (w//2, 0), (w//2, h), (0, 160, 255), 1)
    cv2.line(ov, (0, h//2), (w, h//2), (0, 160, 255), 1)
    return cv2.addWeighted(ov, 0.30, img, 0.70, 0)


def resize_for_display(img, tw, th):
    h, w = img.shape[:2]
    s = min(tw/w, th/h)
    return cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s))),
                      interpolation=cv2.INTER_AREA)


def cv2_to_tk(img_bgr):
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))


# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────
class FisheyeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fisheye Correction Studio")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)

        # Folder state
        self.folder_path   = None
        self.folder_images = []
        self.folder_index  = 0
        self._img_cache    = {}
        self._cache_order  = []

        # All params are global — one value for every image
        self.params: dict[str, tk.DoubleVar] = {
            k: tk.DoubleVar(value=d) for k, _, _, _, d, *_ in PARAM_DEFS
        }
        self.flip_h     = tk.BooleanVar(value=False)
        self.flip_v     = tk.BooleanVar(value=False)
        self.lens_model = tk.StringVar(value="fisheye")
        self.dftype     = tk.StringVar(value="equalarea")

        self.show_grid   = tk.BooleanVar(value=True)
        self.sensitivity = tk.StringVar(value="NORMAL")
        self._after_id   = None
        self._sliders: dict[str, tuple] = {}  # key -> (Scale, lo, hi)

        self._build_ui()
        self.bind("<Left>",  lambda e: self._nav(-1))
        self.bind("<Right>", lambda e: self._nav(+1))
        self._refresh_preview()

    # ─────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────
    def _build_ui(self):
        # ── top bar
        top = tk.Frame(self, bg=DARK_BG, pady=8)
        top.pack(fill="x", padx=16)
        tk.Label(top, text="⬡  FISHEYE STUDIO",
                 font=("Courier New", 14, "bold"),
                 fg=ACCENT, bg=DARK_BG).pack(side="left")
        tk.Label(top, text="all settings apply to every image",
                 font=("Courier New", 8), fg=TEXT_DIM,
                 bg=DARK_BG).pack(side="left", padx=12)
        tk.Checkbutton(top, text="Grid",
                       variable=self.show_grid,
                       command=self._schedule_refresh,
                       bg=DARK_BG, fg=TEXT_MAIN, selectcolor=TROUGH,
                       activebackground=DARK_BG,
                       font=("Courier New", 8)).pack(side="right")

        # ── folder nav
        nav = tk.Frame(self, bg=PANEL_BG, pady=7)
        nav.pack(fill="x", padx=16, pady=(0, 5))

        tk.Button(nav, text="📂  Open Folder",
                  font=("Courier New", 9, "bold"),
                  bg=ACCENT, fg=DARK_BG,
                  activebackground=ACCENT2, activeforeground=DARK_BG,
                  relief="flat", bd=0, padx=10, pady=3,
                  command=self._open_folder).pack(side="left", padx=(8, 12))

        self.btn_prev = tk.Button(nav, text="◀",
                                  font=("Courier New", 11, "bold"),
                                  bg=TROUGH, fg=TEXT_MAIN,
                                  activebackground=ACCENT3, activeforeground=DARK_BG,
                                  relief="flat", bd=0, padx=10, pady=3,
                                  command=lambda: self._nav(-1), state="disabled")
        self.btn_prev.pack(side="left", padx=2)

        self.nav_label = tk.Label(nav, text="no folder  ·  ← → to browse",
                                  font=("Courier New", 9), fg=TEXT_DIM,
                                  bg=PANEL_BG, width=48, anchor="center")
        self.nav_label.pack(side="left", padx=4)

        self.btn_next = tk.Button(nav, text="▶",
                                  font=("Courier New", 11, "bold"),
                                  bg=TROUGH, fg=TEXT_MAIN,
                                  activebackground=ACCENT3, activeforeground=DARK_BG,
                                  relief="flat", bd=0, padx=10, pady=3,
                                  command=lambda: self._nav(+1), state="disabled")
        self.btn_next.pack(side="left", padx=2)

        self.folder_label = tk.Label(nav, text="", font=("Courier New", 8),
                                     fg=TEXT_DIM, bg=PANEL_BG)
        self.folder_label.pack(side="left", padx=10)

        # ── main body
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        # Previews
        pv = tk.Frame(body, bg=DARK_BG)
        pv.pack(side="left", fill="both", expand=True)

        lrow = tk.Frame(pv, bg=DARK_BG)
        lrow.pack(fill="x")
        tk.Label(lrow, text="ORIGINAL", font=("Courier New", 8, "bold"),
                 fg=TEXT_DIM, bg=DARK_BG,
                 width=int(PREVIEW_W/7)).pack(side="left", padx=PANEL_PAD)
        tk.Label(lrow, text="CORRECTED", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=DARK_BG,
                 width=int(PREVIEW_W/7)).pack(side="left", padx=PANEL_PAD)

        crow = tk.Frame(pv, bg=DARK_BG)
        crow.pack(fill="both", expand=True)
        self.canvas_orig = tk.Canvas(crow, width=PREVIEW_W, height=PREVIEW_H,
                                     bg="#09090e", highlightthickness=1,
                                     highlightbackground=BORDER)
        self.canvas_orig.pack(side="left", padx=PANEL_PAD)
        self.canvas_corr = tk.Canvas(crow, width=PREVIEW_W, height=PREVIEW_H,
                                     bg="#09090e", highlightthickness=1,
                                     highlightbackground=ACCENT)
        self.canvas_corr.pack(side="left", padx=PANEL_PAD)
        self._draw_placeholder()

        # ── right panel (sliders)
        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side="right", fill="y", padx=(8, 0))

        # Sensitivity bar
        sens_bar = tk.Frame(right, bg=PANEL_BG, pady=5, padx=8)
        sens_bar.pack(fill="x")
        tk.Label(sens_bar, text="SENSITIVITY",
                 font=("Courier New", 7, "bold"),
                 fg=TEXT_DIM, bg=PANEL_BG).pack(side="left", padx=(0, 6))
        for mode, col in [("COARSE", ACCENT2), ("NORMAL", TEXT_MAIN),
                          ("FINE", ACCENT), ("ULTRA", ACCENT3)]:
            tk.Radiobutton(sens_bar, text=mode,
                           variable=self.sensitivity, value=mode,
                           command=self._on_sensitivity_change,
                           font=("Courier New", 7, "bold"),
                           fg=col, bg=PANEL_BG, selectcolor=TROUGH,
                           activebackground=PANEL_BG,
                           indicatoron=0, relief="flat", bd=0,
                           padx=6, pady=2,
                           ).pack(side="left", padx=2)

        # Profile buttons
        prof_bar = tk.Frame(right, bg=PANEL_BG, pady=4, padx=8)
        prof_bar.pack(fill="x")
        for lbl, cmd, col in [
            ("💾 Save profile", self._save_profile, TEXT_MAIN),
            ("📂 Load profile", self._load_profile, ACCENT),
            ("↺ Reset all",    self._reset_all,     ACCENT2),
        ]:
            tk.Button(prof_bar, text=lbl,
                      font=("Courier New", 8),
                      bg=TROUGH, fg=col,
                      activebackground=ACCENT, activeforeground=DARK_BG,
                      relief="flat", bd=0, padx=7, pady=3,
                      command=cmd).pack(side="left", padx=2)

        # Scrollable slider area
        sc_outer = tk.Frame(right, bg=DARK_BG)
        sc_outer.pack(fill="both", expand=True)
        self.sl_canvas = tk.Canvas(sc_outer, bg=PANEL_BG,
                                   width=330, highlightthickness=0)
        vsb = tk.Scrollbar(sc_outer, orient="vertical",
                           command=self.sl_canvas.yview)
        self.sl_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.sl_canvas.pack(side="left", fill="both", expand=True)

        self.sl_frame = tk.Frame(self.sl_canvas, bg=PANEL_BG)
        self._sl_win = self.sl_canvas.create_window(
            (0, 0), window=self.sl_frame, anchor="nw")
        self.sl_frame.bind("<Configure>",
                           lambda e: self.sl_canvas.configure(
                               scrollregion=self.sl_canvas.bbox("all")))
        self.sl_canvas.bind("<Configure>",
                            lambda e: self.sl_canvas.itemconfig(
                                self._sl_win, width=e.width))
        self.sl_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._on_global_mousewheel(e))

        self._build_slider_groups()

        # Export buttons
        exp = tk.Frame(right, bg=PANEL_BG, pady=6, padx=8)
        exp.pack(fill="x")
        tk.Button(exp, text="⬇ Save current",
                  font=("Courier New", 8, "bold"),
                  bg=ACCENT, fg=DARK_BG,
                  activebackground=ACCENT2, activeforeground=DARK_BG,
                  relief="flat", bd=0, padx=8, pady=4,
                  command=self._save_image).pack(side="left", padx=(0, 4))
        tk.Button(exp, text="⬇ Export folder",
                  font=("Courier New", 8),
                  bg=TROUGH, fg=ACCENT,
                  activebackground=ACCENT, activeforeground=DARK_BG,
                  relief="flat", bd=0, padx=8, pady=4,
                  command=self._export_folder).pack(side="left")

        # Status
        self.status_var = tk.StringVar(value="Open a folder to begin")
        tk.Label(self, textvariable=self.status_var,
                 font=("Courier New", 8), fg=TEXT_DIM,
                 bg=DARK_BG, anchor="w").pack(fill="x", padx=20, pady=(0, 4))

    def _build_slider_groups(self):
        self.val_labels: dict[str, tk.Label] = {}
        current_group = None

        for key, label, lo, hi, default, group, color in PARAM_DEFS:
            # Section header
            if group != current_group:
                current_group = group
                gname, gdesc, gcol = GROUP_INFO[group]

                card = tk.Frame(self.sl_frame, bg=CARD_BG,
                                highlightthickness=1,
                                highlightbackground=gcol)
                card.pack(fill="x", padx=6, pady=(8, 1))

                hdr = tk.Frame(card, bg=CARD_BG)
                hdr.pack(fill="x", padx=8, pady=(5, 0))
                tk.Label(hdr, text=gname,
                         font=("Courier New", 9, "bold"),
                         fg=gcol, bg=CARD_BG).pack(side="left")
                tk.Label(hdr, text=gdesc,
                         font=("Courier New", 7),
                         fg=TEXT_DIM, bg=CARD_BG).pack(side="left", padx=8)

                # Fisheye model toggle
                if group == "fisheye":
                    mrow = tk.Frame(card, bg=CARD_BG)
                    mrow.pack(fill="x", padx=8, pady=(4, 2))
                    tk.Label(mrow, text="MODEL:",
                             font=("Courier New", 7, "bold"), fg=TEXT_DIM,
                             bg=CARD_BG).pack(side="left", padx=(0,4))
                    for val, lbl, col in [
                        ("fisheye",   "CV FISHEYE",  ACCENT3),
                        ("standard",  "CV STANDARD", ACCENT4),
                        ("defisheye", "DEFISHEYE",   ACCENT ),
                    ]:
                        tk.Radiobutton(mrow, text=lbl,
                                       variable=self.lens_model, value=val,
                                       command=self._schedule_refresh,
                                       font=("Courier New", 7, "bold"),
                                       fg=col, bg=CARD_BG, selectcolor=TROUGH,
                                       activebackground=CARD_BG,
                                       indicatoron=0, relief="flat", bd=0,
                                       padx=5, pady=2,
                                       ).pack(side="left", padx=2)

                    # defisheye projection type
                    dtrow = tk.Frame(card, bg=CARD_BG)
                    dtrow.pack(fill="x", padx=8, pady=(2, 2))
                    tk.Label(dtrow, text="projection (DEFISHEYE only):",
                             font=("Courier New", 7), fg=TEXT_DIM,
                             bg=CARD_BG).pack(side="left")
                    for val, lbl in [("equalarea","equal-area"),
                                     ("equidistant","equidist."),
                                     ("orthographic","ortho"),
                                     ("stereographic","stereo")]:
                        tk.Radiobutton(dtrow, text=lbl,
                                       variable=self.dftype, value=val,
                                       command=self._schedule_refresh,
                                       font=("Courier New", 7),
                                       fg=ACCENT, bg=CARD_BG,
                                       selectcolor=TROUGH,
                                       activebackground=CARD_BG,
                                       indicatoron=0, relief="flat", bd=0,
                                       padx=4, pady=1,
                                       ).pack(side="left", padx=1)

                    # k1 quick-set presets
                    prow = tk.Frame(card, bg=CARD_BG)
                    prow.pack(fill="x", padx=8, pady=(2, 6))
                    tk.Label(prow, text="k1 presets:",
                             font=("Courier New", 7), fg=TEXT_DIM,
                             bg=CARD_BG).pack(side="left")
                    for v, lbl in [(0.0,"0"), (-0.2,"−0.2"), (-0.4,"−0.4"),
                                   (-0.6,"−0.6"), (0.2,"+0.2"), (0.4,"+0.4"),
                                   (0.6,"+0.6")]:
                        tk.Button(prow, text=lbl,
                                  font=("Courier New", 7),
                                  bg=TROUGH, fg=TEXT_MAIN,
                                  activebackground=ACCENT3,
                                  activeforeground=DARK_BG,
                                  relief="flat", bd=0, padx=4, pady=1,
                                  command=lambda val=v: self._set_param("k1", val)
                                  ).pack(side="left", padx=1)

                # Orientation extras
                if group == "orient":
                    extras = tk.Frame(card, bg=CARD_BG)
                    extras.pack(fill="x", padx=8, pady=(2, 4))
                    # Flip toggles
                    for var, lbl in [(self.flip_h, "Flip H"), (self.flip_v, "Flip V")]:
                        tk.Checkbutton(extras, text=lbl, variable=var,
                                       command=self._schedule_refresh,
                                       bg=CARD_BG, fg=TEXT_MAIN,
                                       selectcolor=TROUGH,
                                       activebackground=CARD_BG,
                                       font=("Courier New", 8)
                                       ).pack(side="left", padx=(0, 8))
                    # Quick rotation buttons
                    tk.Label(extras, text="|", fg=BORDER, bg=CARD_BG,
                             font=("Courier New", 8)).pack(side="left", padx=4)
                    for deg, lbl in [(-90,"−90°"),(180,"180°"),(90,"+90°"),(0,"0°")]:
                        tk.Button(extras, text=lbl,
                                  font=("Courier New", 7, "bold"),
                                  bg=TROUGH, fg=TEXT_MAIN,
                                  activebackground=ACCENT2,
                                  activeforeground=DARK_BG,
                                  relief="flat", bd=0, padx=5, pady=1,
                                  command=lambda d=deg: self._set_param("rotation", d)
                                  ).pack(side="left", padx=1)

            self._make_slider(card, key, label, lo, hi, color)

    def _make_slider(self, parent, key, label, lo, hi, color):
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill="x", padx=6, pady=2)

        # Label
        tk.Label(row, text=label,
                 font=("Courier New", 8),
                 fg=TEXT_MAIN, bg=CARD_BG,
                 width=19, anchor="w").pack(side="left")

        # Value label — click to type
        val_lbl = tk.Label(row,
                           font=("Courier New", 8, "bold"),
                           fg=color, bg=CARD_BG,
                           width=9, anchor="e",
                           cursor="xterm")
        val_lbl.pack(side="right")
        val_lbl.bind("<Button-1>",
                     lambda e, k=key: self._inline_edit(k, val_lbl))
        self.val_labels[key] = val_lbl

        # − / + nudge buttons
        tk.Button(row, text="−",
                  font=("Courier New", 9, "bold"),
                  bg=TROUGH, fg=ACCENT2,
                  activebackground=ACCENT2, activeforeground=DARK_BG,
                  relief="flat", bd=0, width=2,
                  command=lambda k=key, r=(hi-lo): self._nudge(k, -r)
                  ).pack(side="left", padx=(2, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("F.Horizontal.TScale",
                        background=CARD_BG, troughcolor=TROUGH,
                        sliderlength=16, sliderrelief="flat")

        sc = ttk.Scale(row, from_=lo, to=hi,
                       variable=self.params[key],
                       orient="horizontal", length=145,
                       style="F.Horizontal.TScale",
                       command=lambda v, k=key: self._on_slider_move(k))
        sc.pack(side="left", padx=2)
        # Mouse-wheel on the slider itself
        sc.bind("<MouseWheel>",
                lambda e, k=key, r=(hi-lo): self._wheel_nudge(e, k, r))
        sc.bind("<Button-4>",   # Linux scroll up
                lambda e, k=key, r=(hi-lo): self._wheel_nudge(e, k, r))
        sc.bind("<Button-5>",   # Linux scroll down
                lambda e, k=key, r=(hi-lo): self._wheel_nudge(e, k, r))

        tk.Button(row, text="+",
                  font=("Courier New", 9, "bold"),
                  bg=TROUGH, fg=ACCENT,
                  activebackground=ACCENT, activeforeground=DARK_BG,
                  relief="flat", bd=0, width=2,
                  command=lambda k=key, r=(hi-lo): self._nudge(k, +r)
                  ).pack(side="left", padx=(0, 2))

        self._sliders[key] = (sc, lo, hi)
        self._update_val_label(key)

    # ─────────────────────────────────────────
    #  Slider / param helpers
    # ─────────────────────────────────────────
    def _on_slider_move(self, key):
        self._update_val_label(key)
        self._schedule_refresh()

    def _update_val_label(self, key):
        v = self.params[key].get()
        if key == "rotation":
            txt = f"{v:+.1f}°"
        elif key in ("cx_offset", "cy_offset", "brightness"):
            txt = f"{v:+.1f}"
        elif abs(v) >= 10:
            txt = f"{v:.3f}"
        else:
            txt = f"{v:+.4f}"
        self.val_labels[key].config(text=txt)

    def _set_param(self, key, value):
        _, lo, hi = self._sliders[key]
        self.params[key].set(float(np.clip(value, lo, hi)))
        self._update_val_label(key)
        self._schedule_refresh()

    def _nudge(self, key, full_range):
        """Nudge by sensitivity fraction of the full range."""
        frac  = SENSITIVITY[self.sensitivity.get()]
        delta = full_range * frac
        self._set_param(key, self.params[key].get() + delta)

    def _wheel_nudge(self, event, key, full_range):
        """Mouse wheel on a slider — scroll panel only when pointer is outside."""
        frac  = SENSITIVITY[self.sensitivity.get()]
        delta = full_range * frac
        # Positive delta = scroll up = increase value
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._set_param(key, self.params[key].get() + delta)
        else:
            self._set_param(key, self.params[key].get() - delta)
        return "break"  # prevent panel scroll when wheeling a slider

    def _on_global_mousewheel(self, event):
        """Scroll the slider panel — but only if pointer is over the canvas, not a slider."""
        widget = event.widget
        # If the event is on a Scale, let _wheel_nudge handle it
        if isinstance(widget, ttk.Scale):
            return
        self.sl_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _inline_edit(self, key, val_lbl):
        """Replace value label with an Entry for direct typing."""
        _, lo, hi = self._sliders[key]
        cur = f"{self.params[key].get():.5f}"

        entry_var = tk.StringVar(value=cur)
        entry = tk.Entry(val_lbl.master,
                         textvariable=entry_var,
                         font=("Courier New", 8, "bold"),
                         fg=ACCENT, bg=TROUGH,
                         insertbackground=ACCENT,
                         relief="flat", width=9,
                         justify="right")
        # Place it exactly where the label is
        entry.place(in_=val_lbl, relx=0, rely=0, relwidth=1, relheight=1)
        entry.select_range(0, "end")
        entry.focus_set()

        def commit(_e=None):
            try:
                v = float(entry_var.get())
                self._set_param(key, v)
            except ValueError:
                pass
            entry.destroy()

        entry.bind("<Return>",   commit)
        entry.bind("<Tab>",      commit)
        entry.bind("<Escape>",   lambda _e: entry.destroy())
        entry.bind("<FocusOut>", commit)

    def _on_sensitivity_change(self):
        # Rescale slider window symmetrically around current value
        mult = SENSITIVITY[self.sensitivity.get()]
        for key, (sc, lo_full, hi_full) in self._sliders.items():
            cur  = self.params[key].get()
            half = (hi_full - lo_full) * mult * 5   # half-window
            # Try symmetric first
            new_lo = cur - half
            new_hi = cur + half
            # If we'd go outside the hard limits, shift the window rather than squash it
            if new_lo < lo_full:
                new_lo = lo_full
                new_hi = min(hi_full, lo_full + half * 2)
            if new_hi > hi_full:
                new_hi = hi_full
                new_lo = max(lo_full, hi_full - half * 2)
            # Safety: never degenerate
            if new_hi - new_lo < 1e-9:
                new_lo, new_hi = lo_full, hi_full
            sc.config(from_=new_lo, to=new_hi)
        self.status_var.set(
            f"Sensitivity: {self.sensitivity.get()}  ·  "
            f"scroll mouse wheel on any slider to nudge")

    def _schedule_refresh(self, *_):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(25, self._refresh_preview)

    def _reset_all(self):
        for key, _, _, _, default, *_ in PARAM_DEFS:
            self.params[key].set(default)
            self._update_val_label(key)
        self.flip_h.set(False)
        self.flip_v.set(False)
        # Restore full slider ranges
        for key, (sc, lo, hi) in self._sliders.items():
            sc.config(from_=lo, to=hi)
        self._schedule_refresh()

    # ─────────────────────────────────────────
    #  Profile save/load
    # ─────────────────────────────────────────
    def _get_profile(self):
        return {
            "params":     {k: v.get() for k, v in self.params.items()},
            "flip_h":     self.flip_h.get(),
            "flip_v":     self.flip_v.get(),
            "lens_model": self.lens_model.get(),
            "dftype":     self.dftype.get(),
        }

    def _apply_profile(self, profile):
        for k, v in profile.get("params", {}).items():
            if k in self.params:
                self.params[k].set(v)
                self._update_val_label(k)
        self.flip_h.set(profile.get("flip_h", False))
        self.flip_v.set(profile.get("flip_v", False))
        self.lens_model.set(profile.get("lens_model", "fisheye"))
        self.dftype.set(profile.get("dftype", "equalarea"))
        self._schedule_refresh()

    def _save_profile(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON profile", "*.json"), ("All", "*.*")],
            initialfile="fisheye_profile.json")
        if path:
            with open(path, "w") as f:
                json.dump(self._get_profile(), f, indent=2)
            self.status_var.set(f"Profile saved → {path}")

    def _load_profile(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON profile", "*.json"), ("All", "*.*")])
        if path:
            try:
                with open(path) as f:
                    self._apply_profile(json.load(f))
                self.status_var.set(f"Profile loaded ← {path}")
            except Exception as e:
                messagebox.showerror("Load error", str(e))

    # ─────────────────────────────────────────
    #  Folder logic
    # ─────────────────────────────────────────
    def _open_folder(self):
        path = filedialog.askdirectory(title="Select image folder")
        if not path:
            return
        files = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        if not files:
            messagebox.showinfo("Empty", "No supported images found.")
            return
        self.folder_path   = path
        self.folder_images = [os.path.join(path, f) for f in files]
        self.folder_index  = 0
        self._img_cache.clear()
        self._cache_order.clear()
        self._update_nav_ui()
        self._load_current()

    def _nav(self, direction):
        if not self.folder_images:
            return
        self.folder_index = (self.folder_index + direction) % len(self.folder_images)
        self._update_nav_ui()
        self._load_current()

    def _update_nav_ui(self):
        n = len(self.folder_images)
        if not n:
            self.nav_label.config(text="no folder loaded", fg=TEXT_DIM)
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            return
        name = os.path.basename(self.folder_images[self.folder_index])
        self.nav_label.config(
            text=f"{self.folder_index+1} / {n}   ·   {name[:44]}",
            fg=TEXT_MAIN)
        s = "normal" if n > 1 else "disabled"
        self.btn_prev.config(state=s)
        self.btn_next.config(state=s)
        short = self.folder_path or ""
        self.folder_label.config(
            text=("…" + short[-48:] if len(short) > 50 else short))

    def _load_current(self):
        if not self.folder_images:
            return
        path = self.folder_images[self.folder_index]
        if path not in self._img_cache:
            img = cv2.imread(path)
            if img is None:
                self.status_var.set(f"⚠ Cannot read: {os.path.basename(path)}")
                return
            if len(self._cache_order) >= 12:
                self._img_cache.pop(self._cache_order.pop(0), None)
            self._img_cache[path] = img
            self._cache_order.append(path)
        self._refresh_preview()

    def _current_image(self):
        if not self.folder_images:
            return None
        return self._img_cache.get(self.folder_images[self.folder_index])

    # ─────────────────────────────────────────
    #  Rendering
    # ─────────────────────────────────────────
    def _get_kw(self):
        kw = {k: v.get() for k, v in self.params.items()}
        kw["flip_h"]     = self.flip_h.get()
        kw["flip_v"]     = self.flip_v.get()
        kw["lens_model"] = self.lens_model.get()
        kw["dftype"]     = self.dftype.get()
        return kw

    def _refresh_preview(self):
        self._after_id = None
        img = self._current_image()
        if img is None:
            self._draw_placeholder()
            return
        kw = self._get_kw()
        try:
            corrected = process_image(img, **kw)
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            return
        orig_d = resize_for_display(img.copy(),       PREVIEW_W, PREVIEW_H)
        corr_d = resize_for_display(corrected.copy(), PREVIEW_W, PREVIEW_H)
        if self.show_grid.get():
            orig_d = draw_grid(orig_d)
            corr_d = draw_grid(corr_d)
        self._render_canvas(self.canvas_orig, orig_d)
        self._render_canvas(self.canvas_corr, corr_d)
        path = self.folder_images[self.folder_index] if self.folder_images else ""
        n, idx = len(self.folder_images), self.folder_index
        self.status_var.set(
            f"{idx+1}/{n}  ·  {os.path.basename(path)}  ·  "
            f"{img.shape[1]}×{img.shape[0]}  ·  "
            f"rot={kw['rotation']:+.1f}°  "
            f"k1={kw['k1']:+.4f}  k2={kw['k2']:+.4f}  "
            f"focal={kw['focal_scale']:.3f}  balance={kw['balance']:.2f}")

    def _render_canvas(self, canvas, img_bgr):
        canvas.delete("all")
        h, w = img_bgr.shape[:2]
        tk_img = cv2_to_tk(img_bgr)
        canvas._tk_img = tk_img
        canvas.create_image((PREVIEW_W-w)//2, (PREVIEW_H-h)//2,
                             anchor="nw", image=tk_img)

    def _draw_placeholder(self):
        for c, t in [(self.canvas_orig, "original"),
                     (self.canvas_corr, "corrected")]:
            c.delete("all")
            c.create_text(PREVIEW_W//2, PREVIEW_H//2,
                          text=f"[ {t} ]\nopen a folder to begin",
                          fill=TEXT_DIM, font=("Courier New", 11),
                          justify="center")

    # ─────────────────────────────────────────
    #  Export
    # ─────────────────────────────────────────
    def _save_image(self):
        img = self._current_image()
        if img is None:
            messagebox.showwarning("No Image", "No image loaded.")
            return
        out  = process_image(img, **self._get_kw())
        base = os.path.splitext(os.path.basename(
            self.folder_images[self.folder_index]))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")],
            initialfile=f"{base}_corrected.png")
        if path:
            cv2.imwrite(path, out)
            self.status_var.set(f"Saved → {path}")

    def _export_folder(self):
        if not self.folder_images:
            messagebox.showwarning("No Folder", "Open a folder first.")
            return
        out_dir = filedialog.askdirectory(title="Choose export folder")
        if not out_dir:
            return
        kw = self._get_kw()
        total, success = len(self.folder_images), 0
        for path in self.folder_images:
            img = self._img_cache.get(path) or cv2.imread(path)
            if img is None:
                continue
            try:
                out  = process_image(img, **kw)
                base = os.path.splitext(os.path.basename(path))[0]
                cv2.imwrite(os.path.join(out_dir, f"{base}_corrected.png"), out)
                success += 1
            except Exception as e:
                print(f"Export failed {path}: {e}")
        self.status_var.set(f"Exported {success}/{total} → {out_dir}")
        messagebox.showinfo("Done",
            f"Exported {success} of {total} images to:\n{out_dir}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = FisheyeApp()
    app.mainloop()