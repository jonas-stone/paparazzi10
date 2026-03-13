# # Note made with the help of AI (Claude)
#
# """
# Image Labeling Tool for Neural Network Data
# - Dual image view (top/bottom), two folders, same filename
# - Draw rectangles on either image — instantly mirrored on the other
# - Zoom (scroll wheel) and pan (middle mouse / right mouse drag) — persists across images
# - Exports binary vector + shape data to JSON & CSV
# """
#
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import json
# import csv
# from pathlib import Path
#
# try:
#     from PIL import Image, ImageTk
#
#     PIL_AVAILABLE = True
# except ImportError:
#     PIL_AVAILABLE = False
#
# # ── Palette ──────────────────────────────────────────────────────────────────
# BG = "#0f1117"
# PANEL = "#1a1d27"
# ACCENT = "#e05c2a"
# ACCENT2 = "#f5a623"
# TEXT = "#e8e8f0"
# SUBTEXT = "#7a7a9a"
# BORDER = "#2a2d3e"
# DANGER_CLR = "#e74c3c"
# HOVER = "#252838"
# LABEL_A = "#a29bfe"
# LABEL_B = "#55efc4"
#
# FONT_TITLE = ("Georgia", 20, "bold")
# FONT_HEAD = ("Georgia", 11, "bold")
# FONT_SMALL = ("Courier New", 9)
# FONT_BTN = ("Georgia", 10, "bold")
#
#
# class DualCanvas:
#     """One image pane. Shares shape state + zoom/pan state with LabelingTool."""
#
#     def __init__(self, parent, label_text, label_color, tool):
#         self.tool = tool
#         self.pil_image = None
#         self.tk_image = None
#
#         outer = tk.Frame(parent, bg=BG)
#         outer.pack(fill=tk.BOTH, expand=True)
#
#         hdr = tk.Frame(outer, bg=PANEL, height=22)
#         hdr.pack(fill=tk.X)
#         hdr.pack_propagate(False)
#         tk.Label(hdr, text=f"  {label_text}", font=FONT_SMALL,
#                  fg=label_color, bg=PANEL).pack(side=tk.LEFT)
#         self.path_var = tk.StringVar(value="No folder selected")
#         tk.Label(hdr, textvariable=self.path_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT, padx=6)
#
#         self.canvas = tk.Canvas(outer, bg="#080a10",
#                                 highlightthickness=0, cursor="crosshair")
#         self.canvas.pack(fill=tk.BOTH, expand=True)
#
#         # Drawing bindings
#         self.canvas.bind("<ButtonPress-1>", self._mouse_down)
#         self.canvas.bind("<B1-Motion>", self._mouse_drag)
#         self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
#
#         # Zoom: scroll wheel
#         self.canvas.bind("<MouseWheel>", self._on_scroll)  # Windows/Mac
#         self.canvas.bind("<Button-4>", self._on_scroll)  # Linux scroll up
#         self.canvas.bind("<Button-5>", self._on_scroll)  # Linux scroll down
#
#         # Pan: middle mouse drag
#         self.canvas.bind("<ButtonPress-2>", self._pan_start)
#         self.canvas.bind("<B2-Motion>", self._pan_move)
#         # Also support right-drag for pan (convenient on trackpads)
#         self.canvas.bind("<ButtonPress-3>", self._pan_start)
#         self.canvas.bind("<B3-Motion>", self._pan_move)
#
#         self.canvas.bind("<Configure>", lambda e: self._render())
#
#         self.drawing = False
#         self.draw_start = None
#         self.current_rect = None
#         self._pan_last = None
#
#     # ── image ─────────────────────────────────────────────────────────────────
#
#     def load_image(self, path):
#         if not PIL_AVAILABLE:
#             return
#         try:
#             self.pil_image = Image.open(path)
#         except Exception as e:
#             self.pil_image = None
#             messagebox.showerror("Load Error", f"Cannot open {path}:\n{e}")
#             return
#         self.path_var.set(str(Path(path).parent))
#         self._render()
#
#     def clear_image(self):
#         self.pil_image = None
#         self.tk_image = None
#         self.canvas.delete("all")
#
#     # ── render ────────────────────────────────────────────────────────────────
#
#     def _render(self):
#         """Render the image using the shared zoom/pan state from tool."""
#         self.canvas.delete("all")
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         if cw < 2 or ch < 2:
#             self.canvas.after(80, self._render)
#             return
#         if not self.pil_image:
#             return
#
#         iw, ih = self.pil_image.size
#         zoom = self.tool.zoom  # current zoom multiplier
#         pan_x = self.tool.pan_x  # pan offset in image pixels
#         pan_y = self.tool.pan_y
#
#         # Base scale so image fits the canvas at zoom=1
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale = base_scale * zoom
#
#         # Visible region in image coordinates
#         # pan_x/pan_y is the image-coord of the canvas top-left corner
#         vis_w = cw / scale
#         vis_h = ch / scale
#
#         # Clamp pan so we don't go way outside the image
#         pan_x = max(-vis_w * 0.5, min(pan_x, iw - vis_w * 0.5))
#         pan_y = max(-vis_h * 0.5, min(pan_y, ih - vis_h * 0.5))
#         self.tool.pan_x = pan_x
#         self.tool.pan_y = pan_y
#
#         # Crop region from original image
#         crop_x0 = int(max(0, pan_x))
#         crop_y0 = int(max(0, pan_y))
#         crop_x1 = int(min(iw, pan_x + vis_w))
#         crop_y1 = int(min(ih, pan_y + vis_h))
#
#         if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
#             return
#
#         cropped = self.pil_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
#
#         # Scale cropped region to canvas size
#         display_w = int((crop_x1 - crop_x0) * scale)
#         display_h = int((crop_y1 - crop_y0) * scale)
#         if display_w < 1 or display_h < 1:
#             return
#
#         resized = cropped.resize((display_w, display_h), Image.LANCZOS)
#         self.tk_image = ImageTk.PhotoImage(resized)
#
#         # Offset on canvas (image may not fill full canvas if zoomed out)
#         ox = int((pan_x - crop_x0) * -scale) if pan_x < 0 else 0
#         oy = int((pan_y - crop_y0) * -scale) if pan_y < 0 else 0
#
#         self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_image)
#         self._redraw_shapes()
#
#     def rerender(self):
#         self._render()
#
#     # ── coordinate helpers ────────────────────────────────────────────────────
#
#     def _canvas_to_image(self, cx, cy):
#         """Convert canvas pixel → image pixel."""
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale = base_scale * self.tool.zoom
#         ix = cx / scale + self.tool.pan_x
#         iy = cy / scale + self.tool.pan_y
#         return ix, iy
#
#     def _image_to_canvas(self, ix, iy):
#         """Convert image pixel → canvas pixel."""
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale = base_scale * self.tool.zoom
#         cx = (ix - self.tool.pan_x) * scale
#         cy = (iy - self.tool.pan_y) * scale
#         return cx, cy
#
#     # ── zoom ──────────────────────────────────────────────────────────────────
#
#     def _on_scroll(self, event):
#         # Determine scroll direction
#         if event.num == 4 or event.delta > 0:
#             factor = 1.1
#         else:
#             factor = 1 / 1.1
#
#         # Zoom toward the mouse cursor
#         if self.pil_image:
#             ix, iy = self._canvas_to_image(event.x, event.y)
#
#         self.tool.zoom = max(0.2, min(20.0, self.tool.zoom * factor))
#
#         # Adjust pan so the point under the cursor stays fixed
#         if self.pil_image:
#             cw = self.canvas.winfo_width()
#             ch = self.canvas.winfo_height()
#             iw, ih = self.pil_image.size
#             base_scale = min(cw / iw, ch / ih, 1.0)
#             scale = base_scale * self.tool.zoom
#             self.tool.pan_x = ix - event.x / scale
#             self.tool.pan_y = iy - event.y / scale
#
#         # Update zoom label
#         self.tool.zoom_var.set(f"Zoom: {self.tool.zoom:.1f}×")
#
#         # Re-render both panes
#         self.tool.pane_a._render()
#         self.tool.pane_b._render()
#
#     # ── pan ───────────────────────────────────────────────────────────────────
#
#     def _pan_start(self, event):
#         self._pan_last = (event.x, event.y)
#
#     def _pan_move(self, event):
#         if self._pan_last is None or not self.pil_image:
#             return
#         dx = event.x - self._pan_last[0]
#         dy = event.y - self._pan_last[1]
#         self._pan_last = (event.x, event.y)
#
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale = base_scale * self.tool.zoom
#
#         self.tool.pan_x -= dx / scale
#         self.tool.pan_y -= dy / scale
#
#         self.tool.pane_a._render()
#         self.tool.pane_b._render()
#
#     # ── drawing ───────────────────────────────────────────────────────────────
#
#     def _mouse_down(self, event):
#         if not self.tool.image_names:
#             return
#         self.drawing = True
#         self.draw_start = (event.x, event.y)
#         self.current_rect = self.canvas.create_rectangle(
#             event.x, event.y, event.x, event.y,
#             outline=DANGER_CLR, width=2, dash=(4, 3))
#
#     def _mouse_drag(self, event):
#         if self.drawing and self.current_rect:
#             x0, y0 = self.draw_start
#             self.canvas.coords(self.current_rect, x0, y0, event.x, event.y)
#
#     def _mouse_up(self, event):
#         if not self.drawing:
#             return
#         self.drawing = False
#         if self.current_rect:
#             self.canvas.delete(self.current_rect)
#             self.current_rect = None
#
#         x0, y0 = self.draw_start
#         x1, y1 = event.x, event.y
#         if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
#             return
#
#         if not self.pil_image:
#             return
#
#         ix0, iy0 = self._canvas_to_image(x0, y0)
#         ix1, iy1 = self._canvas_to_image(x1, y1)
#
#         iw, ih = self.pil_image.size
#         ix0 = max(0, min(ix0, iw));
#         ix1 = max(0, min(ix1, iw))
#         iy0 = max(0, min(iy0, ih));
#         iy1 = max(0, min(iy1, ih))
#
#         shape = {
#             "type": "rect",
#             "coords": (min(ix0, ix1), min(iy0, iy1),
#                        max(ix0, ix1), max(iy0, iy1))
#         }
#         self.tool._add_shape(shape)
#
#     # ── redraw shapes ─────────────────────────────────────────────────────────
#
#     def _redraw_shapes(self):
#         self.canvas.delete("shape")
#         if not self.pil_image:
#             return
#         for i, shape in enumerate(self.tool.shapes):
#             x0, y0, x1, y1 = shape["coords"]
#             cx0, cy0 = self._image_to_canvas(x0, y0)
#             cx1, cy1 = self._image_to_canvas(x1, y1)
#             self.canvas.create_rectangle(
#                 cx0, cy0, cx1, cy1,
#                 fill=DANGER_CLR, stipple="gray25", outline="", tags="shape")
#             self.canvas.create_rectangle(
#                 cx0, cy0, cx1, cy1,
#                 outline=DANGER_CLR, width=2, tags="shape")
#             self.canvas.create_text(
#                 cx0 + 4, cy0 + 4, anchor=tk.NW,
#                 text=f"⚠ {i + 1}", fill=TEXT,
#                 font=("Courier New", 8, "bold"), tags="shape")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
#
# class LabelingTool:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Danger Zone Labeler")
#         self.root.configure(bg=BG)
#         self.root.minsize(1100, 760)
#
#         self.folder_a = None
#         self.folder_b = None
#         self.image_names = []
#         self.current_index = 0
#         self.shapes = []
#         self.labels = {}
#         self.vectors = {}
#
#         # Shared zoom/pan state — persists across image navigation
#         self.zoom = 1.0
#         self.pan_x = 0.0
#         self.pan_y = 0.0
#
#         self._build_ui()
#         self._bind_keys()
#
#     # ── UI ───────────────────────────────────────────────────────────────────
#
#     def _build_ui(self):
#         title_bar = tk.Frame(self.root, bg=BG, height=58)
#         title_bar.pack(fill=tk.X)
#         title_bar.pack_propagate(False)
#         tk.Label(title_bar, text="◈", font=("Georgia", 24), fg=ACCENT,
#                  bg=BG).pack(side=tk.LEFT, padx=(18, 5), pady=8)
#         tk.Label(title_bar, text="DANGER ZONE LABELER",
#                  font=FONT_TITLE, fg=TEXT, bg=BG).pack(side=tk.LEFT)
#         tk.Label(title_bar, text="  ·  dual-view sync labeling",
#                  font=("Georgia", 10, "italic"), fg=SUBTEXT, bg=BG
#                  ).pack(side=tk.LEFT, pady=14)
#         tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)
#
#         body = tk.Frame(self.root, bg=BG)
#         body.pack(fill=tk.BOTH, expand=True)
#         self._build_sidebar(body)
#         tk.Frame(body, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
#
#         right = tk.Frame(body, bg=BG)
#         right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#
#         self.pane_a = DualCanvas(right, "▲  FOLDER A", LABEL_A, self)
#         tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)
#         self.pane_b = DualCanvas(right, "▼  FOLDER B", LABEL_B, self)
#
#         tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)
#         strip_hdr = tk.Frame(right, bg=PANEL, height=22)
#         strip_hdr.pack(fill=tk.X)
#         strip_hdr.pack_propagate(False)
#         tk.Label(strip_hdr, text="  COLUMN VECTOR  [ 0 = safe · 1 = danger ]",
#                  font=FONT_SMALL, fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
#         self.vec_info_var = tk.StringVar(value="")
#         tk.Label(strip_hdr, textvariable=self.vec_info_var,
#                  font=FONT_SMALL, fg=ACCENT2, bg=PANEL).pack(side=tk.RIGHT, padx=10)
#         self.vector_canvas = tk.Canvas(right, bg="#080a10",
#                                        height=36, highlightthickness=0)
#         self.vector_canvas.pack(fill=tk.X)
#
#         tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)
#         sf = tk.Frame(self.root, bg=PANEL, height=28)
#         sf.pack(fill=tk.X)
#         sf.pack_propagate(False)
#         self.status_var = tk.StringVar(value="Select Folder A to begin.")
#         tk.Label(sf, textvariable=self.status_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL, anchor="w").pack(side=tk.LEFT, padx=12)
#
#     def _build_sidebar(self, parent):
#         sb = tk.Frame(parent, bg=PANEL, width=252)
#         sb.pack(side=tk.LEFT, fill=tk.Y)
#         sb.pack_propagate(False)
#
#         def section(t):
#             f = tk.Frame(sb, bg=PANEL)
#             f.pack(fill=tk.X, padx=14, pady=(13, 3))
#             tk.Label(f, text=t, font=FONT_HEAD, fg=ACCENT2, bg=PANEL).pack(anchor="w")
#             tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=14)
#
#         def btn(pw, t, cmd, color=ACCENT):
#             b = tk.Button(pw, text=t, command=cmd, font=FONT_BTN,
#                           fg=TEXT, bg=color, activebackground=HOVER,
#                           activeforeground=TEXT, relief=tk.FLAT,
#                           cursor="hand2", padx=10, pady=5, bd=0)
#             b.pack(fill=tk.X, padx=14, pady=2)
#             return b
#
#         # FOLDERS
#         section("FOLDERS")
#         btn(sb, "📂  Select Folder A  (primary)", self.select_folder_a)
#         self.lbl_a = tk.Label(sb, text="Not selected", font=FONT_SMALL,
#                               fg=LABEL_A, bg=PANEL, wraplength=220, anchor="w")
#         self.lbl_a.pack(fill=tk.X, padx=18, pady=(0, 6))
#         btn(sb, "📂  Select Folder B  (secondary)", self.select_folder_b, color="#2a2d3e")
#         self.lbl_b = tk.Label(sb, text="Not selected", font=FONT_SMALL,
#                               fg=LABEL_B, bg=PANEL, wraplength=220, anchor="w")
#         self.lbl_b.pack(fill=tk.X, padx=18, pady=(0, 6))
#
#         # IMAGE QUEUE
#         section("IMAGE QUEUE")
#         lf = tk.Frame(sb, bg=BORDER, bd=1, relief=tk.FLAT)
#         lf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 4))
#         scr = tk.Scrollbar(lf, bg=PANEL, troughcolor=PANEL, relief=tk.FLAT)
#         scr.pack(side=tk.RIGHT, fill=tk.Y)
#         self.image_listbox = tk.Listbox(
#             lf, font=FONT_SMALL, fg=TEXT, bg=PANEL,
#             selectbackground=ACCENT, selectforeground=TEXT,
#             relief=tk.FLAT, bd=0, activestyle="none",
#             yscrollcommand=scr.set)
#         self.image_listbox.pack(fill=tk.BOTH, expand=True)
#         scr.config(command=self.image_listbox.yview)
#         self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)
#
#         # NAVIGATE
#         section("NAVIGATE")
#         nav = tk.Frame(sb, bg=PANEL)
#         nav.pack(fill=tk.X, padx=14, pady=3)
#         tk.Button(nav, text="◀  Prev", command=self.prev_image,
#                   font=FONT_BTN, fg=TEXT, bg="#2a2d3e",
#                   activebackground=HOVER, activeforeground=TEXT,
#                   relief=tk.FLAT, cursor="hand2", padx=8, pady=4, bd=0
#                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
#         tk.Button(nav, text="Next  ▶", command=self.next_image,
#                   font=FONT_BTN, fg=TEXT, bg="#2a2d3e",
#                   activebackground=HOVER, activeforeground=TEXT,
#                   relief=tk.FLAT, cursor="hand2", padx=8, pady=4, bd=0
#                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
#         self.progress_var = tk.StringVar(value="— / —")
#         tk.Label(sb, textvariable=self.progress_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL).pack(pady=2)
#
#         # ZOOM
#         section("ZOOM")
#         self.zoom_var = tk.StringVar(value="Zoom: 1.0×")
#         tk.Label(sb, textvariable=self.zoom_var, font=FONT_SMALL,
#                  fg=ACCENT2, bg=PANEL).pack(anchor="w", padx=14, pady=(2, 2))
#         btn(sb, "⊙  Reset Zoom", self._reset_zoom, color="#2a2d3e")
#         tk.Label(sb, text="Scroll = zoom  |  Right-drag = pan",
#                  font=FONT_SMALL, fg=SUBTEXT, bg=PANEL).pack(
#             anchor="w", padx=14, pady=(1, 4))
#
#         # TOOLS
#         section("TOOLS")
#         btn(sb, "↩  Undo Last Box", self.undo_shape, color="#2a2d3e")
#         btn(sb, "✕  Clear This Image", self.clear_shapes, color="#2a2d3e")
#
#         # EXPORT
#         section("EXPORT")
#         btn(sb, "💾  Save Labels (JSON)", self.save_json)
#         btn(sb, "📊  Export Vectors (CSV)", self.export_csv, color="#2a2d3e")
#
#         # STATS
#         section("STATS")
#         self.info_var = tk.StringVar(value="No image loaded.")
#         tk.Label(sb, textvariable=self.info_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL, justify=tk.LEFT,
#                  wraplength=220).pack(anchor="w", padx=14, pady=4)
#
#     def _bind_keys(self):
#         self.root.bind("<Left>", lambda e: self.prev_image())
#         self.root.bind("<Right>", lambda e: self.next_image())
#         self.root.bind("<Control-z>", lambda e: self.undo_shape())
#         self.root.bind("<Delete>", lambda e: self.clear_shapes())
#         self.root.bind("<Control-s>", lambda e: self.save_json())
#         self.root.bind("<r>", lambda e: self._reset_zoom())
#
#     def _reset_zoom(self):
#         self.zoom = 1.0
#         self.pan_x = 0.0
#         self.pan_y = 0.0
#         self.zoom_var.set("Zoom: 1.0×")
#         self.pane_a._render()
#         self.pane_b._render()
#
#     # ── Folders ───────────────────────────────────────────────────────────────
#
#     def select_folder_a(self):
#         folder = filedialog.askdirectory(title="Select Folder A (primary)")
#         if not folder:
#             return
#         self.folder_a = folder
#         self.lbl_a.config(text=folder)
#         self._refresh_image_list()
#
#     def select_folder_b(self):
#         folder = filedialog.askdirectory(title="Select Folder B (secondary)")
#         if not folder:
#             return
#         self.folder_b = folder
#         self.lbl_b.config(text=folder)
#         if self.image_names:
#             self._show_current()
#
#     def _refresh_image_list(self):
#         exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
#         names = sorted(p.name for p in Path(self.folder_a).iterdir()
#                        if p.suffix.lower() in exts)
#         if not names:
#             messagebox.showinfo("No Images", "No supported images found in Folder A.")
#             return
#         self.image_names = names
#         self.current_index = 0
#         self.image_listbox.delete(0, tk.END)
#         for n in names:
#             self.image_listbox.insert(tk.END, "  " + n)
#         self._show_current()
#
#     # ── Display ───────────────────────────────────────────────────────────────
#
#     def _show_current(self):
#         if not self.image_names:
#             return
#         name = self.image_names[self.current_index]
#         self.image_listbox.selection_clear(0, tk.END)
#         self.image_listbox.selection_set(self.current_index)
#         self.image_listbox.see(self.current_index)
#         self.shapes = list(self.labels.get(name, []))
#         self.progress_var.set(f"{self.current_index + 1} / {len(self.image_names)}")
#
#         if self.folder_a:
#             p = Path(self.folder_a) / name
#             self.pane_a.load_image(str(p)) if p.exists() else self.pane_a.clear_image()
#         if self.folder_b:
#             p = Path(self.folder_b) / name
#             self.pane_b.load_image(str(p)) if p.exists() else self.pane_b.clear_image()
#         else:
#             self.pane_b.clear_image()
#
#         self._update_vector()
#         self._update_info()
#
#     def _update_info(self):
#         name = self.image_names[self.current_index] if self.image_names else "—"
#         iw = ih = "?"
#         if self.pane_a.pil_image:
#             iw, ih = self.pane_a.pil_image.size
#         self.info_var.set(
#             f"File: {name}\n"
#             f"Size (A): {iw} × {ih} px\n"
#             f"Boxes: {len(self.shapes)}"
#         )
#
#     # ── Shapes ────────────────────────────────────────────────────────────────
#
#     def _add_shape(self, shape):
#         self.shapes.append(shape)
#         self.labels[self.image_names[self.current_index]] = list(self.shapes)
#         self.pane_a._redraw_shapes()
#         self.pane_b._redraw_shapes()
#         self._update_vector()
#         self._update_info()
#         self.status_var.set(f"Box #{len(self.shapes)} drawn — mirrored on both views.")
#
#     def undo_shape(self):
#         if not self.shapes:
#             return
#         self.shapes.pop()
#         self.labels[self.image_names[self.current_index]] = list(self.shapes)
#         self.pane_a._redraw_shapes()
#         self.pane_b._redraw_shapes()
#         self._update_vector()
#         self._update_info()
#         self.status_var.set("Last box removed.")
#
#     def clear_shapes(self):
#         if not self.image_names:
#             return
#         self.shapes = []
#         self.labels[self.image_names[self.current_index]] = []
#         self.pane_a._redraw_shapes()
#         self.pane_b._redraw_shapes()
#         self._update_vector()
#         self._update_info()
#         self.status_var.set("All boxes cleared for this image.")
#
#     # ── Navigation ────────────────────────────────────────────────────────────
#
#     def prev_image(self):
#         if self.image_names and self.current_index > 0:
#             self.current_index -= 1
#             self._show_current()
#
#     def next_image(self):
#         if self.image_names and self.current_index < len(self.image_names) - 1:
#             self.current_index += 1
#             self._show_current()
#
#     def _on_list_select(self, event):
#         sel = self.image_listbox.curselection()
#         if sel:
#             self.current_index = sel[0]
#             self._show_current()
#
#     # ── Vector ────────────────────────────────────────────────────────────────
#
#     def _compute_vector(self):
#         pane = self.pane_a if self.pane_a.pil_image else self.pane_b
#         if not pane.pil_image:
#             return []
#         iw, _ = pane.pil_image.size
#         vector = [0] * iw
#         for shape in self.shapes:
#             x0, _, x1, _ = shape["coords"]
#             for col in range(max(0, int(x0)), min(iw, int(x1) + 1)):
#                 vector[col] = 1
#         return vector
#
#     def _update_vector(self):
#         vector = self._compute_vector()
#         if not vector:
#             self.vector_canvas.delete("all")
#             return
#         if self.image_names:
#             self.vectors[self.image_names[self.current_index]] = vector
#         vc = self.vector_canvas
#         vc.delete("all")
#         vw = vc.winfo_width()
#         vh = vc.winfo_height() or 36
#         n = len(vector)
#         if n == 0 or vw < 2:
#             return
#         px = max(1, vw / n)
#         for i, val in enumerate(vector):
#             x0 = int(i * px);
#             x1 = int((i + 1) * px)
#             vc.create_rectangle(x0, 0, x1, vh,
#                                 fill=DANGER_CLR if val else "#1a2a1a", outline="")
#         danger = sum(vector)
#         pct = danger / n * 100
#         self.vec_info_var.set(
#             f"Width: {n}  |  Danger: {danger} ({pct:.1f}%)  |  Safe: {n - danger}"
#         )
#
#     # ── Export ────────────────────────────────────────────────────────────────
#
#     def save_json(self):
#         if not self.labels:
#             messagebox.showinfo("Nothing to save", "No boxes drawn yet.")
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".json",
#                                             filetypes=[("JSON", "*.json")])
#         if not path:
#             return
#         output = {}
#         for name, shapes in self.labels.items():
#             output[name] = {
#                 "filename": name,
#                 "path_a": str(Path(self.folder_a) / name) if self.folder_a else None,
#                 "path_b": str(Path(self.folder_b) / name) if self.folder_b else None,
#                 "boxes": [
#                     {"x0": round(s["coords"][0], 2), "y0": round(s["coords"][1], 2),
#                      "x1": round(s["coords"][2], 2), "y1": round(s["coords"][3], 2)}
#                     for s in shapes
#                 ],
#                 "vector": self.vectors.get(name, [])
#             }
#         with open(path, "w") as f:
#             json.dump(output, f, indent=2)
#         self.status_var.set(f"Saved → {path}")
#         messagebox.showinfo("Saved", f"Labels saved to:\n{path}")
#
#     def export_csv(self):
#         if not self.vectors:
#             messagebox.showinfo("Nothing to export", "No vectors computed yet.")
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".csv",
#                                             filetypes=[("CSV", "*.csv")])
#         if not path:
#             return
#         with open(path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["filename", "path_a", "path_b", "vector"])
#             for name, vec in self.vectors.items():
#                 writer.writerow([
#                     name,
#                     str(Path(self.folder_a) / name) if self.folder_a else "",
#                     str(Path(self.folder_b) / name) if self.folder_b else "",
#                     "[" + ",".join(str(v) for v in vec) + "]"
#                 ])
#         self.status_var.set(f"Exported → {path}")
#         messagebox.showinfo("Exported", f"Vectors exported to:\n{path}")
#
#
# # ── Entry point ───────────────────────────────────────────────────────────────
#
# def main():
#     if not PIL_AVAILABLE:
#         root2 = tk.Tk()
#         root2.withdraw()
#         go = messagebox.askyesno(
#             "Pillow not installed",
#             "Pillow is required.\n\nInstall with:  pip install Pillow\n\nContinue anyway?"
#         )
#         root2.destroy()
#         if not go:
#             return
#
#     root = tk.Tk()
#     root.geometry("1280x860")
#     LabelingTool(root)
#     root.mainloop()
#
#
# if __name__ == "__main__":
#     main()
#
#
# """
# Image Labeling Tool for Neural Network Data
# - Dual image view (top/bottom), two folders, same filename
# - Draw rectangles on either image — instantly mirrored on the other
# - Scroll to zoom, right-drag to pan — both persist across image switches
# - Exports binary vector + shape data to JSON & CSV
# """
#
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import json
# import csv
# from pathlib import Path
#
# try:
#     from PIL import Image, ImageTk
#     PIL_AVAILABLE = True
# except ImportError:
#     PIL_AVAILABLE = False
#
# # ── Palette ──────────────────────────────────────────────────────────────────
# BG         = "#0f1117"
# PANEL      = "#1a1d27"
# ACCENT     = "#e05c2a"
# ACCENT2    = "#f5a623"
# TEXT       = "#e8e8f0"
# SUBTEXT    = "#7a7a9a"
# BORDER     = "#2a2d3e"
# DANGER_CLR = "#e74c3c"
# HOVER      = "#252838"
# LABEL_A    = "#a29bfe"
# LABEL_B    = "#55efc4"
#
# FONT_TITLE = ("Georgia", 20, "bold")
# FONT_HEAD  = ("Georgia", 11, "bold")
# FONT_SMALL = ("Courier New", 9)
# FONT_BTN   = ("Georgia", 10, "bold")
#
#
# class DualCanvas:
#     """One image pane. Zoom/pan state lives in LabelingTool and is shared."""
#
#     def __init__(self, parent, label_text, label_color, tool):
#         self.tool      = tool
#         self.pil_image = None
#         self.tk_image  = None
#
#         outer = tk.Frame(parent, bg=BG)
#         outer.pack(fill=tk.BOTH, expand=True)
#
#         hdr = tk.Frame(outer, bg=PANEL, height=22)
#         hdr.pack(fill=tk.X)
#         hdr.pack_propagate(False)
#         tk.Label(hdr, text=f"  {label_text}", font=FONT_SMALL,
#                  fg=label_color, bg=PANEL).pack(side=tk.LEFT)
#         self.path_var = tk.StringVar(value="No folder selected")
#         tk.Label(hdr, textvariable=self.path_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT, padx=6)
#
#         self.canvas = tk.Canvas(outer, bg="#080a10",
#                                 highlightthickness=0, cursor="crosshair")
#         self.canvas.pack(fill=tk.BOTH, expand=True)
#
#         # Drawing
#         self.canvas.bind("<ButtonPress-1>",   self._mouse_down)
#         self.canvas.bind("<B1-Motion>",        self._mouse_drag)
#         self.canvas.bind("<ButtonRelease-1>",  self._mouse_up)
#
#         # Zoom — scroll wheel (Windows/Mac) and Button-4/5 (Linux)
#         self.canvas.bind("<MouseWheel>", self._on_scroll)
#         self.canvas.bind("<Button-4>",   self._on_scroll)
#         self.canvas.bind("<Button-5>",   self._on_scroll)
#
#         # Pan — right-click drag
#         self.canvas.bind("<ButtonPress-3>",  self._pan_start)
#         self.canvas.bind("<B3-Motion>",       self._pan_move)
#
#         self.canvas.bind("<Configure>", lambda e: self._render())
#
#         self.drawing      = False
#         self.draw_start   = None
#         self.current_rect = None
#         self._pan_last    = None
#
#     # ── image ─────────────────────────────────────────────────────────────────
#
#     def load_image(self, path):
#         if not PIL_AVAILABLE:
#             return
#         try:
#             self.pil_image = Image.open(path)
#         except Exception as e:
#             self.pil_image = None
#             messagebox.showerror("Load Error", f"Cannot open {path}:\n{e}")
#             return
#         self.path_var.set(str(Path(path).parent))
#         self._render()
#
#     def clear_image(self):
#         self.pil_image = None
#         self.tk_image  = None
#         self.canvas.delete("all")
#
#     # ── render ────────────────────────────────────────────────────────────────
#
#     def _render(self):
#         self.canvas.delete("all")
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         if cw < 2 or ch < 2:
#             self.canvas.after(80, self._render)
#             return
#         if not self.pil_image:
#             return
#
#         iw, ih     = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)   # fit-to-canvas scale at zoom=1
#         scale      = base_scale * self.tool.zoom   # actual render scale
#
#         # Visible region in image coordinates given current pan
#         vis_w = cw / scale
#         vis_h = ch / scale
#
#         # Clamp pan so we don't wander too far outside the image
#         self.tool.pan_x = max(-vis_w * 0.5, min(self.tool.pan_x, iw - vis_w * 0.5))
#         self.tool.pan_y = max(-vis_h * 0.5, min(self.tool.pan_y, ih - vis_h * 0.5))
#
#         # Crop the part of the image that's visible
#         cx0 = int(max(0, self.tool.pan_x))
#         cy0 = int(max(0, self.tool.pan_y))
#         cx1 = int(min(iw, self.tool.pan_x + vis_w))
#         cy1 = int(min(ih, self.tool.pan_y + vis_h))
#         if cx1 <= cx0 or cy1 <= cy0:
#             return
#
#         cropped = self.pil_image.crop((cx0, cy0, cx1, cy1))
#
#         dw = int((cx1 - cx0) * scale)
#         dh = int((cy1 - cy0) * scale)
#         if dw < 1 or dh < 1:
#             return
#
#         resized       = cropped.resize((dw, dh), Image.LANCZOS)
#         self.tk_image = ImageTk.PhotoImage(resized)
#
#         # Canvas offset — non-zero only when zoomed out and image doesn't fill canvas
#         ox = int(-self.tool.pan_x * scale) if self.tool.pan_x < 0 else 0
#         oy = int(-self.tool.pan_y * scale) if self.tool.pan_y < 0 else 0
#
#         self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_image)
#         self._redraw_shapes()
#
#     def rerender(self):
#         self._render()
#
#     # ── coordinate helpers ────────────────────────────────────────────────────
#
#     def _canvas_to_image(self, cx, cy):
#         """Canvas pixel → image pixel."""
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih     = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale      = base_scale * self.tool.zoom
#         return cx / scale + self.tool.pan_x, cy / scale + self.tool.pan_y
#
#     def _image_to_canvas(self, ix, iy):
#         """Image pixel → canvas pixel."""
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih     = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale      = base_scale * self.tool.zoom
#         return (ix - self.tool.pan_x) * scale, (iy - self.tool.pan_y) * scale
#
#     # ── zoom ──────────────────────────────────────────────────────────────────
#
#     def _on_scroll(self, event):
#         if not self.pil_image:
#             return
#
#         # Direction
#         if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
#             factor = 1.15
#         else:
#             factor = 1 / 1.15
#
#         # Image coordinate under the cursor before zoom
#         ix, iy = self._canvas_to_image(event.x, event.y)
#
#         # Apply zoom (clamped)
#         self.tool.zoom = max(0.2, min(20.0, self.tool.zoom * factor))
#         self.tool.zoom_var.set(f"Zoom: {self.tool.zoom:.1f}×")
#
#         # Adjust pan so the point under the cursor stays fixed
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih     = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale      = base_scale * self.tool.zoom
#         self.tool.pan_x = ix - event.x / scale
#         self.tool.pan_y = iy - event.y / scale
#
#         self.tool.pane_a._render()
#         self.tool.pane_b._render()
#
#     # ── pan ───────────────────────────────────────────────────────────────────
#
#     def _pan_start(self, event):
#         self._pan_last = (event.x, event.y)
#
#     def _pan_move(self, event):
#         if self._pan_last is None or not self.pil_image:
#             return
#         dx = event.x - self._pan_last[0]
#         dy = event.y - self._pan_last[1]
#         self._pan_last = (event.x, event.y)
#
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         iw, ih     = self.pil_image.size
#         base_scale = min(cw / iw, ch / ih, 1.0)
#         scale      = base_scale * self.tool.zoom
#
#         self.tool.pan_x -= dx / scale
#         self.tool.pan_y -= dy / scale
#
#         self.tool.pane_a._render()
#         self.tool.pane_b._render()
#
#     # ── drawing ───────────────────────────────────────────────────────────────
#
#     def _mouse_down(self, event):
#         if not self.tool.image_names:
#             return
#         self.drawing    = True
#         self.draw_start = (event.x, event.y)
#         self.current_rect = self.canvas.create_rectangle(
#             event.x, event.y, event.x, event.y,
#             outline=DANGER_CLR, width=2, dash=(4, 3))
#
#     def _mouse_drag(self, event):
#         if self.drawing and self.current_rect:
#             x0, y0 = self.draw_start
#             self.canvas.coords(self.current_rect, x0, y0, event.x, event.y)
#
#     def _mouse_up(self, event):
#         if not self.drawing:
#             return
#         self.drawing = False
#         if self.current_rect:
#             self.canvas.delete(self.current_rect)
#             self.current_rect = None
#
#         x0, y0 = self.draw_start
#         x1, y1 = event.x, event.y
#         if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
#             return
#         if not self.pil_image:
#             return
#
#         ix0, iy0 = self._canvas_to_image(x0, y0)
#         ix1, iy1 = self._canvas_to_image(x1, y1)
#         iw, ih   = self.pil_image.size
#         ix0 = max(0, min(ix0, iw));  ix1 = max(0, min(ix1, iw))
#         iy0 = max(0, min(iy0, ih));  iy1 = max(0, min(iy1, ih))
#
#         self.tool._add_shape({
#             "type":   "rect",
#             "coords": (min(ix0, ix1), min(iy0, iy1),
#                        max(ix0, ix1), max(iy0, iy1))
#         })
#
#     # ── redraw shapes ─────────────────────────────────────────────────────────
#
#     def _redraw_shapes(self):
#         self.canvas.delete("shape")
#         if not self.pil_image:
#             return
#         for i, shape in enumerate(self.tool.shapes):
#             x0, y0, x1, y1 = shape["coords"]
#             cx0, cy0 = self._image_to_canvas(x0, y0)
#             cx1, cy1 = self._image_to_canvas(x1, y1)
#             self.canvas.create_rectangle(
#                 cx0, cy0, cx1, cy1,
#                 fill=DANGER_CLR, stipple="gray25", outline="", tags="shape")
#             self.canvas.create_rectangle(
#                 cx0, cy0, cx1, cy1,
#                 outline=DANGER_CLR, width=2, tags="shape")
#             self.canvas.create_text(
#                 cx0 + 4, cy0 + 4, anchor=tk.NW,
#                 text=f"⚠ {i + 1}", fill=TEXT,
#                 font=("Courier New", 8, "bold"), tags="shape")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
#
# class LabelingTool:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Danger Zone Labeler")
#         self.root.configure(bg=BG)
#         self.root.minsize(1100, 760)
#
#         self.folder_a      = None
#         self.folder_b      = None
#         self.image_names   = []
#         self.current_index = 0
#         self.shapes        = []
#         self.labels        = {}
#         self.vectors       = {}
#
#         # How many path parts to match when loading filter JSON (1 = filename only, 2 = folder/filename, etc.)
#         self.filter_depth = 1
#
#         # Set of filenames explicitly excluded from export (shown red in list)
#         self.excluded = set()
#
#         # Optional filter: set of filenames (just the name, no path) to show.
#         # If None, all images in Folder A are shown.
#         self.filter_names  = None
#
#         # Zoom/pan — stored here so they persist when switching images
#         self.zoom  = 1.0
#         self.pan_x = 0.0
#         self.pan_y = 0.0
#
#         self._build_ui()
#         self._bind_keys()
#
#     # ── UI ───────────────────────────────────────────────────────────────────
#
#     def _build_ui(self):
#         title_bar = tk.Frame(self.root, bg=BG, height=58)
#         title_bar.pack(fill=tk.X)
#         title_bar.pack_propagate(False)
#         tk.Label(title_bar, text="◈", font=("Georgia", 24), fg=ACCENT,
#                  bg=BG).pack(side=tk.LEFT, padx=(18, 5), pady=8)
#         tk.Label(title_bar, text="DANGER ZONE LABELER",
#                  font=FONT_TITLE, fg=TEXT, bg=BG).pack(side=tk.LEFT)
#         tk.Label(title_bar, text="  ·  dual-view sync labeling",
#                  font=("Georgia", 10, "italic"), fg=SUBTEXT, bg=BG
#                  ).pack(side=tk.LEFT, pady=14)
#         tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)
#
#         body = tk.Frame(self.root, bg=BG)
#         body.pack(fill=tk.BOTH, expand=True)
#         self._build_sidebar(body)
#         tk.Frame(body, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
#
#         right = tk.Frame(body, bg=BG)
#         right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#
#         self.pane_a = DualCanvas(right, "▲  FOLDER A", LABEL_A, self)
#         tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)
#         self.pane_b = DualCanvas(right, "▼  FOLDER B", LABEL_B, self)
#
#         tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)
#         strip_hdr = tk.Frame(right, bg=PANEL, height=22)
#         strip_hdr.pack(fill=tk.X)
#         strip_hdr.pack_propagate(False)
#         tk.Label(strip_hdr, text="  COLUMN VECTOR  [ 0 = safe · 1 = danger ]",
#                  font=FONT_SMALL, fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
#         self.vec_info_var = tk.StringVar(value="")
#         tk.Label(strip_hdr, textvariable=self.vec_info_var,
#                  font=FONT_SMALL, fg=ACCENT2, bg=PANEL).pack(side=tk.RIGHT, padx=10)
#         self.vector_canvas = tk.Canvas(right, bg="#080a10",
#                                         height=36, highlightthickness=0)
#         self.vector_canvas.pack(fill=tk.X)
#
#         tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)
#         sf = tk.Frame(self.root, bg=PANEL, height=28)
#         sf.pack(fill=tk.X)
#         sf.pack_propagate(False)
#         self.status_var = tk.StringVar(value="Select Folder A to begin.")
#         tk.Label(sf, textvariable=self.status_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL, anchor="w").pack(side=tk.LEFT, padx=12)
#
#     def _build_sidebar(self, parent):
#         sb = tk.Frame(parent, bg=PANEL, width=252)
#         sb.pack(side=tk.LEFT, fill=tk.Y)
#         sb.pack_propagate(False)
#
#         def section(t):
#             f = tk.Frame(sb, bg=PANEL)
#             f.pack(fill=tk.X, padx=14, pady=(8, 2))
#             tk.Label(f, text=t, font=FONT_HEAD, fg=ACCENT2, bg=PANEL).pack(anchor="w")
#             tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=14)
#
#         def btn(pw, t, cmd, color=ACCENT):
#             b = tk.Button(pw, text=t, command=cmd, font=FONT_BTN,
#                           fg=TEXT, bg=color, activebackground=HOVER,
#                           activeforeground=TEXT, relief=tk.FLAT,
#                           cursor="hand2", padx=10, pady=5, bd=0)
#             b.pack(fill=tk.X, padx=14, pady=2)
#             return b
#
#         # FOLDERS
#         section("FOLDERS")
#         btn(sb, "📂  Select Folder A  (primary)", self.select_folder_a)
#         self.lbl_a = tk.Label(sb, text="Not selected", font=FONT_SMALL,
#                                fg=LABEL_A, bg=PANEL, wraplength=220, anchor="w")
#         self.lbl_a.pack(fill=tk.X, padx=18, pady=(0, 6))
#         btn(sb, "📂  Select Folder B  (secondary)",
#             self.select_folder_b, color="#2a2d3e")
#         self.lbl_b = tk.Label(sb, text="Not selected", font=FONT_SMALL,
#                                fg=LABEL_B, bg=PANEL, wraplength=220, anchor="w")
#         self.lbl_b.pack(fill=tk.X, padx=18, pady=(0, 6))
#
#         # FILTER
#         section("FILTER  (optional)")
#         btn(sb, "🔍  Load Filter JSON", self.load_filter_json, color="#2a2d3e")
#         self.lbl_filter = tk.Label(sb, text="No filter — showing all images",
#                                     font=FONT_SMALL, fg=SUBTEXT, bg=PANEL,
#                                     wraplength=220, anchor="w")
#         self.lbl_filter.pack(fill=tk.X, padx=18, pady=(0, 2))
#         # Path depth selector
#         depth_row = tk.Frame(sb, bg=PANEL)
#         depth_row.pack(fill=tk.X, padx=14, pady=(0, 4))
#         tk.Label(depth_row, text="Match last", font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
#         self.depth_var = tk.IntVar(value=1)
#         depth_spin = tk.Spinbox(depth_row, from_=1, to=10, width=3,
#                                 textvariable=self.depth_var,
#                                 font=FONT_SMALL, fg=TEXT, bg=PANEL,
#                                 buttonbackground=PANEL, relief=tk.FLAT,
#                                 command=self._on_depth_change)
#         depth_spin.pack(side=tk.LEFT, padx=4)
#         tk.Label(depth_row, text="path parts", font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
#         tk.Label(sb, text="(1 = filename only, 2 = folder/file)",
#                  font=FONT_SMALL, fg=SUBTEXT, bg=PANEL
#                  ).pack(anchor="w", padx=14, pady=(0, 2))
#         btn(sb, "✕  Clear Filter", self.clear_filter, color="#2a2d3e")
#
#         # IMAGE QUEUE
#         section("IMAGE QUEUE")
#         lf = tk.Frame(sb, bg=BORDER, bd=1, relief=tk.FLAT)
#         lf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 4))
#         scr = tk.Scrollbar(lf, bg=PANEL, troughcolor=PANEL, relief=tk.FLAT)
#         scr.pack(side=tk.RIGHT, fill=tk.Y)
#         self.image_listbox = tk.Listbox(
#             lf, font=FONT_SMALL, fg=TEXT, bg=PANEL,
#             selectbackground=ACCENT, selectforeground=TEXT,
#             relief=tk.FLAT, bd=0, activestyle="none",
#             yscrollcommand=scr.set)
#         self.image_listbox.pack(fill=tk.BOTH, expand=True)
#         scr.config(command=self.image_listbox.yview)
#         self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)
#
#         # NAVIGATE
#         section("NAVIGATE")
#         nav = tk.Frame(sb, bg=PANEL)
#         nav.pack(fill=tk.X, padx=14, pady=3)
#         tk.Button(nav, text="◀  Prev", command=self.prev_image,
#                   font=FONT_BTN, fg=TEXT, bg="#2a2d3e",
#                   activebackground=HOVER, activeforeground=TEXT,
#                   relief=tk.FLAT, cursor="hand2", padx=8, pady=4, bd=0
#                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
#         tk.Button(nav, text="Next  ▶", command=self.next_image,
#                   font=FONT_BTN, fg=TEXT, bg="#2a2d3e",
#                   activebackground=HOVER, activeforeground=TEXT,
#                   relief=tk.FLAT, cursor="hand2", padx=8, pady=4, bd=0
#                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
#         self.progress_var = tk.StringVar(value="— / —")
#         tk.Label(sb, textvariable=self.progress_var, font=FONT_SMALL,
#                  fg=SUBTEXT, bg=PANEL).pack(pady=2)
#
#         # zoom_var still exists so the rest of the code doesn't break
#         self.zoom_var = tk.StringVar(value="Zoom: 1.0×")
#
#         # TOOLS
#         section("TOOLS")
#         btn(sb, "↩  Undo Last Box",      self.undo_shape,   color="#2a2d3e")
#         btn(sb, "✕  Clear This Image",    self.clear_shapes, color="#2a2d3e")
#         btn(sb, "🚫  Toggle Exclude",     self.toggle_exclude, color="#5a1a1a")
#
#         # EXPORT
#         section("EXPORT")
#         btn(sb, "💾  Save Labels (JSON)",   self.save_json)
#         btn(sb, "📊  Export Vectors (CSV)", self.export_csv, color="#2a2d3e")
#
#         # info_var still used internally
#         self.info_var = tk.StringVar(value="No image loaded.")
#
#     def _bind_keys(self):
#         self.root.bind("<Left>",      lambda e: self.prev_image())
#         self.root.bind("<Right>",     lambda e: self.next_image())
#         self.root.bind("<Control-z>", lambda e: self.undo_shape())
#         self.root.bind("<Delete>",    lambda e: self.clear_shapes())
#         self.root.bind("<Control-s>", lambda e: self.save_json())
#         self.root.bind("<r>",         lambda e: self._reset_zoom())
#
#     def _reset_zoom(self):
#         self.zoom  = 1.0
#         self.pan_x = 0.0
#         self.pan_y = 0.0
#         self.zoom_var.set("Zoom: 1.0×")
#         self.pane_a._render()
#         self.pane_b._render()
#
#     # ── Folders ───────────────────────────────────────────────────────────────
#
#     def select_folder_a(self):
#         folder = filedialog.askdirectory(title="Select Folder A (primary)")
#         if not folder:
#             return
#         self.folder_a = folder
#         self.lbl_a.config(text=folder)
#         self._refresh_image_list()
#
#     def select_folder_b(self):
#         folder = filedialog.askdirectory(title="Select Folder B (secondary)")
#         if not folder:
#             return
#         self.folder_b = folder
#         self.lbl_b.config(text=folder)
#         if self.image_names:
#             self._show_current()
#
#     def _refresh_image_list(self):
#         exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
#         all_names = sorted(p.name for p in Path(self.folder_a).iterdir()
#                            if p.suffix.lower() in exts)
#         # If a filter is loaded, match using the last filter_depth parts of each file's path
#         if self.filter_names is not None:
#             def _tail(name):
#                 parts = (Path(self.folder_a) / name).parts
#                 return "/".join(parts[-self.filter_depth:])
#             names = [n for n in all_names if _tail(n) in self.filter_names]
#         else:
#             names = all_names
#         if not names:
#             msg = ("No matching images found in Folder A.\nCheck that the filter JSON matches this folder."
#                    if self.filter_names else
#                    "No supported images found in Folder A.")
#             messagebox.showinfo("No Images", msg)
#             return
#         self.image_names   = names
#         self.current_index = 0
#         self.image_listbox.delete(0, tk.END)
#         for n in names:
#             self.image_listbox.insert(tk.END, "  " + n)
#         self._show_current()
#         self._refresh_listbox_colors()
#
#     def load_filter_json(self):
#         path = filedialog.askopenfilename(
#             title="Load Filter JSON",
#             filetypes=[("JSON", "*.json"), ("All files", "*.*")])
#         if not path:
#             return
#         try:
#             with open(path) as f:
#                 data = json.load(f)
#         except Exception as e:
#             messagebox.showerror("JSON Error", f"Could not read filter file:\n{e}")
#             return
#         # Format: {folder_path: [full_file_path, ...], ...}
#         # Build a set of the last `filter_depth` path parts joined, e.g. "folder/file.png"
#         self.filter_depth = self.depth_var.get()
#         names = set()
#         for file_list in data.values():
#             for fpath in file_list:
#                 parts = Path(fpath).parts
#                 tail = "/".join(parts[-self.filter_depth:])
#                 names.add(tail)
#         if not names:
#             messagebox.showwarning("Empty Filter", "No filenames found in the JSON.")
#             return
#         self.filter_names = names
#         self.lbl_filter.config(text=f"{len(names)} images (depth {self.filter_depth})", fg=ACCENT2)
#         self.status_var.set(f"Filter loaded: {len(names)} images.")
#         if self.folder_a:
#             self._refresh_image_list()
#
#     def clear_filter(self):
#         self.filter_names = None
#         self.lbl_filter.config(text="No filter — showing all images", fg=SUBTEXT)
#         self.status_var.set("Filter cleared.")
#         if self.folder_a:
#             self._refresh_image_list()
#
#     def _on_depth_change(self):
#         """Re-apply the filter with the new depth when the spinbox changes."""
#         self.filter_depth = self.depth_var.get()
#         if self.filter_names is not None:
#             # Reload the filter JSON isn't available anymore, but we can just
#             # re-run the image list refresh — the depth is already updated
#             if self.folder_a:
#                 self._refresh_image_list()
#
#     # ── Display ───────────────────────────────────────────────────────────────
#
#     def _show_current(self):
#         if not self.image_names:
#             return
#         name = self.image_names[self.current_index]
#         self.image_listbox.selection_clear(0, tk.END)
#         self.image_listbox.selection_set(self.current_index)
#         self.image_listbox.see(self.current_index)
#         self.shapes = list(self.labels.get(name, []))
#         self.progress_var.set(f"{self.current_index + 1} / {len(self.image_names)}")
#
#         if self.folder_a:
#             p = Path(self.folder_a) / name
#             self.pane_a.load_image(str(p)) if p.exists() else self.pane_a.clear_image()
#         if self.folder_b:
#             p = Path(self.folder_b) / name
#             self.pane_b.load_image(str(p)) if p.exists() else self.pane_b.clear_image()
#         else:
#             self.pane_b.clear_image()
#
#         self._update_vector()
#         self._update_info()
#
#     def _update_info(self):
#         name = self.image_names[self.current_index] if self.image_names else "—"
#         iw = ih = "?"
#         if self.pane_a.pil_image:
#             iw, ih = self.pane_a.pil_image.size
#         self.info_var.set(
#             f"File: {name}\n"
#             f"Size (A): {iw} × {ih} px\n"
#             f"Boxes: {len(self.shapes)}"
#         )
#
#     # ── Shapes ────────────────────────────────────────────────────────────────
#
#     def _add_shape(self, shape):
#         self.shapes.append(shape)
#         self.labels[self.image_names[self.current_index]] = list(self.shapes)
#         self.pane_a._redraw_shapes()
#         self.pane_b._redraw_shapes()
#         self._update_vector()
#         self._update_info()
#         self.status_var.set(f"Box #{len(self.shapes)} drawn — mirrored on both views.")
#
#     def undo_shape(self):
#         if not self.shapes:
#             return
#         self.shapes.pop()
#         self.labels[self.image_names[self.current_index]] = list(self.shapes)
#         self.pane_a._redraw_shapes()
#         self.pane_b._redraw_shapes()
#         self._update_vector()
#         self._update_info()
#         self.status_var.set("Last box removed.")
#
#     def clear_shapes(self):
#         if not self.image_names:
#             return
#         self.shapes = []
#         self.labels[self.image_names[self.current_index]] = []
#         self.pane_a._redraw_shapes()
#         self.pane_b._redraw_shapes()
#         self._update_vector()
#         self._update_info()
#         self.status_var.set("All boxes cleared for this image.")
#
#     # ── Navigation ────────────────────────────────────────────────────────────
#
#     def prev_image(self):
#         if self.image_names and self.current_index > 0:
#             self.current_index -= 1
#             self._show_current()
#
#     def next_image(self):
#         if self.image_names and self.current_index < len(self.image_names) - 1:
#             self.current_index += 1
#             self._show_current()
#
#     def _on_list_select(self, event):
#         sel = self.image_listbox.curselection()
#         if sel:
#             self.current_index = sel[0]
#             self._show_current()
#
#     # ── Vector ────────────────────────────────────────────────────────────────
#
#     def _compute_vector(self):
#         pane = self.pane_a if self.pane_a.pil_image else self.pane_b
#         if not pane.pil_image:
#             return []
#         iw, _ = pane.pil_image.size
#         vector = [0] * iw
#         for shape in self.shapes:
#             x0, _, x1, _ = shape["coords"]
#             for col in range(max(0, int(x0)), min(iw, int(x1) + 1)):
#                 vector[col] = 1
#         return vector
#
#     def _update_vector(self):
#         vector = self._compute_vector()
#         if not vector:
#             self.vector_canvas.delete("all")
#             return
#         if self.image_names:
#             self.vectors[self.image_names[self.current_index]] = vector
#         vc  = self.vector_canvas
#         vc.delete("all")
#         vw  = vc.winfo_width()
#         vh  = vc.winfo_height() or 36
#         n   = len(vector)
#         if n == 0 or vw < 2:
#             return
#         px = max(1, vw / n)
#         for i, val in enumerate(vector):
#             x0 = int(i * px);  x1 = int((i + 1) * px)
#             vc.create_rectangle(x0, 0, x1, vh,
#                                  fill=DANGER_CLR if val else "#1a2a1a", outline="")
#         danger = sum(vector)
#         pct    = danger / n * 100
#         self.vec_info_var.set(
#             f"Width: {n}  |  Danger: {danger} ({pct:.1f}%)  |  Safe: {n - danger}"
#         )
#
#     # ── Export ────────────────────────────────────────────────────────────────
#
#     def toggle_exclude(self):
#         if not self.image_names:
#             return
#         name = self.image_names[self.current_index]
#         if name in self.excluded:
#             self.excluded.discard(name)
#             self.status_var.set(f"{name} restored.")
#         else:
#             self.excluded.add(name)
#             self.shapes = []
#             self.labels[name] = []
#             self.pane_a._redraw_shapes()
#             self.pane_b._redraw_shapes()
#             self._update_vector()
#             self.status_var.set(f"{name} excluded from export.")
#         self._refresh_listbox_colors()
#         self._update_info()
#
#     def _refresh_listbox_colors(self):
#         for i, name in enumerate(self.image_names):
#             if name in self.excluded:
#                 self.image_listbox.itemconfig(i, fg="#e74c3c", bg="#2a0a0a")
#             else:
#                 self.image_listbox.itemconfig(i, fg=TEXT, bg=PANEL)
#
#     def save_json(self):
#         if not self.labels:
#             messagebox.showinfo("Nothing to save", "No boxes drawn yet.")
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".json",
#                                              filetypes=[("JSON", "*.json")])
#         if not path:
#             return
#         output = {}
#         for name, shapes in self.labels.items():
#             if name in self.excluded:
#                 continue
#             output[name] = {
#                 "filename": name,
#                 "path_a":   str(Path(self.folder_a) / name) if self.folder_a else None,
#                 "path_b":   str(Path(self.folder_b) / name) if self.folder_b else None,
#                 "boxes": [
#                     {"x0": round(s["coords"][0], 2), "y0": round(s["coords"][1], 2),
#                      "x1": round(s["coords"][2], 2), "y1": round(s["coords"][3], 2)}
#                     for s in shapes
#                 ],
#                 "vector": self.vectors.get(name, [])
#             }
#         with open(path, "w") as f:
#             json.dump(output, f, indent=2)
#         self.status_var.set(f"Saved → {path}")
#         messagebox.showinfo("Saved", f"Labels saved to:\n{path}")
#
#     def export_csv(self):
#         if not self.vectors:
#             messagebox.showinfo("Nothing to export", "No vectors computed yet.")
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".csv",
#                                              filetypes=[("CSV", "*.csv")])
#         if not path:
#             return
#         with open(path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["filename", "path_a", "path_b", "vector"])
#             for name, vec in self.vectors.items():
#                 if name in self.excluded:
#                     continue
#                 writer.writerow([
#                     name,
#                     str(Path(self.folder_a) / name) if self.folder_a else "",
#                     str(Path(self.folder_b) / name) if self.folder_b else "",
#                     "[" + ",".join(str(v) for v in vec) + "]"
#                 ])
#         self.status_var.set(f"Exported → {path}")
#         messagebox.showinfo("Exported", f"Vectors exported to:\n{path}")
#
#
# # ── Entry point ───────────────────────────────────────────────────────────────
#
# def main():
#     if not PIL_AVAILABLE:
#         root2 = tk.Tk()
#         root2.withdraw()
#         go = messagebox.askyesno(
#             "Pillow not installed",
#             "Pillow is required.\n\nInstall with:  pip install Pillow\n\nContinue anyway?"
#         )
#         root2.destroy()
#         if not go:
#             return
#
#     root = tk.Tk()
#     root.geometry("1280x860")
#     LabelingTool(root)
#     root.mainloop()
#
#
# if __name__ == "__main__":
#     main()

"""
Image Labeling Tool for Neural Network Data
- Dual image view (top/bottom), two folders, same filename
- Draw rectangles on either image — instantly mirrored on the other
- Scroll to zoom, right-drag to pan — both persist across image switches
- Exports binary vector + shape data to JSON & CSV
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import csv
from pathlib import Path

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Palette ──────────────────────────────────────────────────────────────────
BG         = "#0f1117"
PANEL      = "#1a1d27"
ACCENT     = "#e05c2a"
ACCENT2    = "#f5a623"
TEXT       = "#e8e8f0"
SUBTEXT    = "#7a7a9a"
BORDER     = "#2a2d3e"
DANGER_CLR = "#e74c3c"
HOVER      = "#252838"
LABEL_A    = "#a29bfe"
LABEL_B    = "#55efc4"

FONT_TITLE = ("Georgia", 20, "bold")
FONT_HEAD  = ("Georgia", 11, "bold")
FONT_SMALL = ("Courier New", 9)
FONT_BTN   = ("Georgia", 10, "bold")


class DualCanvas:
    """One image pane. Zoom/pan state lives in LabelingTool and is shared."""

    def __init__(self, parent, label_text, label_color, tool):
        self.tool      = tool
        self.pil_image = None
        self.tk_image  = None

        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(outer, bg=PANEL, height=22)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  {label_text}", font=FONT_SMALL,
                 fg=label_color, bg=PANEL).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="No folder selected")
        tk.Label(hdr, textvariable=self.path_var, font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT, padx=6)

        self.canvas = tk.Canvas(outer, bg="#080a10",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Drawing
        self.canvas.bind("<ButtonPress-1>",   self._mouse_down)
        self.canvas.bind("<B1-Motion>",        self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._mouse_up)

        # Zoom — scroll wheel (Windows/Mac) and Button-4/5 (Linux)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>",   self._on_scroll)
        self.canvas.bind("<Button-5>",   self._on_scroll)

        # Pan — right-click drag
        self.canvas.bind("<ButtonPress-3>",  self._pan_start)
        self.canvas.bind("<B3-Motion>",       self._pan_move)

        self.canvas.bind("<Configure>", lambda e: self._render())

        self.drawing      = False
        self.draw_start   = None
        self.current_rect = None
        self._pan_last    = None

    # ── image ─────────────────────────────────────────────────────────────────

    def load_image(self, path):
        if not PIL_AVAILABLE:
            return
        try:
            self.pil_image = Image.open(path)
        except Exception as e:
            self.pil_image = None
            messagebox.showerror("Load Error", f"Cannot open {path}:\n{e}")
            return
        self.path_var.set(str(Path(path).parent))
        self._render()

    def clear_image(self):
        self.pil_image = None
        self.tk_image  = None
        self.canvas.delete("all")

    # ── render ────────────────────────────────────────────────────────────────

    def _render(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            self.canvas.after(80, self._render)
            return
        if not self.pil_image:
            return

        iw, ih     = self.pil_image.size
        base_scale = min(cw / iw, ch / ih, 1.0)   # fit-to-canvas scale at zoom=1
        scale      = base_scale * self.tool.zoom   # actual render scale

        # Visible region in image coordinates given current pan
        vis_w = cw / scale
        vis_h = ch / scale

        # Clamp pan so we don't wander too far outside the image
        self.tool.pan_x = max(-vis_w * 0.5, min(self.tool.pan_x, iw - vis_w * 0.5))
        self.tool.pan_y = max(-vis_h * 0.5, min(self.tool.pan_y, ih - vis_h * 0.5))

        # Crop the part of the image that's visible
        cx0 = int(max(0, self.tool.pan_x))
        cy0 = int(max(0, self.tool.pan_y))
        cx1 = int(min(iw, self.tool.pan_x + vis_w))
        cy1 = int(min(ih, self.tool.pan_y + vis_h))
        if cx1 <= cx0 or cy1 <= cy0:
            return

        cropped = self.pil_image.crop((cx0, cy0, cx1, cy1))

        dw = int((cx1 - cx0) * scale)
        dh = int((cy1 - cy0) * scale)
        if dw < 1 or dh < 1:
            return

        resized       = cropped.resize((dw, dh), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        # Canvas offset — non-zero only when zoomed out and image doesn't fill canvas
        ox = int(-self.tool.pan_x * scale) if self.tool.pan_x < 0 else 0
        oy = int(-self.tool.pan_y * scale) if self.tool.pan_y < 0 else 0

        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_image)
        self._redraw_shapes()

    def rerender(self):
        self._render()

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _canvas_to_image(self, cx, cy):
        """Canvas pixel → image pixel."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih     = self.pil_image.size
        base_scale = min(cw / iw, ch / ih, 1.0)
        scale      = base_scale * self.tool.zoom
        return cx / scale + self.tool.pan_x, cy / scale + self.tool.pan_y

    def _image_to_canvas(self, ix, iy):
        """Image pixel → canvas pixel."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih     = self.pil_image.size
        base_scale = min(cw / iw, ch / ih, 1.0)
        scale      = base_scale * self.tool.zoom
        return (ix - self.tool.pan_x) * scale, (iy - self.tool.pan_y) * scale

    # ── zoom ──────────────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if not self.pil_image:
            return

        # Direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.15
        else:
            factor = 1 / 1.15

        # Image coordinate under the cursor before zoom
        ix, iy = self._canvas_to_image(event.x, event.y)

        # Apply zoom (clamped)
        self.tool.zoom = max(0.2, min(20.0, self.tool.zoom * factor))
        self.tool.zoom_var.set(f"Zoom: {self.tool.zoom:.1f}×")

        # Adjust pan so the point under the cursor stays fixed
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih     = self.pil_image.size
        base_scale = min(cw / iw, ch / ih, 1.0)
        scale      = base_scale * self.tool.zoom
        self.tool.pan_x = ix - event.x / scale
        self.tool.pan_y = iy - event.y / scale

        self.tool.pane_a._render()
        self.tool.pane_b._render()

    # ── pan ───────────────────────────────────────────────────────────────────

    def _pan_start(self, event):
        self._pan_last = (event.x, event.y)

    def _pan_move(self, event):
        if self._pan_last is None or not self.pil_image:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih     = self.pil_image.size
        base_scale = min(cw / iw, ch / ih, 1.0)
        scale      = base_scale * self.tool.zoom

        self.tool.pan_x -= dx / scale
        self.tool.pan_y -= dy / scale

        self.tool.pane_a._render()
        self.tool.pane_b._render()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _mouse_down(self, event):
        if not self.tool.image_names:
            return
        self.drawing    = True
        self.draw_start = (event.x, event.y)
        self.current_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=DANGER_CLR, width=2, dash=(4, 3))

    def _mouse_drag(self, event):
        if self.drawing and self.current_rect:
            x0, y0 = self.draw_start
            self.canvas.coords(self.current_rect, x0, y0, event.x, event.y)

    def _mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        x0, y0 = self.draw_start
        x1, y1 = event.x, event.y
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return
        if not self.pil_image:
            return

        ix0, iy0 = self._canvas_to_image(x0, y0)
        ix1, iy1 = self._canvas_to_image(x1, y1)
        iw, ih   = self.pil_image.size
        ix0 = max(0, min(ix0, iw));  ix1 = max(0, min(ix1, iw))
        iy0 = max(0, min(iy0, ih));  iy1 = max(0, min(iy1, ih))

        self.tool._add_shape({
            "type":   "rect",
            "coords": (min(ix0, ix1), min(iy0, iy1),
                       max(ix0, ix1), max(iy0, iy1))
        })

    # ── redraw shapes ─────────────────────────────────────────────────────────

    def _redraw_shapes(self):
        self.canvas.delete("shape")
        if not self.pil_image:
            return
        for i, shape in enumerate(self.tool.shapes):
            x0, y0, x1, y1 = shape["coords"]
            cx0, cy0 = self._image_to_canvas(x0, y0)
            cx1, cy1 = self._image_to_canvas(x1, y1)
            self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1,
                fill=DANGER_CLR, stipple="gray25", outline="", tags="shape")
            self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1,
                outline=DANGER_CLR, width=2, tags="shape")
            self.canvas.create_text(
                cx0 + 4, cy0 + 4, anchor=tk.NW,
                text=f"⚠ {i + 1}", fill=TEXT,
                font=("Courier New", 8, "bold"), tags="shape")


# ─────────────────────────────────────────────────────────────────────────────

class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Danger Zone Labeler")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 760)

        self.folder_a      = None
        self.folder_b      = None
        self.image_names   = []
        self.current_index = 0
        self.shapes        = []
        self.labels        = {}
        self.vectors       = {}

        # How many path parts to match when loading filter JSON (1 = filename only, 2 = folder/filename, etc.)
        self.filter_depth = 1

        # Set of filenames explicitly excluded from export (shown red in list)
        self.excluded = set()

        # Optional filter: set of filenames (just the name, no path) to show.
        # If None, all images in Folder A are shown.
        self.filter_names  = None

        # Zoom/pan — stored here so they persist when switching images
        self.zoom  = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._build_ui()
        self._bind_keys()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        title_bar = tk.Frame(self.root, bg=BG, height=58)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="◈", font=("Georgia", 24), fg=ACCENT,
                 bg=BG).pack(side=tk.LEFT, padx=(18, 5), pady=8)
        tk.Label(title_bar, text="DANGER ZONE LABELER",
                 font=FONT_TITLE, fg=TEXT, bg=BG).pack(side=tk.LEFT)
        tk.Label(title_bar, text="  ·  dual-view sync labeling",
                 font=("Georgia", 10, "italic"), fg=SUBTEXT, bg=BG
                 ).pack(side=tk.LEFT, pady=14)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar(body)
        tk.Frame(body, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pane_a = DualCanvas(right, "▲  FOLDER A", LABEL_A, self)
        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)
        self.pane_b = DualCanvas(right, "▼  FOLDER B", LABEL_B, self)

        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)
        strip_hdr = tk.Frame(right, bg=PANEL, height=22)
        strip_hdr.pack(fill=tk.X)
        strip_hdr.pack_propagate(False)
        tk.Label(strip_hdr, text="  COLUMN VECTOR  [ 0 = safe · 1 = danger ]",
                 font=FONT_SMALL, fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
        self.vec_info_var = tk.StringVar(value="")
        tk.Label(strip_hdr, textvariable=self.vec_info_var,
                 font=FONT_SMALL, fg=ACCENT2, bg=PANEL).pack(side=tk.RIGHT, padx=10)
        self.vector_canvas = tk.Canvas(right, bg="#080a10",
                                        height=36, highlightthickness=0)
        self.vector_canvas.pack(fill=tk.X)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)
        sf = tk.Frame(self.root, bg=PANEL, height=28)
        sf.pack(fill=tk.X)
        sf.pack_propagate(False)
        self.status_var = tk.StringVar(value="Select Folder A to begin.")
        tk.Label(sf, textvariable=self.status_var, font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL, anchor="w").pack(side=tk.LEFT, padx=12)

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=PANEL, width=252)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        def section(t):
            f = tk.Frame(sb, bg=PANEL)
            f.pack(fill=tk.X, padx=14, pady=(8, 2))
            tk.Label(f, text=t, font=FONT_HEAD, fg=ACCENT2, bg=PANEL).pack(anchor="w")
            tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=14)

        def btn(pw, t, cmd, color=ACCENT):
            b = tk.Button(pw, text=t, command=cmd, font=FONT_BTN,
                          fg=TEXT, bg=color, activebackground=HOVER,
                          activeforeground=TEXT, relief=tk.FLAT,
                          cursor="hand2", padx=10, pady=5, bd=0)
            b.pack(fill=tk.X, padx=14, pady=2)
            return b

        # FOLDERS
        section("FOLDERS")
        btn(sb, "📂  Select Folder A  (primary)", self.select_folder_a)
        self.lbl_a = tk.Label(sb, text="Not selected", font=FONT_SMALL,
                               fg=LABEL_A, bg=PANEL, wraplength=220, anchor="w")
        self.lbl_a.pack(fill=tk.X, padx=18, pady=(0, 6))
        btn(sb, "📂  Select Folder B  (secondary)",
            self.select_folder_b, color="#2a2d3e")
        self.lbl_b = tk.Label(sb, text="Not selected", font=FONT_SMALL,
                               fg=LABEL_B, bg=PANEL, wraplength=220, anchor="w")
        self.lbl_b.pack(fill=tk.X, padx=18, pady=(0, 6))

        # FILTER
        section("FILTER  (optional)")
        btn(sb, "🔍  Load Filter JSON", self.load_filter_json, color="#2a2d3e")
        self.lbl_filter = tk.Label(sb, text="No filter — showing all images",
                                    font=FONT_SMALL, fg=SUBTEXT, bg=PANEL,
                                    wraplength=220, anchor="w")
        self.lbl_filter.pack(fill=tk.X, padx=18, pady=(0, 2))
        # Path depth selector
        depth_row = tk.Frame(sb, bg=PANEL)
        depth_row.pack(fill=tk.X, padx=14, pady=(0, 4))
        tk.Label(depth_row, text="Match last", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
        self.depth_var = tk.IntVar(value=1)
        depth_spin = tk.Spinbox(depth_row, from_=1, to=10, width=3,
                                textvariable=self.depth_var,
                                font=FONT_SMALL, fg=TEXT, bg=PANEL,
                                buttonbackground=PANEL, relief=tk.FLAT,
                                command=self._on_depth_change)
        depth_spin.pack(side=tk.LEFT, padx=4)
        tk.Label(depth_row, text="path parts", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(side=tk.LEFT)
        tk.Label(sb, text="(1 = filename only, 2 = folder/file)",
                 font=FONT_SMALL, fg=SUBTEXT, bg=PANEL
                 ).pack(anchor="w", padx=14, pady=(0, 2))
        btn(sb, "✕  Clear Filter", self.clear_filter, color="#2a2d3e")

        # IMAGE QUEUE
        section("IMAGE QUEUE")
        lf = tk.Frame(sb, bg=BORDER, bd=1, relief=tk.FLAT)
        lf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 4))
        scr = tk.Scrollbar(lf, bg=PANEL, troughcolor=PANEL, relief=tk.FLAT)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(
            lf, font=FONT_SMALL, fg=TEXT, bg=PANEL,
            selectbackground=ACCENT, selectforeground=TEXT,
            relief=tk.FLAT, bd=0, activestyle="none",
            yscrollcommand=scr.set)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        scr.config(command=self.image_listbox.yview)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # NAVIGATE
        section("NAVIGATE")
        nav = tk.Frame(sb, bg=PANEL)
        nav.pack(fill=tk.X, padx=14, pady=3)
        tk.Button(nav, text="◀  Prev", command=self.prev_image,
                  font=FONT_BTN, fg=TEXT, bg="#2a2d3e",
                  activebackground=HOVER, activeforeground=TEXT,
                  relief=tk.FLAT, cursor="hand2", padx=8, pady=4, bd=0
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(nav, text="Next  ▶", command=self.next_image,
                  font=FONT_BTN, fg=TEXT, bg="#2a2d3e",
                  activebackground=HOVER, activeforeground=TEXT,
                  relief=tk.FLAT, cursor="hand2", padx=8, pady=4, bd=0
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        self.progress_var = tk.StringVar(value="— / —")
        tk.Label(sb, textvariable=self.progress_var, font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(pady=2)

        # zoom_var still exists so the rest of the code doesn't break
        self.zoom_var = tk.StringVar(value="Zoom: 1.0×")

        # TOOLS
        section("TOOLS")
        btn(sb, "↩  Undo Last Box",      self.undo_shape,   color="#2a2d3e")
        btn(sb, "✕  Clear This Image",    self.clear_shapes, color="#2a2d3e")
        btn(sb, "🚫  Toggle Exclude",     self.toggle_exclude, color="#5a1a1a")

        # EXPORT
        section("EXPORT")
        btn(sb, "💾  Save Labels (JSON)",   self.save_json)
        btn(sb, "📊  Export Vectors (CSV)", self.export_csv, color="#2a2d3e")

        # info_var still used internally
        self.info_var = tk.StringVar(value="No image loaded.")

    def _bind_keys(self):
        self.root.bind("<Left>",      lambda e: self.prev_image())
        self.root.bind("<Right>",     lambda e: self.next_image())
        self.root.bind("<Control-z>", lambda e: self.undo_shape())
        self.root.bind("<Delete>",    lambda e: self.clear_shapes())
        self.root.bind("<Control-s>", lambda e: self.save_json())
        self.root.bind("<r>",         lambda e: self._reset_zoom())

    def _reset_zoom(self):
        self.zoom  = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom_var.set("Zoom: 1.0×")
        self.pane_a._render()
        self.pane_b._render()

    # ── Folders ───────────────────────────────────────────────────────────────

    def select_folder_a(self):
        folder = filedialog.askdirectory(title="Select Folder A (primary)")
        if not folder:
            return
        self.folder_a = folder
        self.lbl_a.config(text=folder)
        self._refresh_image_list()

    def select_folder_b(self):
        folder = filedialog.askdirectory(title="Select Folder B (secondary)")
        if not folder:
            return
        self.folder_b = folder
        self.lbl_b.config(text=folder)
        if self.image_names:
            self._show_current()

    def _refresh_image_list(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        all_names = sorted(p.name for p in Path(self.folder_a).iterdir()
                           if p.suffix.lower() in exts)
        # If a filter is loaded, match using the last filter_depth parts of each file's path
        if self.filter_names is not None:
            def _tail(name):
                parts = (Path(self.folder_a) / name).parts
                return "/".join(parts[-self.filter_depth:])
            names = [n for n in all_names if _tail(n) in self.filter_names]
        else:
            names = all_names
        if not names:
            msg = ("No matching images found in Folder A.\nCheck that the filter JSON matches this folder."
                   if self.filter_names else
                   "No supported images found in Folder A.")
            messagebox.showinfo("No Images", msg)
            return
        self.image_names   = names
        self.current_index = 0
        self.image_listbox.delete(0, tk.END)
        for n in names:
            self.image_listbox.insert(tk.END, "  " + n)
        self._show_current()
        self._refresh_listbox_colors()

    def load_filter_json(self):
        path = filedialog.askopenfilename(
            title="Load Filter JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("JSON Error", f"Could not read filter file:\n{e}")
            return
        # Format: {folder_path: [full_file_path, ...], ...}
        # Build a set of the last `filter_depth` path parts joined, e.g. "folder/file.png"
        self.filter_depth = self.depth_var.get()
        names = set()
        for file_list in data.values():
            for fpath in file_list:
                parts = Path(fpath).parts
                tail = "/".join(parts[-self.filter_depth:])
                names.add(tail)
        if not names:
            messagebox.showwarning("Empty Filter", "No filenames found in the JSON.")
            return
        self.filter_names = names
        self.lbl_filter.config(text=f"{len(names)} images (depth {self.filter_depth})", fg=ACCENT2)
        self.status_var.set(f"Filter loaded: {len(names)} images.")
        if self.folder_a:
            self._refresh_image_list()

    def clear_filter(self):
        self.filter_names = None
        self.lbl_filter.config(text="No filter — showing all images", fg=SUBTEXT)
        self.status_var.set("Filter cleared.")
        if self.folder_a:
            self._refresh_image_list()

    def _on_depth_change(self):
        """Re-apply the filter with the new depth when the spinbox changes."""
        self.filter_depth = self.depth_var.get()
        if self.filter_names is not None:
            # Reload the filter JSON isn't available anymore, but we can just
            # re-run the image list refresh — the depth is already updated
            if self.folder_a:
                self._refresh_image_list()

    # ── Display ───────────────────────────────────────────────────────────────

    def _show_current(self):
        if not self.image_names:
            return
        name = self.image_names[self.current_index]
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        self.shapes = list(self.labels.get(name, []))
        self.progress_var.set(f"{self.current_index + 1} / {len(self.image_names)}")

        if self.folder_a:
            p = Path(self.folder_a) / name
            self.pane_a.load_image(str(p)) if p.exists() else self.pane_a.clear_image()
        if self.folder_b:
            p = Path(self.folder_b) / name
            self.pane_b.load_image(str(p)) if p.exists() else self.pane_b.clear_image()
        else:
            self.pane_b.clear_image()

        self._update_vector()
        self._update_info()

    def _update_info(self):
        name = self.image_names[self.current_index] if self.image_names else "—"
        iw = ih = "?"
        if self.pane_a.pil_image:
            iw, ih = self.pane_a.pil_image.size
        self.info_var.set(
            f"File: {name}\n"
            f"Size (A): {iw} × {ih} px\n"
            f"Boxes: {len(self.shapes)}"
        )

    # ── Shapes ────────────────────────────────────────────────────────────────

    def _add_shape(self, shape):
        self.shapes.append(shape)
        self.labels[self.image_names[self.current_index]] = list(self.shapes)
        self.pane_a._redraw_shapes()
        self.pane_b._redraw_shapes()
        self._update_vector()
        self._update_info()
        self.status_var.set(f"Box #{len(self.shapes)} drawn — mirrored on both views.")

    def undo_shape(self):
        if not self.shapes:
            return
        self.shapes.pop()
        self.labels[self.image_names[self.current_index]] = list(self.shapes)
        self.pane_a._redraw_shapes()
        self.pane_b._redraw_shapes()
        self._update_vector()
        self._update_info()
        self.status_var.set("Last box removed.")

    def clear_shapes(self):
        if not self.image_names:
            return
        self.shapes = []
        self.labels[self.image_names[self.current_index]] = []
        self.pane_a._redraw_shapes()
        self.pane_b._redraw_shapes()
        self._update_vector()
        self._update_info()
        self.status_var.set("All boxes cleared for this image.")

    # ── Navigation ────────────────────────────────────────────────────────────

    def prev_image(self):
        if self.image_names and self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def next_image(self):
        if self.image_names and self.current_index < len(self.image_names) - 1:
            self.current_index += 1
            self._show_current()

    def _on_list_select(self, event):
        sel = self.image_listbox.curselection()
        if sel:
            self.current_index = sel[0]
            self._show_current()

    # ── Vector ────────────────────────────────────────────────────────────────

    def _compute_vector(self):
        pane = self.pane_a if self.pane_a.pil_image else self.pane_b
        if not pane.pil_image:
            return []
        iw, _ = pane.pil_image.size
        vector = [0] * iw
        for shape in self.shapes:
            x0, _, x1, _ = shape["coords"]
            for col in range(max(0, int(x0)), min(iw, int(x1) + 1)):
                vector[col] = 1
        return vector

    def _update_vector(self):
        vector = self._compute_vector()
        if not vector:
            self.vector_canvas.delete("all")
            return
        if self.image_names:
            self.vectors[self.image_names[self.current_index]] = vector
        vc  = self.vector_canvas
        vc.delete("all")
        vw  = vc.winfo_width()
        vh  = vc.winfo_height() or 36
        n   = len(vector)
        if n == 0 or vw < 2:
            return
        px = max(1, vw / n)
        for i, val in enumerate(vector):
            x0 = int(i * px);  x1 = int((i + 1) * px)
            vc.create_rectangle(x0, 0, x1, vh,
                                 fill=DANGER_CLR if val else "#1a2a1a", outline="")
        danger = sum(vector)
        pct    = danger / n * 100
        self.vec_info_var.set(
            f"Width: {n}  |  Danger: {danger} ({pct:.1f}%)  |  Safe: {n - danger}"
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def toggle_exclude(self):
        if not self.image_names:
            return
        name = self.image_names[self.current_index]
        if name in self.excluded:
            self.excluded.discard(name)
            self.status_var.set(f"{name} restored.")
        else:
            self.excluded.add(name)
            self.shapes = []
            self.labels[name] = []
            self.pane_a._redraw_shapes()
            self.pane_b._redraw_shapes()
            self._update_vector()
            self.status_var.set(f"{name} excluded from export.")
        self._refresh_listbox_colors()
        self._update_info()

    def _refresh_listbox_colors(self):
        for i, name in enumerate(self.image_names):
            if name in self.excluded:
                self.image_listbox.itemconfig(i, fg="#e74c3c", bg="#2a0a0a")
            else:
                self.image_listbox.itemconfig(i, fg=TEXT, bg=PANEL)

    def save_json(self):
        if not self.image_names:
            messagebox.showinfo("Nothing to save", "No images loaded yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON", "*.json")])
        if not path:
            return
        output = {}
        for name in self.image_names:
            if name in self.excluded:
                continue
            shapes = self.labels.get(name, [])
            output[name] = {
                "filename": name,
                "path_a":   str(Path(self.folder_a) / name) if self.folder_a else None,
                "path_b":   str(Path(self.folder_b) / name) if self.folder_b else None,
                "boxes": [
                    {"x0": round(s["coords"][0], 2), "y0": round(s["coords"][1], 2),
                     "x1": round(s["coords"][2], 2), "y1": round(s["coords"][3], 2)}
                    for s in shapes
                ],
                "vector": self.vectors.get(name, [])
            }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Labels saved to:\n{path}")

    def export_csv(self):
        if not self.image_names:
            messagebox.showinfo("Nothing to export", "No images loaded yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "path_a", "path_b", "vector"])
            for name in self.image_names:
                if name in self.excluded:
                    continue
                vec = self.vectors.get(name, [])
                writer.writerow([
                    name,
                    str(Path(self.folder_a) / name) if self.folder_a else "",
                    str(Path(self.folder_b) / name) if self.folder_b else "",
                    "[" + ",".join(str(v) for v in vec) + "]"
                ])
        self.status_var.set(f"Exported → {path}")
        messagebox.showinfo("Exported", f"Vectors exported to:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not PIL_AVAILABLE:
        root2 = tk.Tk()
        root2.withdraw()
        go = messagebox.askyesno(
            "Pillow not installed",
            "Pillow is required.\n\nInstall with:  pip install Pillow\n\nContinue anyway?"
        )
        root2.destroy()
        if not go:
            return

    root = tk.Tk()
    root.geometry("1280x860")
    LabelingTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()