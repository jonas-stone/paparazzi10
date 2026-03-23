# # """
# # gate_detection_v2.py
# # ====================
# #
# # Upgraded gate detector for full-resolution images with robust rectangle detection.
# # Detects two parallel blue bars, computes midpoint, confidence, and smooths over frames.
# #
# # Pipeline:
# # 1. Blue mask via YUV + optional HSV
# # 2. Find contours and extract elongated rectangles
# # 3. Filter rectangles based on area/aspect ratio
# # 4. Test all rectangle pairs for:
# #     - Parallel angle
# #     - Similar size/aspect
# #     - Reasonable distance
# # 5. Compute midpoint and confidence
# # 6. Temporal smoothing via EMA tracker
# # 7. Display side-by-side visualization at full resolution
# # """
# #
# # import os
# # import sys
# # import cv2
# # import numpy as np
# # from glob import glob
# # from dataclasses import dataclass
# # from typing import Optional, List, Tuple
# #
# # # ─────────────────────────────────────────────
# # # CONFIGURATION
# # # ─────────────────────────────────────────────
# #
# # @dataclass
# # class Config:
# #     # Blue thresholds (YUV + optional HSV)
# #     blue_y_min: int = 0
# #     blue_y_max: int = 220
# #     blue_u_min: int = 122
# #     blue_u_max: int = 255
# #     blue_v_min: int = 0
# #     blue_v_max: int = 123
# #
# #     use_hsv_confirm: bool = True
# #     hsv_hue_min: int = 100
# #     hsv_hue_max: int = 130
# #     hsv_sat_min: int = 80
# #     hsv_val_min: int = 40
# #
# #     blue_morph_k: int = 5  # morphological kernel size
# #
# #     # Blob/rectangle filtering
# #     blob_min_area_ratio: float = 0.003
# #     min_aspect: float = 3.0  # minimum elongation for gate bars
# #
# #     # Parallelism & geometry
# #     max_angle_diff: float = 10.0  # degrees
# #     max_area_diff: float = 0.6
# #     max_aspect_diff: float = 0.5
# #     min_gate_dist_ratio: float = 0.5
# #     max_gate_dist_ratio: float = 3.0
# #
# #     # Tracker
# #     tracker_alpha: float = 0.4
# #     tracker_max_miss: int = 10
# #
# #     # Display
# #     display_max_w: int = 1800
# #     display_max_h: int = 900
# #
# # # ─────────────────────────────────────────────
# # # BLUE MASK
# # # ─────────────────────────────────────────────
# #
# # def blue_mask(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
# #     yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
# #     Y, U, V = cv2.split(yuv)
# #
# #     yuv_mask = (
# #         (Y >= cfg.blue_y_min) & (Y <= cfg.blue_y_max) &
# #         (U >= cfg.blue_u_min) & (U <= cfg.blue_u_max) &
# #         (V >= cfg.blue_v_min) & (V <= cfg.blue_v_max)
# #     ).astype(np.uint8) * 255
# #
# #     if cfg.use_hsv_confirm:
# #         hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# #         hsv_mask = cv2.inRange(
# #             hsv,
# #             (cfg.hsv_hue_min, cfg.hsv_sat_min, cfg.hsv_val_min),
# #             (cfg.hsv_hue_max, 255, 255)
# #         )
# #         mask = cv2.bitwise_and(yuv_mask, hsv_mask)
# #     else:
# #         mask = yuv_mask
# #
# #     # Morphology
# #     k = cfg.blue_morph_k
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# #     return mask
# #
# # # ─────────────────────────────────────────────
# # # RECTANGLE EXTRACTION
# # # ─────────────────────────────────────────────
# #
# # def extract_rectangles(mask: np.ndarray, image_shape: tuple, cfg: Config) -> List[dict]:
# #     min_area = int(image_shape[0] * image_shape[1] * cfg.blob_min_area_ratio)
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     rects = []
# #     for cnt in contours:
# #         area = cv2.contourArea(cnt)
# #         if area < min_area:
# #             continue
# #         rect = cv2.minAreaRect(cnt)
# #         (cx, cy), (w, h), angle = rect
# #         if min(w, h) == 0:
# #             continue
# #         aspect = max(w, h) / min(w, h)
# #         if aspect < cfg.min_aspect:
# #             continue
# #         rects.append({
# #             "cx": int(cx), "cy": int(cy),
# #             "w": w, "h": h,
# #             "angle": angle,
# #             "area": area,
# #             "contour": cnt
# #         })
# #     return rects
# #
# # # ─────────────────────────────────────────────
# # # PARALLEL PAIR DETECTION
# # # ─────────────────────────────────────────────
# #
# # def are_parallel(r1: dict, r2: dict, cfg: Config) -> bool:
# #     # 1. Angle similarity
# #     angle_diff = abs(r1["angle"] - r2["angle"])
# #     if angle_diff > cfg.max_angle_diff:
# #         return False
# #     # 2. Area similarity
# #     area_diff = abs(r1["area"] - r2["area"]) / max(r1["area"], r2["area"])
# #     if area_diff > cfg.max_area_diff:
# #         return False
# #     # 3. Aspect similarity
# #     aspect_diff = abs((r1["w"]/r1["h"]) - (r2["w"]/r2["h"])) / max(r1["w"]/r1["h"], r2["w"]/r2["h"])
# #     if aspect_diff > cfg.max_aspect_diff:
# #         return False
# #     # 4. Distance sanity (relative to height)
# #     dist = np.hypot(r1["cx"] - r2["cx"], r1["cy"] - r2["cy"])
# #     avg_h = (r1["h"] + r2["h"]) / 2
# #     if not (cfg.min_gate_dist_ratio*avg_h < dist < cfg.max_gate_dist_ratio*avg_h):
# #         return False
# #     return True
# #
# # def find_gate_pair(rects: List[dict], cfg: Config) -> Optional[Tuple[dict, dict]]:
# #     best_pair = None
# #     best_area = 0
# #     for i in range(len(rects)):
# #         for j in range(i+1, len(rects)):
# #             r1, r2 = rects[i], rects[j]
# #             if are_parallel(r1, r2, cfg):
# #                 combined = r1["area"] + r2["area"]
# #                 if combined > best_area:
# #                     best_area = combined
# #                     best_pair = (r1, r2)
# #     return best_pair
# #
# # # ─────────────────────────────────────────────
# # # MIDPOINT & CONFIDENCE
# # # ─────────────────────────────────────────────
# #
# # def gate_midpoint(r1: dict, r2: dict) -> Tuple[int, int]:
# #     return ((r1["cx"] + r2["cx"]) // 2, (r1["cy"] + r2["cy"]) // 2)
# #
# # def gate_confidence(r1: dict, r2: dict, cfg: Config) -> float:
# #     ar_diff = abs((r1["w"]/r1["h"]) - (r2["w"]/r2["h"])) / max(r1["w"]/r1["h"], r2["w"]/r2["h"])
# #     area_diff = abs(r1["area"] - r2["area"]) / max(r1["area"], r2["area"])
# #     angle_diff = abs(r1["angle"] - r2["angle"])
# #     score = 1.0
# #     score *= np.exp(-ar_diff / cfg.max_aspect_diff)
# #     score *= np.exp(-area_diff / cfg.max_area_diff)
# #     score *= np.exp(-angle_diff / cfg.max_angle_diff)
# #     return round(max(0.0, min(1.0, score)), 3)
# #
# # # ─────────────────────────────────────────────
# # # TEMPORAL SMOOTHING
# # # ─────────────────────────────────────────────
# #
# # class GateTracker:
# #     def __init__(self, cfg: Config):
# #         self.alpha = cfg.tracker_alpha
# #         self.max_miss = cfg.tracker_max_miss
# #         self.mid_x = None
# #         self.mid_y = None
# #         self.missing = 0
# #     def update(self, mid: Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
# #         if mid is None:
# #             self.missing += 1
# #             if self.missing > self.max_miss:
# #                 self.mid_x = None
# #                 self.mid_y = None
# #             return self.get()
# #         self.missing = 0
# #         if self.mid_x is None:
# #             self.mid_x, self.mid_y = mid
# #         else:
# #             self.mid_x = int(self.alpha*mid[0] + (1-self.alpha)*self.mid_x)
# #             self.mid_y = int(self.alpha*mid[1] + (1-self.alpha)*self.mid_y)
# #         return self.get()
# #     def get(self) -> Optional[Tuple[int,int]]:
# #         if self.mid_x is None:
# #             return None
# #         return (self.mid_x, self.mid_y)
# #
# # # ─────────────────────────────────────────────
# # # DISPLAY
# # # ─────────────────────────────────────────────
# #
# # def _rot(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# #
# # def _fit_to_screen(img, max_w, max_h):
# #     h, w = img.shape[:2]
# #     scale = min(max_w/w, max_h/h, 1.0)
# #     if scale < 1.0:
# #         return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
# #     return img
# #
# # def build_display(image_bgr, mask, rects, pair, mid, confidence, cfg):
# #     anno = image_bgr.copy()
# #     overlay = anno.copy()
# #     overlay[mask>0] = [200,100,0]
# #     cv2.addWeighted(overlay, 0.35, anno, 0.65, 0, anno)
# #     for r in rects:
# #         box = cv2.boxPoints(((r["cx"], r["cy"]), (r["w"], r["h"]), r["angle"]))
# #         box = np.int0(box)
# #         cv2.drawContours(anno, [box], 0, (120,120,120), 1)
# #     if pair:
# #         for r in pair:
# #             box = cv2.boxPoints(((r["cx"], r["cy"]), (r["w"], r["h"]), r["angle"]))
# #             box = np.int0(box)
# #             cv2.drawContours(anno, [box], 0, (0,165,255), 2)
# #     if mid:
# #         cv2.drawMarker(anno, mid, (0,255,0), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
# #         cv2.circle(anno, mid, 8, (0,255,0), -1, cv2.LINE_AA)
# #         label = f"gate  conf={confidence:.2f}" if confidence else "gate"
# #         cv2.putText(anno, label, (max(0, mid[0]-60), max(20, mid[1]-14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0),2)
# #     else:
# #         cv2.putText(anno,"no gate",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,80,80),1)
# #     mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# #     p_mask, p_anno = _rot(mask_bgr), _rot(anno)
# #     cv2.putText(p_mask,"blue mask",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
# #     cv2.putText(p_anno,"detections",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
# #     combined = np.hstack([p_mask,p_anno])
# #     return _fit_to_screen(combined, cfg.display_max_w, cfg.display_max_h)
# #
# # # ─────────────────────────────────────────────
# # # MAIN PROCESSOR
# # # ─────────────────────────────────────────────
# #
# # def process_sequence(image_folder: str, cfg: Optional[Config]=None):
# #     if cfg is None: cfg = Config()
# #     paths = sorted(glob(os.path.join(image_folder,"*.jpg")))+sorted(glob(os.path.join(image_folder,"*.png")))
# #     if not paths: sys.exit(f"No images in {image_folder}")
# #     print(f"Processing {len(paths)} images from '{image_folder}'")
# #     WIN = "Gate detector  |  q = quit  |  any key = next"
# #     cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
# #     tracker = GateTracker(cfg)
# #     for idx, path in enumerate(paths):
# #         img = cv2.imread(path)
# #         if img is None: continue
# #         mask = blue_mask(img, cfg)
# #         rects = extract_rectangles(mask, img.shape, cfg)
# #         pair = find_gate_pair(rects, cfg)
# #         raw_mid = gate_midpoint(*pair) if pair else None
# #         confidence = gate_confidence(*pair, cfg) if pair else None
# #         mid = tracker.update(raw_mid)
# #         fname = os.path.basename(path)
# #         if mid: print(f"[{idx:04d}] {fname}  GATE  mid={mid}  conf={confidence}  rects={len(rects)}")
# #         else: print(f"[{idx:04d}] {fname}  no gate  rects={len(rects)}")
# #         vis = build_display(img, mask, rects, pair, mid, confidence, cfg)
# #         cv2.imshow(WIN, vis)
# #         cv2.resizeWindow(WIN, vis.shape[1], vis.shape[0])
# #         if (cv2.waitKey(30)&0xFF)==ord("q"): break
# #     cv2.destroyAllWindows()
# #
# # # ─────────────────────────────────────────────
# # # ENTRY POINT
# # # ─────────────────────────────────────────────
# #
# # if __name__ == "__main__":
# #     IMAGE_FOLDER = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320"
# #     cfg = Config(
# #         display_max_w=1800,
# #         display_max_h=900,
# #         use_hsv_confirm=True,
# #         tracker_alpha=0.4
# #     )
# #     process_sequence(IMAGE_FOLDER, cfg)
#
# """
# gate_detection_hybrid.py
# ========================
#
# Hybrid gate detector for full-resolution images.
# - Blob-based detection (for robustness)
# - Optional rectangle check for precision
# - EMA smoothing of midpoint
# - Full-resolution display
# """
#
# import os
# import sys
# import cv2
# import numpy as np
# from glob import glob
# from dataclasses import dataclass
# from typing import Optional, List, Tuple
#
# # ────────────────────────────────
# # CONFIGURATION
# # ────────────────────────────────
# @dataclass
# class Config:
#     # Blue thresholds (YUV + optional HSV)
#     blue_y_min: int = 0
#     blue_y_max: int = 220
#     blue_u_min: int = 122
#     blue_u_max: int = 255
#     blue_v_min: int = 0
#     blue_v_max: int = 123
#
#     use_hsv_confirm: bool = True
#     hsv_hue_min: int = 100
#     hsv_hue_max: int = 130
#     hsv_sat_min: int = 80
#     hsv_val_min: int = 40
#
#     blue_morph_k: int = 5
#
#     # Blob filtering
#     blob_min_area_ratio: float = 0.003
#
#     # Parallelism criteria
#     max_aspect_diff: float = 0.5
#     max_area_diff: float = 0.6
#     max_skew_deg: float = 10
#
#     # Tracker
#     tracker_alpha: float = 0.4
#     tracker_max_miss: int = 10
#
#     # Display
#     display_max_w: int = 1800
#     display_max_h: int = 900
#
# # ────────────────────────────────
# # BLUE MASK
# # ────────────────────────────────
# def blue_mask(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
#     yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
#     Y, U, V = cv2.split(yuv)
#     yuv_mask = ((Y >= cfg.blue_y_min) & (Y <= cfg.blue_y_max) &
#                 (U >= cfg.blue_u_min) & (U <= cfg.blue_u_max) &
#                 (V >= cfg.blue_v_min) & (V <= cfg.blue_v_max)).astype(np.uint8) * 255
#     if cfg.use_hsv_confirm:
#         hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
#         hsv_mask = cv2.inRange(hsv,
#                                (cfg.hsv_hue_min, cfg.hsv_sat_min, cfg.hsv_val_min),
#                                (cfg.hsv_hue_max, 255, 255))
#         mask = cv2.bitwise_and(yuv_mask, hsv_mask)
#     else:
#         mask = yuv_mask
#     # Morphology
#     k = cfg.blue_morph_k
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     return mask
#
# # ────────────────────────────────
# # BLOB EXTRACTION
# # ────────────────────────────────
# def extract_blobs(mask: np.ndarray, image_shape: tuple, cfg: Config) -> List[dict]:
#     H, W = image_shape[:2]
#     min_area = int(H * W * cfg.blob_min_area_ratio)
#     n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     blobs = []
#     for i in range(1, n):
#         area = int(stats[i, cv2.CC_STAT_AREA])
#         if area < min_area:
#             continue
#         x = int(stats[i, cv2.CC_STAT_LEFT])
#         y = int(stats[i, cv2.CC_STAT_TOP])
#         bw = int(stats[i, cv2.CC_STAT_WIDTH])
#         bh = int(stats[i, cv2.CC_STAT_HEIGHT])
#         blobs.append({
#             "x_min": x, "y_min": y, "x_max": x+bw, "y_max": y+bh,
#             "cx": int(cents[i,0]), "cy": int(cents[i,1]),
#             "area": area,
#             "aspect": bw / max(bh,1)
#         })
#     return blobs
#
# # ────────────────────────────────
# # PARALLEL PAIR DETECTION
# # ────────────────────────────────
# def _are_parallel(b1: dict, b2: dict, cfg: Config) -> bool:
#     dx = b2["cx"] - b1["cx"]
#     dy = b2["cy"] - b1["cy"]
#     if abs(dy) <= abs(dx):
#         return False
#     # Aspect ratio similarity
#     ar1, ar2 = b1["aspect"], b2["aspect"]
#     if (max(ar1,ar2)-min(ar1,ar2))/max(ar1,ar2) > cfg.max_aspect_diff:
#         return False
#     # Area similarity
#     a1, a2 = b1["area"], b2["area"]
#     if (max(a1,a2)-min(a1,a2))/max(a1,a2) > cfg.max_area_diff:
#         return False
#     # Skew
#     avg_w = (b1["x_max"]-b1["x_min"] + b2["x_max"]-b2["x_min"])/2
#     avg_h = (b1["y_max"]-b1["y_min"] + b2["y_max"]-b2["y_min"])/2
#     long_ax = np.array([1.0,0.0]) if avg_w>=avg_h else np.array([0.0,1.0])
#     join = np.array([dx,dy],dtype=float)
#     join /= np.linalg.norm(join)
#     dot = float(np.clip(np.dot(join,long_ax),-1.0,1.0))
#     angle = np.degrees(np.arccos(abs(dot)))
#     skew = abs(90.0 - angle)
#     return skew <= cfg.max_skew_deg
#
# def find_gate_pair(blobs: List[dict], cfg: Config) -> Optional[Tuple[dict,dict]]:
#     best = None
#     best_area = 0
#     for i in range(len(blobs)):
#         for j in range(i+1,len(blobs)):
#             b1,b2 = blobs[i],blobs[j]
#             if not _are_parallel(b1,b2,cfg):
#                 continue
#             combined = b1["area"]+b2["area"]
#             if combined > best_area:
#                 best_area = combined
#                 best = (b1,b2)
#     return best
#
# # ────────────────────────────────
# # MIDPOINT & CONFIDENCE
# # ────────────────────────────────
# def gate_midpoint(b1: dict, b2: dict) -> Tuple[int,int]:
#     return ((b1["cx"]+b2["cx"])//2,(b1["cy"]+b2["cy"])//2)
#
# def gate_confidence(b1: dict, b2: dict, cfg: Config) -> float:
#     ar_diff = abs(b1['aspect']-b2['aspect']) / max(b1['aspect'], b2['aspect'],1e-5)
#     a_diff  = abs(b1['area']-b2['area']) / max(b1['area'], b2['area'],1e-5)
#     score = 1.0
#     score *= (1.0 - ar_diff / cfg.max_aspect_diff)
#     score *= (1.0 - a_diff  / cfg.max_area_diff)
#     return round(max(0.0,min(1.0,score)),3)
#
# # ────────────────────────────────
# # TRACKER
# # ────────────────────────────────
# class GateTracker:
#     def __init__(self, cfg: Config):
#         self.alpha = cfg.tracker_alpha
#         self.max_miss = cfg.tracker_max_miss
#         self.mid_x = None
#         self.mid_y = None
#         self.missing = 0
#     def update(self, mid: Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
#         if mid is None:
#             self.missing += 1
#             if self.missing > self.max_miss:
#                 self.mid_x = None
#                 self.mid_y = None
#             return self.get()
#         self.missing = 0
#         if self.mid_x is None:
#             self.mid_x,self.mid_y = mid
#         else:
#             self.mid_x = int(self.alpha*mid[0] + (1-self.alpha)*self.mid_x)
#             self.mid_y = int(self.alpha*mid[1] + (1-self.alpha)*self.mid_y)
#         return self.get()
#     def get(self) -> Optional[Tuple[int,int]]:
#         if self.mid_x is None: return None
#         return (self.mid_x,self.mid_y)
#
# # ────────────────────────────────
# # DISPLAY
# # ────────────────────────────────
# def _rot(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# def _fit_to_screen(img,max_w,max_h):
#     h,w = img.shape[:2]
#     scale = min(max_w/w,max_h/h,1.0)
#     if scale < 1.0:
#         return cv2.resize(img,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
#     return img
#
# def build_display(image_bgr, mask, blobs, pair, mid, confidence, cfg):
#     anno = image_bgr.copy()
#     overlay = anno.copy()
#     overlay[mask>0] = [200,100,0]
#     cv2.addWeighted(overlay,0.35,anno,0.65,0,anno)
#     for b in blobs:
#         cv2.rectangle(anno,(b["x_min"],b["y_min"]),(b["x_max"],b["y_max"]),(120,120,120),1)
#     if pair is not None:
#         for b in pair:
#             cv2.rectangle(anno,(b["x_min"],b["y_min"]),(b["x_max"],b["y_max"]),(0,165,255),2)
#     if mid is not None:
#         cv2.drawMarker(anno,mid,(0,255,0),cv2.MARKER_CROSS,26,2,cv2.LINE_AA)
#         cv2.circle(anno,mid,8,(0,255,0),-1,cv2.LINE_AA)
#         label = f"gate  conf={confidence:.2f}" if confidence is not None else "gate"
#         cv2.putText(anno,label,(max(0,mid[0]-60),max(20,mid[1]-14)),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
#     else:
#         cv2.putText(anno,"no gate",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,80,80),1)
#     mask_bgr = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#     p_mask = _rot(mask_bgr)
#     p_anno = _rot(anno)
#     cv2.putText(p_mask,"blue mask",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
#     cv2.putText(p_anno,"detections",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
#     combined = np.hstack([p_mask,p_anno])
#     return _fit_to_screen(combined,cfg.display_max_w,cfg.display_max_h)
#
# # ────────────────────────────────
# # MAIN SEQUENCE
# # ────────────────────────────────
# def process_sequence(image_folder: str, cfg: Optional[Config]=None):
#     if cfg is None: cfg = Config()
#     paths = sorted(glob(os.path.join(image_folder,"*.jpg")))+sorted(glob(os.path.join(image_folder,"*.png")))
#     if not paths: sys.exit(f"No images in {image_folder}")
#     print(f"Processing {len(paths)} images from '{image_folder}'")
#     WIN = "Gate detector  |  q = quit  |  any key = next"
#     cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
#     tracker = GateTracker(cfg)
#     for idx,path in enumerate(paths):
#         img = cv2.imread(path)
#         if img is None: continue
#         mask = blue_mask(img,cfg)
#         blobs = extract_blobs(mask,img.shape,cfg)
#         pair = find_gate_pair(blobs,cfg)
#         raw_mid = gate_midpoint(*pair) if pair else None
#         confidence = gate_confidence(*pair,cfg) if pair else None
#         mid = tracker.update(raw_mid)
#         fname = os.path.basename(path)
#         if mid:
#             print(f"[{idx:04d}] {fname}  GATE  mid={mid}  conf={confidence}  blobs={len(blobs)}")
#         else:
#             print(f"[{idx:04d}] {fname}  no gate  blobs={len(blobs)}")
#         vis = build_display(img,mask,blobs,pair,mid,confidence,cfg)
#         cv2.imshow(WIN,vis)
#         cv2.resizeWindow(WIN,vis.shape[1],vis.shape[0])
#         if (cv2.waitKey(30)&0xFF)==ord("q"): break
#     cv2.destroyAllWindows()
#
# # ────────────────────────────────
# # ENTRY POINT
# # ────────────────────────────────
# if __name__ == "__main__":
#     IMAGE_FOLDER = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320"
#     cfg = Config(display_max_w=1800, display_max_h=900, use_hsv_confirm=True, tracker_alpha=0.4)
#     process_sequence(IMAGE_FOLDER,cfg)
"""
gate_detection_hybrid_v2.py
===========================

Hybrid gate detector for full-resolution images.
- Blue mask via YUV + optional HSV confirmation
- Connected-component blob extraction
- Parallel-pair detection (same logic as v1, proven to work)
- Confidence score based on aspect/area similarity
- EMA smoothing of midpoint with motion trail
- Fit-to-screen side-by-side display (mask | annotated)

Usage
-----
Set IMAGE_FOLDER at the bottom and run:
    python gate_detection_hybrid_v2.py
"""

import os
import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from glob import glob
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── YUV blue thresholds ───────────────────────────────────────────────────
    blue_y_min: int = 0
    blue_y_max: int = 220
    blue_u_min: int = 122
    blue_u_max: int = 255
    blue_v_min: int = 0
    blue_v_max: int = 123

    # ── Optional HSV confirmation (AND-ed with YUV mask) ──────────────────────
    use_hsv_confirm: bool = True
    hsv_hue_min: int = 100
    hsv_hue_max: int = 130
    hsv_sat_min: int = 80
    hsv_val_min: int = 40

    # ── Morphology ────────────────────────────────────────────────────────────
    blue_morph_k: int = 5

    # ── Blob filtering ────────────────────────────────────────────────────────
    blob_min_area_ratio: float = 0.003

    # ── Parallelism criteria ──────────────────────────────────────────────────
    max_aspect_diff: float = 0.5   # max relative AR difference
    max_area_diff:   float = 0.6   # max relative area difference
    max_skew_deg:    float = 10.0  # max deviation from perpendicular centroid join

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker_alpha:    float = 0.4  # EMA weight for new measurement
    tracker_max_miss: int   = 10   # frames without detection before reset
    trail_length:     int   = 20   # number of history points to draw

    # ── Display ───────────────────────────────────────────────────────────────
    display_max_w:    int  = 1800
    display_max_h:    int  = 900
    show_motion_trail: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Blue mask
# ─────────────────────────────────────────────────────────────────────────────

def blue_mask(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Binary mask of blue pixels.

    YUV gives lighting-robust chroma separation.
    Optional HSV confirmation (AND) reduces non-blue false positives.
    """
    yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    yuv_mask = (
        (Y >= cfg.blue_y_min) & (Y <= cfg.blue_y_max) &
        (U >= cfg.blue_u_min) & (U <= cfg.blue_u_max) &
        (V >= cfg.blue_v_min) & (V <= cfg.blue_v_max)
    ).astype(np.uint8) * 255

    if cfg.use_hsv_confirm:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(
            hsv,
            (cfg.hsv_hue_min, cfg.hsv_sat_min, cfg.hsv_val_min),
            (cfg.hsv_hue_max, 255, 255),
        )
        mask = cv2.bitwise_and(yuv_mask, hsv_mask)
    else:
        mask = yuv_mask

    k      = cfg.blue_morph_k
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Blob extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_blobs(mask: np.ndarray, image_shape: tuple, cfg: Config) -> List[dict]:
    """Connected-component blobs above the minimum area threshold."""
    H, W     = image_shape[:2]
    min_area = int(H * W * cfg.blob_min_area_ratio)

    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append({
            "x_min":  x,
            "y_min":  y,
            "x_max":  x + bw,
            "y_max":  y + bh,
            "cx":     int(cents[i, 0]),
            "cy":     int(cents[i, 1]),
            "area":   area,
            "aspect": bw / max(bh, 1),
        })
    return blobs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Parallel pair detection
# ─────────────────────────────────────────────────────────────────────────────

def _are_parallel(b1: dict, b2: dict, cfg: Config) -> bool:
    """
    Return True if two blobs look like the two bars of a gate.

    Conditions:
      0. Image is rotated 90° CW → real-world vertical pillars appear stacked
         along the image Y-axis. Require |dy| > |dx|.
      1. Similar aspect ratio (w/h).
      2. Similar area.
      3. Vector joining centroids is perpendicular to the blobs' long axis
         (within max_skew_deg).
    """
    dx = b2["cx"] - b1["cx"]
    dy = b2["cy"] - b1["cy"]

    # 0. Must be stacked vertically in image coordinates
    if abs(dy) <= abs(dx):
        return False

    # 1. Aspect ratio similarity
    ar1, ar2 = b1["aspect"], b2["aspect"]
    larger   = max(ar1, ar2)
    if larger == 0:
        return False
    if (larger - min(ar1, ar2)) / larger > cfg.max_aspect_diff:
        return False

    # 2. Area similarity
    a1, a2   = b1["area"], b2["area"]
    larger_a = max(a1, a2)
    if (larger_a - min(a1, a2)) / larger_a > cfg.max_area_diff:
        return False

    # 3. Centroid join ⊥ long axis
    #    Determine each blob's long axis independently, then average.
    def _long_ax(b: dict) -> np.ndarray:
        bw = b["x_max"] - b["x_min"]
        bh = b["y_max"] - b["y_min"]
        return np.array([1.0, 0.0]) if bw >= bh else np.array([0.0, 1.0])

    long_ax = _long_ax(b1) + _long_ax(b2)
    norm    = np.linalg.norm(long_ax)
    if norm == 0:
        return False
    long_ax /= norm

    join  = np.array([dx, dy], dtype=float)
    join /= np.linalg.norm(join)
    dot   = float(np.clip(np.dot(join, long_ax), -1.0, 1.0))
    angle = np.degrees(np.arccos(abs(dot)))   # 0° = parallel to long axis
    skew  = abs(90.0 - angle)                 # deviation from perpendicular

    return skew <= cfg.max_skew_deg


def find_gate_pair(
    blobs: List[dict], cfg: Config
) -> Optional[Tuple[dict, dict]]:
    """Best (largest combined area) valid parallel pair, or None."""
    best      = None
    best_area = 0
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            b1, b2 = blobs[i], blobs[j]
            if not _are_parallel(b1, b2, cfg):
                continue
            combined = b1["area"] + b2["area"]
            if combined > best_area:
                best_area = combined
                best      = (b1, b2)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Midpoint & confidence
# ─────────────────────────────────────────────────────────────────────────────

def gate_midpoint(b1: dict, b2: dict) -> Tuple[int, int]:
    return (
        (b1["cx"] + b2["cx"]) // 2,
        (b1["cy"] + b2["cy"]) // 2,
    )


def gate_confidence(b1: dict, b2: dict, cfg: Config) -> float:
    """
    Score in [0, 1] based on how similar the two blobs are.

    Uses aspect-ratio and area differences, both normalised by their
    respective thresholds so the score is 1.0 for a perfect match and
    approaches 0 near the rejection boundary.
    """
    ar_diff = (
        abs(b1["aspect"] - b2["aspect"])
        / max(b1["aspect"], b2["aspect"], 1e-5)
    )
    a_diff = (
        abs(b1["area"] - b2["area"])
        / max(b1["area"], b2["area"], 1e-5)
    )
    score  = (1.0 - ar_diff / cfg.max_aspect_diff)
    score *= (1.0 - a_diff  / cfg.max_area_diff)
    return round(max(0.0, min(1.0, score)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class GateTracker:
    """
    Exponential moving average (EMA) smoother for the gate midpoint.

    Keeps a fixed-length history of smoothed positions for the motion trail.
    Resets after max_miss consecutive missed detections.
    """

    def __init__(self, cfg: Config):
        self.alpha     = cfg.tracker_alpha
        self.max_miss  = cfg.tracker_max_miss
        self.trail_len = cfg.trail_length
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self.missing   = 0
        self.history:  List[Tuple[int, int]] = []

    def update(self, mid: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if mid is None:
            self.missing += 1
            if self.missing > self.max_miss:
                self._x = None
                self._y = None
                self.history.clear()
            return self.get()

        self.missing = 0
        if self._x is None:
            self._x, self._y = float(mid[0]), float(mid[1])
        else:
            self._x = self.alpha * mid[0] + (1 - self.alpha) * self._x
            self._y = self.alpha * mid[1] + (1 - self.alpha) * self._y

        self.history.append((int(self._x), int(self._y)))
        if len(self.history) > self.trail_len:
            self.history.pop(0)

        return self.get()

    def get(self) -> Optional[Tuple[int, int]]:
        if self._x is None:
            return None
        return (int(self._x), int(self._y))


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rot(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _fit_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w  = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def _transform_trail_point(
    pt: Tuple[int, int], orig_h: int, orig_w: int
) -> Tuple[int, int]:
    """
    Map a point from the original (pre-rotation) image space to the
    rotated (post-ROTATE_90_COUNTERCLOCKWISE) image space.

    After CCW rotation: new_x = y,  new_y = (orig_w - 1) - x
    """
    x, y = pt
    return (y, (orig_w - 1) - x)


def build_display(
    image_bgr:  np.ndarray,
    mask:       np.ndarray,
    blobs:      List[dict],
    pair:       Optional[Tuple[dict, dict]],
    mid:        Optional[Tuple[int, int]],
    confidence: Optional[float],
    tracker:    GateTracker,
    cfg:        Config,
) -> np.ndarray:
    """
    Side-by-side panel: [ blue mask | annotated frame ], rotated CCW, scaled.
    """
    orig_h, orig_w = image_bgr.shape[:2]

    anno    = image_bgr.copy()
    overlay = anno.copy()
    overlay[mask > 0] = [200, 100, 0]
    cv2.addWeighted(overlay, 0.35, anno, 0.65, 0, anno)

    # All blobs — thin grey boxes
    for b in blobs:
        cv2.rectangle(
            anno,
            (b["x_min"], b["y_min"]),
            (b["x_max"], b["y_max"]),
            (120, 120, 120), 1,
        )

    # Gate pair — bright orange boxes
    if pair is not None:
        for b in pair:
            cv2.rectangle(
                anno,
                (b["x_min"], b["y_min"]),
                (b["x_max"], b["y_max"]),
                (0, 165, 255), 2,
            )

    # Midpoint marker
    if mid is not None:
        cv2.drawMarker(anno, mid, (0, 255, 0),
                       cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(anno, mid, 8, (0, 255, 0), -1, cv2.LINE_AA)
        label = f"gate  conf={confidence:.2f}" if confidence is not None else "gate"
        cv2.putText(
            anno, label,
            (max(0, mid[0] - 60), max(20, mid[1] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(anno, "no gate", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

    # Rotate both panels
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    p_mask   = _rot(mask_bgr)
    p_anno   = _rot(anno)

    cv2.putText(p_mask, "blue mask",  (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(p_anno, "detections", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Motion trail — coords must be transformed to post-rotation space
    if cfg.show_motion_trail and len(tracker.history) >= 2:
        rot_trail = [
            _transform_trail_point(pt, orig_h, orig_w)
            for pt in tracker.history
        ]
        for i in range(1, len(rot_trail)):
            alpha_fade = i / len(rot_trail)
            color      = (0, int(255 * alpha_fade), 0)
            cv2.line(p_anno, rot_trail[i - 1], rot_trail[i], color, 1, cv2.LINE_AA)

    combined = np.hstack([p_mask, p_anno])
    return _fit_to_screen(combined, cfg.display_max_w, cfg.display_max_h)


# ─────────────────────────────────────────────────────────────────────────────
# Main sequence runner (live window)
# ─────────────────────────────────────────────────────────────────────────────

def process_sequence(image_folder: str, cfg: Optional[Config] = None):
    """Process every image in image_folder. Press 'q' to quit."""
    if cfg is None:
        cfg = Config()

    paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not paths:
        paths = sorted(glob(os.path.join(image_folder, "*.png")))
    if not paths:
        sys.exit(f"No images found in: {image_folder}")

    print(f"Processing {len(paths)} images from '{image_folder}'")

    WIN     = "Gate detector  |  q = quit"
    tracker = GateTracker(cfg)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for idx, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] {path}")
            continue

        mask       = blue_mask(img, cfg)
        blobs      = extract_blobs(mask, img.shape, cfg)
        pair       = find_gate_pair(blobs, cfg)
        raw_mid    = gate_midpoint(*pair) if pair is not None else None
        confidence = gate_confidence(*pair, cfg) if pair is not None else None
        mid        = tracker.update(raw_mid)

        fname = os.path.basename(path)
        if mid is not None:
            print(f"  [{idx:04d}] {fname}  GATE  mid={mid}  conf={confidence}  blobs={len(blobs)}")
        else:
            print(f"  [{idx:04d}] {fname}  no gate  blobs={len(blobs)}")

        vis = build_display(img, mask, blobs, pair, mid, confidence, tracker, cfg)
        cv2.imshow(WIN, vis)
        cv2.resizeWindow(WIN, vis.shape[1], vis.shape[0])
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Random preview (matplotlib grid)
# ─────────────────────────────────────────────────────────────────────────────

def process_random_preview(
    image_folder: str,
    cfg:          Optional[Config] = None,
    n_images:     int = 20,
):
    """Sample n_images at random and show results in a matplotlib grid."""
    if cfg is None:
        cfg = Config()

    paths = (
        sorted(glob(os.path.join(image_folder, "*.jpg")))
        + sorted(glob(os.path.join(image_folder, "*.png")))
    )
    if not paths:
        sys.exit(f"No images found in: {image_folder}")

    selected = random.sample(paths, min(n_images, len(paths)))
    print(f"Selected {len(selected)} random images for preview.")

    tracker = GateTracker(cfg)
    cols    = 5
    rows    = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes      = axes.flatten()

    for idx, path in enumerate(selected):
        img = cv2.imread(path)
        if img is None:
            axes[idx].axis("off")
            continue

        mask       = blue_mask(img, cfg)
        blobs      = extract_blobs(mask, img.shape, cfg)
        pair       = find_gate_pair(blobs, cfg)
        raw_mid    = gate_midpoint(*pair) if pair is not None else None
        confidence = gate_confidence(*pair, cfg) if pair is not None else None
        mid        = tracker.update(raw_mid)

        vis = build_display(img, mask, blobs, pair, mid, confidence, tracker, cfg)
        axes[idx].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[idx].axis("off")
        axes[idx].set_title("GATE" if mid is not None else "no gate")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    IMAGE_FOLDER = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320"

    cfg = Config(
        display_max_w=1800,
        display_max_h=900,
        use_hsv_confirm=True,
        tracker_alpha=0.4,
    )

    # Uncomment whichever mode you want:
    process_random_preview(IMAGE_FOLDER, cfg, n_images=20)
    # process_sequence(IMAGE_FOLDER, cfg)