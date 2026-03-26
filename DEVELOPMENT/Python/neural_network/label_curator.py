"""
label_curator_gui.py — GUI tool to curate and label drone zoo images.
MADE WITH HELP OF AI CLAUDE

Controls:
    y / Enter          = keep with current label
    n / Delete         = discard
    a / Left arrow     = previous image
    d / Right arrow    = next image
    1-8                = override label
    r                  = reset override
    z                  = undo last action
    s                  = save JSON
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os, glob, sys, json
from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_FOLDER      = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
MASKS_OUTPUT_ROOT = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\latest_flight_all_generated_masks'
OUTPUT_JSON       = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\labels.json'

GROUND_MASK_DIR = os.path.join(MASKS_OUTPUT_ROOT, 'ground_masks')
POLE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'pole_masks')
TREE_MASK_DIR   = os.path.join(MASKS_OUTPUT_ROOT, 'tree_masks')

GROUND_COLOUR         = (0, 255, 0)
POLE_COLOUR           = (0, 165, 255)
TREE_COLOUR           = (255, 0, 200)
GATE_COLOUR           = (255, 255, 0)
OPACITY               = 0.5
N_STRIPS              = 5
BIEXP_EXP             = 0.9
SPIN_THRESHOLD        = 5.0
TREE_PAD_W            = 0
TREE_PAD_H            = 30
TREE_CLUSTER_M        = 35
GATE_MIN_REM          = 30
ROOT_EXTRA            = 2    # extra px below tree bbox to catch root-line artifact
BLOCK_MIN_WIDTH_FRAC  = 0.05  # default; adjustable at runtime with [ and ] keys
BLOCK_FRAC_STEP       = 0.01  # step size for adjusting threshold

# ══════════════════════════════════════════════════════════════════════════════
# GATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GateCfg:
    blue_y_min:int=0; blue_y_max:int=220
    blue_u_min:int=122; blue_u_max:int=255
    blue_v_min:int=0; blue_v_max:int=123
    blue_morph_k:int=5; blob_min_area_ratio:float=0.003
    max_aspect_diff:float=0.5; max_area_diff:float=0.6; max_skew_deg:float=10

def detect_gate(img, cfg):
    yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV); Y,U,V=cv2.split(yuv)
    mask=((Y>=cfg.blue_y_min)&(Y<=cfg.blue_y_max)&(U>=cfg.blue_u_min)&
          (U<=cfg.blue_u_max)&(V>=cfg.blue_v_min)&(V<=cfg.blue_v_max)).astype(np.uint8)*255
    k=cfg.blue_morph_k; ker=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,ker)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,ker)
    H,W=img.shape[:2]; mna=int(H*W*cfg.blob_min_area_ratio)
    n,_,stats,cents=cv2.connectedComponentsWithStats(mask,connectivity=8)
    blobs=[]
    for i in range(1,n):
        a=int(stats[i,cv2.CC_STAT_AREA])
        if a<mna: continue
        x=int(stats[i,cv2.CC_STAT_LEFT]); y=int(stats[i,cv2.CC_STAT_TOP])
        bw=int(stats[i,cv2.CC_STAT_WIDTH]); bh=int(stats[i,cv2.CC_STAT_HEIGHT])
        blobs.append({"x_min":x,"y_min":y,"x_max":x+bw,"y_max":y+bh,
                      "cx":int(cents[i,0]),"cy":int(cents[i,1]),"area":a,"aspect":bw/max(bh,1)})
    def par(b1,b2):
        dx,dy=b2["cx"]-b1["cx"],b2["cy"]-b1["cy"]
        if abs(dy)<=abs(dx): return False
        lrg=max(b1["aspect"],b2["aspect"])
        if lrg==0 or (lrg-min(b1["aspect"],b2["aspect"]))/lrg>cfg.max_aspect_diff: return False
        la=max(b1["area"],b2["area"])
        if (la-min(b1["area"],b2["area"]))/la>cfg.max_area_diff: return False
        if dx==0 and dy==0: return False
        aw=(b1["x_max"]-b1["x_min"]+b2["x_max"]-b2["x_min"])/2
        ah=(b1["y_max"]-b1["y_min"]+b2["y_max"]-b2["y_min"])/2
        lax=np.array([1.,0.]) if aw>=ah else np.array([0.,1.])
        j=np.array([dx,dy],dtype=float); j/=np.linalg.norm(j)
        dot=float(np.clip(np.dot(j,lax),-1.,1.))
        return abs(90.-np.degrees(np.arccos(abs(dot))))<=cfg.max_skew_deg
    best,ba=None,0
    for i in range(len(blobs)):
        for j in range(i+1,len(blobs)):
            b1,b2=blobs[i],blobs[j]
            if not par(b1,b2): continue
            c=b1["area"]+b2["area"]
            if c>ba: ba,best=c,(b1,b2)
    mid=None
    if best: b1,b2=best; mid=((b1["cx"]+b2["cx"])//2,(b1["cy"]+b2["cy"])//2)
    return mask,blobs,best,mid

# ══════════════════════════════════════════════════════════════════════════════
# GROUND PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_mask(d, stem, suf):
    p=os.path.join(d,f'{stem}_mask_{suf}.png')
    return cv2.imread(p,cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None

def get_partitions(W):
    t=np.linspace(-1,1,N_STRIPS+1); w=np.sign(t)*(np.abs(t)**BIEXP_EXP)
    w=(w-w[0])/(w[-1]-w[0]); edges=(w*W).astype(int)
    return list(reversed([(int(edges[i]),int(edges[i+1])) for i in range(N_STRIPS)]))

def part_counts(cc, parts, H):
    return [(int(cc[x0:x1].sum()),
             100.*cc[x0:x1].sum()/((x1-x0)*H) if (x1-x0)*H>0 else 0.)
            for x0,x1 in parts]

def clean_blobs(m, frac=0.025):
    if m is None: return None
    H,W=m.shape; b=(m>0).astype(np.uint8); se=max(1,int(W*frac))
    _,lab=cv2.connectedComponents(b,connectivity=8)
    sl=set(np.unique(lab[:,:se])); sl.discard(0)
    if not sl: return m
    k=np.isin(lab,list(sl)).astype(np.uint8)
    return (k*255).astype(np.uint8) if k.sum()>0 else m

def carve_poles(g, p):
    if g is None or p is None: return g
    r=g.copy(); r[p>0]=0; return r

def fill_ground_holes(ground_mask, closing_radius=0.04, min_hole_area=200):
    if ground_mask is None: return None
    H,W = ground_mask.shape
    binary = (ground_mask > 0).astype(np.uint8)
    rows_with_ground = np.any(binary > 0, axis=1)
    if not rows_with_ground.any(): return ground_mask
    top_ground_row = int(np.argmax(rows_with_ground))
    ksize  = max(3, int(H * closing_radius) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    new_pixels = ((closed == 1) & (binary == 0)).astype(np.uint8)
    new_pixels[:top_ground_row, :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(new_pixels, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
            new_pixels[labels == i] = 0
    result = binary.copy(); result[new_pixels == 1] = 1
    return (result * 255).astype(np.uint8)

def get_tree_boxes(tm, pw=0, ph=0, cm=10):
    if tm is None: return []
    H,W=tm.shape; b=(tm>0).astype(np.uint8)
    if b.sum()==0: return []
    if cm>0:
        k=cm*2+1; ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        dil=cv2.dilate(b,ker,iterations=1)
    else: dil=b
    _,cl=cv2.connectedComponents(dil,connectivity=8)
    cc2={}
    for cid in np.unique(cl):
        if cid==0: continue
        op=(b==1)&(cl==cid)
        if not op.any(): continue
        cols=np.where(op.any(axis=0))[0]; cc2[cid]=(int(cols[0]),int(cols[-1]))
    par={c:c for c in cc2}
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    def union(a,b_): par[find(a)]=find(b_)
    cids=list(cc2.keys())
    for i in range(len(cids)):
        for j in range(i+1,len(cids)):
            a,b_=cids[i],cids[j]; ax0,ax1=cc2[a]; bx0,bx1=cc2[b_]
            if ax0<=bx1+cm and bx0<=ax1+cm: union(a,b_)
    groups={}
    for c in cids: groups.setdefault(find(c),[]).append(c)
    boxes=[]
    for root_,mems in groups.items():
        comb=np.zeros_like(b,dtype=bool)
        for c in mems: comb|=((b==1)&(cl==c))
        rows=np.where(comb.any(axis=1))[0]; cols=np.where(comb.any(axis=0))[0]
        ty,by_=int(rows[0]),int(rows[-1]); tx,bx_=int(cols[0]),int(cols[-1])
        tw=bx_-tx+1; th=by_-ty+1
        px=max(0,tx-pw); py=max(0,ty-ph)
        px2=min(W-1,tx+tw+pw); py2=min(H-1,ty+th+ph)
        boxes.append({'padded':(px,py,px2-px,py2-py)})
    return boxes

def carve_trees_rot(g, tm, pw, ph, cm):
    """Carve tree bboxes from ground. Extends ROOT_EXTRA px below bbox to kill root-line artifact."""
    if g is None or tm is None: return g
    H = g.shape[0]
    boxes=get_tree_boxes(tm,pw,ph,cm); r=g.copy()
    for b in boxes:
        px,py,pw_,ph_=b['padded']
        carve_to = min(H, py + ph_ + ROOT_EXTRA)
        r[py:carve_to, px:px+pw_]=0
    return r

def gate_boost_fn(ground_rot, raw, gate_data, W_rot, H_rot, parts, orig_shape):
    _,_,pair,mid=gate_data
    if mid is None or pair is None: return ground_rot,None,None
    Ho,Wo=orig_shape; b1,b2=pair
    if b1['y_min']>b2['y_min']: b1,b2=b2,b1
    iy0=b1['y_max']; iy1=b2['y_min']
    if iy0>=iy1: iy0=min(b1['y_min'],b2['y_min']); iy1=max(b1['y_max'],b2['y_max'])
    ix0=max(b1['x_min'],b2['x_min']); ix1=min(b1['x_max'],b2['x_max'])
    if ix0>=ix1: ix0=min(b1['x_min'],b2['x_min']); ix1=max(b1['x_max'],b2['x_max'])
    ix0=max(0,min(ix0,Wo-1)); ix1=max(0,min(ix1,Wo-1))
    iy0=max(0,min(iy0,Ho-1)); iy1=max(0,min(iy1,Ho-1))
    inner_rect=(ix0,iy0,ix1,iy1)
    if ix0>=ix1 or iy0>=iy1: return ground_rot,None,inner_rect
    if raw is not None:
        cy1=min(Ho-1,iy1+(iy1-iy0))
        rp=int((raw[iy0:cy1,ix0:ix1]>0).sum())
        cl=clean_blobs(raw)
        cp=int((cl[iy0:cy1,ix0:ix1]>0).sum()) if cl is not None else 0
        if rp-cp<GATE_MIN_REM: return ground_rot,None,inner_rect
    ox,oy=mid; rc=Ho-1-oy
    gs=None
    for si,(x0,x1) in enumerate(parts):
        if x0<=rc<x1: gs=si; break
    if gs is None: return ground_rot,None,inner_rect
    sx0,sx1=parts[gs]
    bst=ground_rot.copy() if ground_rot is not None else np.zeros((H_rot,W_rot),dtype=np.uint8)
    bst[:,sx0:sx1]=255
    return bst,gs,inner_rect

def get_blocked_strips(ground_rot, partitions, W_rot, H_rot,
                       boost_strip_idx=None, gate_pair=None, orig_shape=None,
                       block_frac=None):
    """
    A strip is BLOCKED if >BLOCK_MIN_WIDTH_FRAC of its columns have zero
    ground pixels. Gate strip and gate blob column range are exempt.
    """
    blocked = set()
    if ground_rot is None:
        return blocked

    if block_frac is None:
        block_frac = BLOCK_MIN_WIDTH_FRAC

    col_has_ground = (ground_rot > 0).any(axis=0)

    # Gate-exempt column range in rotated space
    gate_exempt_x0 = None
    gate_exempt_x1 = None
    if gate_pair is not None and orig_shape is not None:
        b1, b2 = gate_pair
        H_orig = orig_shape[0]
        all_orig_y = [b1['y_min'], b1['y_max'], b2['y_min'], b2['y_max']]
        rot_xs = [H_orig - 1 - y for y in all_orig_y]
        gate_exempt_x0 = max(0,       min(rot_xs))
        gate_exempt_x1 = min(W_rot-1, max(rot_xs))

    for si, (x0, x1) in enumerate(partitions):
        if si == boost_strip_idx:
            continue
        strip_w = x1 - x0
        if strip_w <= 0:
            continue

        check_cols = np.ones(strip_w, dtype=bool)
        if gate_exempt_x0 is not None:
            lx0 = max(0,       gate_exempt_x0 - x0)
            lx1 = min(strip_w, gate_exempt_x1 - x0 + 1)
            if lx0 < lx1:
                check_cols[lx0:lx1] = False

        n_checked = check_cols.sum()
        if n_checked == 0:
            continue

        strip_ground = col_has_ground[x0:x1]
        empty_cols   = (~strip_ground[check_cols]).sum()
        empty_frac   = empty_cols / n_checked

        if empty_frac > block_frac:
            blocked.add(si)

    return blocked

# ══════════════════════════════════════════════════════════════════════════════
# FRAME BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_frame(img, gm, pm, tm, block_frac=BLOCK_MIN_WIDTH_FRAC):
    """Build full annotated BGR frame at native resolution. Caller scales it."""
    top=img.copy()
    gm_display = carve_poles(fill_ground_holes(clean_blobs(gm)), pm)
    for mask,col in [(gm_display,GROUND_COLOUR),(pm,POLE_COLOUR),(tm,TREE_COLOUR)]:
        if mask is not None:
            c=np.zeros_like(top); c[mask>0]=col
            top=cv2.addWeighted(top,1.,c,OPACITY,0)

    gate_data=detect_gate(img,GateCfg())
    gm_vis,blobs,pair,mid=gate_data
    c=np.zeros_like(top); c[gm_vis>0]=GATE_COLOUR
    top=cv2.addWeighted(top,1.,c,OPACITY,0)
    if pair:
        for b in pair:
            cv2.rectangle(top,(b['x_min'],b['y_min']),(b['x_max'],b['y_max']),GATE_COLOUR,2)
        b1v,b2v=pair
        if b1v['y_min']>b2v['y_min']: b1v,b2v=b2v,b1v
        iy0=b1v['y_max']; iy1=b2v['y_min']
        if iy0>=iy1: iy0=min(b1v['y_min'],b2v['y_min']); iy1=max(b1v['y_max'],b2v['y_max'])
        ix0=max(b1v['x_min'],b2v['x_min']); ix1=min(b1v['x_max'],b2v['x_max'])
        if ix0>=ix1: ix0=min(b1v['x_min'],b2v['x_min']); ix1=max(b1v['x_max'],b2v['x_max'])
        Ho_,Wo_=top.shape[:2]
        ix0=max(0,min(ix0,Wo_-1)); ix1=max(0,min(ix1,Wo_-1))
        iy0=max(0,min(iy0,Ho_-1)); iy1=max(0,min(iy1,Ho_-1))
        if ix0<ix1 and iy0<iy1:
            ov=top.copy(); cv2.rectangle(ov,(ix0,iy0),(ix1,iy1),(0,0,255),-1)
            cv2.addWeighted(ov,0.4,top,0.6,0,top)
            cv2.rectangle(top,(ix0,iy0),(ix1,iy1),(0,0,255),2)
    if mid: cv2.drawMarker(top,mid,(0,255,0),cv2.MARKER_CROSS,20,2,cv2.LINE_AA)

    top_rot=cv2.rotate(top,cv2.ROTATE_90_COUNTERCLOCKWISE)
    Hr,Wr=top_rot.shape[:2]; Ho,Wo=img.shape[:2]

    # Full ground pipeline (same order as mask_viewer)
    dg=clean_blobs(gm)
    dg=fill_ground_holes(dg)
    dg=carve_poles(dg,pm)
    if tm is not None and dg is not None:
        tr=cv2.rotate(tm,cv2.ROTATE_90_COUNTERCLOCKWISE)
        gr=cv2.rotate(dg,cv2.ROTATE_90_COUNTERCLOCKWISE)
        gr=carve_trees_rot(gr,tr,TREE_PAD_W,TREE_PAD_H,TREE_CLUSTER_M)
        dg=cv2.rotate(gr,cv2.ROTATE_90_CLOCKWISE)
    ground_rot=cv2.rotate(dg,cv2.ROTATE_90_COUNTERCLOCKWISE) if dg is not None else None

    parts=get_partitions(Wr)
    grb,bs,_=gate_boost_fn(ground_rot,gm,gate_data,Wr,Hr,parts,(Ho,Wo))

    # Blocked strip detection — on processed ground, before boost
    blocked = get_blocked_strips(
        ground_rot, parts, Wr, Hr,
        boost_strip_idx=bs,
        gate_pair=pair,
        orig_shape=(Ho,Wo),
        block_frac=block_frac)

    if grb is not None:
        cc=(grb>0).astype(np.int32).sum(axis=0).astype(float)
    else:
        cc=np.zeros(Wr,dtype=float)
    pd=part_counts(cc,parts,Hr)

    # Best strip excludes blocked ones
    scores = [pct if si not in blocked else -1.0 for si,(_,pct) in enumerate(pd)]
    avail  = [s for s in scores if s >= 0]
    mp     = max(avail) if avail else 0.0
    must_spin = mp < SPIN_THRESHOLD
    bi = int(np.argmax(scores)) if not must_spin else 0

    # Strip lines + numbers
    for si,(x0,_) in enumerate(parts):
        if si>0: cv2.line(top_rot,(x0,0),(x0,Hr),(160,160,160),1)
    for si,(x0,x1) in enumerate(parts):
        cv2.putText(top_rot,str(si+1),((x0+x1)//2-5,16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)

    # Gate boost highlight
    if bs is not None:
        sx0,sx1=parts[bs]; ind=top_rot.copy()
        cv2.rectangle(ind,(sx0,0),(sx1,Hr),(255,255,0),-1)
        cv2.addWeighted(ind,0.18,top_rot,0.82,0,top_rot)
        cv2.line(top_rot,(sx0,0),(sx0,Hr),(255,255,0),2)
        cv2.line(top_rot,(sx1,0),(sx1,Hr),(255,255,0),2)

    # Blocked strip overlays
    for si in blocked:
        bkx0,bkx1=parts[si]
        red_ov=top_rot.copy()
        cv2.rectangle(red_ov,(bkx0,0),(bkx1,Hr),(0,0,180),-1)
        cv2.addWeighted(red_ov,0.35,top_rot,0.65,0,top_rot)
        cv2.line(top_rot,(bkx0,0),(bkx0,Hr),(0,0,255),2)
        cv2.line(top_rot,(bkx1,0),(bkx1,Hr),(0,0,255),2)
        cv2.putText(top_rot,'X',((bkx0+bkx1)//2-8,Hr//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)

    if must_spin:
        ov=top_rot.copy(); cv2.rectangle(ov,(0,0),(Wr,Hr),(0,0,180),-1)
        cv2.addWeighted(ov,0.3,top_rot,0.7,0,top_rot)
        cv2.putText(top_rot,'TURN AROUND',(Wr//2-80,Hr//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
    else:
        bx0,bx1=parts[bi]; hl=top_rot.copy()
        cv2.rectangle(hl,(bx0,0),(bx1,Hr),(0,220,80),-1)
        cv2.addWeighted(hl,0.15,top_rot,0.85,0,top_rot)
        cv2.line(top_rot,(bx0,0),(bx0,Hr),(0,220,80),2)
        cv2.line(top_rot,(bx1,0),(bx1,Hr),(0,220,80),2)

    # Tree bounding boxes on rotated image
    if tm is not None:
        tree_rot_disp = cv2.rotate(tm, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for b in get_tree_boxes(tree_rot_disp, TREE_PAD_W, TREE_PAD_H, TREE_CLUSTER_M):
            tx,ty,tw_,th_ = b['padded']
            cv2.rectangle(top_rot, (tx,ty), (tx+tw_,ty+th_), (255,80,255), 2)

    # Bar chart with per-column line
    ph_ = max(80, Hr//4)
    bpanel = np.full((ph_, Wr, 3), (30,30,30), dtype=np.uint8)
    pad_t=12; pad_b=8; pad_l=4; pad_r=4
    dw = Wr - pad_l - pad_r
    dh = ph_ - pad_t - pad_b
    max_count = cc.max() if cc.max() > 0 else 1.0

    for si, (x0, x1) in enumerate(parts):
        px0 = pad_l + int(x0 / Wr * dw)
        px1 = pad_l + int(x1 / Wr * dw)
        shade = (50,50,50) if si%2==0 else (42,42,42)
        cv2.rectangle(bpanel, (px0, pad_t), (px1, pad_t+dh), shade, -1)
        _, pct = pd[si]
        bar_h = int(pct / 100.0 * dh)
        if si in blocked:
            bar_col = (30, 30, 100)
        elif si == bi:
            bar_col = (0, 100, 40)
        else:
            bar_col = (40, 70, 100)
        cv2.rectangle(bpanel, (px0, pad_t+dh-bar_h), (px1, pad_t+dh), bar_col, -1)
        if bar_h > 14:
            lbl = f'{pct:.0f}%'
            lx = (px0+px1)//2 - len(lbl)*3
            cv2.putText(bpanel, lbl, (lx, pad_t+dh-bar_h+11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (180,255,180) if si==bi else (140,180,220), 1)
        if si > 0:
            cv2.line(bpanel, (px0, pad_t), (px0, pad_t+dh), (80,80,80), 1)
        mid_px = (px0+px1)//2
        cv2.putText(bpanel, str(si+1), (mid_px-4, pad_t-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160,160,160), 1)

    xs = np.linspace(0, len(cc)-1, len(cc))
    px_arr = (pad_l + xs / Wr * dw).astype(int)
    py_arr = (pad_t + dh - (cc / max_count) * dh).astype(int)
    pts = np.stack([px_arr, py_arr], axis=1).reshape(-1,1,2).astype(np.int32)
    cv2.polylines(bpanel, [pts], False, (0,255,100), 1, cv2.LINE_AA)

    auto_lbl = 6 if must_spin else bi+1
    return np.vstack((top_rot, bpanel)), pd, auto_lbl

# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

class CuratorApp:
    def __init__(self, root):
        self.root     = root
        self.root.title('Drone Zoo — Label Curator')
        self.root.configure(bg='#1e1e1e')
        self.root.geometry('1400x820')

        self.image_paths = []
        self.idx         = 0
        self.labels      = {}
        self.history     = []
        self.override    = None
        self._tk_img     = None
        self._frame_bgr  = None
        self._auto_lbl   = 1
        self._block_frac = BLOCK_MIN_WIDTH_FRAC

        self._load_images()
        self._load_json()
        self._build_ui()
        self.root.after(150, self._render)

    def _load_images(self):
        all_imgs=sorted(glob.glob(os.path.join(IMAGE_FOLDER,'*.jpg'))+
                        glob.glob(os.path.join(IMAGE_FOLDER,'*.png')),
                        key=lambda p: int(''.join(filter(str.isdigit, os.path.splitext(os.path.basename(p))[0])) or '0'))
        all_imgs=[p for p in all_imgs if '_mask' not in os.path.basename(p).lower()]
        def has_mask(p):
            stem=os.path.splitext(os.path.basename(p))[0]
            return any(os.path.exists(os.path.join(d,f'{stem}_mask_{s}.png'))
                       for d,s in [(GROUND_MASK_DIR,'ground'),(POLE_MASK_DIR,'pole'),(TREE_MASK_DIR,'tree')])
        self.image_paths=[p for p in all_imgs if has_mask(p)]
        if not self.image_paths:
            messagebox.showerror('No images','No images with masks found.')
            sys.exit(1)

    def _load_json(self):
        if os.path.exists(OUTPUT_JSON):
            with open(OUTPUT_JSON) as f:
                for e in json.load(f):
                    self.labels[e['path']]=e
        print(f'Loaded {len(self.labels)} existing labels.')

    def _build_ui(self):
        BTN_BASE = dict(fg='white', relief=tk.FLAT, padx=10, pady=5,
                        cursor='hand2', activeforeground='white',
                        font=('Helvetica',11))
        BTN   = dict(**BTN_BASE, bg='#444',    activebackground='#666')
        BTN_G = dict(**BTN_BASE, bg='#1a5c2e', activebackground='#267a3e')
        BTN_R = dict(**BTN_BASE, bg='#5c1a1a', activebackground='#7a2626')
        BTN_O = dict(**BTN_BASE, bg='#5c3d0a', activebackground='#7a5210')

        toolbar = tk.Frame(self.root, bg='#2d2d2d', pady=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text='✔  Keep',    command=self._keep,    **BTN_G).pack(side=tk.LEFT, padx=6)
        tk.Button(toolbar, text='✘  Discard', command=self._discard, **BTN_R).pack(side=tk.LEFT, padx=6)
        tk.Button(toolbar, text='↩  Undo',    command=self._undo,    **BTN_O).pack(side=tk.LEFT, padx=6)
        tk.Button(toolbar, text='◀  Prev',    command=self._prev,    **BTN  ).pack(side=tk.LEFT, padx=6)
        tk.Button(toolbar, text='▶  Next',    command=self._next,    **BTN  ).pack(side=tk.LEFT, padx=6)
        tk.Button(toolbar, text='💾  Save',   command=self._save,    **BTN  ).pack(side=tk.LEFT, padx=6)

        tk.Label(toolbar, text='  Override:', bg='#2d2d2d', fg='#aaa',
                 font=('Helvetica',11)).pack(side=tk.LEFT, padx=(12,4))
        self._strip_btns = []
        for i in range(1,7):
            lbl = str(i) if i<=5 else '↺6'
            b = tk.Button(toolbar, text=lbl, width=3,
                          command=lambda n=i: self._set_override(n), **BTN)
            b.pack(side=tk.LEFT, padx=2)
            self._strip_btns.append(b)
        tk.Button(toolbar, text='Reset', command=self._reset_override, **BTN_O).pack(side=tk.LEFT, padx=6)

        self._lbl_var = tk.StringVar(value='Label: —')
        tk.Label(toolbar, textvariable=self._lbl_var, bg='#2d2d2d',
                 fg='#00dc64', font=('Helvetica',13,'bold')).pack(side=tk.LEFT, padx=16)

        tk.Label(toolbar, text='y=keep  n=disc  1-8=override  r=reset  z=undo  a/d=nav  s=save',
                 bg='#2d2d2d', fg='#555', font=('Helvetica',9)).pack(side=tk.RIGHT, padx=10)

        self._status_var = tk.StringVar(value='Loading...')
        tk.Label(self.root, textvariable=self._status_var, bg='#1e1e1e', fg='#aaa',
                 anchor='w', font=('Helvetica',10)).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=3)

        main = tk.Frame(self.root, bg='#1e1e1e')
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, bg='#111', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6,3), pady=6)
        self.canvas.bind('<Configure>', lambda _: self._display_frame())

        lf = tk.Frame(main, bg='#2d2d2d', width=300)
        lf.pack(side=tk.RIGHT, fill=tk.Y, padx=(3,6), pady=6)
        lf.pack_propagate(False)

        tk.Label(lf, text='Images', bg='#2d2d2d', fg='white',
                 font=('Helvetica',12,'bold'), anchor='w').pack(fill=tk.X, padx=8, pady=(8,2))
        self._stats_var = tk.StringVar(value='')
        tk.Label(lf, textvariable=self._stats_var, bg='#2d2d2d', fg='#888',
                 font=('Courier',9), anchor='w').pack(fill=tk.X, padx=8)

        self._search_var = tk.StringVar()
        self._search_var.trace_add('write', lambda *_: self._filter_list())
        tk.Entry(lf, textvariable=self._search_var, bg='#3a3a3a', fg='white',
                 insertbackground='white', relief=tk.FLAT,
                 font=('Courier',9)).pack(fill=tk.X, padx=8, pady=4)

        sb = tk.Scrollbar(lf, orient=tk.VERTICAL, bg='#3a3a3a',
                          troughcolor='#2d2d2d', activebackground='#555')
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(lf, bg='#2d2d2d', fg='#ccc', font=('Courier',9),
                                  selectbackground='#3a3a3a', selectforeground='#00dc64',
                                  activestyle='none', relief=tk.FLAT, bd=0,
                                  yscrollcommand=sb.set, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,0))
        sb.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self._on_list_select)

        self._list_indices = []
        self._populate_list()
        self.root.bind('<KeyPress>', self._on_key)
        self.root.focus_set()

    def _populate_list(self, fs=''):
        self.listbox.delete(0, tk.END)
        self._list_indices = []
        fsl = fs.lower()
        for i, path in enumerate(self.image_paths):
            name = os.path.basename(path)
            if fsl and fsl not in name.lower(): continue
            self._list_indices.append(i)
            entry = self.labels.get(path)
            if entry is None:
                self.listbox.insert(tk.END, f'  {name}')
                self.listbox.itemconfig(tk.END, fg='#888')
            elif entry.get('discarded'):
                self.listbox.insert(tk.END, f'✘ {name}')
                self.listbox.itemconfig(tk.END, fg='#ff4444', bg='#2a1010')
            else:
                star = '*' if entry.get('overridden') else ' '
                self.listbox.insert(tk.END, f'✔{star}[{entry["label"]}] {name}')
                self.listbox.itemconfig(tk.END, fg='#00dc64', bg='#0a2010')
        kept   = sum(1 for e in self.labels.values() if not e.get('discarded'))
        disc   = sum(1 for e in self.labels.values() if e.get('discarded'))
        unseen = len(self.image_paths) - len(self.labels)
        self._stats_var.set(f'kept={kept}  disc={disc}  unseen={unseen}')

    def _filter_list(self):
        self._populate_list(self._search_var.get())
        self._scroll_to_current()

    def _scroll_to_current(self):
        try:
            lb = self._list_indices.index(self.idx)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(lb)
            self.listbox.see(lb)
        except ValueError:
            pass

    def _on_list_select(self, event):
        sel = self.listbox.curselection()
        if not sel: return
        li = sel[0]
        if li < len(self._list_indices):
            self.idx = self._list_indices[li]
            self.override = None
            self._render()

    def _render(self):
        try:
            path = self.image_paths[self.idx]
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]

            img = cv2.imread(path)
            gm  = load_mask(GROUND_MASK_DIR, stem, 'ground')
            pm  = load_mask(POLE_MASK_DIR,   stem, 'pole')
            tm  = load_mask(TREE_MASK_DIR,   stem, 'tree')

            frame_bgr, pd, auto_lbl = build_frame(img, gm, pm, tm, self._block_frac)
            self._frame_bgr = frame_bgr
            self._auto_lbl  = auto_lbl

            self._display_frame()

            label = self.override if self.override is not None else auto_lbl
            entry = self.labels.get(path)

            if self.override is not None:
                self._lbl_var.set(f'Label: {label}  (override, auto={auto_lbl})')
            elif entry and not entry.get('discarded'):
                star = '  ★ corrected' if entry.get('overridden') else '  auto'
                self._lbl_var.set(f'Label: {entry["label"]}{star}')
            else:
                self._lbl_var.set(f'Label: {auto_lbl}  (auto, unsaved)')

            for i, b in enumerate(self._strip_btns):
                b.config(bg='#1a5c2e' if i+1==label else '#444')

            disc_tag = '  [DISCARDED]' if entry and entry.get('discarded') else ''
            kept = sum(1 for e in self.labels.values() if not e.get('discarded'))
            self._status_var.set(
                f'[{self.idx+1}/{len(self.image_paths)}]  {name}{disc_tag}  | kept={kept}  | block_thresh={self._block_frac:.2f} ([/])')

            self._populate_list(self._search_var.get())
            self._scroll_to_current()

        except Exception as e:
            print(f'[render error] {e}')
            import traceback; traceback.print_exc()

    def _display_frame(self):
        if self._frame_bgr is None: return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return
        fh, fw = self._frame_bgr.shape[:2]
        scale = min(cw/fw, ch/fh)
        nw = max(1, int(fw*scale))
        nh = max(1, int(fh*scale))
        scaled = cv2.resize(self._frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete('all')
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self._tk_img)

    def _keep(self):
        path = self.image_paths[self.idx]
        name = os.path.basename(path)
        self.history.append((path, self.labels.get(path)))
        label = self.override if self.override is not None else self._auto_lbl
        self.labels[path] = {'path':path,'name':name,'label':label,
                              'overridden':self.override is not None,'discarded':False}
        print(f'[KEPT] {name}  label={label}')
        self.override = None; self._next()

    def _discard(self):
        path = self.image_paths[self.idx]
        name = os.path.basename(path)
        self.history.append((path, self.labels.get(path)))
        self.labels[path] = {'path':path,'name':name,'label':None,'discarded':True}
        print(f'[DISCARDED] {name}')
        self.override = None; self._next()

    def _undo(self):
        if not self.history: print('[UNDO] nothing to undo'); return
        path, old = self.history.pop()
        if old is None: self.labels.pop(path, None)
        else: self.labels[path] = old
        self.idx = self.image_paths.index(path)
        self.override = None; self._render()
        print(f'[UNDO] reverted {os.path.basename(path)}')

    def _prev(self):
        self.override=None; self.idx=(self.idx-1)%len(self.image_paths); self._render()

    def _next(self):
        self.override=None; self.idx=(self.idx+1)%len(self.image_paths); self._render()

    def _set_override(self, n):
        self.override=n; self._render()

    def _reset_override(self):
        self.override=None; self._render()

    def _save(self):
        entries=[e for e in self.labels.values() if not e.get('discarded')]
        with open(OUTPUT_JSON,'w', encoding='utf-8') as f:
            json.dump(sorted(entries,key=lambda e:e['path']),f,indent=2)
        kept=len(entries); disc=sum(1 for e in self.labels.values() if e.get('discarded'))
        print(f'[SAVED] {kept} kept, {disc} discarded')
        messagebox.showinfo('Saved',f'Saved {kept} labels to:\n{OUTPUT_JSON}')

    def _on_key(self, event):
        k=event.keysym
        if   k in ('d','Right'):  self._next()
        elif k in ('a','Left'):   self._prev()
        elif k in ('y','Return'): self._keep()
        elif k in ('n','Delete'): self._discard()
        elif k=='z':              self._undo()
        elif k=='r':              self._reset_override()
        elif k=='s':              self._save()
        elif k=='bracketleft':
            self._block_frac = max(0.01, round(self._block_frac - BLOCK_FRAC_STEP, 2))
            self._render()
        elif k=='bracketright':
            self._block_frac = min(0.99, round(self._block_frac + BLOCK_FRAC_STEP, 2))
            self._render()
        elif k in ('1','2','3','4','5','6'): self._set_override(int(k))


def main():
    root = tk.Tk()
    CuratorApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()