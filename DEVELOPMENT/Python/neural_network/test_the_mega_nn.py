"""
test_nn.py
==========
Visually test the trained neural network on labelled images.

Shows each image with:
  - The neural network's prediction
  - The ground truth label
  - A confidence bar for each class
  - GREEN border = correct, RED border = wrong

Controls
--------
  d / Right  — next image
  a / Left   — previous image
  r          — random image
  w          — next WRONG prediction only
  q          — quit
"""

import numpy as np
import cv2
import os
import json
import sys

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR  = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\model'
LABELS_JSON = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\labels_25_03_attempt.json'

IMG_W = 64
IMG_H = 48

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    W1      = np.load(os.path.join(MODEL_DIR, 'model_W1.npy')).astype(np.float32)
    b1      = np.load(os.path.join(MODEL_DIR, 'model_b1.npy')).astype(np.float32)
    W2      = np.load(os.path.join(MODEL_DIR, 'model_W2.npy')).astype(np.float32)
    b2      = np.load(os.path.join(MODEL_DIR, 'model_b2.npy')).astype(np.float32)
    classes = np.load(os.path.join(MODEL_DIR, 'model_classes.npy'))
    return W1, b1, W2, b2, classes


def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return yuv.astype(np.float32).flatten() / 255.0


def predict_proba(x, W1, b1, W2, b2):
    """Returns softmax probabilities for each class."""
    h      = np.maximum(0, W1 @ x + b1)
    logits = W2 @ h + b2
    e      = np.exp(logits - logits.max())
    return e / e.sum()


def predict(img_bgr, W1, b1, W2, b2, classes):
    x     = preprocess(img_bgr)
    probs = predict_proba(x, W1, b1, W2, b2)
    idx   = np.argmax(probs)
    return int(classes[idx]), probs

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_display(img, pred, true_lbl, probs, classes, img_idx, total):
    """Build a display frame with the image, prediction, and confidence bars."""
    H, W = img.shape[:2]
    correct = (pred == true_lbl)

    # Scale image to fixed display height
    disp_h = 400
    scale  = disp_h / H
    disp_w = int(W * scale)
    disp   = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    # Border: green = correct, red = wrong
    border_col = (0, 220, 80) if correct else (0, 0, 255)
    bw = 6
    disp = cv2.copyMakeBorder(disp, bw, bw, bw, bw,
                               cv2.BORDER_CONSTANT, value=border_col)

    # Sidebar for confidence bars
    sidebar_w = 320
    sidebar   = np.full((disp_h + 2*bw, sidebar_w, 3), (30, 30, 30), dtype=np.uint8)

    # Title
    result_txt = 'CORRECT' if correct else 'WRONG'
    result_col = (0, 220, 80) if correct else (0, 0, 255)
    cv2.putText(sidebar, result_txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_col, 2, cv2.LINE_AA)

    cv2.putText(sidebar, f'Pred : Strip {pred}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(sidebar, f'True : Strip {true_lbl}', (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(sidebar, f'[{img_idx+1}/{total}]', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    # Confidence bars
    bar_x0  = 60
    bar_maxw = sidebar_w - bar_x0 - 20
    bar_h   = 22
    bar_gap = 8
    top     = 140

    for i, (cls, prob) in enumerate(zip(classes, probs)):
        y0  = top + i * (bar_h + bar_gap)
        y1  = y0 + bar_h
        bw_ = int(prob * bar_maxw)

        # Bar colour: gold for predicted, green for true, grey otherwise
        if cls == pred and cls == true_lbl:
            col = (0, 220, 80)     # both — bright green
        elif cls == pred:
            col = (0, 180, 255)    # predicted only — orange
        elif cls == true_lbl:
            col = (0, 220, 80)     # true only — green
        else:
            col = (80, 80, 80)

        # Background
        cv2.rectangle(sidebar, (bar_x0, y0), (bar_x0 + bar_maxw, y1),
                      (50, 50, 50), -1)
        # Fill
        if bw_ > 0:
            cv2.rectangle(sidebar, (bar_x0, y0), (bar_x0 + bw_, y1), col, -1)

        # Label
        lbl = '↺ 8' if cls == 8 else f'S{cls}'
        cv2.putText(sidebar, lbl, (8, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Percentage
        pct_txt = f'{prob*100:.0f}%'
        cv2.putText(sidebar, pct_txt, (bar_x0 + bw_ + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Controls hint
    cv2.putText(sidebar, 'a/d=prev/next  r=random',
                (8, disp_h + 2*bw - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
    cv2.putText(sidebar, 'w=next wrong  q=quit',
                (8, disp_h + 2*bw - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

    return np.hstack([disp, sidebar])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Loading model...')
    W1, b1, W2, b2, classes = load_model()
    print(f'  Classes: {classes.tolist()}')

    print(f'Loading labels from {LABELS_JSON}...')
    with open(LABELS_JSON) as f:
        entries = json.load(f)
    print(f'  {len(entries)} labelled images')

    # Pre-run all predictions so we know which are wrong
    print('Running predictions on all images...')
    results = []
    for i, e in enumerate(entries):
        if i % 200 == 0: print(f'  {i}/{len(entries)}', end='\r')
        img = cv2.imread(e['path'])
        if img is None: continue
        pred, probs = predict(img, W1, b1, W2, b2, classes)
        results.append({
            'path':     e['path'],
            'true':     int(e['label']),
            'pred':     pred,
            'probs':    probs,
            'correct':  pred == int(e['label']),
        })

    total   = len(results)
    correct = sum(r['correct'] for r in results)
    print(f'\nOverall accuracy: {correct}/{total} = {100*correct/total:.1f}%')
    print()

    # Per-class accuracy
    from collections import defaultdict
    cls_correct = defaultdict(int)
    cls_total   = defaultdict(int)
    for r in results:
        cls_total[r['true']]   += 1
        cls_correct[r['true']] += int(r['correct'])
    print('Per-class accuracy:')
    for c in sorted(cls_total):
        n = cls_total[c]; k = cls_correct[c]
        bar = '█' * int(30 * k / max(n, 1))
        lbl = 'TURN' if c == 8 else f'Strip {c}'
        print(f'  {lbl:8s}: {k:3d}/{n:3d}  {100*k/n:5.1f}%  {bar}')

    wrong_indices = [i for i, r in enumerate(results) if not r['correct']]
    print(f'\n{len(wrong_indices)} wrong predictions')
    print('Controls: a/d=prev/next  r=random  w=next wrong  q=quit')

    WINDOW = 'NN Tester  |  a/d=nav  r=random  w=next-wrong  q=quit'
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    idx        = 0
    wrong_ptr  = 0
    rng        = np.random.default_rng(42)
    needs_draw = True

    while True:
        if needs_draw:
            r   = results[idx]
            img = cv2.imread(r['path'])
            if img is None:
                idx = (idx + 1) % total
                continue

            frame = build_display(img, r['pred'], r['true'], r['probs'],
                                  classes, idx, total)
            cv2.setWindowTitle(WINDOW,
                f'[{idx+1}/{total}]  {os.path.basename(r["path"])}  '
                f'pred={r["pred"]}  true={r["true"]}  '
                f'{"✓" if r["correct"] else "✗"}')
            cv2.imshow(WINDOW, frame)
            needs_draw = False

        key = cv2.waitKey(15)
        if key < 0: continue

        if key == ord('q'):
            break
        elif key in (ord('d'), 83, 65363):   # next
            idx = (idx + 1) % total
            needs_draw = True
        elif key in (ord('a'), 81, 65361):   # prev
            idx = (idx - 1) % total
            needs_draw = True
        elif key == ord('r'):                 # random
            idx = int(rng.integers(0, total))
            needs_draw = True
        elif key == ord('w'):                 # next wrong
            if wrong_indices:
                wrong_ptr = (wrong_ptr + 1) % len(wrong_indices)
                idx       = wrong_indices[wrong_ptr]
                needs_draw = True
            else:
                print('No wrong predictions!')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()