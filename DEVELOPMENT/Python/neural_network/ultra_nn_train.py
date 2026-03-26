"""
train_nn.py
===========
Trains a two-layer neural network to predict drone direction.

Supports multiple label formats via LABEL_MODE:
  '3strip'  — labels 1=left, 2=centre, 3=right, 4=turn  (labels_3strip.json)
  '5strip'  — labels 1-5=strips, 6=turn                 (labels.json)
  '5merged' — same JSON as 5strip but merges 1+2→1, 3→2, 4+5→3, 6→4

Architecture
------------
  Input  : 64 x 48 x 3 = 9216 features (YUV, normalised 0-1)
  Hidden : HIDDEN_SIZE units, ReLU
  Output : N classes, softmax

Inference snippet (copy to drone)
----------------------------------
    import numpy as np, cv2
    W1=np.load('model_W1.npy'); b1=np.load('model_b1.npy')
    W2=np.load('model_W2.npy'); b2=np.load('model_b2.npy')
    classes=np.load('model_classes.npy')

    def predict(img_bgr):
        yuv = cv2.cvtColor(cv2.resize(img_bgr,(64,48)),cv2.COLOR_BGR2YUV)
        x   = yuv.astype(np.float32).flatten() / 255.0
        h   = np.maximum(0, W1 @ x + b1)
        return int(classes[np.argmax(W2 @ h + b2)])
"""

import os, json, time
import numpy as np
import cv2
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# ── Label mode ────────────────────────────────────────────────────────────────
# '3strip'  : 3-strip labels (1=left, 2=centre, 3=right, 4=turn)
# '5strip'  : 5-strip labels (1-5=strips, 6=turn)
# '5merged' : 5-strip JSON but merged to 4 classes (1+2→1, 3→2, 4+5→3, 6→4)
LABEL_MODE = '3strip'

LABELS_JSON_3STRIP = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\labels_3strip.json'
LABELS_JSON_5STRIP = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\labels.json'

OUTPUT_DIR = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\model'

# ── Split ─────────────────────────────────────────────────────────────────────
TRAIN_FRACTION = 0.70   # rest split 50/50 val/test

# ── Architecture ──────────────────────────────────────────────────────────────
IMG_W        = 64
IMG_H        = 48
HIDDEN_SIZE  = 64

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 300
BATCH_SIZE    = 32
LEARNING_RATE = 0.003
L2_REG        = 2e-3
RANDOM_SEED   = 42
PATIENCE      = 60

# ── Class balancing ───────────────────────────────────────────────────────────
# Cap the number of samples per class before training.
# Set to None to use all samples.
# e.g. MAX_SAMPLES_PER_CLASS = 150 caps majority classes to match minorities
MAX_SAMPLES_PER_CLASS = None
AUGMENT              = True
AUG_FLIP             = True
AUG_BRIGHTNESS       = True
AUG_NOISE            = True
AUG_NOISE_STD        = 0.02
AUG_BRIGHTNESS_RANGE = 0.25

# ══════════════════════════════════════════════════════════════════════════════
# LABEL MODE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

MODE_CONFIGS = {
    '3strip': {
        'json':      LABELS_JSON_3STRIP,
        'merge_map': None,
        'flip_map':  {1:3, 2:2, 3:1, 4:4},
        'flip_skip': set(),
        'class_names': {1:'LEFT', 2:'CENTRE', 3:'RIGHT', 4:'TURN'},
    },
    '5strip': {
        'json':      LABELS_JSON_5STRIP,
        'merge_map': None,
        'flip_map':  {1:5, 2:4, 3:3, 4:2, 5:1, 6:6},
        'flip_skip': {1, 5},
        'class_names': {1:'S1', 2:'S2', 3:'S3', 4:'S4', 5:'S5', 6:'TURN'},
    },
    '5merged': {
        'json':      LABELS_JSON_5STRIP,
        'merge_map': {1:1, 2:1, 3:2, 4:3, 5:3, 6:4},
        'flip_map':  {1:3, 2:2, 3:1, 4:4},
        'flip_skip': set(),
        'class_names': {1:'LEFT', 2:'CENTRE', 3:'RIGHT', 4:'TURN'},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return yuv.astype(np.float32).flatten() / 255.0

def flip_vec(vec):
    return np.fliplr(vec.reshape(IMG_H, IMG_W, 3)).flatten()

def augment_sample(vec, label, rng):
    img = vec.reshape(IMG_H, IMG_W, 3).copy()
    if AUG_BRIGHTNESS and rng.random() > 0.5:
        delta = rng.uniform(-AUG_BRIGHTNESS_RANGE, AUG_BRIGHTNESS_RANGE)
        img[:, :, 0] = np.clip(img[:, :, 0] + delta, 0.0, 1.0)
    if AUG_NOISE and rng.random() > 0.5:
        noise = rng.normal(0, AUG_NOISE_STD, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
    return img.flatten(), label

def build_augmented_dataset(X, y_raw, rng, flip_map, flip_skip):
    X_out = [X]; y_out = [y_raw]

    if AUG_FLIP:
        X_flip, y_flip = [], []
        for vec, lbl in zip(X, y_raw):
            if lbl in flip_skip or lbl not in flip_map: continue
            X_flip.append(flip_vec(vec))
            y_flip.append(flip_map[lbl])
        if X_flip:
            X_out.append(np.array(X_flip))
            y_out.append(np.array(y_flip))
            print(f'  +{len(X_flip)} flipped (skipped {flip_skip})')

    if AUG_BRIGHTNESS or AUG_NOISE:
        X_aug, y_aug = [], []
        for vec, lbl in zip(X, y_raw):
            v2, l2 = augment_sample(vec, lbl, rng)
            X_aug.append(v2); y_aug.append(l2)
        X_out.append(np.array(X_aug))
        y_out.append(np.array(y_aug))
        print(f'  +{len(X_aug)} brightness/noise augmented')

    return np.vstack(X_out).astype(np.float32), np.concatenate(y_out)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def relu(x):      return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(np.float32)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    h_pre = X @ W1.T + b1
    h     = relu(h_pre)
    return softmax(h @ W2.T + b2), h_pre, h

def predict_batch(X, W1, b1, W2, b2):
    probs, _, _ = forward(X, W1, b1, W2, b2)
    return np.argmax(probs, axis=1)

def accuracy_score(X, y, W1, b1, W2, b2):
    return (predict_batch(X, W1, b1, W2, b2) == y).mean() * 100.0

def confusion_str(X, y, W1, b1, W2, b2, classes, class_names):
    preds = predict_batch(X, W1, b1, W2, b2)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, preds): cm[t, p] += 1
    col_w = 8
    names = [class_names.get(c, str(c)) for c in classes]
    header = ' ' * 10 + ''.join(f'{nm:>{col_w}}' for nm in names)
    lines  = [header, ' ' * 10 + '-' * col_w * n]
    for i, row in enumerate(cm):
        lines.append(f'{names[i]:>9} |' + ''.join(f'{v:{col_w}d}' for v in row))
    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    cfg = MODE_CONFIGS[LABEL_MODE]
    print(f'Mode: {LABEL_MODE}')
    print(f'Loading {cfg["json"]} ...')

    with open(cfg['json']) as f:
        entries = json.load(f)
    print(f'  {len(entries)} labelled entries.')

    # ── Preprocess ────────────────────────────────────────────────────────────
    n_feat = IMG_W * IMG_H * 3
    print(f'Preprocessing -> {IMG_W}x{IMG_H}x3 YUV = {n_feat} features ...')
    X_list, y_list = [], []
    skipped = 0; t0 = time.time()
    for i, e in enumerate(entries):
        if i % 200 == 0: print(f'  {i}/{len(entries)}', end='\r')
        vec = load_and_preprocess(e['path'])
        if vec is None: skipped += 1; continue
        X_list.append(vec); y_list.append(int(e['label']))
    print(f'  Done in {time.time()-t0:.1f}s  ({skipped} skipped)        ')

    X     = np.stack(X_list).astype(np.float32)
    y_raw = np.array(y_list)

    # ── Label merge (if applicable) ───────────────────────────────────────────
    if cfg['merge_map']:
        y_raw = np.array([cfg['merge_map'][c] for c in y_raw])
        print(f'\n[MERGE] Applied: {cfg["merge_map"]}')

    classes     = np.array(sorted(set(y_raw.tolist())))
    n_cls       = len(classes)
    label2idx   = {c: i for i, c in enumerate(classes)}
    class_names = cfg['class_names']

    # ── Per-class cap (undersampling) ─────────────────────────────────────────
    if MAX_SAMPLES_PER_CLASS is not None:
        keep_idx = []
        for c in classes:
            idx_c = np.where(y_raw == c)[0]
            if len(idx_c) > MAX_SAMPLES_PER_CLASS:
                idx_c = rng.choice(idx_c, MAX_SAMPLES_PER_CLASS, replace=False)
            keep_idx.append(idx_c)
        keep_idx = np.concatenate(keep_idx)
        keep_idx = rng.permutation(keep_idx)  # shuffle
        X     = X[keep_idx]
        y_raw = y_raw[keep_idx]
        print(f'\n[CAP] Capped to {MAX_SAMPLES_PER_CLASS} per class → {len(X)} total samples')

    print(f'\nDataset: {len(X)} samples  |  {n_feat} features  |  {n_cls} classes')

    # ── Class distribution ────────────────────────────────────────────────────
    counts = Counter(y_raw.tolist())
    print('\nClass distribution:')
    max_count = max(counts.values())
    for c in sorted(counts):
        bar = 'X' * (counts[c] // max(1, max_count // 30))
        pct = 100 * counts[c] / len(X)
        name = class_names.get(c, str(c))
        print(f'  {name:8s} ({c}): {counts[c]:4d}  ({pct:4.1f}%)  {bar}')

    imb = max_count / max(min(counts.values()), 1)
    print(f'\n  Imbalance ratio: {imb:.1f}x  -> balanced class weights')

    # ── Three-way split ───────────────────────────────────────────────────────
    idx     = rng.permutation(len(X))
    n_train = int(len(X) * TRAIN_FRACTION)
    n_val   = (len(X) - n_train) // 2
    n_test  = len(X) - n_train - n_val

    X_trn, y_trn = X[idx[:n_train]],               y_raw[idx[:n_train]]
    X_val, y_val = X[idx[n_train:n_train+n_val]],   y_raw[idx[n_train:n_train+n_val]]
    X_tst, y_tst = X[idx[n_train+n_val:]],          y_raw[idx[n_train+n_val:]]

    print(f'\nBase split ({TRAIN_FRACTION*100:.0f}%): '
          f'{len(X_trn)} train / {n_val} val / {n_test} test')

    # ── Augment training only ─────────────────────────────────────────────────
    if AUGMENT:
        print(f'\nAugmenting ({len(X_trn)} -> ', end='')
        X_trn, y_trn = build_augmented_dataset(
            X_trn, y_trn, rng, cfg['flip_map'], cfg['flip_skip'])
        shuf = rng.permutation(len(X_trn))
        X_trn, y_trn = X_trn[shuf], y_trn[shuf]
        print(f'{len(X_trn)} samples)')

    # Remap to indices
    y_trn = np.array([label2idx[c] for c in y_trn])
    y_val = np.array([label2idx[c] for c in y_val])
    y_tst = np.array([label2idx[c] for c in y_tst])

    aug_counts = Counter(y_trn.tolist())
    cw = np.array([len(y_trn) / (n_cls * max(1, aug_counts.get(i, 1)))
                   for i in range(n_cls)], dtype=np.float32)
    cw /= cw.mean()

    print(f'\nAfter augmentation: train={len(X_trn)}  val={len(X_val)}  test={len(X_tst)}')

    # ── Initialise weights (He) ───────────────────────────────────────────────
    W1 = (rng.standard_normal((HIDDEN_SIZE, n_feat)) * np.sqrt(2.0/n_feat)).astype(np.float32)
    b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    W2 = (rng.standard_normal((n_cls, HIDDEN_SIZE)) * np.sqrt(2.0/HIDDEN_SIZE)).astype(np.float32)
    b2 = np.zeros(n_cls, dtype=np.float32)

    params = HIDDEN_SIZE*n_feat + HIDDEN_SIZE + n_cls*HIDDEN_SIZE + n_cls
    print(f'\nArchitecture: {n_feat} -> {HIDDEN_SIZE} (ReLU) -> {n_cls} (softmax)')
    print(f'Parameters  : {params:,}  ({params*4/1024:.1f} KB)')

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f'\nTraining  epochs={EPOCHS}  lr={LEARNING_RATE}  l2={L2_REG}  patience={PATIENCE}')
    print('-' * 60)

    best_val = 0.0
    bW1,bb1,bW2,bb2 = W1.copy(),b1.copy(),W2.copy(),b2.copy()
    no_improve = 0; t1 = time.time()

    for epoch in range(1, EPOCHS+1):
        idx_e = rng.permutation(len(X_trn))
        for start in range(0, len(X_trn), BATCH_SIZE):
            mb  = idx_e[start:start+BATCH_SIZE]
            Xb  = X_trn[mb]; yb = y_trn[mb]; sw = cw[yb]
            probs, h_pre, h = forward(Xb, W1, b1, W2, b2)
            oh = np.zeros((len(mb), n_cls), dtype=np.float32)
            oh[np.arange(len(mb)), yb] = 1.0
            dL  = (probs - oh) * sw[:, None] / len(mb)
            dW2 = dL.T @ h + L2_REG * W2;  db2 = dL.sum(0)
            dh  = dL @ W2
            dp  = dh * relu_grad(h_pre)
            dW1 = dp.T @ Xb + L2_REG * W1; db1 = dp.sum(0)
            W1 -= LEARNING_RATE*dW1; b1 -= LEARNING_RATE*db1
            W2 -= LEARNING_RATE*dW2; b2 -= LEARNING_RATE*db2

        if epoch % 10 == 0 or epoch == 1:
            ta = accuracy_score(X_trn, y_trn, W1, b1, W2, b2)
            va = accuracy_score(X_val, y_val, W1, b1, W2, b2)
            t  = time.time()-t1
            is_best = va > best_val
            print(f'  epoch {epoch:4d}/{EPOCHS}  train={ta:5.1f}%  val={va:5.1f}%  t={t:.1f}s'
                  + ('  <- best' if is_best else ''))
            if is_best:
                best_val = va
                bW1,bb1,bW2,bb2 = W1.copy(),b1.copy(),W2.copy(),b2.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE // 10:
                    print(f'  Early stopping at epoch {epoch}')
                    break

    W1,b1,W2,b2 = bW1,bb1,bW2,bb2
    print(f'\nBest val: {best_val:.1f}%')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    ta = accuracy_score(X_trn, y_trn, W1, b1, W2, b2)
    va = accuracy_score(X_val, y_val, W1, b1, W2, b2)
    te = accuracy_score(X_tst, y_tst, W1, b1, W2, b2)
    print(f'\n{"-"*40}')
    print(f'  Train : {ta:.1f}%')
    print(f'  Val   : {va:.1f}%')
    print(f'  TEST  : {te:.1f}%  <- held-out')
    print(f'{"-"*40}')

    cm_str = confusion_str(X_tst, y_tst, W1, b1, W2, b2, classes, class_names)
    print('\nConfusion matrix (TEST):')
    print(cm_str)

    print('\nPer-class TEST accuracy:')
    preds_tst = predict_batch(X_tst, W1, b1, W2, b2)
    for ci, c in enumerate(classes):
        mask = y_tst == ci
        if mask.sum() == 0: continue
        acc = (preds_tst[mask] == ci).mean() * 100
        print(f'  {class_names.get(c,str(c)):8s} ({c}): {acc:5.1f}%  ({mask.sum()} samples)')

    # ── Save model ────────────────────────────────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, 'model_W1.npy'),      W1)
    np.save(os.path.join(OUTPUT_DIR, 'model_b1.npy'),      b1)
    np.save(os.path.join(OUTPUT_DIR, 'model_W2.npy'),      W2)
    np.save(os.path.join(OUTPUT_DIR, 'model_b2.npy'),      b2)
    np.save(os.path.join(OUTPUT_DIR, 'model_classes.npy'), classes)

    total_kb = (W1.nbytes+b1.nbytes+W2.nbytes+b2.nbytes)/1024
    print(f'\nModel saved -> {OUTPUT_DIR}  ({total_kb:.1f} KB)')

    # ── Report ────────────────────────────────────────────────────────────────
    report = os.path.join(OUTPUT_DIR, 'training_report.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write(f'Training Report\n{"="*50}\n\n')
        f.write(f'Mode           : {LABEL_MODE}\n')
        f.write(f'Labels JSON    : {cfg["json"]}\n')
        f.write(f'Architecture   : {n_feat} -> {HIDDEN_SIZE} (ReLU) -> {n_cls}\n')
        f.write(f'Parameters     : {params:,}\n\n')
        f.write(f'Split ({TRAIN_FRACTION*100:.0f}% train): '
                f'{len(X_trn)} / {len(X_val)} / {len(X_tst)}\n\n')
        f.write('Class distribution:\n')
        for c in sorted(counts):
            f.write(f'  {class_names.get(c,str(c))}: {counts[c]}\n')
        f.write(f'\nTrain : {ta:.1f}%\nVal   : {va:.1f}%\nTEST  : {te:.1f}%\n\n')
        f.write('Confusion matrix (TEST):\n')
        f.write(cm_str + '\n')
    print(f'Report -> {report}')

    print(f"""
Inference snippet:
--------------------------------------------------
import numpy as np, cv2
W1=np.load('model_W1.npy'); b1=np.load('model_b1.npy')
W2=np.load('model_W2.npy'); b2=np.load('model_b2.npy')
classes=np.load('model_classes.npy')
# classes = {classes.tolist()}  ({LABEL_MODE})

def predict(img_bgr):
    yuv = cv2.cvtColor(cv2.resize(img_bgr,({IMG_W},{IMG_H})),cv2.COLOR_BGR2YUV)
    x   = yuv.astype(np.float32).flatten() / 255.0
    h   = np.maximum(0, W1 @ x + b1)
    return int(classes[np.argmax(W2 @ h + b2)])
--------------------------------------------------
""")


if __name__ == '__main__':
    main()