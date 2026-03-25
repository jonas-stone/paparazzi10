"""
train_nn.py
===========
Trains a two-layer neural network (input → hidden ReLU → softmax output)
to predict the drone's direction (1-7 = strip, 8 = turn around) from a
downscaled 64x48 YUV image (all 3 channels).

Architecture (Option 5)
-----------------------
  Input  : 64 x 48 x 3 = 9216 features  (YUV, normalised 0-1)
  Hidden : 64 units, ReLU activation
  Output : 8 units, softmax

Split
-----
  TRAIN_FRACTION  — fraction of labelled data used for training
  The remaining fraction is split equally into validation and test sets.
  e.g. TRAIN_FRACTION=0.70 → 70% train, 15% val, 15% test

Output files
------------
  model_W1.npy      — hidden layer weights  (64, 9216)
  model_b1.npy      — hidden layer biases   (64,)
  model_W2.npy      — output layer weights  (8, 64)
  model_b2.npy      — output layer biases   (8,)
  model_classes.npy — class labels          [1,2,...,8]
  training_report.txt

Inference on drone (5 lines of numpy)
--------------------------------------
    import numpy as np, cv2
    W1=np.load('model_W1.npy'); b1=np.load('model_b1.npy')
    W2=np.load('model_W2.npy'); b2=np.load('model_b2.npy')
    classes=np.load('model_classes.npy')

    def predict(img_bgr):
        yuv = cv2.cvtColor(cv2.resize(img_bgr,(64,48)),cv2.COLOR_BGR2YUV)
        x   = yuv.astype(np.float32).flatten() / 255.0
        h   = np.maximum(0, W1 @ x + b1)      # ReLU hidden
        return int(classes[np.argmax(W2 @ h + b2)])
"""

import os, sys, json, time
import numpy as np
import cv2
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these
# ══════════════════════════════════════════════════════════════════════════════

LABELS_JSON    = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\labels_25_03_attempt.json'
OUTPUT_DIR     = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\model'

# ── Split ─────────────────────────────────────────────────────────────────────
# TRAIN_FRACTION = fraction of data used for training.
# Remainder is split 50/50 into validation (used during training) and
# held-out test set (final unbiased evaluation).
# e.g. 0.70 → 70% train | 15% val | 15% test
#      0.80 → 80% train | 10% val | 10% test
TRAIN_FRACTION = 0.70

# ── Architecture ──────────────────────────────────────────────────────────────
IMG_W          = 64       # resize width
IMG_H          = 48       # resize height
HIDDEN_SIZE    = 64       # number of hidden units

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS         = 300
BATCH_SIZE     = 32
LEARNING_RATE  = 0.003
L2_REG         = 2e-3     # stronger L2 on W1 to reduce overfitting
RANDOM_SEED    = 42

# ── Augmentation ──────────────────────────────────────────────────────────────
AUGMENT        = True
AUG_FLIP       = True     # horizontal flip with mirrored label (1↔7, 2↔6, 3↔5)
AUG_FLIP_SKIP  = {1, 7}   # skip flip for these classes (too few / ambiguous)
AUG_BRIGHTNESS = True
AUG_NOISE      = True
AUG_NOISE_STD  = 0.02
AUG_BRIGHTNESS_RANGE = 0.25

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(img_path):
    """Load → resize to IMG_W×IMG_H → YUV all channels → flat float32 [0,1]."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return yuv.astype(np.float32).flatten() / 255.0   # shape: (9216,)


# Label mirror map for horizontal flip: strip 1↔7, 2↔6, 3↔5, 4→4, 8→8
FLIP_LABEL = {1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1, 8:8}

def flip_vec(vec):
    """Horizontally flip a flattened YUV image vector."""
    img = vec.reshape(IMG_H, IMG_W, 3)
    return np.fliplr(img).flatten()

def augment_sample(vec, label, rng):
    """Apply random augmentations to one sample. Returns (new_vec, new_label)."""
    img = vec.reshape(IMG_H, IMG_W, 3).copy()

    # Brightness/contrast jitter on Y channel only
    if AUG_BRIGHTNESS and rng.random() > 0.5:
        delta = rng.uniform(-AUG_BRIGHTNESS_RANGE, AUG_BRIGHTNESS_RANGE)
        img[:, :, 0] = np.clip(img[:, :, 0] + delta, 0.0, 1.0)

    # Gaussian noise on all channels
    if AUG_NOISE and rng.random() > 0.5:
        noise = rng.normal(0, AUG_NOISE_STD, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return img.flatten(), label


def build_augmented_dataset(X, y_raw, rng):
    """
    Build augmented dataset from base samples.
    - Always includes all original samples
    - Adds horizontally flipped copies with mirrored labels
    - Adds brightness/noise augmented copies
    Returns (X_aug, y_aug) as numpy arrays.
    """
    X_out  = [X]
    y_out  = [y_raw]

    if AUG_FLIP:
        X_flip, y_flip = [], []
        for vec, lbl in zip(X, y_raw):
            if lbl in AUG_FLIP_SKIP:
                continue
            X_flip.append(flip_vec(vec))
            y_flip.append(FLIP_LABEL[lbl])
        X_flip = np.array(X_flip); y_flip = np.array(y_flip)
        X_out.append(X_flip); y_out.append(y_flip)
        print(f'  +{len(X_flip)} flipped samples (skipped classes {AUG_FLIP_SKIP})')

    if AUG_BRIGHTNESS or AUG_NOISE:
        X_aug_list, y_aug_list = [], []
        for vec, lbl in zip(X, y_raw):
            v2, l2 = augment_sample(vec, lbl, rng)
            X_aug_list.append(v2)
            y_aug_list.append(l2)
        X_out.append(np.array(X_aug_list))
        y_out.append(np.array(y_aug_list))
        print(f'  +{len(X_aug_list)} brightness/noise augmented samples')

    X_combined = np.vstack(X_out).astype(np.float32)
    y_combined = np.concatenate(y_out)
    return X_combined, y_combined

# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (two-layer net, pure numpy)
# ══════════════════════════════════════════════════════════════════════════════

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    """Returns (probs, hidden_pre_relu, hidden_post_relu)."""
    h_pre  = X @ W1.T + b1          # (N, hidden)
    h      = relu(h_pre)             # (N, hidden)
    logits = h @ W2.T + b2           # (N, n_cls)
    probs  = softmax(logits)
    return probs, h_pre, h

def predict_batch(X, W1, b1, W2, b2):
    probs, _, _ = forward(X, W1, b1, W2, b2)
    return np.argmax(probs, axis=1)

def cross_entropy_loss(probs, y_oh, W1, W2, l2):
    ce   = -np.sum(y_oh * np.log(np.clip(probs, 1e-9, 1))) / len(probs)
    reg  = 0.5 * l2 * (np.sum(W1**2) + np.sum(W2**2))
    return ce + reg

def accuracy_score(X, y, W1, b1, W2, b2):
    preds = predict_batch(X, W1, b1, W2, b2)
    return (preds == y).mean() * 100.0

# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def confusion_str(X, y, W1, b1, W2, b2, classes):
    preds = predict_batch(X, W1, b1, W2, b2)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    header = '      ' + ''.join(f'{c:5d}' for c in classes)
    lines  = [header, '      ' + '─────'*n]
    for i, row in enumerate(cm):
        lines.append(f'  {classes[i]:2d} │' + ''.join(f'{v:5d}' for v in row))
    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Load labels ────────────────────────────────────────────────────────
    print(f'Loading {LABELS_JSON} ...')
    with open(LABELS_JSON) as f:
        entries = json.load(f)
    print(f'  {len(entries)} labelled entries.')

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    n_feat = IMG_W * IMG_H * 3
    print(f'Preprocessing → {IMG_W}×{IMG_H}×3 YUV = {n_feat} features ...')
    X_list, y_list = [], []
    skipped = 0
    t0 = time.time()
    for i, e in enumerate(entries):
        if i % 200 == 0: print(f'  {i}/{len(entries)}', end='\r')
        vec = load_and_preprocess(e['path'])
        if vec is None: skipped += 1; continue
        X_list.append(vec)
        y_list.append(int(e['label']))
    print(f'  Done in {time.time()-t0:.1f}s  ({skipped} skipped)        ')

    X     = np.stack(X_list).astype(np.float32)
    y_raw = np.array(y_list)
    classes   = np.array(sorted(set(y_raw.tolist())))
    n_cls     = len(classes)
    label2idx = {c: i for i, c in enumerate(classes)}

    print(f'\nDataset: {len(X)} samples  |  {n_feat} features  |  {n_cls} classes')

    # ── 3. Class distribution ─────────────────────────────────────────────────
    counts = Counter(y_raw.tolist())
    print('\nClass distribution:')
    for c in sorted(counts):
        bar = '█' * (counts[c] // max(1, max(counts.values()) // 30))
        pct = 100 * counts[c] / len(X)
        print(f'  Strip {c}: {counts[c]:4d}  ({pct:4.1f}%)  {bar}')

    imb = max(counts.values()) / max(min(counts.values()), 1)
    print(f'\n  Imbalance ratio: {imb:.1f}x  → using balanced class weights')

    # ── 4. Three-way split ────────────────────────────────────────────────────
    idx       = rng.permutation(len(X))
    n_train   = int(len(X) * TRAIN_FRACTION)
    n_remain  = len(X) - n_train
    n_val     = n_remain // 2
    n_test    = n_remain - n_val

    trn_idx   = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    tst_idx   = idx[n_train+n_val:]

    X_trn, y_trn = X[trn_idx], y_raw[trn_idx]
    X_val, y_val = X[val_idx], y_raw[val_idx]
    X_tst, y_tst = X[tst_idx], y_raw[tst_idx]

    print(f'\nBase split ({TRAIN_FRACTION*100:.0f}% train): '
          f'{len(trn_idx)} train / {n_val} val / {n_test} test')

    # ── 5. Augment training set only ──────────────────────────────────────────
    if AUGMENT:
        print(f'\nAugmenting training set ({len(X_trn)} → ', end='')
        X_trn, y_trn = build_augmented_dataset(X_trn, y_trn, rng)
        # Shuffle augmented set
        shuf = rng.permutation(len(X_trn))
        X_trn, y_trn = X_trn[shuf], y_trn[shuf]
        print(f'{len(X_trn)} samples)')

    # Remap raw labels → class indices for all splits
    y_trn = np.array([label2idx[c] for c in y_trn])
    y_val = np.array([label2idx[c] for c in y_val])
    y_tst = np.array([label2idx[c] for c in y_tst])

    # Recompute class weights on augmented training set
    aug_counts = Counter(y_trn.tolist())
    cw = np.array([len(y_trn) / (n_cls * max(1, aug_counts.get(i, 1)))
                   for i in range(n_cls)], dtype=np.float32)
    cw /= cw.mean()

    print(f'\nSplit after augmentation:')
    print(f'  Train : {len(X_trn)}')
    print(f'  Val   : {len(X_val)}  (original, no augmentation)')
    print(f'  Test  : {len(X_tst)}  (original, no augmentation)')

    # ── 5. Initialise weights (He initialisation for ReLU) ────────────────────
    W1 = (rng.standard_normal((HIDDEN_SIZE, n_feat))
          * np.sqrt(2.0 / n_feat)).astype(np.float32)
    b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    W2 = (rng.standard_normal((n_cls, HIDDEN_SIZE))
          * np.sqrt(2.0 / HIDDEN_SIZE)).astype(np.float32)
    b2 = np.zeros(n_cls, dtype=np.float32)

    print(f'\nArchitecture:')
    print(f'  Input  : {n_feat}')
    print(f'  Hidden : {HIDDEN_SIZE}  (ReLU)')
    print(f'  Output : {n_cls}  (softmax)')
    params = HIDDEN_SIZE*n_feat + HIDDEN_SIZE + n_cls*HIDDEN_SIZE + n_cls
    print(f'  Params : {params:,}  ({params*4/1024:.1f} KB)')

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print(f'\nTraining  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LEARNING_RATE}  l2={L2_REG}')
    print('─' * 60)

    best_val_acc  = 0.0
    best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    patience      = 60    # increased from 40
    no_improve    = 0
    t1 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Shuffle
        idx_e = rng.permutation(len(X_trn))

        for start in range(0, len(X_trn), BATCH_SIZE):
            mb   = idx_e[start:start+BATCH_SIZE]
            Xb   = X_trn[mb]
            yb   = y_trn[mb]
            sw   = cw[yb]                         # sample weights

            # Forward
            probs, h_pre, h = forward(Xb, W1, b1, W2, b2)

            # One-hot
            oh = np.zeros((len(mb), n_cls), dtype=np.float32)
            oh[np.arange(len(mb)), yb] = 1.0

            # Backward — output layer
            dL_dlogits = (probs - oh) * sw[:, None] / len(mb)   # (batch, n_cls)
            dW2 = dL_dlogits.T @ h + L2_REG * W2                # (n_cls, hidden)
            db2 = dL_dlogits.sum(axis=0)

            # Backward — hidden layer
            dL_dh   = dL_dlogits @ W2                            # (batch, hidden)
            dL_dpre = dL_dh * relu_grad(h_pre)
            dW1     = dL_dpre.T @ Xb + L2_REG * W1              # (hidden, n_feat)
            db1     = dL_dpre.sum(axis=0)

            # Update
            W1 -= LEARNING_RATE * dW1
            b1 -= LEARNING_RATE * db1
            W2 -= LEARNING_RATE * dW2
            b2 -= LEARNING_RATE * db2

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            trn_acc = accuracy_score(X_trn, y_trn, W1, b1, W2, b2)
            val_acc = accuracy_score(X_val, y_val, W1, b1, W2, b2)
            elapsed = time.time() - t1
            print(f'  epoch {epoch:4d}/{EPOCHS}  '
                  f'train={trn_acc:5.1f}%  val={val_acc:5.1f}%  '
                  f't={elapsed:.1f}s'
                  + ('  ← best' if val_acc > best_val_acc else ''))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_W1, best_b1 = W1.copy(), b1.copy()
                best_W2, best_b2 = W2.copy(), b2.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience // 10:
                    print(f'  Early stopping at epoch {epoch} (no val improvement for {patience} epochs)')
                    break

    # Restore best weights
    W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2
    print(f'\nBest val accuracy: {best_val_acc:.1f}%')

    # ── 7. Final evaluation ───────────────────────────────────────────────────
    trn_acc = accuracy_score(X_trn, y_trn, W1, b1, W2, b2)
    val_acc = accuracy_score(X_val, y_val, W1, b1, W2, b2)
    tst_acc = accuracy_score(X_tst, y_tst, W1, b1, W2, b2)

    print(f'\n{"─"*40}')
    print(f'  Train accuracy : {trn_acc:.1f}%')
    print(f'  Val   accuracy : {val_acc:.1f}%')
    print(f'  TEST  accuracy : {tst_acc:.1f}%  ← held-out, unbiased')
    print(f'{"─"*40}')

    print('\nConfusion matrix on TEST set (rows=true, cols=predicted):')
    cm_str = confusion_str(X_tst, y_tst, W1, b1, W2, b2, classes)
    print(cm_str)

    print('\nPer-class TEST accuracy:')
    preds_tst = predict_batch(X_tst, W1, b1, W2, b2)
    for ci, c in enumerate(classes):
        mask = y_tst == ci
        if mask.sum() == 0:
            print(f'  Strip {c}: no test samples')
            continue
        cls_acc = (preds_tst[mask] == ci).mean() * 100
        print(f'  Strip {c}: {cls_acc:5.1f}%  ({mask.sum()} samples)')

    # ── 8. Save model ─────────────────────────────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, 'model_W1.npy'),      W1)
    np.save(os.path.join(OUTPUT_DIR, 'model_b1.npy'),      b1)
    np.save(os.path.join(OUTPUT_DIR, 'model_W2.npy'),      W2)
    np.save(os.path.join(OUTPUT_DIR, 'model_b2.npy'),      b2)
    np.save(os.path.join(OUTPUT_DIR, 'model_classes.npy'), classes)
    print(f'\nModel saved to {OUTPUT_DIR}')
    print(f'  W1: {W1.shape}  ({W1.nbytes/1024:.0f} KB)')
    print(f'  W2: {W2.shape}  ({W2.nbytes/1024:.0f} KB)')
    total_kb = (W1.nbytes + b1.nbytes + W2.nbytes + b2.nbytes) / 1024
    print(f'  Total model size: {total_kb:.1f} KB')

    # ── 9. Save report ────────────────────────────────────────────────────────
    report = os.path.join(OUTPUT_DIR, 'training_report.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write(f'Training Report\n{"="*50}\n\n')
        f.write(f'Labels JSON    : {LABELS_JSON}\n')
        f.write(f'Image size     : {IMG_W}x{IMG_H}x3 YUV\n')
        f.write(f'Features       : {n_feat}\n')
        f.write(f'Architecture   : {n_feat} -> {HIDDEN_SIZE} (ReLU) -> {n_cls} (softmax)\n')
        f.write(f'Parameters     : {params:,}\n\n')
        f.write(f'Split ({TRAIN_FRACTION*100:.0f}% train):\n')
        f.write(f'  Train : {len(X_trn)}\n')
        f.write(f'  Val   : {len(X_val)}\n')
        f.write(f'  Test  : {len(X_tst)}\n\n')
        f.write('Class distribution:\n')
        for c in sorted(counts):
            f.write(f'  Strip {c}: {counts[c]}\n')
        f.write(f'\nTrain accuracy : {trn_acc:.1f}%\n')
        f.write(f'Val   accuracy : {val_acc:.1f}%\n')
        f.write(f'TEST  accuracy : {tst_acc:.1f}%\n\n')
        f.write('Confusion matrix (TEST, rows=true, cols=predicted):\n')
        f.write(cm_str + '\n')
    print(f'Report saved to {report}')

    # ── 10. Inference snippet ─────────────────────────────────────────────────
    print(f"""
Inference snippet for the drone:
─────────────────────────────────────────────────────
import numpy as np, cv2
W1=np.load('model_W1.npy'); b1=np.load('model_b1.npy')
W2=np.load('model_W2.npy'); b2=np.load('model_b2.npy')
classes=np.load('model_classes.npy')

def predict(img_bgr):
    yuv = cv2.cvtColor(cv2.resize(img_bgr,({IMG_W},{IMG_H})),cv2.COLOR_BGR2YUV)
    x   = yuv.astype(np.float32).flatten() / 255.0
    h   = np.maximum(0, W1 @ x + b1)      # ReLU hidden layer
    return int(classes[np.argmax(W2 @ h + b2)])
─────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()