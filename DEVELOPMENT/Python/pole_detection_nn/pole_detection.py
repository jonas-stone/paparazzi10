import cv2
import numpy as np
import glob
import os
from random import randrange
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ── Folders — edit these two lines ───────────────────────────────────────────
IMAGES_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
MASKS_FOLDER  = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\pole_detection_nn\pole_detection_nn_data'

# ── Load the images ──────────────────────────────────────────────────────────
labels = glob.glob(MASKS_FOLDER + '\\*_mask.jpg', recursive=True)

images = []
for lf in labels:
    basename = os.path.basename(lf).replace('_mask.jpg', '')
    matches = glob.glob(IMAGES_FOLDER + '\\' + basename + '.*')
    matches = [m for m in matches if '_mask' not in m]
    if matches:
        images.append(matches[0])

print(images)

# ── Convert images to training data (vector) ─────────────────────────────────
X_vec = []
y_vec = []

maxfiles = 500
samples_per_image = 300000

def extract_features(yuv, hsv, lab, y, x):
    p_yuv = yuv[y, x]
    p_hsv = hsv[y, x]
    p_lab = lab[y, x]
    patch = yuv[y-1:y+2, x-1:x+2].reshape(-1, 3)
    mean  = patch.mean(axis=0)
    std   = patch.std(axis=0)
    grad  = np.abs(yuv[y, x+1].astype(int) - yuv[y, x-1].astype(int))
    return [
        int(p_yuv[0]), int(p_yuv[1]), int(p_yuv[2]),
        int(p_hsv[0]), int(p_hsv[1]), int(p_hsv[2]),
        int(p_lab[0]), int(p_lab[1]), int(p_lab[2]),
        *mean.tolist(), *std.tolist(), *grad.tolist()
    ]

for f in images:
    basename = os.path.splitext(os.path.basename(f))[0]
    lf = os.path.join(MASKS_FOLDER, basename + '_mask.jpg')

    if os.path.exists(lf):
        maxfiles = maxfiles - 1
        if maxfiles <= 0:
            break

        img = cv2.imread(f)
        msk = cv2.imread(lf)
        h, w, d = img.shape

        print('img=', f, 'lbl=', lf, w, 'x', h)

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img[:, :, 1] = msk[:, :, 0]

        for i in range(0, samples_per_image):
            x = randrange(2, w-3)
            y = randrange(4, h-2)

            m = int(msk[y, x, 0])
            if m < 127:
                m = 0
            else:
                m = 255

            X_vec.append(extract_features(yuv, hsv, lab, y, x))
            y_vec.append([m])

print('Dataset', len(X_vec), len(y_vec))

# ── Split training data and test data ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_vec, test_size=0.2, stratify=y_vec, random_state=1
)

print('Train', len(X_train), len(y_train))
print('Test', len(X_test), len(y_test))

# ── Train ─────────────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    n_jobs=-1,
    random_state=0
)
clf.fit(X_train, np.array(y_train).ravel())

y_pred = clf.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Sensitivity:', round(score, 3))

# ── Save the model ────────────────────────────────────────────────────────────
MODEL_PATH = 'pole_detector.pkl'
joblib.dump(clf, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')

# ── Result: show classification by overwriting the green channel ──────────────
for f in images:
    img = cv2.imread(f)
    h, w, d = img.shape

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    pixels = np.array([
        extract_features(yuv, hsv, lab, y, x)
        for y in range(4, h-2)
        for x in range(2, w-3)
    ])
    y_pred_flat = clf.predict(pixels)

    msk_cropped = y_pred_flat.reshape(h-6, w-5)
    msk = np.zeros((h, w), dtype=np.uint8)
    msk[4:h-2, 2:w-3] = msk_cropped

    img[:, :, 1] = msk[:, :]

    plt.imshow(img)
    plt.show()

# import cv2
# import numpy as np
# import glob
# import os
# from random import randrange
#
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
#
# # ── Folders — edit these two lines ───────────────────────────────────────────
# IMAGES_FOLDER = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320'
# MASKS_FOLDER  = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\pole_detection_nn\pole_detection_nn_data'
#
# # ── Load the images ──────────────────────────────────────────────────────────
# labels = glob.glob(MASKS_FOLDER + '\\*_mask.jpg', recursive=True)
#
# images = []
# for lf in labels:
#     basename = os.path.basename(lf).replace('_mask.jpg', '')
#     matches = glob.glob(IMAGES_FOLDER + '\\' + basename + '.*')
#     matches = [m for m in matches if '_mask' not in m]
#     if matches:
#         images.append(matches[0])
#
# print(images)
#
# # ── Convert images to training data (vector) ─────────────────────────────────
# X_vec = []
# y_vec = []
#
# maxfiles = 500
# samples_per_image = 75000
#
# for f in images:
#     basename = os.path.splitext(os.path.basename(f))[0]
#     lf = os.path.join(MASKS_FOLDER, basename + '_mask.jpg')
#
#     if os.path.exists(lf):
#         maxfiles = maxfiles - 1
#         if maxfiles <= 0:
#             break
#
#         img = cv2.imread(f)
#         msk = cv2.imread(lf)
#         h, w, d = img.shape
#
#         print('img=', f, 'lbl=', lf, w, 'x', h)
#
#         # Color space of RAW Bebop Images
#         yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#         img[:, :, 1] = msk[:, :, 0]
#
#         for i in range(0, samples_per_image):
#             x = randrange(2, w-3)
#             y = randrange(4, h-2)
#
#             # Pixels
#             p = yuv[y, x]
#             # Ground truth
#             m = int(msk[y, x, 0])
#             if m < 127:
#                 m = 0
#             else:
#                 m = 255
#
#             # Y, U, V
#             X_vec.append([int(p[0]), int(p[1]), int(p[2])])
#             y_vec.append([m])
#
# print('Dataset', len(X_vec), len(y_vec))
#
# # ── Split training data and test data ────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vec, y_vec, test_size=0.2, stratify=y_vec, random_state=1
# )
#
# print('Train', len(X_train), len(y_train))
# print('Test', len(X_test), len(y_test))
#
# # ── Train ─────────────────────────────────────────────────────────────────────
# dt = DecisionTreeClassifier(max_depth=2, random_state=0)
# dt.fit(X_train, y_train)
#
# y_pred = dt.predict(X_test)
#
# score = accuracy_score(y_test, y_pred)
# print('Sensitivity:', round(score, 3))
#
# # ── Export ────────────────────────────────────────────────────────────────────
# text_representation = export_text(dt, feature_names=['Y', 'U', 'V'])
# print(text_representation)
#
# # ── Result: show classification by overwriting the V color channel ────────────
# for f in images:
#     img = cv2.imread(f)
#     h, w, d = img.shape
#
#     # Load an image
#     yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#
#     X_run = yuv.reshape(int(h*w), int(d))
#     y_pred = dt.predict(X_run)
#     msk = y_pred.reshape(h, w)
#
#     img[:, :, 1] = msk[:, :]
#
#     plt.imshow(img)
#     plt.show()