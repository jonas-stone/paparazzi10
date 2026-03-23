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


def train_detector(
    images_folder,                   # path to folder containing raw drone images
    masks_folder,                    # path to folder containing *_mask.jpg ground truth files
    model_save_path  = 'detector.pkl',  # where to save the trained model on disk
    show_figures     = False,           # switch: set True to visualise results on training images after training
    samples_per_image = 300000,         # number of random pixels sampled per image — higher = better but slower
    maxfiles          = 500,            # safety cap on number of images processed — rarely needs changing
):
    """
    Train a pixel-level object detector from image/mask pairs and save it to disk.

    Parameters
    ----------
    images_folder     : str  — folder containing the raw drone images
    masks_folder      : str  — folder containing the corresponding *_mask.jpg files
    model_save_path   : str  — where to save the trained .pkl model (default: 'detector.pkl')
    show_figures      : bool — if True, displays original vs detection overlay after training
    samples_per_image : int  — random pixels sampled per image (default: 300000)
    maxfiles          : int  — cap on number of images processed (default: 500)

    Returns
    -------
    clf : trained RandomForestClassifier — also saved to model_save_path
    """

    # ── Feature extractor ─────────────────────────────────────────────────────
    # Converts a single pixel location into a 21-value feature vector
    # combining color information from three color spaces (YUV, HSV, LAB)
    # plus neighbourhood texture statistics and edge information.
    # This function must remain identical between training and inference —
    # changing it after training will silently break predictions.
    def extract_features(yuv, hsv, lab, y, x):
        p_yuv = yuv[y, x]       # pixel values in YUV color space
        p_hsv = hsv[y, x]       # pixel values in HSV color space
        p_lab = lab[y, x]       # pixel values in LAB color space

        patch = yuv[y-1:y+2, x-1:x+2].reshape(-1, 3)  # 3x3 neighbourhood around pixel in YUV
        mean  = patch.mean(axis=0)                      # mean color of neighbourhood — smooths noise
        std   = patch.std(axis=0)                       # std of neighbourhood — high = edge/texture, low = flat region
        grad  = np.abs(yuv[y, x+1].astype(int) - yuv[y, x-1].astype(int))  # horizontal gradient — detects edges

        # Return all 21 features concatenated into one flat list:
        # 3 YUV + 3 HSV + 3 LAB + 3 neighbourhood mean + 3 neighbourhood std + 3 gradient = 21
        return [
            int(p_yuv[0]), int(p_yuv[1]), int(p_yuv[2]),  # Y, U, V
            int(p_hsv[0]), int(p_hsv[1]), int(p_hsv[2]),  # H, S, V
            int(p_lab[0]), int(p_lab[1]), int(p_lab[2]),  # L, A, B
            *mean.tolist(), *std.tolist(), *grad.tolist()  # neighbourhood stats + gradient
        ]

    # ── Load image/mask pairs ─────────────────────────────────────────────────
    # Scans the masks folder for all *_mask.jpg files, then finds the
    # corresponding original image in the images folder by stripping the
    # _mask suffix and searching for any file with that basename.
    labels = glob.glob(masks_folder + '\\*_mask.jpg', recursive=True)

    images = []
    for lf in labels:
        basename = os.path.basename(lf).replace('_mask.jpg', '')  # strip _mask suffix to get base filename
        matches  = glob.glob(images_folder + '\\' + basename + '.*')  # find matching original image
        matches  = [m for m in matches if '_mask' not in m]  # exclude any mask files from matches
        if matches:
            images.append(matches[0])  # take first match if multiple extensions exist

    print(f'Found {len(images)} image/mask pairs.')
    print(images)

    # ── Build dataset ─────────────────────────────────────────────────────────
    # For each image pair, randomly samples pixels and records their
    # 21-value feature vector (X_vec) and ground truth label (y_vec).
    # Labels are binarised: mask pixel >= 127 → 255 (target), else → 0 (background).
    X_vec = []  # feature vectors — one per sampled pixel
    y_vec = []  # labels — 255 = target object, 0 = background
    files_remaining = maxfiles  # countdown to enforce the maxfiles cap

    for f in images:
        basename = os.path.splitext(os.path.basename(f))[0]  # filename without extension
        lf       = os.path.join(masks_folder, basename + '_mask.jpg')  # expected mask path

        if os.path.exists(lf):
            files_remaining -= 1
            if files_remaining <= 0:
                break  # stop if we've hit the maxfiles cap

            img = cv2.imread(f)   # load original image in BGR
            msk = cv2.imread(lf)  # load mask image in BGR (we only use channel 0)
            h, w, d = img.shape

            print('img=', f, 'lbl=', lf, w, 'x', h)

            # Convert to all three color spaces upfront — done once per image for efficiency
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            img[:, :, 1] = msk[:, :, 0]  # overlay mask on green channel for visual reference

            # Randomly sample pixels from this image
            for i in range(samples_per_image):
                x = randrange(2, w-3)  # x margin of 2 keeps 3x3 patch within image bounds
                y = randrange(4, h-2)  # y margin of 4 keeps 3x3 patch within image bounds

                # Binarise the mask pixel into a clean 0/255 label
                m = int(msk[y, x, 0])
                m = 255 if m >= 127 else 0  # >= 127 = target object, < 127 = background

                X_vec.append(extract_features(yuv, hsv, lab, y, x))  # 21-value feature vector
                y_vec.append([m])                                      # ground truth label

    print('Dataset', len(X_vec), len(y_vec))

    # ── Train/test split ──────────────────────────────────────────────────────
    # 80% of samples go to training, 20% to testing.
    # stratify=y_vec ensures both splits have the same target/background ratio —
    # important because target pixels are typically rare compared to background.
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_vec, test_size=0.2, stratify=y_vec, random_state=1
    )

    print('Train', len(X_train), '| Test', len(X_test))

    # ── Train ─────────────────────────────────────────────────────────────────
    # Random Forest: 200 fully grown decision trees each trained on a random
    # subset of the data, voting together at inference time.
    # class_weight='balanced' compensates for target pixels being much rarer
    # than background pixels, preventing the model from ignoring the target class.
    # n_jobs=-1 uses all available CPU cores in parallel.
    clf = RandomForestClassifier(
        n_estimators=200,        # number of trees — more = better accuracy but slower
        max_depth=None,          # trees grow until leaves are pure — most expressive setting
        class_weight='balanced', # upweights rare target class to prevent bias toward background
        n_jobs=-1,               # use all CPU cores
        random_state=0           # seed for reproducibility
    )
    clf.fit(X_train, np.array(y_train).ravel())  # ravel() flattens y from [[m],...] to [m,...]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # Runs the trained model on the held-out 20% test set and reports accuracy.
    # A score above ~0.90 is generally good for this type of task.
    y_pred = clf.predict(X_test)
    score  = accuracy_score(y_test, y_pred)
    print('Sensitivity:', round(score, 3))

    # ── Save model ────────────────────────────────────────────────────────────
    # Serialises the entire trained model to disk as a .pkl file.
    # Load it later with: clf = joblib.load('your_model.pkl')
    # No retraining needed — model is immediately ready to predict.
    joblib.dump(clf, model_save_path)
    print(f'Model saved to {model_save_path}')

    # ── Visualise ─────────────────────────────────────────────────────────────
    # Only runs if show_figures=True — useful for a quick sanity check after training.
    # Shows the original image alongside the detection overlay (target pixels in green)
    # for each training image. Slow due to the pixel-by-pixel Python loop —
    # fine for offline inspection but not suitable for real-time use.
    if show_figures:
        for f in images:
            img      = cv2.imread(f)
            h, w, d  = img.shape

            # Recompute color spaces for inference
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Build feature matrix for every pixel in the image
            # Border margins match those used during training to keep patch within bounds
            pixels = np.array([
                extract_features(yuv, hsv, lab, y, x)
                for y in range(4, h-2)
                for x in range(2, w-3)
            ])

            # Run classifier on all pixels
            y_pred_flat = clf.predict(pixels)

            # Reconstruct the prediction as a 2D mask
            # The cropped shape accounts for the border margins used above
            msk_cropped = y_pred_flat.reshape(h-6, w-5)
            msk         = np.zeros((h, w), dtype=np.uint8)  # full canvas, borders stay black
            msk[4:h-2, 2:w-3] = msk_cropped                # paste predictions into valid region

            # Build overlay: replace green channel with detection mask
            img_rgb          = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # original for display
            overlay          = img.copy()
            overlay[:, :, 1] = msk[:, :]                             # green = detections
            overlay_rgb      = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Display side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_rgb);     axes[0].set_title('Original');   axes[0].axis('off')
            axes[1].imshow(overlay_rgb); axes[1].set_title('Detection');  axes[1].axis('off')
            plt.suptitle(f, fontsize=8)
            plt.tight_layout()
            plt.show()

    return clf  # also returned in case the caller wants to use the model directly in the same session


# ── Entry point ───────────────────────────────────────────────────────────────
# Edit the three lines below and run the script.
# To train a different detector (trees, ground, etc.) just change the
# masks_folder and model_save_path — everything else stays the same.
if __name__ == '__main__':
    train_detector(
        images_folder   = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\downloads from drone\20260320',
        masks_folder    = r'C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_CW\DEVELOPMENT\Python\neural_network\masks_data\tree_detection_masks',
        model_save_path = 'tree_detector.pkl',
        show_figures    = True,  # flip to True to see visualisations after training
    )

