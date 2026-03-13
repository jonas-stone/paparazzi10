import cv2
import numpy as np
import glob
from sklearn.tree import DecisionTreeClassifier

# Global lists for training data
X_vec = [] # Stores Y, U, V
y_vec = [] # Stores 1 (Ground) or 0 (Not Ground)

current_yuv_image = None
current_display_image = None

def mouse_callback(event, x, y, flags, param):
    global current_yuv_image, current_display_image, X_vec, y_vec
    
    # LEFT CLICK: Ground
    if event == cv2.EVENT_LBUTTONDOWN:
        p = current_yuv_image[y, x]
        X_vec.append([int(p[0]), int(p[1]), int(p[2])])
        y_vec.append(1)
        cv2.circle(current_display_image, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Teach the Drone", current_display_image)
        
    # RIGHT CLICK: Obstacles / Background
    elif event == cv2.EVENT_RBUTTONDOWN:
        p = current_yuv_image[y, x]
        X_vec.append([int(p[0]), int(p[1]), int(p[2])])
        y_vec.append(0)
        cv2.circle(current_display_image, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("Teach the Drone", current_display_image)

def export_tree_to_python(tree_model, feature_names):
    """
    Translates the trained scikit-learn Decision Tree directly into Python code.
    """
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    print("\n" + "="*50)
    print(" COPY THIS PYTHON CODE TO YOUR TEST SCRIPT ")
    print("="*50 + "\n")
    print("def is_ground(Y, U, V):")

    def recurse(node, depth):
        indent = "    " * depth
        # If the node is a decision node
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(f"{indent}if {name} <= {threshold:.2f}:")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}else:")
            recurse(tree_.children_right[node], depth + 1)
        # If the node is a leaf (final decision)
        else:
            class_idx = np.argmax(tree_.value[node][0])
            class_val = tree_model.classes_[class_idx]
            result = 255 if class_val == 1 else 0
            print(f"{indent}return {result}")

    recurse(0, 1)
    print("\n" + "="*50)

def main():
    global current_yuv_image, current_display_image, X_vec, y_vec
    
    # Update this to point to your Bebop images
    folder_path = "TEAM-10-PROTOTYPING/downloads from drone/20260306-095826/*.jpg"
    images = glob.glob(folder_path, recursive=True)

    if not images:
        print("No images found! Check your folder path.")
        return

    cv2.namedWindow("Teach the Drone")
    cv2.setMouseCallback("Teach the Drone", mouse_callback)

    print("\n--- INSTRUCTIONS ---")
    print("LEFT CLICK  -> Green Ground")
    print("RIGHT CLICK -> Obstacles, Sky, Background")
    print("Press 'n'   -> Next image")
    print("Press 'q'   -> Train model & generate Python code\n")

    for f in images:
        img = cv2.imread(f)
        if img is None: continue
        
        # Resize if massive
        h, w = img.shape[:2]
        if max(h, w) > 900:
            scale = 900 / max(h, w)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)

        current_yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        current_display_image = img.copy()

        cv2.imshow("Teach the Drone", current_display_image)
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n') or key == ord('q'): break
        if key == ord('q'): break

    cv2.destroyAllWindows()

    if len(X_vec) < 20:
        print("Not enough clicks! Try clicking at least 15 ground and 15 background points.")
        return

    print(f"\nTraining on {len(X_vec)} clicked pixels...")

    # Train the Decision Tree. 
    dt = DecisionTreeClassifier(max_depth=4, random_state=0)
    dt.fit(X_vec, y_vec)

    # Generate the hardcoded Python logic
    export_tree_to_python(dt, ['Y', 'U', 'V'])

if __name__ == "__main__":
    main()