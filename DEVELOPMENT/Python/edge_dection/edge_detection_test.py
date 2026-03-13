"""
Edge Detection Test File
AI Has been used to help with debugging
"""

import cv2
import numpy as np
import random
import os

def detect_edges(img_in, sigma=0):
    img = cv2.imread(img_in)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_in}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(input=gray, sigmaX=5, sigmaY=5, order=0.000000000000001)
    blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Partition into 3 vertical sections
    h, w = edges.shape
    third = w // 3
    left   = edges[:, :third]
    center = edges[:, third:2*third]
    right  = edges[:, 2*third:]

    for name, part in [("left", left), ("center", center), ("right", right)]:
        density = np.sum(part > 0) / part.size
        print(f"{name} density: {density:.4f}")

    # cv2.imshow("Original", img) # Comment or un-comment to see the original
    # cv2.imshow("Canny Edges", edges)
    # Convert gray edges to BGR so it can be stacked with the color image
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stack side by side
    combined = np.hstack((img, edges_bgr))

    cv2.imshow("Original | Canny Edges", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




detect_edges("../../downloads from drone/20240322-084506/350680718.jpg")
# # def random_image(folder):
# #     """
# #     Function to select a random image in a folder
# #     """
# #     exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
# #     images = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
# #     return os.path.join(folder, random.choice(images))
#
# class ImagePicker:
#     def __init__(self, folder, n=1, state_file="image_sample.txt"):
#         self.folder = folder
#         self.n = n
#         self.state_file = state_file
#         exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         self.images = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
#
#         if os.path.exists(state_file):
#             with open(state_file, 'r') as f:
#                 self.current = [line.strip() for line in f.readlines()]
#         else:
#             self.current = []
#
#     def random(self):
#         self.current = [os.path.join(self.folder, p) for p in random.sample(self.images, self.n)]
#         with open(self.state_file, 'w') as f:
#             f.write('\n'.join(self.current))
#         return self.current
#
#     def same(self):
#         if not self.current:
#             return self.random()
#         return self.current
#
#     def pick(self, mode="random"):
#         if mode == "random":
#             return self.random()
#         elif mode == "same":
#             return self.same()
#         else:
#             raise ValueError(f"mode must be 'random' or 'same', got '{mode}'")
#
# '''
# TESTSING THE FUNCTION TO FIND THE CORRECT BLUE LEVEL FOR EACH PICTURE
#
# NEED TO RANDOMISE SAMPLING A BIT MORE BUT WILL BE DONE
# '''
#
# # Setup - do this once at the top of your script
# picker = ImagePicker("../../downloads from drone/20240322-084506", n=4)
#
# # Pick 4 new random images
# for img in picker.pick("random"):
#     detect_edges(img_in=img, sigma=0)
#
# # Reuse the same 4 images
# for img in picker.pick("same"):
#     detect_edges(img_in=img, sigma=0)
#
# # From footage made on 06/03/2026
# # detect_edges("../../downloads from drone/20240322-084506/350680718.jpg")
#
# ## From provided test data
#
# # CALIBRATION BOTTOM CAM
# # detect_edges("../../AE4317_2019_datasets/calibration_bottomcam/20190121-163846/27146925.jpg", 1)
#
# # CALIBRATION Front CAM
# # detect_edges("../../AE4317_2019_datasets/calibration_frontcam/20190121-163447/30315196.jpg", 1)
#
# # CYBERZOO AGGRESSIVE FLIGHT
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646/25181969.jpg", 0.1)
#
# # CYBERZOO AGGRESSIVE FLIGHT BOTTOM CAM
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_aggressive_flight_bottomcam/20190121-143427/242384873.jpg", 0)
#
# # CYBERZOO BOTTOM CAM
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_bottomcam/20190121-152231/27752131.jpg", 1)
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_bottomcam/20190121-152231/27752131.jpg", 0.1)
#
# # CYBERZOO CANVAS APPROACH TBD
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_canvas_approach/20190121-151448/57414791.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_poles/20190121-135009/75044792.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205/60482805.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/11549407.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/cyberzoo_poles_panels_mats_bottomcam/20190121-143427/12824780.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/sim_poles/20190121-160844/12529000.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/sim_poles_bottomcam/20190121-160605/16536999.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/sim_poles_panels/20190121-161422/15610000.jpg", 0)
#
# # detect_edges("../../AE4317_2019_datasets/sim_poles_panels_mats/20190121-161931/21477000.jpg", 0)
#
