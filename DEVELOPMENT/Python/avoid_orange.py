import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def output(img_name=""):
    """
    Get images sequentially from folder -- as if they came from a video feed.
    Parse them and output the orange pixel count using the functions below.
    """
    
    dir_name = "AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"
    # all_image_names = os.listdir(dir_name)

    # # sort based on the image name (without the .jpg extension) -> sequential list of images
    # all_image_names.sort(key=lambda x: int(x[:-4]))

    # # get random image from list
    # img_name = all_image_names[32]

    # process and print image
    img = get_image(dir_name, img_name)

    # count orange pixels with a hue range of 10 (over 180 possible H values)
    counter = orange_pixel_count(img, type="HSV", range=15)
    # print(f"orange pixels: {counter}")

    return counter / img[..., 1].size


def get_image(dir_name: str, file_name: str):
    """
    Normally images would come directly from the drone camera.
    Now they come from the directory: 'AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/'
    """
    
    file_path = f"{dir_name}/{file_name}"
    
    # flag=1 means read image with color channels -- BGR format, not RGB.
    img = cv2.imread(file_path, flags=1)

    return np.asarray(img)


def parse_colors(img: np.ndarray, type: str):

    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]
        
    if type.upper() == "HSV":
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        H = hsv_img[..., 0]
        S = hsv_img[..., 1]
        V = hsv_img[..., 2]

        return H, S, V

    elif type.upper() == "RGB": pass

    else: print("Warning: invalid color channel type. Returning RGB channels.")

    return R, G, B


def orange_pixel_count(img: np.ndarray, type: str, range: int):
    
    H, _, _ = parse_colors(img, "HSV")

    # for now, suppose the channels are HSV.
    hsv_center_orange   = 20
    left_limit          = hsv_center_orange - range // 2
    right_limit         = hsv_center_orange + range // 2

    # analyze the H channel for pixel that fall into this range.
    orange_pixels = (H >= left_limit) & (H <= right_limit)
    counter = np.count_nonzero(orange_pixels)

    return counter


def get_centroid(mask: np.ndarray):

    mass_x = 0
    mass_y = 0
    total  = 0 
    for row in mask:
        for col in row:

            if mask[row, col] is False:
                continue

            x = row
            y = col

            mass_x += x
            mass_y += y

            total  += 1
    
    return (mass_x / total, mass_y / total)
    

if __name__ == "__main__":
    output()
