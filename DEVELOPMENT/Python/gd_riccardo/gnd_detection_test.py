import numpy.random as rd
import numpy as np
import cv2


def find_average_color_ground(img_path: str) -> tuple:
    
    img_original = np.array(cv2.imread(img_path, flags=1))
    img, points = subsample(img_original, 30)

    H, S, V = parse_colors(img, type="HSV")

    avg_H = np.mean(H)
    avg_S = np.mean(S)
    avg_V = np.mean(V)

    std_H = np.std(H)
    std_S = np.std(S)
    std_V = np.std(V)

    return avg_H, avg_S, avg_V, std_H, std_S, std_V


def parse_colors(img: np.ndarray, type: str) -> tuple:

    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]
        
    if type.upper() == "HSV":
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        H = np.asarray(hsv_img[..., 0])
        S = np.asarray(hsv_img[..., 1])
        V = np.asarray(hsv_img[..., 2])

        return H, S, V

    elif type.upper() == "RGB": pass

    else: print("Warning: invalid color channel type. Returning RGB channels.")

    return R, G, B


def falls_into_range(img: np.ndarray, avg_color: tuple, std_color: tuple):

    H, S, V = parse_colors(img, "HSV")
    avg_H, avg_S, avg_V = avg_color
    std_H, std_S, std_V = std_color

    for row in img:
        for col in row:

            h = H[row, col]
            s = S[row, col]
            v = V[row, col]

            if -std_H <= h - avg_H <= std_H & \
               -std_S <= s - avg_S <= std_S & \
               -std_V <= v - avg_V <= std_V:
                
                pass

    pass


def detect_ground(img: np.ndarray):
    pass


def subsample(img: np.ndarray, number_of_points: int) -> tuple:

    height  = img.shape[0]
    width   = img.shape[1]

    points = list()

    img_subsampled = np.zeros([height, width])

    n = 1
    while n < number_of_points:
        
        pi = rd.randint(0, height)
        pj = rd.randint(0, width)

        if (pi, pj) not in points:

            img_subsampled[pi, pj] = img[pi, pj]
            points.append((pi, pj))
            n += 1

    return img_subsampled, points


if __name__ == "__main__":
    
    # perform average calculation on many bottom camera photos
    pass