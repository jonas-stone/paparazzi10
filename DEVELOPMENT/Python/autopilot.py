import avoid_orange

import numpy as np
import time
import os

penis = 2
def main():
    
    # define absolute orange pixel threshold (orange / total)
    threshold = 0.05

    # define initial quantities + vision code framerate
    timers = []
    beginning = time.process_time()
    image_index = 0
    polling_Hz = 25

    # get all image file names inside the wanted directory
    dir_name = "AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"
    all_image_names = os.listdir(dir_name)

    # sort based on the image name (without the .jpg extension) -> sequential list of images
    all_image_names.sort(key=lambda x: int(x[:-4]))

    # loop until 50 sec have passed.
    current_time = time.process_time()
    while current_time - beginning < 50:

        # get image from the list
        img_name = all_image_names[image_index]
        
        current_time = time.process_time()

        # ========== COMPUTE ORANGE PIXELS ============
        start           = time.process_time_ns()
        orange_coverage = avoid_orange.output(img_name)
        # centroid        = avoid_orange.get_centroid(np.asarray([1]))
        end             = time.process_time_ns()
        # =============================================

        if orange_coverage > threshold:
            action = "rotate"
        else:
            action = "move forward"

        # print action every second
        if current_time % 1 < 5e-3:
            print()
            print(action)

        # simulate visual code framerate (get next image every 1 / Hz seconds)
        if current_time % (1 / polling_Hz) < 1e-3:
            image_index += 1
            print(img_name)

        
        timers.append(end-start)
    
    # print(timers)
    # print(f"number of iterations: {len(timers)}")
    # print(f"average time: {sum(timers) / len(timers) / 1e9} seconds")


if __name__ == "__main__":
    main()