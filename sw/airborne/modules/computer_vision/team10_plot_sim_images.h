#ifndef TEAM10_TEST_CODE_ON_SIM_IMAGES_H
#define TEAM10_TEST_CODE_ON_SIM_IMAGES_H

#include "modules/computer_vision/lib/vision/image.h"

/**
 * Loads a JPEG from the simulation directory into an image_t struct.
 * @param filename The full path to the JPEG file.
 * @param img      Pointer to an image_t struct to be populated.
 * @return         1 on success, 0 on failure.
 */

void wait_for_keypress(void);
int load_jpeg_to_image_t(const char *filename, struct image_t *img);
int numeric_sort(const struct dirent **a, const struct dirent **b);

/**
 * Iterates through the sim directory and displays images.
 */
void run_sim_viewer(struct image_t *img);

#endif /* SIM_IMAGE_LOADER_H */