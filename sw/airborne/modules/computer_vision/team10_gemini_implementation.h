#ifndef TEAM10_GET_OBSTACLE_INFO_H
#define TEAM10_GET_OBSTACLE_INFO_H

#include "modules/computer_vision/lib/vision/image.h"
#include <stdint.h>
#include <stdbool.h>

#define MAX_IMAGE_WIDTH 640
#define MAX_OBSTACLE_REGIONS 20

#define DET_OBSTACLE 0
#define DET_PLANT    1

struct obstacle_region_t {
    int start;
    int width;
    int type;
};

// Core processing function to be called from the video callback
uint8_t get_obstacle_info(
    struct image_t *img,
    float *ground_baseline,
    int *baseline_inited,
    float oa_color_count_frac,
    int median_ksize,
    int min_width,
    struct obstacle_region_t *obstacles,
    int *boundary_rows_out,
    int *ground_found_out,
    float *green_frac_out
);

#endif // TEAM10_GET_OBSTACLE_INFO_H