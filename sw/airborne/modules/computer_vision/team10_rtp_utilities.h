#ifndef TEAM10_RTP_UTILITIES_H
#define TEAM10_RTP_UTILITIES_H

#include <stdint.h>
#include "modules/computer_vision/lib/vision/image.h"
#include "team10_get_obstacle_info.h"

void draw_mask_printer(struct image_t *img, uint8_t *work_mask, int w, int h);
void draw_toolbar_vertical(struct image_t *img, int w, int h, float gf, uint8_t no, struct obstacle_region_t oo[]);
void draw_obstacle_detection_bar(struct image_t *img, int w, int h, uint8_t no, struct obstacle_region_t oo[]);
int get_safest_column(int h, uint8_t no, struct obstacle_region_t oo[]);
void draw_safe_direction_bar(struct image_t *img, int w, int h,
                             uint8_t no, struct obstacle_region_t oo[],
                             uint8_t np, struct obstacle_region_t po[],
                             const float gb[]);
#endif