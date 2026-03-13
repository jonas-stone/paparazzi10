#ifndef GROUND_DETECTION_H
#define GROUND_DETECTION_H

#include "std.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "config.h"

// Obstacle region structure
struct obstacle_region_t {
  uint16_t start_col;
  uint16_t end_col;
  uint16_t width;
};

// Results accessible from outside the module
extern volatile int    green_pixel_count;
extern volatile float  green_fraction;
extern volatile bool   ground_found;

// Number of obstacle regions found in last frame
extern volatile uint8_t obstacle_count;

// Obstacle regions found in last frame (max 20)
extern struct obstacle_region_t obstacle_regions[20];

// Classifies a single pixel as ground (255) or not (0)
// using a trained decision tree on YUV values
uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V);

// Module init function - registers module on Bebop front camera
extern void ground_detection_init(void);

#endif