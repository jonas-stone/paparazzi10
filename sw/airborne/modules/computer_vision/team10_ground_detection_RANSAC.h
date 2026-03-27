#ifndef TEAM10_GROUND_DETECTION_RANSAC_H
#define TEAM10_GROUND_DETECTION_RANSAC_H

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// for the YUV color picker type
#include "team10_RANSAC_adaptation.h"
 
#ifndef NUMBER_VERTICAL_BUCKETS
#define NUMBER_VERTICAL_BUCKETS  13
#endif

#ifndef MAX_EDGE_PIXELS
#define MAX_EDGE_PIXELS 5000
#endif

// controllable slider on Paparazzi GCS
extern int area_threshold;
extern uint8_t draw_parabola;
extern uint8_t draw_ground_mask;
extern float score_multiplier;

// color picker variables
extern yuv_color_e parabola_color;
extern yuv_color_e bucket_color;

// ABI messaging functions
extern void ground_detection_init(void);
extern void ground_detection_periodic(void);

#endif