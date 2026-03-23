#ifndef TEAM10_LOGIC_H
#define TEAM10_LOGIC_H

#include <stdint.h>
#include "modules/computer_vision/lib/vision/image.h"
#include "team10_get_obstacle_info.h"

#define DEFAULT_OBS_BIAS    50
#define DEFAULT_PLANT_BIAS   70
#define DEFAULT_OBS_BIAS_FRAC   0.05f
#define DEFAULT_PLANT_BIAS_FRAC 0.1f

/*
 * ══════════════════════════════════════════════════════════════════════════════
 *  GREENEST SECTION -- Column version
 *  Returns 1-based index of the greenest section out of ns sections.
 * ══════════════════════════════════
 */
int greenest_section_column(const float gb[], int w, int ns);

/*
 * ══════════════════════════════════════════════════════════════════════════════
 *  GREENEST PIXEL
 *  Returns the column index (0 to w-1) where the baseline is at its highest.
 * ══════════════════════════════════════════════════════════════════════════════ 
 */
int greenest_pixel(const float gb[], int w);

/* ══════════════════════════════════════════════════════════════════════════════
 *  OBSTACLE TOUCHES THE IMAGE BOTTOM
 *  Fills touches_out[i] = 1 if obstacle i has no ground beneath it.
 *  Returns 1 if any obstacle touches, 0 otherwise.
 * ══════════════════════════════════════════════════════════════════════════════ */
uint8_t obstacle_touches_ground(const struct obstacle_region_t oo[], uint8_t no,
                                int h, uint8_t touches_out[], uint8_t *touch);

/* ══════════════════════════════════════════════════════════════════════════════
 *  BIAS LOGIC
 *  Erases dangerous zones from a copy of gb[] and returns the safest column.
 * ══════════════════════════════════════════════════════════════════════════════ */
int bias_logic(const struct obstacle_region_t oo[], uint8_t no,
               const struct obstacle_region_t po[], uint8_t np,
               const float gb[], int w, int h,
               const uint8_t touches_out[],
               int obs_bias, int plant_bias);

/* ══════════════════════════════════════════════════════════════════════════════
 *  NORMALISED BIAS
 *  Computes a bias in pixels as a fraction of the obstacle/plant width.
 *
 *  Inputs:
 *    o    - pointer to the obstacle/plant region
 *    frac - fraction of the width to use as bias (e.g. 0.5 = 50%)
 *
 *  Output:
 *    int  - bias in pixels (minimum 1)
 * ══════════════════════════════════════════════════════════════════════════════ */
int normalised_bias(const struct obstacle_region_t *o, float frac);

/* ══════════════════════════════════════════════════════════════════════════════
 *  MOTION LOGIC
 *  Main waypoint selection function. Returns column index (0 to w-1).
 * ══════════════════════════════════════════════════════════════════════════════ */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                 const struct obstacle_region_t po[], uint8_t np,
                 const float gb[], int w, int h,
                 int obs_bias, int plant_bias);

/* ══════════════════════════════════════════════════════════════════════════════
 *  MOTION LOGIC NORMALISED
 *  Same as motion_logic but uses normalised_bias per obstacle/plant width
 *  instead of fixed pixel values.
 *
 *  Inputs:
 *    obs_bias_frac   - fraction of each obstacle width to use as bias
 *    plant_bias_frac - fraction of each plant width to use as bias
 * ══════════════════════════════════════════════════════════════════════════════ */
int motion_logic_normalised(const struct obstacle_region_t oo[], uint8_t no,
                            const struct obstacle_region_t po[], uint8_t np,
                            const float gb[], int w, int h,
                            float obs_bias_frac, float plant_bias_frac);

#endif /* TEAM10_LOGIC_H */