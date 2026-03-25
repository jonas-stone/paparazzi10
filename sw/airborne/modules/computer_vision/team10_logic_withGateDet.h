#ifndef TEAM10_LOGIC_H
#define TEAM10_LOGIC_H

#include <stdint.h>
#include "modules/computer_vision/lib/vision/image.h"
#include "team10_get_obstacle_info.h"

/* ── Legacy bias defaults (kept for callers that still use fixed values) ──── */
#define DEFAULT_OBS_BIAS         50
#define DEFAULT_PLANT_BIAS       70
#define DEFAULT_OBS_BIAS_FRAC    0.065f
#define DEFAULT_PLANT_BIAS_FRAC  0.15f

/* ── Function declarations ────────────────────────────────────────────────── */

/*
 * greenest_section_column
 * Returns 1-based index of the greenest section out of ns sections.
 */
int greenest_section_column(const float gb[], int w, int ns);

/*
 * greenest_pixel
 * Returns the column index (0 to w-1) where the baseline is lowest.
 * Now delegates to widest_corridor_centre internally.
 */
int greenest_pixel(const float gb[], int w);

/*
 * widest_corridor_centre
 * Returns the centre column of the widest contiguous run of clear columns.
 * A column is clear if gb[col] < clear_frac * h.
 * Runs narrower than min_corridor_width_px are discarded.
 * Falls back to the minimum-value plateau centre if no run qualifies.
 */
int widest_corridor_centre(const float gb[], int w, int h);

/*
 * obstacle_touches_ground
 * Fills touches_out[i] = 1 if obstacle i has no ground beneath it.
 * Returns 1 if any obstacle touches the ground, 0 otherwise.
 */
uint8_t obstacle_touches_ground(const struct obstacle_region_t oo[], uint8_t no,
                                int h, uint8_t touches_out[], uint8_t *touch);

/*
 * normalised_bias
 * Returns a bias in pixels as a fraction of the obstacle/plant width (min 1).
 */
int normalised_bias(const struct obstacle_region_t *o, float frac);

/*
 * bias_logic
 * Erases dangerous zones from a copy of gb[] and returns the safest column.
 */
int bias_logic(const struct obstacle_region_t oo[], uint8_t no,
               const struct obstacle_region_t po[], uint8_t np,
               const float gb[], int w, int h,
               const uint8_t touches_out[],
               int obs_bias, int plant_bias);

/*
 * motion_logic
 * Main waypoint selection with fixed pixel bias values.
 * Returns column index (0 to w-1).
 */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                 const struct obstacle_region_t po[], uint8_t np,
                 const float gb[], int w, int h,
                 int obs_bias, int plant_bias);

/*
 * motion_logic_normalised
 * Same as motion_logic but bias is a fraction of each obstacle/plant width.
 */
int motion_logic_normalised(const struct obstacle_region_t oo[], uint8_t no,
                            const struct obstacle_region_t po[], uint8_t np,
                            const float gb[], int w, int h,
                            float obs_bias_frac, float plant_bias_frac);

/*
 * motion_logic_gate_aware
 * Same as motion_logic_normalised but skips erasing the gate column if a
 * gate is detected, so the drone can steer toward it instead of away.
 */
int motion_logic_gate_aware(const struct obstacle_region_t oo[], uint8_t no,
                            const struct obstacle_region_t po[], uint8_t np,
                            const float gb[], int w, int h,
                            float obs_bias_frac, float plant_bias_frac,
                            uint8_t gate_detected, int gate_center_col);

#endif /* TEAM10_LOGIC_H */