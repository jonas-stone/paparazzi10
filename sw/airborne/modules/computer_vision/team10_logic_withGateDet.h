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

/* ── Runtime-tunable corridor parameters ──────────────────────────────────── */
/*
 * These variables are defined in team10_logic_withGateDet.c and can be
 * adjusted via GCS sliders at runtime without recompiling. The extern
 * declarations here make them accessible to any file that includes this header,
 * including the autopilot module and the GCS slider system.
 *
 * clear_frac           — a column is "clear" if its baseline value is below
 *                        this fraction of the image height. Range [0.5, 1.0].
 * min_corridor_width_px — runs of clear columns shorter than this are
 *                        discarded. Set to at least the drone's body width in
 *                        pixels so we never steer toward a gap too narrow to
 *                        fit through. Range [10, 200].
 */
extern float clear_frac;
extern int   min_corridor_width_px;

/* ── Function declarations ────────────────────────────────────────────────── */

/*
 * greenest_section_column
 * Returns 1-based index of the greenest section out of ns sections.
 */
int greenest_section_column(const float gb[], int w, int ns);

/*
 * greenest_pixel
 * Delegates to widest_corridor_centre() internally.
 * Returns -1 if no corridor wide enough to fly through exists.
 */
int greenest_pixel(const float gb[], int w);

/*
 * widest_corridor_centre
 * Returns the centre column of the widest contiguous run of clear columns.
 * A column is clear if gb[col] < clear_frac * h.
 * Runs narrower than min_corridor_width_px are discarded.
 * Returns -1 if no qualifying corridor is found — the caller should keep
 * rotating rather than committing to a direction.
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
 * Returns column index (0 to w-1), or -1 if no passable corridor exists.
 */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                 const struct obstacle_region_t po[], uint8_t np,
                 const float gb[], int w, int h,
                 int obs_bias, int plant_bias);

/*
 * motion_logic_normalised
 * Same as motion_logic but bias is a fraction of each obstacle/plant width.
 * Returns column index (0 to w-1), or -1 if no passable corridor exists.
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