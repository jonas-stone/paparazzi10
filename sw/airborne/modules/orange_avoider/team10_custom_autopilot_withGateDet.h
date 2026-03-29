#ifndef TEAM10_CUSTOM_AUTOPILOT_WITHGATEDET_H
#define TEAM10_CUSTOM_AUTOPILOT_WITHGATEDET_H

/* ── Tunable parameters (exposed as GCS sliders via module XML) ───────────── */
/* These are non-static globals defined in team10_custom_autopilot_withGateDet.c.
   The GCS slider system writes to them by name at runtime.                    */
extern float    obstacle_width_threshold;  /* fraction of image width that obstacles must cover before confidence decays */
extern float    obs_bias_frac;             /* safety margin around each obstacle, as a fraction of its width            */
extern float    plant_bias_frac;           /* safety margin around each plant, as a fraction of its width               */
extern int      wp_update_period_ticks;    /* number of 10 Hz ticks between WP_GOAL advances in SAFE state              */
extern float    maxDistance;               /* maximum waypoint displacement [m]                                          */
extern float    speed_multiplier;          /* overall speed scaling factor                                               */
extern float    heading_increment_setting; /* rotation step size when searching for a safe heading [deg]                 */

/* Total pixel width of all ground-touching obstacles in the current frame.
   Drives the confidence counter — exposed so other modules can read it.       */
extern uint16_t total_obstacle_width;

/* obstacle_free_confidence and heading_increment are internal to the state
   machine and not exposed as GCS sliders, but declared here so other modules
   can read the current confidence level if needed.                             */
extern int16_t  obstacle_free_confidence;
extern float    heading_increment;

/* NOTE: clear_frac and min_corridor_width_px are NOT declared here.
   They are defined in team10_logic_withGateDet.c and declared as extern in
   team10_logic_withGateDet.h. The GCS sliders for those variables must point
   to modules/computer_vision/team10_logic_withGateDet, not to this file.     */

/* ── Module entry points ──────────────────────────────────────────────────── */
extern void ground_obstacle_avoidance_init(void);
extern void ground_obstacle_avoidance_periodic(void);

#endif /* TEAM10_CUSTOM_AUTOPILOT_WITHGATEDET_H */