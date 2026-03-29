/*
 * team10_custom_autopilot_withGateDet_riccardo.h
 *
 * Public interface for Riccardo's gate-aware obstacle avoidance autopilot.
 *
 * This module uses a GO/ROTATE state machine rather than the confidence-counter
 * approach used in other autopilot variants. The drone alternates between
 * rotating to align with the best corridor and driving forward for a fixed
 * number of ticks. Gate detection data is also received but the gate approach
 * logic is not yet implemented in this version.
 *
 * All globals below are exposed as GCS sliders so they can be tuned in flight
 * without recompiling.
 */

#include <stdint.h>

#ifndef TEAM10_AUTOPILOT_H
#define TEAM10_AUTOPILOT_H

/* Overall speed multiplier applied on top of the distance calculation.
 * Default: 1.0 (no additional scaling). */
extern float   speed_multiplier;

/* Maximum distance the GOAL waypoint is pushed forward during the GO state [m].
 * Default: 1.5 m. */
extern float   maxDistance;

/* Number of pixels either side of the image centre within which the safe column
 * is considered "centred" — i.e. the drone is already pointing the right way
 * and can transition to GO without rotating first.
 * Default: 10% of image width. */
extern uint8_t centerline_tolerance;

/* Rotation step size applied each 10 Hz tick while in the ROTATE state [deg].
 * Default: 3°. */
extern uint8_t heading_increment_degrees_setting;

/* Number of 10 Hz ticks the drone spends rotating before re-evaluating.
 * Default: 30 ticks = 3 seconds. */
extern uint8_t locked_rotate_cooldown_frames_setting;

/* Number of 10 Hz ticks the drone drives forward before switching back to ROTATE.
 * Default: 30 ticks = 3 seconds. */
extern uint8_t locked_go_cooldown_frames_setting;

/* Fraction of image width that obstacles must cover before OBSTACLE_FOUND is
 * triggered. Lower = more cautious. Default: 0.3 (30% of image width). */
extern float   obstacle_width_threshold;

/* Number of consecutive frames where total_obstacle_width exceeds the threshold
 * before the drone transitions to OBSTACLE_FOUND. Prevents single noisy frames
 * from triggering a full stop. Default: 5. */
extern uint8_t max_trajectory_confidence;

/* ── Module entry points ──────────────────────────────────────────────────── */

/* Called once at startup. Initialises state machine variables, resets counters,
 * and registers both ABI callbacks (ground detection + gate detection). */
extern void ground_obstacle_avoidance_init(void);

/* Called at 10 Hz while the drone is in flight. Runs the corridor logic to find
 * the best column, maps it to a rotation direction, updates the obstacle counter,
 * and executes the current GO/ROTATE/OBSTACLE_FOUND/OUT_OF_BOUNDS state. */
extern void ground_obstacle_avoidance_periodic(void);

#endif /* TEAM10_AUTOPILOT_H */