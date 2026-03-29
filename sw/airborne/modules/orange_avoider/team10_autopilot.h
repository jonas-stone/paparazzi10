/*
 * team10_custom_autopilot.h
 *
 * Public interface for the Team 10 custom ground-baseline obstacle avoidance autopilot.
 *
 * This module is an improved version of team10_autopilot that uses
 * motion_logic_normalised() to intelligently choose which direction to rotate
 * when an obstacle is found, rather than picking a random direction.
 * It reads obstacle, plant, and ground boundary data from the ground detection
 * module via ABI, then drives a four-state navigation state machine.
 *
 * The three tunable globals below are exposed as GCS sliders so they can
 * be adjusted in flight without recompiling.
 */

#ifndef TEAM10_AUTOPILOT_H
#define TEAM10_AUTOPILOT_H

/* Fraction of image width that ground-touching obstacles must cover before
 * the confidence counter starts decrementing. Lower = more cautious (reacts
 * to smaller obstacles sooner). Default: 0.2 (20% of image width). */
extern float obstacle_width_threshold;

/* Maximum distance the GOAL waypoint can be pushed forward in one step [m].
 * The actual step is scaled down by the confidence counter so the drone
 * moves cautiously after finding a clear heading. Default: 2.25 m. */
extern float maxDistance;

/* Overall speed multiplier applied on top of the confidence-based distance
 * scaling. Default: 0.5. Lower values make the drone move more slowly overall. */
extern float speed_multiplier;

/* ── Module entry points ──────────────────────────────────────────────────── */

/* Called once at startup. Seeds the random number generator, runs the
 * corridor logic once to set an initial sensible turn direction, and
 * registers the ABI callback so the module starts receiving detection data. */
extern void ground_obstacle_avoidance_init(void);

/* Called at 10 Hz while the drone is in flight. Updates the confidence
 * counter based on total obstacle width, then runs the navigation state
 * machine to move or rotate the drone. */
extern void ground_obstacle_avoidance_periodic(void);

#endif /* TEAM10_AUTOPILOT_H */