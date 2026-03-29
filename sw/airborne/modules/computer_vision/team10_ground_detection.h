/*
 * team10_ground_detection.h
 *
 * Public interface for the ground detection module.
 *
 * This module runs on the camera thread and processes each frame to detect:
 *   - Obstacle regions (columns of the image where the floor is blocked)
 *   - Plant regions (potted plants detected separately from floor obstacles)
 *   - Gate presence and centre column (only when an obstacle is present)
 *
 * Results are published over the ABI message bus so the autopilot module
 * can read them without sharing memory directly with the camera thread.
 *
 * The two globals below (gate_detected, gate_center_x) are also written
 * directly and can be read by any module that includes this header after
 * ground_detection_periodic() has run.
 */

#ifndef TEAM10_GROUND_DETECTION_H
#define TEAM10_GROUND_DETECTION_H

#include <stdint.h>
#include <stdbool.h>

/* ── Gate detection result ────────────────────────────────────────────────── */

/* Set to 1 if a gate was found in the most recent frame, 0 otherwise.
   Only updated when at least one obstacle is present in the frame —
   gate detection does not run on clear frames. */
extern uint8_t gate_detected;

/* Pixel distance from the LEFT edge of the raw image to the gate centre.
   This corresponds to the distance from the TOP edge when the image is
   displayed upright after the 90° CCW rotation.
   Only meaningful when gate_detected == 1. */
extern int gate_center_x;

/* ── Module entry points ──────────────────────────────────────────────────── */

/* Called once at startup. Initialises internal state, sets up the mutex,
   and registers the camera callback so processing starts automatically. */
extern void ground_detection_init(void);

/* Called at 10 Hz by the Paparazzi scheduler. Reads the latest results from
   the camera thread (protected by mutex) and publishes them as ABI messages
   for the autopilot to consume. */
extern void ground_detection_periodic(void);

#endif /* TEAM10_GROUND_DETECTION_H */