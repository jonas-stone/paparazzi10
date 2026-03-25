/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/team10_custom_autopilot.h"
 * @author Roland Meertens / Team 10
 */

#ifndef TEAM10_AUTOPILOT_H
#define TEAM10_AUTOPILOT_H

/* ── Tunable parameters (exposed as GCS sliders via module XML) ───────────── */
extern float    obstacle_width_threshold;  /* fraction of image width, default 0.15  */
extern float    obs_bias_frac;             /* obstacle safety margin frac, default 0.50 */
extern float    plant_bias_frac;           /* plant safety margin frac,    default 0.75 */
extern int      wp_update_period_ticks;    /* WP_GOAL update interval (ticks), default 5 */
extern float    maxDistance;               /* max waypoint displacement [m], default 2.25 */
extern float    speed_multiplier;          /* overall speed multiplier, default 0.5   */
extern float    clear_frac;               /* corridor clear threshold, default 0.90  */
extern int      min_corridor_width_px;    /* minimum passable corridor (px), default 60 */
extern float    heading_increment_setting;

/* ── Module entry points ──────────────────────────────────────────────────── */
extern void ground_obstacle_avoidance_init(void);
extern void ground_obstacle_avoidance_periodic(void);

#endif /* TEAM10_AUTOPILOT_H */