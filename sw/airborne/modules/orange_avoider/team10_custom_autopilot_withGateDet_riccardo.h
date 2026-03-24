/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */
#include <stdint.h>

#ifndef TEAM10_AUTOPILOT_H
#define TEAM10_AUTOPILOT_H

// GCS settings
extern float   speed_multiplier;
extern float   maxDistance;    // meters
extern uint8_t centerline_tolerance;
extern uint8_t heading_increment_degrees_setting;
extern uint8_t locked_rotate_cooldown_frames_setting;
extern uint8_t locked_go_cooldown_frames_setting ;
extern float   obstacle_width_threshold;
extern uint8_t max_trajectory_confidence;

// functions (TEAM 10 GROUND DETECTION)
extern void ground_obstacle_avoidance_init(void);
extern void ground_obstacle_avoidance_periodic(void);

#endif

