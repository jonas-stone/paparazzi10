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

extern float obstacle_width_threshold;
extern float maxDistance;
extern float speed_multiplier;

extern float setting_heading_increment;
extern uint8_t locked_state_cooldown_frames;
extern float centerline_tolerance;

// functions (TEAM 10 GROUND DETECTION)
extern void ground_obstacle_avoidance_init(void);
extern void ground_obstacle_avoidance_periodic(void);

#endif

