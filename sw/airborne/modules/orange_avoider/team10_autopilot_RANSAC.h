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

#ifndef TEAM10_AUTOPILOT_H
#define TEAM10_AUTOPILOT_H

extern float maxDistance;

// functions (TEAM 10 GROUND DETECTION)
extern void ground_obstacle_avoidance_init(void);
extern void ground_obstacle_avoidance_periodic(void);

#endif

