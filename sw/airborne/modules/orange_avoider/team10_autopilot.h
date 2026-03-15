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

// settings
extern float oa_color_count_frac;

// functions (ORANGE AVOIDER)
extern void orange_avoider_init(void);
extern void orange_avoider_periodic(void);

// functions (TEAM 10 GROUND DETECTION)
extern void ground_detection_init(void)

#endif

