/*
 * Copyright (C) 2019 Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/cv_detect_color_object.h
 * Assumes the color_object consists of a continuous color and checks
 * if you are over the defined color_object or not
 */

#ifndef TEAM10_GROUND_DETECTION_H
#define TEAM10_GROUND_DETECTION_H

#include <stdint.h>
#include <stdbool.h>

/* ── Gate detection result (read from any module after ground_detection_periodic) ── */
/* gate_detected : 1 if a gate was found in the most recent frame, 0 otherwise       */
/* gate_center_x : pixel distance from the LEFT edge of the raw image to the gate    */
/*                 centre (= distance from the TOP edge in the upright display view)  */
extern uint8_t gate_detected;
extern int     gate_center_x;

// Module functions
extern void ground_detection_init(void);
extern void ground_detection_periodic(void);

#endif /* COLOR_OBJECT_DETECTOR_CV_H */