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

/*
 * @file modules/computer_vision/cv_detect_object.h
 * Assumes the object consists of a continuous color and checks
 * if you are over the defined object or not
 */

// Own header
#include "modules/computer_vision/team10_ground_detection.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"

// Dependencies
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

// Libraries
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

// declare mutex variable
static pthread_mutex_t mutex;

// leave this unchanged (i'm scared)
#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0 ///< Default FPS (zero means run at camera fps)
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0 ///< Default FPS (zero means run at camera fps)
#endif

// ======================================================================================================
// NOTE: the variables MAX_IMAGE_WIDTH, MAX_OBSTACLE_REGIONS, etc. are defined inside get_obstacle_info.h
// ======================================================================================================

// define persistent variables (across frames)
static float    ground_baseline[MAX_IMAGE_WIDTH];
static int      baseline_inited       = 0;
static bool     obstacles_updated     = false;
static uint8_t  global_obstacle_count = 0;

// define global variable
struct obstacle_region_t global_obstacles[MAX_OBSTACLE_REGIONS];

// heavy function that runs as video callback
static struct image_t *detect_obstacles_from_ground(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    int                      boundary_rows[MAX_IMAGE_WIDTH];
    struct obstacle_region_t local_obstacles[MAX_OBSTACLE_REGIONS];

    // uint8_t obstacle_count = get_obstacle_info(
    //         img,
    //         ground_baseline,
    //         &baseline_inited,
    //         0.001f,   /* oa_color_count_frac */
    //         5,       /* median_ksize        */
    //         20,      /* min_width           */
    //         local_obstacles,
    //         boundary_rows,   /* int[MAX_IMAGE_WIDTH], local to the callback */
    //         NULL,            /* ground_found_out, or pass a local int if you need it */
    //         NULL             /* green_frac_out,   or pass a local float if you need it */
    // );

    uint8_t obstacle_count = get_obstacle_info(
            img,
            ground_baseline,
            &baseline_inited,
            0.001f,          /* oa_color_count_frac */
            5,               /* median_ksize        */
            20,              /* min_width           */
            5,               /* NEW: min_ground_pixels */
            5,               /* NEW: max_gap           */
            3,               /* NEW: smooth_kernel     */
            5,               /* NEW: max_col_gap       */
            1,               /* NEW: use_sim           */
            local_obstacles,
            boundary_rows,   /* int[MAX_IMAGE_WIDTH], local to the callback */
            NULL,            /* ground_found_out, or pass a local int if you need it */
            NULL             /* green_frac_out,   or pass a local float if you need it */
    );

    // lock mutex and modify global_obstacles
    pthread_mutex_lock(&mutex);
    memcpy(global_obstacles, local_obstacles, MAX_OBSTACLE_REGIONS * sizeof(struct obstacle_region_t));
    global_obstacle_count = obstacle_count;
    obstacles_updated = true;
    
    // print info about current obstacles
    for (int i = 0; i < obstacle_count; i++) {
        printf("Obstacle %d: left=%d width=%d\n",
               i, global_obstacles[i].start, global_obstacles[i].width);
    }
    pthread_mutex_unlock(&mutex);

    return img;
}

// Remember to change function name in XML file
void ground_detection_init(void)
{
  memset(global_obstacles, 0, MAX_OBSTACLE_REGIONS * sizeof(struct obstacle_region_t));
  pthread_mutex_init(&mutex, NULL);
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, detect_obstacles_from_ground, COLOR_OBJECT_DETECTOR_FPS1, 0);
}

// Remember to change function name in XML file
void ground_detection_periodic(void)
{
  struct obstacle_region_t  local_obstacles[MAX_OBSTACLE_REGIONS];
  uint8_t                   obstacle_count;

  pthread_mutex_lock(&mutex);

  // if nothing new, don't even copy the global variable into local
  if (!obstacles_updated) {
        pthread_mutex_unlock(&mutex);
        return;
  }

  // create a local copy of global variables
  obstacle_count = global_obstacle_count;
  memcpy(local_obstacles, global_obstacles, MAX_OBSTACLE_REGIONS * sizeof(struct obstacle_region_t));

  // reset updated state
  obstacles_updated = false;
  pthread_mutex_unlock(&mutex);

  // ABI function:      42
  // ABI message ID:    1
  AbiSendMsgTEAM10_GROUND_DETECTION(TEAM10_GROUND_DETECTION_ID, local_obstacles, obstacle_count);
  
}

// /* 
//  * FOR NOW -> USE THE ORANGE_AVOIDER FUNCTIONS 
//  *
//  * THIS FUNCTION ACTUALLY GETS MESSAGED BACK TO TEAM10_AUTOPILOT_CONTROL.C
//  */
// void color_object_detector_periodic(void)
// {
//     AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, 0, 0, 0, 0, 0, 0);

// }