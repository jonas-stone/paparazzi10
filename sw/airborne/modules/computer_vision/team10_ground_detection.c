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
 * @file modules/computer_vision/cv_detect_object.h
 * Assumes the object consists of a continuous color and checks
 * if you are over the defined object or not
 */

// Own header
#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include "modules/computer_vision/team10_ground_detection.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

#define PRINT(string,...) fprintf(stderr, "[object_detector->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if OBJECT_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0 ///< Default FPS (zero means run at camera fps)
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0 ///< Default FPS (zero means run at camera fps)
#endif

// MAX. number of obstacles to report (must match the size of the output array in get_obstacle_info)
#define OBSTACLE_ALPHA          0.6f
#define OBSTACLE_THRESHOLD      50.0f
#define NO_GROUND_BASELINE      220.0f
#define MAX_OBSTACLE_REGIONS    7     /* max regions we can return    */
#define MAX_IMAGE_WIDTH         520   /* adjust to your camera width  */

// Filter Settings
uint8_t cod_lum_min1 = 0;
uint8_t cod_lum_max1 = 0;
uint8_t cod_cb_min1 = 0;
uint8_t cod_cb_max1 = 0;
uint8_t cod_cr_min1 = 0;
uint8_t cod_cr_max1 = 0;

uint8_t cod_lum_min2 = 0;
uint8_t cod_lum_max2 = 0;
uint8_t cod_cb_min2 = 0;
uint8_t cod_cb_max2 = 0;
uint8_t cod_cr_min2 = 0;
uint8_t cod_cr_max2 = 0;

bool cod_draw1 = false;
bool cod_draw2 = false;

// define global variables
struct obstacle_t {
    int left;   ///< leftmost column in original (unflipped) image coordinates
    int width;  ///< width in columns
};

struct ground_debug_t {
    int     ground_found;   ///< 1 = "GROUND FOUND", 0 = "NO GROUND"
    float   green_frac;     ///< fraction of pixels classified as ground
    /* boundary_rows and obstacle_regions_raw are owned by the caller --
       pass the same arrays you supply to get_obstacle_info()            */
};

// Global variables
struct obstacle_t     global_obstacles[MAX_OBSTACLE_REGIONS];
int                   boundary_rows[MAX_IMAGE_WIDTH];
struct ground_debug_t debug;


/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
int get_obstacle_info(struct image_t           *input,
                      float                    *ground_baseline,
                      int                      *baseline_inited,
                      float                     oa_color_count_frac,
                      int                       median_ksize,
                      int                       min_width,
                      int                      *boundary_rows_out,
                      struct ground_debug_t    *debug_out)
{
    int W = input->w;
    int H = input->h;

    /* ---- Step 1: ground mask ---- */
    struct image_t mask;
    image_create(&mask, W, H, IMAGE_GRAYSCALE);

    float green_frac  = 0.0f;
    int   apply_median = (median_ksize >= 3 && median_ksize % 2 == 1);
    int   ground_found = detect_green_ground_ml(input, &mask,
                                                oa_color_count_frac,
                                                apply_median,
                                                &green_frac);

    if (debug_out != NULL) {
        debug_out->ground_found = ground_found;
        debug_out->green_frac   = green_frac;
    }

    if (!ground_found) {
        /* no ground: leave baseline unchanged, return zero obstacles */
        image_free(&mask);
        return 0;
    }

    /* ---- Step 2: flip mask horizontally (mask[:, ::-1]) ---- */
    struct image_t mask_flipped;
    image_create(&mask_flipped, W, H, IMAGE_GRAYSCALE);
    flip_horizontal(&mask, &mask_flipped);
    image_free(&mask); /* no longer needed */

    /* ---- Step 3: find ground boundary ---- */
    find_ground_boundary(&mask_flipped,
                         boundary_rows_out,
                         /* min_ground_pixels= */ 5,
                         /* max_gap=           */ 10,
                         /* smooth_kernel=     */ 5);
    image_free(&mask_flipped);

    /* ---- Step 4: update baseline and detect obstacle regions ---- */
    struct obstacle_region_t raw_regions[MAX_OBSTACLE_REGIONS];
    int n_regions = update_and_detect(
                        (const uint16_t *)boundary_rows_out,
                        W, H,
                        ground_baseline,
                        baseline_inited,
                        min_width,
                        raw_regions);

    /* ---- Step 5: convert from flipped coords to image coords ----
     *
     * In the flipped mask a column index c corresponds to column
     * (W - 1 - c) in the original image.  An obstacle region [s, e]
     * in flipped coords therefore spans:
     *
     *   original right edge : W - 1 - s
     *   original left  edge : W - 1 - e   ← this is "left" in left→right order
     *   width               : e - s + 1   (unchanged)
     */
    
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < n_regions; i++) {
        global_obstacles[i].left  = W - 1 - (int)raw_regions[i].end;
        global_obstacles[i].width = (int)raw_regions[i].width;
    }
    pthread_mutex_unlock(&mutex);

    return n_regions;
}

void color_object_detector_init(void)
{

  memset(global_obstacles, 0, MAX_OBSTACLE_REGIONS*sizeof(struct color_object_t));
  pthread_mutex_init(&mutex, NULL);

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, get_obstacle_info, COLOR_OBJECT_DETECTOR_FPS1, 0);

}


void color_object_detector_periodic(void)
{
  static struct color_object_t local_filters[2];
  pthread_mutex_lock(&mutex);
  memcpy(local_filters, global_filters, 2*sizeof(struct color_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_filters[0].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, local_filters[0].x_c, local_filters[0].y_c,
        0, 0, local_filters[0].color_count, 0);
    local_filters[0].updated = false;
  }
  if(local_filters[1].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, local_filters[1].x_c, local_filters[1].y_c,
        0, 0, local_filters[1].color_count, 1);
    local_filters[1].updated = false;
  }
}

/* --- called each frame --- */
void vision_periodic(void)
{
  // create a local immutable copy of global_obstacles
  static struct obstacle_t local_obstacles;
  pthread_mutex_lock(&mutex);
  memcpy(local_obstacles, global_obstacles, MAX_OBSTACLE_REGIONS*sizeof(struct obstacle_t));
  pthread_mutex_unlock(&mutex);

  // send local_obstacles over to autopilot code
  AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, local_obstacles, 1);

  for (int i = 0; i < n; i++) {
    printf("Obstacle %d: left=%d width=%d\n",
          i, obstacles[i].left, obstacles[i].width);
  }
}