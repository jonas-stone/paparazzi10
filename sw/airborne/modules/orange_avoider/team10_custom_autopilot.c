/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 */

 // Team 10 inclusions
#include "modules/orange_avoider/team10_autopilot.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/team10_logic.h"

// Other inclusions
#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// flight plan inclusions
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t moveWaypointWithCorrection(uint8_t waypoint, float distanceMeters, float lateral_frac);        // <---
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
float heading_increment = 5.f;          // heading angle increment [deg]
float maxDistance = 0.5;//2.25;               // max waypoint displacement [m]
float speed_multiplier = 0.05;
// define script-level variables
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];
static int16_t                  boundary_rows[MAX_IMAGE_HEIGHT];
static float boundary_rows_f[MAX_IMAGE_HEIGHT];
static uint8_t                  obstacle_count = 0;
static uint8_t                  plant_count    = 0;
static uint16_t                 boundary_len   = 0;

// define threshold settings -> lower, drone is more scared
float obstacle_width_threshold = 0.2f;

const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */
#ifndef TEAM10_GROUND_DETECTION_ID
#define TEAM10_GROUND_DETECTION_ID ABI_BROADCAST
#endif

// ABI event declaration (used to bind to callback function)
static abi_event ground_detection_ev;

// ABI video callback function, gets the data from the computer vision code
static void ground_detection_callback(
    uint8_t __attribute__((unused)) sender_id,
    struct obstacle_region_t *in_obs,   uint8_t  in_oc,
    struct obstacle_region_t *in_plants, uint8_t  in_pc,
    int16_t                  *in_br,    uint16_t in_bl)
{
    obstacle_count = in_oc;
    memcpy(obstacles, in_obs, in_oc * sizeof(struct obstacle_region_t));

    plant_count = in_pc;
    memcpy(plants, in_plants, in_pc * sizeof(struct obstacle_region_t));

    boundary_len = in_bl;
    if (boundary_len > MAX_IMAGE_HEIGHT) boundary_len = MAX_IMAGE_HEIGHT;
    memcpy(boundary_rows, in_br, boundary_len * sizeof(int16_t));

    for (uint16_t i = 0; i < boundary_len; i++)
    boundary_rows_f[i] = (float)in_br[i];


}

/*
 * Initialisation function, random seed and heading_increment
 */
void ground_obstacle_avoidance_init(void) {
  srand(time(NULL));
  chooseRandomIncrementAvoidance();
  AbiBindMsgTEAM10_GROUND_DETECTION(TEAM10_GROUND_DETECTION_ID, &ground_detection_ev, ground_detection_callback);
}

/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
 */
void ground_obstacle_avoidance_periodic(void)
{
  if (!autopilot_in_flight()) return;

  /* ════════════════════════════════════════════════════════════════════════
   *  STEP 1 — Derive useful quantities from raw data
   * ════════════════════════════════════════════════════════════════════════ */

  // ── obstacle summary ────────────────────────────────────────────────────
  uint16_t total_obstacle_width = 0;
  float    widest_obs_center    = 0.5f;   // as fraction of image height [0=right, 1=left]
  int16_t  widest_obs_width     = 0;
  int16_t  tallest_obs_height   = 0;
  uint8_t touches_out_step1[MAX_OBSTACLE_REGIONS];
  uint8_t touch = 0;
  obstacle_touches_ground(obstacles, obstacle_count, MAX_IMAGE_HEIGHT, touches_out_step1, &touch);                    // whether any obstacle touches the ground (1) or not (0)

  for (uint8_t i = 0; i < obstacle_count; i++) {
    total_obstacle_width += obstacles[i].width;
    if (obstacles[i].width > widest_obs_width) {
        widest_obs_width  = obstacles[i].width;
        widest_obs_center = (obstacles[i].start + obstacles[i].width / 2.0f)
                            / (float)MAX_IMAGE_HEIGHT;
        tallest_obs_height = obstacles[i].baseline_height;
    }
  }

  float obstacle_frac = total_obstacle_width / (float)MAX_IMAGE_HEIGHT;

  for (uint8_t i = 0; i < obstacle_count; i++) {
    total_obstacle_width += obstacles[i].width;
  }

  // ── plant summary ───────────────────────────────────────────────────────
  uint16_t total_plant_width = 0;
  for (uint8_t i = 0; i < plant_count; i++) {
    total_plant_width += plants[i].width;
  }
  float plant_frac = total_plant_width / (float)MAX_IMAGE_HEIGHT;

  // ── boundary analysis (left half vs right half of view) ─────────────────
  // lower boundary value = more ground visible = safer
  // higher boundary value = ground occluded = obstacle
  float left_avg  = 0.0f;
  float right_avg = 0.0f;

  if (boundary_len > 0) {
    uint16_t mid = boundary_len / 2;
    for (uint16_t i = 0; i < mid; i++) {
      right_avg += boundary_rows[i];         // rows 0..mid = right side
    }
    for (uint16_t i = mid; i < boundary_len; i++) {
      left_avg += boundary_rows[i];          // rows mid..end = left side
    }
    right_avg /= (float)mid;
    left_avg  /= (float)(boundary_len - mid);
  }

  // higher average = more occluded = more dangerous on that side
  // so the SAFER side has the LOWER average
  float safer_direction = (left_avg < right_avg) ? -1.0f : 1.0f;
  // -1 = left is safer (turn left), +1 = right is safer (turn right)


  /* ════════════════════════════════════════════════════════════════════════
   *  STEP 2 — Confidence update (same logic, now using local variable)
   * ════════════════════════════════════════════════════════════════════════ */

  if (obstacle_frac < obstacle_width_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float moveDistance = maxDistance;  /* sempre alla velocità massima */
  bool if_plant_frac = (plant_frac > 0.1f);
  if (if_plant_frac || touch)
    moveDistance *= 0.3f;  /* rallenta vicino alle piante */


  /* ════════════════════════════════════════════════════════════════════════
   *  STEP 3 — State machine (your new logic goes here)
   * ════════════════════════════════════════════════════════════════════════ */

  switch (navigation_state) {
    case SAFE:
    moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

    if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;

    } else {
        uint8_t touches_out[MAX_OBSTACLE_REGIONS];
        obstacle_touches_ground(obstacles, obstacle_count,
                                MAX_IMAGE_HEIGHT, touches_out, &touch);

        if (touch) {
            navigation_state = OBSTACLE_FOUND;
        } else {
            int safe_col = motion_logic_normalised(obstacles, obstacle_count,
                                                   plants, plant_count,
                                                   boundary_rows_f,
                                                   boundary_len, MAX_IMAGE_HEIGHT,
                                                   DEFAULT_OBS_BIAS_FRAC,
                                                   DEFAULT_PLANT_BIAS_FRAC);
            float center      = boundary_len / 2.0f;
            float offset_frac = (center - safe_col) / center;
            float effective_distance = moveDistance;
            if (plant_frac > 0.1f)
                effective_distance *= 0.5f;
            moveWaypointWithCorrection(WP_GOAL, effective_distance, offset_frac);
        }
    }
    break;

    case OBSTACLE_FOUND:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // ── NEW: smart turn direction instead of random ──
      // use widest obstacle position OR boundary analysis
      if (widest_obs_width > 0) {
        // obstacle center < 0.5 means it's on the right → turn left
        heading_increment = (widest_obs_center < 0.5f) ? -5.0f : 5.0f;
      } else {
        // fall back to boundary: turn toward safer side
        heading_increment = safer_direction * 5.0f;
      }

      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
    {
        int safe_col = motion_logic_normalised(obstacles, obstacle_count,
                                       plants, plant_count,
                                       boundary_rows_f,
                                       boundary_len, MAX_IMAGE_HEIGHT,
                                       DEFAULT_OBS_BIAS_FRAC,
                                       DEFAULT_PLANT_BIAS_FRAC);
        int center    = boundary_len / 2;
        int tolerance = boundary_len / 7;  /* circa 1 sezione su 7 */

        if (abs(safe_col - center) <= tolerance) {
            /* Il punto sicuro è abbastanza centrato → torna a SAFE */
            navigation_state = SAFE;
        } else {
            /* Gira verso il punto sicuro */
            float offset_frac = (float)(center - safe_col) / (float)center;
            increase_nav_heading(offset_frac * 10.0f);
        }
    }
    break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;

    default:
      break;
  }
}

/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 */
uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

  // normalize heading to [-pi, pi]
  FLOAT_ANGLE_NORMALIZE(new_heading);

  // set heading, declared in firmwares/rotorcraft/navigation.h
  nav.heading = new_heading;

  VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
  return false;
}

/*
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
 */
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading  = stateGetNedToBodyEulers_f()->psi;

  // Now determine where to place the waypoint you want to go to
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,	
                POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
  return false;
}

/*
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
 */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
 * Only claude
 * Moves waypoint forward AND laterally based on safe column offset.
 * lateral_frac: -1.0 = full left, 0.0 = straight, +1.0 = full right
 */
uint8_t moveWaypointWithCorrection(uint8_t waypoint, float distanceMeters, float lateral_frac)
{
    struct EnuCoor_i new_coor;
    float heading = stateGetNedToBodyEulers_f()->psi;

    /* Forward component */
    float forward_x = sinf(heading) * distanceMeters;
    float forward_y = cosf(heading) * distanceMeters;

    /* Lateral component (perpendicular to heading) */
    float lateral_dist = lateral_frac * distanceMeters * 0.5f;
    float lateral_x = cosf(heading) * lateral_dist;
    float lateral_y = -sinf(heading) * lateral_dist;

    new_coor.x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(forward_x + lateral_x);
    new_coor.y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(forward_y + lateral_y);

    moveWaypoint(waypoint, &new_coor);
    return false;
}

/*
 * Sets the variable 'heading_increment' randomly positive/negative
 */
uint8_t chooseRandomIncrementAvoidance(void)
{
  // Randomly choose CW or CCW avoiding direction
  if (rand() % 2 == 0) {
    heading_increment = 5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  } else {
    heading_increment = -5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  }
  return false;
}

