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
#include "modules/orange_avoider/team10_custom_autopilot_withGateDet.h"
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

// flight plan inclusions
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE FALSE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointToImageColumn(uint8_t waypoint, int image_col, float forward_distance, float lateral_range_m);
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static uint8_t chooseWiseIncrementAvoidance(int safe_col);

#ifndef TEAM10_GATE_DETECTION_ID
#define TEAM10_GATE_DETECTION_ID ABI_BROADCAST
#endif

#ifndef TEAM10_GROUND_DETECTION_ID
#define TEAM10_GROUND_DETECTION_ID ABI_BROADCAST
#endif

enum NavigationState {
  GO,
  ROTATE,
  OBSTACLE_FOUND,
  OUT_OF_BOUNDS
};

enum ObjectiveLocation {
  LEFT,
  RIGHT,
  CENTERLINE
};

// obstacle global variables
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];
static uint16_t boundary_rows[MAX_IMAGE_HEIGHT];
static float    boundary_rows_f[MAX_IMAGE_HEIGHT];
static uint8_t  obstacle_count = 0;
static uint8_t  plant_count    = 0;
static uint16_t boundary_len   = 0;
uint16_t total_obstacle_width  = 0;
int safe_col;

// navigation global variables
static enum ObjectiveLocation point_location;
static enum ObjectiveLocation target_location;
static enum NavigationState   nav_state;
float  heading_increment;      // degrees

// settings
float   speed_multiplier     = 1;
float   maxDistance          = 2.5;    // meters
uint8_t centerline_tolerance = 0.1 * MAX_IMAGE_WIDTH;
float   heading_increment_degrees_setting     = 1;
uint8_t locked_rotate_cooldown_frames_setting = 10;
uint8_t locked_go_cooldown_frames_setting     = 40;
float   obstacle_width_threshold  = 0.3f;
uint8_t max_trajectory_confidence = 5;

// counters
uint8_t locked_rotate_cooldown;
uint8_t locked_go_cooldown;
uint8_t obstacle_found_countdown;

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */

// ABI event declaration (used to bind to callback function)
static abi_event ground_detection_ev;
static abi_event gate_detection_ev;

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

    // update for obstacle detection
    for (uint8_t i = 0; i < in_oc; i++) {
      total_obstacle_width += in_obs[i].width + in_plants[i].width;
    }
}

uint8_t gate_seen;
int     gate_center_col;
static void gate_detection_callback(
    uint8_t __attribute__((unused)) sender_id,
    uint8_t in_gate_detected,
    int     in_gate_center_x)
{
    gate_seen = in_gate_detected;
    gate_center_col = (int16_t)in_gate_center_x;
}

/* Initialization Function */
void ground_obstacle_avoidance_init(void) {

  // initialize locations + nav. state
  nav_state       = ROTATE;
  target_location = RIGHT;
  point_location  = CENTERLINE;

  // initialize necessary obstacle variables
  total_obstacle_width = 0;
  obstacle_found_countdown = 0; 

  // srand(time(NULL));
  // int safe_col = motion_logic_normalised(obstacles, obstacle_count,
  //                                        plants, plant_count,
  //                                        boundary_rows_f, boundary_len, 
  //                                        MAX_IMAGE_HEIGHT,
  //                                        DEFAULT_OBS_BIAS_FRAC,
  //                                        DEFAULT_PLANT_BIAS_FRAC);
  // chooseWiseIncrementAvoidance(safe_col);

  AbiBindMsgTEAM10_GROUND_DETECTION(TEAM10_GROUND_DETECTION_ID, &ground_detection_ev, ground_detection_callback);
  AbiBindMsgTEAM10_GATE_DETECTION(TEAM10_GATE_DETECTION_ID, &gate_detection_ev, gate_detection_callback);
}


void ground_obstacle_avoidance_periodic(void)
{
  // only evaluate our state machine if we are flying
  if(!autopilot_in_flight()){
    return;
  }

  int   best_col;
  float confidence;
  best_col = motion_logic_normalised(obstacles, obstacle_count,
                                     plants, plant_count,
                                     boundary_rows_f, boundary_len, 
                                     MAX_IMAGE_HEIGHT,
                                     DEFAULT_OBS_BIAS_FRAC,
                                     DEFAULT_PLANT_BIAS_FRAC);

  // map pixel to ObjectiveLocation
  if (best_col < (MAX_IMAGE_WIDTH/2) - centerline_tolerance) {
    point_location    = LEFT;
    heading_increment = -heading_increment_degrees_setting;
  } else if (best_col > (MAX_IMAGE_WIDTH/2) + centerline_tolerance) {
    point_location    = RIGHT;
    heading_increment = +heading_increment_degrees_setting;
  } else {
    point_location    = CENTERLINE;
    heading_increment = +heading_increment_degrees_setting; // dummy setting
  }

  if (total_obstacle_width > obstacle_width_threshold) {
      obstacle_found_countdown += 1;
    }

  // obstacles > threshold for 5 consecutive frames
  if (obstacle_found_countdown == 5) {
    nav_state = OBSTACLE_FOUND;
  }
  
  // state machine
  switch (nav_state) 
  {
  case ROTATE:
    if (locked_rotate_cooldown != 0) {
      locked_rotate_cooldown -= 1;
      increase_nav_heading(heading_increment);
    }
    if (point_location == CENTERLINE) {
      nav_state = GO;
      locked_go_cooldown = locked_go_cooldown_frames_setting;
    } else {
      increase_nav_heading(heading_increment);
    }
    break;
  
  case GO:
    if (locked_go_cooldown == locked_go_cooldown_frames_setting - 1) {
      printf("=====================================\n");
    }
    // keep updating waypoint position every frame, 
    // because the drone computes its relative position
    // w.r.t . the waypoint to compute its speed.
    moveWaypointForward(WP_TRAJECTORY, maxDistance);  

    // check if out of bounds
    if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
      nav_state = OUT_OF_BOUNDS;
    } else { // not OOB, actually move goal
      moveWaypointForward(WP_GOAL, maxDistance);
    }

    if (locked_go_cooldown != 0) { 
      locked_go_cooldown -= 1;
      printf("Going...\n");
      break;
    }
    if (locked_go_cooldown == 0) {
      nav_state = ROTATE;
      locked_rotate_cooldown = locked_rotate_cooldown_frames_setting;
      target_location = point_location;
      printf("GO finished. Setting new target.\n");
    }
    break;
  
  case OBSTACLE_FOUND:
    waypoint_move_here_2d(WP_GOAL);
    waypoint_move_here_2d(WP_TRAJECTORY);
    printf("Obstacle found.\n");
    nav_state = ROTATE;
    obstacle_found_countdown = 0;
    break;

  case OUT_OF_BOUNDS:
    increase_nav_heading(heading_increment);
    moveWaypointForward(WP_TRAJECTORY, 1.5f);

    if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);
        nav_state = ROTATE;
    }
    break;
    
  default: break;
  }

  return;
}

/* Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function. */
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

/* Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/* Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading */
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

/* Sets waypoint 'waypoint' to the coordinates of 'new_coor' */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/* Sets the variable 'heading_increment' randomly positive/negative */
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

uint8_t chooseWiseIncrementAvoidance(int safe_direction)
{
  if (safe_direction > MAX_IMAGE_WIDTH / 2) {
    heading_increment = 5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  } else {
    heading_increment = -5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  }
  return false;
}

static float clampf_local(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static uint8_t moveWaypointToImageColumn(uint8_t waypoint, int image_col, float forward_distance, float lateral_range_m) {
  struct EnuCoor_i new_coor;
  float heading = stateGetNedToBodyEulers_f()->psi;
  float image_center = MAX_IMAGE_WIDTH / 2.0f;

  float normalized = ((float)image_col - image_center) / image_center;
  normalized = clampf_local(normalized, -1.0f, 1.0f);

  float lateral = normalized * lateral_range_m;
  float forward = forward_distance;

  float dx = sinf(heading) * forward + cosf(heading) * lateral;
  float dy = cosf(heading) * forward - sinf(heading) * lateral;

  new_coor.x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(dx);
  new_coor.y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(dy);

  moveWaypoint(waypoint, &new_coor);
  return false;
}