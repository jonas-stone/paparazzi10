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
#include "team10_autopilot_RANSAC.h"
#include "modules/computer_vision/team10_RANSAC_adaptation.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/team10_ground_detection_RANSAC.h"

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

#define ORANGE_AVOIDER_VERBOSE FALSE
#define PRINT_RANSAC_BUCKET_INFO true

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};
// just to print the state
const char get_state_name(enum navigation_state_t s) {
    switch (s) {
        case SAFE:                      return "SAFE";
        case OBSTACLE_FOUND:            return "OBSTACLE FOUND";
        case SEARCH_FOR_SAFE_HEADING:   return "SEARCH FOR SAFE HEADING";
        case OUT_OF_BOUNDS:             return "OUT OF BOUNDS";
        default:                        return "UNKNOWN";
    }
}

// define global navigation variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
static float heading_confidence = 0;     // a measure of how certain we are that the way ahead is safe.
float heading_increment = 3.f;          // heading angle increment [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]

// define global obstacle variables
// static float max_confidence_bucket = (uint8_t)(NUMBER_VERTICAL_BUCKETS / 2); // <-- initialize as center bucket
struct column_bucket_t AP_buckets[NUMBER_VERTICAL_BUCKETS];
float  confidence_level[NUMBER_VERTICAL_BUCKETS];
// int    total_edge_pixel_count = 0;
float   AP_max_confidence_level;
uint8_t AP_max_confidence_bucket;


const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */
#ifndef TEAM10_RANSAC_DETECTION_ID
#define TEAM10_RANSAC_DETECTION_ID ABI_BROADCAST
#endif

// callback function, get info from "team10_ground_detection_RANSAC.c"
static abi_event RANSAC_detection_ev;
static void RANSAC_detection_callback(uint8_t __attribute__((unused)) sender_id, 
                                      float incoming_max_confidence_level, 
                                      float incoming_max_confidence_bucket,
                                      struct column_bucket_t *incoming_buckets,
                                      float *incoming_confidence_levels) {
  memcpy(confidence_level, incoming_confidence_levels, NUMBER_VERTICAL_BUCKETS * sizeof(float));
  memcpy(AP_buckets, incoming_buckets, NUMBER_VERTICAL_BUCKETS * sizeof(struct column_bucket_t));
  AP_max_confidence_level  = incoming_max_confidence_level;
  AP_max_confidence_bucket = (uint8_t) incoming_max_confidence_bucket;
}

/*
 * Initialisation function, random seed and heading_increment
 */
void ground_obstacle_avoidance_init(void) {
  srand(time(NULL));
  chooseRandomIncrementAvoidance();
  AbiBindMsgTEAM10_RANSAC_DETECTION(TEAM10_RANSAC_DETECTION_ID, &RANSAC_detection_ev, RANSAC_detection_callback);
}


void ground_obstacle_avoidance_periodic(void) {
  if (!autopilot_in_flight()) return;

  if (PRINT_RANSAC_BUCKET_INFO) {
    printf("---------------------------------------------------------\n");
    for (uint8_t b = 0; b < NUMBER_VERTICAL_BUCKETS; b++) {
      printf("[%d]", b);
      printf("BLACK above: %d, ", AP_buckets[b].black_above_ground_line);
      printf("BLACK below: %d, ", AP_buckets[b].black_below_ground_line);
      printf("WHITE above: %d, ", AP_buckets[b].white_above_ground_line);
      printf("WHITE below: %d, ", AP_buckets[b].white_below_ground_line);
      printf("Confidence: %.2f\n", confidence_level[b]);
    }
    printf("Max confidence bucket = %d\n", (uint8_t)(AP_max_confidence_bucket + 0.5f));
    printf("Confidence:           = %.2f\n", AP_max_confidence_level);
    printf("State: %s\n", get_state_name(navigation_state));
  }

  const uint8_t center_bucket = (uint8_t)(NUMBER_VERTICAL_BUCKETS / 2);
  uint8_t bucket_offset = (AP_max_confidence_bucket - center_bucket);
  printf("bucket offset: %d\n", bucket_offset);
  // If the best path is not centered, rotate toward it
  if (bucket_offset < -4 || bucket_offset > 4) {
    float bucket_increment = (AP_max_confidence_bucket < center_bucket) ? -heading_increment : heading_increment;
    increase_nav_heading(bucket_increment);
    // heading_confidence = 0; // reset confirmation counter while realigning
    // return;
  }

  float average_confidence = 0;
  for (uint8_t b = 0; b < NUMBER_VERTICAL_BUCKETS; b++) {
    average_confidence += confidence_level[b] / NUMBER_VERTICAL_BUCKETS;
  }

  // Accumulate or decay confidence across frames
  if (average_confidence >= 0.3f) {
    heading_confidence += AP_max_confidence_level;
  } else {
    heading_confidence = fmaxf(0.f, heading_confidence - 1.f);
  }

  // Cap it so it doesn't grow unboundedly
  Bound(heading_confidence, 0, max_trajectory_confidence);

  printf("heading_confidence = %.2f\n", heading_confidence);
  printf("average_confidence = %.2f\n", average_confidence);

  // If confidence is too low, stop and search for a safe heading
  if (AP_max_confidence_level < 0.05f) {
    waypoint_move_here_2d(WP_GOAL);
    waypoint_move_here_2d(WP_TRAJECTORY);
    chooseRandomIncrementAvoidance();
    heading_confidence = 0;
    navigation_state = SEARCH_FOR_SAFE_HEADING;
  }

  float moveDistance = fminf(maxDistance, 0.2f * heading_confidence);

  switch (navigation_state) {
    case SAFE:
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
      } else if (heading_confidence == 0) {
        navigation_state = OBSTACLE_FOUND;
      } else {
        moveWaypointForward(WP_GOAL, moveDistance);
      }
      break;

    case OBSTACLE_FOUND:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      chooseRandomIncrementAvoidance();
      heading_confidence = 0;
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(heading_increment);
      // Only transition to SAFE after several consecutive frames of good confidence
      if (heading_confidence >= 3.f) {
        navigation_state = SAFE;
      }
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);
      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);
        heading_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;

    default:
      break;
  }
  return;
}



// /*
//  * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
//  */
// void ground_obstacle_avoidance_periodic(void) {

//   if (!autopilot_in_flight()) return;


//   if (PRINT_RANSAC_BUCKET_INFO) {
//     printf("---------------------------------------------------------\n");
//     for (uint8_t b = 0; b < NUMBER_VERTICAL_BUCKETS; b++) {
//       printf("[%d]", b);
//       printf("BLACK above: %d, ", AP_buckets[b].black_above_ground_line);
//       printf("BLACK below: %d, ", AP_buckets[b].black_below_ground_line);
//       printf("WHITE above: %d, ", AP_buckets[b].white_above_ground_line);
//       printf("WHITE below: %d, ", AP_buckets[b].white_below_ground_line);
//       printf("Confidence: %.2f\n", confidence_level[b]);
//     }
//     printf("Max confidence bucket = %d\n", (uint8_t)(AP_max_confidence_bucket + 0.5f));
//     printf("Confidence:           = %.2f\n", AP_max_confidence_level);
//   }

//   // ======================= LEAVE LIKE THIS FOR NOW =======================
//   heading_confidence = 5 * AP_max_confidence_level;
//   float moveDistance = fminf(maxDistance, 0.2f * heading_confidence);

//   switch (navigation_state){
//     case SAFE:
//       // Move waypoint forward
//       moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
//       if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
//         navigation_state = OUT_OF_BOUNDS;
//       } else if (heading_confidence == 0){
//         navigation_state = OBSTACLE_FOUND;
//       } else {
//         moveWaypointForward(WP_GOAL, moveDistance);
//       }

//       break;
//     case OBSTACLE_FOUND:
//       // stop
//       waypoint_move_here_2d(WP_GOAL);
//       waypoint_move_here_2d(WP_TRAJECTORY);

//       // randomly select new search direction
//       chooseRandomIncrementAvoidance();

//       navigation_state = SEARCH_FOR_SAFE_HEADING;

//       break;
//     case SEARCH_FOR_SAFE_HEADING:
//       increase_nav_heading(heading_increment);

//       // make sure we have a couple of good readings before declaring the way safe
//       if (heading_confidence >= 2){
//         navigation_state = SAFE;
//       }
//       break;
//     case OUT_OF_BOUNDS:
//       increase_nav_heading(heading_increment);
//       moveWaypointForward(WP_TRAJECTORY, 1.5f);

//       if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
//         // add offset to head back into arena
//         increase_nav_heading(heading_increment);

//         // reset safe counter
//         heading_confidence = 0;

//         // ensure direction is safe before continuing
//         navigation_state = SEARCH_FOR_SAFE_HEADING;
//       }
//       break;
//     default:
//       break;
//   }
//   return;
// }

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

