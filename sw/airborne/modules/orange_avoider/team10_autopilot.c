/**
 * Autopilot state machine for ground-baseline obstacle avoidance (Team 10).
 *
 * This module receives obstacle regions, plant regions, and ground boundary
 * data from the ground detection module (team10_ground_detection.c) via ABI
 * messaging, and uses it to steer the drone away from obstacles.
 *
 * The detection pipeline measures the total width of obstacle regions found in
 * the current frame. If that combined width exceeds a fraction of the image
 * width (obstacle_width_threshold), the path is considered blocked and the
 * drone stops and rotates to find a clearer heading.
 *
 * The drone operates in one of four states:
 *   SAFE                  — path looks clear; move the waypoints forward.
 *   OBSTACLE_FOUND        — something is blocking the way; stop and rotate.
 *   SEARCH_FOR_SAFE_HEADING — rotating until obstacles shrink below threshold.
 *   OUT_OF_BOUNDS         — target waypoint left the allowed flight zone;
 *                           rotate back inward.
 *
 * Forward speed scales with obstacle_free_confidence so the drone moves slowly
 * when it has only just found a clear heading and faster once it has had
 * several consecutive clean frames.
 */

 // Team 10 inclusions
#include "modules/orange_avoider/team10_autopilot.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"

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

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// Forward declarations of helper functions defined at the bottom of this file
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
float speed_multiplier = 1.0f;  // reserved for future speed scaling, currently unused

/* ── Navigation state machine ─────────────────────────────────────────────── */
enum navigation_state_t {
  SAFE,                    /* path is clear — move the waypoints forward       */
  OBSTACLE_FOUND,          /* something is blocking us — stop and rotate       */
  SEARCH_FOR_SAFE_HEADING, /* rotating until the scene looks clear             */
  OUT_OF_BOUNDS            /* waypoint left the allowed flight zone            */
};

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
float heading_increment = 5.f;          // heading angle increment [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]

/* ── Module-level state updated by the ABI callback each frame ────────────── */
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];  // detected obstacle regions this frame
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];        // detected plant regions this frame
static int16_t                  boundary_rows[MAX_IMAGE_HEIGHT];  // ground boundary column per row (integer)
static float boundary_rows_f[MAX_IMAGE_HEIGHT];                   // same boundary as floats for logic functions
static uint8_t                  obstacle_count = 0;               // number of obstacles detected
static uint8_t                  plant_count    = 0;               // number of plants detected
static uint16_t                 boundary_len   = 0;               // number of valid entries in boundary_rows
uint16_t  total_obstacle_width  = 0;                              // sum of all obstacle widths in pixels this frame

// Fraction of image width that obstacle columns must exceed before we react.
// Lower values make the drone more cautious (reacts to smaller obstacles).
float obstacle_width_threshold = 0.2f;

// Number of consecutive clean frames required before we are confident the path is clear
const int16_t max_trajectory_confidence = 5;

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

/* Called automatically by Paparazzi whenever the ground detection module sends
   new data. Copies all incoming obstacle, plant, and boundary information into
   our local arrays so the periodic function can read them safely without
   accessing the detection thread's memory directly. */
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

    // Cap boundary length to the maximum buffer size to prevent overflow
    boundary_len = in_bl;
    if (boundary_len > MAX_IMAGE_HEIGHT) boundary_len = MAX_IMAGE_HEIGHT;
    memcpy(boundary_rows, in_br, boundary_len * sizeof(int16_t));

    // Also store the boundary as floats so logic functions can use them directly
    for (uint16_t i = 0; i < boundary_len; i++)
    boundary_rows_f[i] = (float)in_br[i];
}

/*
 * Initialisation function, random seed and heading_increment.
 * Seeds the random number generator so turn directions differ each flight,
 * picks an initial random turn direction, and registers the ABI callback
 * so we start receiving detection data.
 */
void ground_obstacle_avoidance_init(void) {
  srand(time(NULL));
  chooseRandomIncrementAvoidance();
  AbiBindMsgTEAM10_GROUND_DETECTION(TEAM10_GROUND_DETECTION_ID, &ground_detection_ev, ground_detection_callback);
}

/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading.
 * Called at 10 Hz while the drone is in flight.
 *
 * Each cycle:
 *   1. Print the current obstacle width fraction and threshold for debugging.
 *   2. Increment confidence if obstacles are small enough, otherwise decay it.
 *   3. Scale forward step size by confidence — move slowly until we are sure.
 *   4. Execute the current state machine state.
 */
void ground_obstacle_avoidance_periodic(void)
{
  // only evaluate our state machine if we are flying
  if(!autopilot_in_flight()){
    return;
  }

  // print stuff to terminal
  printf("total obstacle width: %.2f\nthreshold (fraction): %.2f\nthreshold (total):    %.2f\n",
       total_obstacle_width / (float)MAX_IMAGE_WIDTH,
       obstacle_width_threshold,
       obstacle_width_threshold * MAX_IMAGE_WIDTH);

  // Increment confidence when obstacles are narrow enough, decay faster when they are not.
  // Decaying by 2 means a single bad frame cancels two good ones, keeping the drone cautious.
  if (total_obstacle_width < obstacle_width_threshold * MAX_IMAGE_WIDTH) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }

  // bound obstacle_free_confidence
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  // Scale forward distance by confidence — start slow, build up as we accumulate clean frames
  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  switch (navigation_state){
    case SAFE:
      // Push TRAJECTORY further ahead than GOAL so the flight controller has
      // time to react before the drone reaches the target
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        // TRAJECTORY left the allowed zone — need to turn back inward
        navigation_state = OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence == 0){
        // Confidence dropped to zero — something is blocking the path
        navigation_state = OBSTACLE_FOUND;
      } else {
        // Path still looks clear — advance the GOAL waypoint too
        moveWaypointForward(WP_GOAL, moveDistance);
      }

      break;
    case OBSTACLE_FOUND:
      // Pull both waypoints back to the current position to stop in place,
      // pick a new random search direction, and start rotating
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // randomly select new search direction
      chooseRandomIncrementAvoidance();

      navigation_state = SEARCH_FOR_SAFE_HEADING;

      break;
    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(heading_increment);

      // make sure we have a couple of good readings before declaring the way safe
      if (obstacle_free_confidence >= 2){
        navigation_state = SAFE;
      }
      break;
    case OUT_OF_BOUNDS:
      // Rotate and probe with TRAJECTORY until it lands back inside the zone,
      // then search for a safe heading before moving forward again
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        // add offset to head back into arena
        increase_nav_heading(heading_increment);

        // reset safe counter
        obstacle_free_confidence = 0;

        // ensure direction is safe before continuing
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;
    default:
      break;
  }
  return;
}

/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 * Converts degrees to radians, adds to the current psi angle, normalises to
 * [-pi, pi] so the flight controller never sees a discontinuity, then sets it.
 * Positive increment = clockwise, negative = counter-clockwise.
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
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates.
 * Convenience wrapper: computes the target ENU position then calls moveWaypoint().
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading.
 * Uses the current heading (psi) to decompose the displacement into East (x)
 * and North (y) components in Paparazzi's fixed-point ENU format.
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
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'.
 * Thin wrapper around Paparazzi's waypoint_move_xy_i().
 */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
 * Sets the variable 'heading_increment' randomly positive/negative.
 * Randomly picking CW or CCW prevents the drone from always circling the same
 * way, which could cause it to spin indefinitely in a tight corner.
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