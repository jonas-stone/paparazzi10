/*
 * team10_custom_autopilot.c
 *
 * Autopilot state machine for ground-baseline obstacle avoidance (Team 10).
 *
 * This is an improved version of team10_autopilot.c. The key difference is
 * that when an obstacle is found, the drone uses the corridor logic
 * (motion_logic_normalised) to work out which side of the image has the most
 * visible ground and rotates toward that side — instead of picking a random
 * direction. This is called "wise" avoidance.
 *
 * The module receives obstacle regions, plant regions, and the ground boundary
 * array from team10_ground_detection.c via ABI messaging, and drives a
 * four-state navigation state machine:
 *   SAFE                  — path looks clear; move the waypoints forward.
 *   OBSTACLE_FOUND        — something is blocking the way; stop and rotate
 *                           toward the corridor identified by the logic module.
 *   SEARCH_FOR_SAFE_HEADING — rotating until confidence recovers.
 *   OUT_OF_BOUNDS         — target waypoint left the allowed flight zone;
 *                           rotate back inward.
 *
 * Forward speed scales with obstacle_free_confidence so the drone moves slowly
 * after finding a new heading and builds up speed over consecutive clear frames.
 */

// Team 10 inclusions
#include "modules/orange_avoider/team10_custom_autopilot.h"
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
static uint8_t chooseWiseIncrementAvoidance(int safe_col);
float speed_multiplier = 0.5;  /* overall speed multiplier applied on top of confidence scaling */

/* ── Navigation state machine ─────────────────────────────────────────────── */
enum navigation_state_t {
  SAFE,                    /* path is clear — move the waypoints forward       */
  OBSTACLE_FOUND,          /* something is blocking — stop and rotate wisely   */
  SEARCH_FOR_SAFE_HEADING, /* rotating until the scene clears                  */
  OUT_OF_BOUNDS            /* waypoint left the allowed flight zone            */
};

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;   // accumulated evidence that the path is clear (+1 per clean frame, -2 per blocked frame)
float heading_increment = 5.f;          // heading angle increment [deg], set by chooseWiseIncrementAvoidance
float maxDistance = 2.25;               // max waypoint displacement [m]

/* ── Module-level state updated by the ABI callback each frame ────────────── */
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];  // obstacle regions from ground detection
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];        // plant regions from ground detection
static int16_t                  boundary_rows[MAX_IMAGE_HEIGHT];  // ground boundary column per row (integer)
static float                    boundary_rows_f[MAX_IMAGE_HEIGHT]; // same boundary as floats for logic functions
static uint8_t                  obstacle_count = 0;               // number of obstacles this frame
static uint8_t                  plant_count    = 0;               // number of plants this frame
static uint16_t                 boundary_len   = 0;               // number of valid boundary entries
uint16_t  total_obstacle_width  = 0;                              // sum of all obstacle widths in pixels

// Fraction of image width that obstacles must cover before the drone reacts.
// Lower = more cautious (reacts to smaller obstacles sooner).
float obstacle_width_threshold = 0.2f;

// Number of consecutive clean frames needed before the path is declared safe
const int16_t max_trajectory_confidence = 5;

/*
 * ABI messaging: the ground detection module runs in a separate camera thread
 * and publishes results via ABI messages. We register a callback here so that
 * every time new detection data is available, our local buffers are updated
 * automatically without directly sharing memory with the camera thread.
 */
#ifndef TEAM10_GROUND_DETECTION_ID
#define TEAM10_GROUND_DETECTION_ID ABI_BROADCAST
#endif

// ABI event handle used to track the registered callback
static abi_event ground_detection_ev;

/* Called automatically each time the ground detection module publishes new data.
   Copies obstacle regions, plant regions, and the ground boundary into our
   local arrays so the periodic function can safely read them. */
static void ground_detection_callback(
    uint8_t __attribute__((unused)) sender_id,
    struct obstacle_region_t *in_obs,    uint8_t  in_oc,
    struct obstacle_region_t *in_plants, uint8_t  in_pc,
    int16_t                  *in_br,     uint16_t in_bl)
{
    obstacle_count = in_oc;
    memcpy(obstacles, in_obs, in_oc * sizeof(struct obstacle_region_t));

    plant_count = in_pc;
    memcpy(plants, in_plants, in_pc * sizeof(struct obstacle_region_t));

    // Cap boundary length to avoid overflowing the fixed-size buffer
    boundary_len = in_bl;
    if (boundary_len > MAX_IMAGE_HEIGHT) boundary_len = MAX_IMAGE_HEIGHT;
    memcpy(boundary_rows, in_br, boundary_len * sizeof(int16_t));

    // Convert to float so the logic functions can use them without casting
    for (uint16_t i = 0; i < boundary_len; i++)
    boundary_rows_f[i] = (float)in_br[i];
}

/*
 * Module initialisation — called once at startup.
 * Runs the corridor logic once with empty data to set an initial turn direction,
 * then registers the ABI callback so detection data starts flowing in.
 */
void ground_obstacle_avoidance_init(void) {
  srand(time(NULL));
  // Run the logic once at startup so heading_increment is set to a sensible
  // value before the first obstacle is encountered. All arrays are empty here
  // so this will almost always return -1, in which case the heading increment
  // stays at its default of +5 degrees.
  int safe_col = motion_logic_normalised(obstacles, obstacle_count,
                                                   plants, plant_count,
                                                   boundary_rows_f,
                                                   boundary_len, MAX_IMAGE_HEIGHT,
                                                   DEFAULT_OBS_BIAS_FRAC,
                                                   DEFAULT_PLANT_BIAS_FRAC);
  chooseWiseIncrementAvoidance(safe_col);
  AbiBindMsgTEAM10_GROUND_DETECTION(TEAM10_GROUND_DETECTION_ID, &ground_detection_ev, ground_detection_callback);
}

/*
 * Module periodic function — called at 10 Hz while the drone is in flight.
 *
 * Each cycle:
 *   1. Print debug info about current obstacle width and threshold.
 *   2. Update obstacle_free_confidence (+1 if clear, -2 if blocked).
 *   3. Scale forward distance by confidence.
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

  // +1 each clean frame, -2 each blocked frame — asymmetry means one bad
  // frame undoes two good ones so the drone reacts quickly to new obstacles
  if (total_obstacle_width < obstacle_width_threshold * MAX_IMAGE_WIDTH) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }

  // bound obstacle_free_confidence
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  // Scale move distance by confidence so the drone starts slow after finding
  // a clear heading and speeds up as it accumulates clean frames
  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  switch (navigation_state){
    case SAFE:
      // Push TRAJECTORY ahead as a lookahead probe for the bounds check;
      // only advance GOAL (the target the flight controller chases) when clear
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        // TRAJECTORY left the allowed zone — turn back inward
        navigation_state = OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence == 0){
        // Confidence exhausted — something is blocking the path
        navigation_state = OBSTACLE_FOUND;
      } else {
        // Path still looks clear — advance the actual goal waypoint too
        moveWaypointForward(WP_GOAL, moveDistance);
      }

      break;
    case OBSTACLE_FOUND:
      // Stop in place, run the corridor logic to find which side is clearest,
      // set heading_increment toward that side, then start rotating
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // logically select new search direction
      int safe_col = motion_logic_normalised(obstacles, obstacle_count,
                                                   plants, plant_count,
                                                   boundary_rows_f,
                                                   boundary_len, MAX_IMAGE_HEIGHT,
                                                   DEFAULT_OBS_BIAS_FRAC,
                                                   DEFAULT_PLANT_BIAS_FRAC);
      chooseWiseIncrementAvoidance(safe_col);

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
      // then transition to SEARCH so the heading is verified before moving
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
 * Rotate the drone's heading by incrementDegrees.
 * Converts to radians, normalises to [-π, π], and sets nav.heading.
 * Positive = clockwise, negative = counter-clockwise.
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
 * Move a waypoint to a point distanceMeters ahead of the current position.
 * Convenience wrapper around calculateForwards() and moveWaypoint().
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Compute ENU coordinates for a point distanceMeters ahead along the current heading.
 * Uses sin/cos of the psi (yaw) Euler angle to decompose the displacement into
 * East (x) and North (y) components in Paparazzi's fixed-point format.
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
 * Move a named waypoint to the given ENU coordinates.
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
 * Fallback: pick a random clockwise or counter-clockwise rotation direction.
 * Not used in normal operation — the drone uses chooseWiseIncrementAvoidance
 * instead — but kept for situations where no corridor information is available.
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

/*
 * Set heading_increment based on which side of the image the safest corridor is.
 *
 * After the 90° CCW rotation applied by the camera pipeline:
 *   column 0        = drone's physical RIGHT
 *   column W-1      = drone's physical LEFT
 *   column W/2      = straight ahead
 *
 * If safe_direction > W/2 the open space is to the right → rotate clockwise (+5°).
 * If safe_direction < W/2 the open space is to the left  → rotate counter-clockwise (-5°).
 * If safe_direction == W/2 or -1 (no corridor) the increment is left unchanged.
 */
uint8_t chooseWiseIncrementAvoidance(int safe_direction)
{
  
  if (safe_direction > MAX_IMAGE_WIDTH / 2) {
    heading_increment = 5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  } else if(safe_direction < MAX_IMAGE_WIDTH / 2){
    heading_increment = -5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  }
  return false;
}