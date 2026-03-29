/*
 * team10_custom_autopilot_withGateDet_riccardo.c
 *
 * Riccardo's gate-aware obstacle avoidance autopilot (Team 10).
 *
 * This is a different state machine architecture from the other autopilot variants.
 * Instead of a confidence counter that scales forward speed, the drone alternates
 * between two primary states with fixed time budgets:
 *
 *   GO     — drive forward for locked_go_cooldown_frames_setting ticks (~3 s),
 *             then switch to ROTATE to re-evaluate the scene.
 *   ROTATE — rotate toward the best corridor column until either the safe
 *             direction is centred or the rotation budget expires, then switch to GO.
 *
 * Two additional states handle exceptional conditions:
 *   OBSTACLE_FOUND — triggered when total_obstacle_width exceeds the threshold
 *                    for 5 consecutive frames. Stops the drone and waits for the
 *                    obstacle count to drop before resuming.
 *   OUT_OF_BOUNDS  — triggered when WP_TRAJECTORY leaves the allowed flight zone.
 *                    Rotates and probes until back inside, then re-enters ROTATE.
 *
 * Gate detection data is received via a second ABI callback but gate approach
 * logic is not yet implemented — the gate_seen and gate_center_col variables are
 * populated but not acted upon in this version.
 *
 * The corridor logic (motion_logic_normalised) is called every tick to find the
 * safest image column, which is then mapped to a rotation direction and used to
 * set heading_increment for that tick's ROTATE action.
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

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// Forward declarations of helper functions defined at the bottom of this file
static uint8_t moveWaypointToImageColumn(uint8_t waypoint, int image_col, float forward_distance, float lateral_range_m);
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static uint8_t chooseWiseIncrementAvoidance(int safe_col);

/* ABI IDs — default to ABI_BROADCAST so the module compiles without any
   airframe override. Set to explicit values in the airframe file so the
   autopilot only listens to the correct sender. */
#ifndef TEAM10_GATE_DETECTION_ID
#define TEAM10_GATE_DETECTION_ID ABI_BROADCAST
#endif

#ifndef TEAM10_GROUND_DETECTION_ID
#define TEAM10_GROUND_DETECTION_ID ABI_BROADCAST
#endif

/* ── Navigation state machine ─────────────────────────────────────────────── */
enum NavigationState {
  GO,             /* driving forward for a fixed number of ticks              */
  ROTATE,         /* rotating toward the best corridor column                 */
  OBSTACLE_FOUND, /* obstacle too large — stop and wait for it to clear       */
  OUT_OF_BOUNDS   /* trajectory waypoint left the allowed flight zone         */
};

/* Which side of the image the safe corridor or target is on */
enum ObjectiveLocation {
  LEFT,       /* best corridor is to the drone's physical left  */
  RIGHT,      /* best corridor is to the drone's physical right */
  CENTERLINE  /* best corridor is straight ahead                */
};

/* ── Module-level state updated by ABI callbacks ─────────────────────────── */
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS]; /* obstacle regions this frame   */
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];       /* plant regions this frame      */
static uint16_t boundary_rows[MAX_IMAGE_HEIGHT];                 /* ground boundary per row (int) */
static float    boundary_rows_f[MAX_IMAGE_HEIGHT];               /* same boundary as floats       */
static uint8_t  obstacle_count = 0;                              /* number of obstacles detected  */
static uint8_t  plant_count    = 0;                              /* number of plants detected     */
static uint16_t boundary_len   = 0;                             /* number of valid boundary rows */
uint16_t total_obstacle_width  = 0;                              /* total width of obstacles [px] */
int safe_col;                                                    /* safest image column this tick */

/* ── Navigation state ─────────────────────────────────────────────────────── */
static enum ObjectiveLocation point_location;  /* where the best corridor is right now */
static enum ObjectiveLocation target_location; /* where we are currently heading toward */
static enum NavigationState   nav_state;       /* current state machine state           */
float  heading_increment;                      /* yaw step applied each ROTATE tick [deg] */

/* ── GCS-tunable settings ─────────────────────────────────────────────────── */
float   speed_multiplier     = 1;    /* overall speed multiplier                           */
float   maxDistance          = 1.5;  /* max waypoint displacement per GO phase [m]         */
uint8_t centerline_tolerance = 0.1 * MAX_IMAGE_WIDTH; /* dead-band around image centre [px] */
float   heading_increment_degrees_setting     = 3;    /* rotation step per tick [deg]        */
uint8_t locked_rotate_cooldown_frames_setting = 30;   /* ticks allowed for one ROTATE phase  */
uint8_t locked_go_cooldown_frames_setting     = 30;   /* ticks allowed for one GO phase      */
float   obstacle_width_threshold  = 0.3f; /* fraction of image width — triggers OBSTACLE_FOUND */
uint8_t max_trajectory_confidence = 5;    /* consecutive blocked frames before stopping        */

/* ── Countdown timers ─────────────────────────────────────────────────────── */
uint8_t locked_rotate_cooldown;  /* ticks remaining in the current ROTATE phase */
uint8_t locked_go_cooldown;      /* ticks remaining in the current GO phase     */
uint8_t obstacle_found_countdown; /* consecutive frames with obstacle above threshold */

/*
 * ABI messaging: the ground detection module runs in a separate camera thread.
 * We register two callbacks — one for obstacle/plant/boundary data and one for
 * gate detection results — so our local buffers are updated automatically each
 * time new data is published without directly sharing memory with the camera thread.
 */

// ABI event handles
static abi_event ground_detection_ev;
static abi_event gate_detection_ev;

/* Called automatically each time the ground detection module publishes new data.
   Copies obstacle, plant, and boundary arrays into local buffers and recomputes
   total_obstacle_width as the sum of all obstacle and plant widths. */
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

    // Convert to float so logic functions can use them without casting
    for (uint16_t i = 0; i < boundary_len; i++)
    boundary_rows_f[i] = (float)in_br[i];

    // Sum widths of all obstacles and plants for the obstacle width check
    total_obstacle_width = 0;
    for (uint8_t i = 0; i < in_oc; i++) {
      total_obstacle_width += in_obs[i].width + in_plants[i].width;
    }
}

/* Gate detection state — populated by the callback but not yet acted upon */
uint8_t gate_seen;       /* 1 if a gate was detected in the most recent frame */
int     gate_center_col; /* image column of the gate centre                   */

/* Called automatically each time gate detection publishes a result.
   Stores whether a gate is visible and where its centre column is.
   Gate approach logic using these values is not yet implemented. */
static void gate_detection_callback(
    uint8_t __attribute__((unused)) sender_id,
    uint8_t in_gate_detected,
    int     in_gate_center_x)
{
    gate_seen = in_gate_detected;
    gate_center_col = (int16_t)in_gate_center_x;
}

/*
 * Module initialisation — called once at startup.
 * Sets the initial state to ROTATE so the drone verifies the heading before
 * moving, initialises counters to zero, and registers both ABI callbacks.
 */
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

/*
 * Module periodic function — called at 10 Hz while the drone is in flight.
 *
 * Each cycle:
 *   1. Run the corridor logic to find the best safe column in the current frame.
 *   2. Re-scale it from boundary coordinates to full image width coordinates.
 *   3. Map the column to a LEFT/RIGHT/CENTERLINE direction and set heading_increment.
 *   4. Update obstacle_found_countdown and check if OBSTACLE_FOUND should trigger.
 *   5. Execute the current state machine state.
 */
void ground_obstacle_avoidance_periodic(void)
{
  // only evaluate our state machine if we are flying
  if(!autopilot_in_flight()){
    return;
  }

  int   best_col;
  float confidence;
  // Find the safest column in boundary coordinates (0..boundary_len)
  best_col = motion_logic_normalised(obstacles, obstacle_count,
                                     plants, plant_count,
                                     boundary_rows_f, boundary_len, 
                                     boundary_len, //  MAX_IMAGE_HEIGHT,
                                     DEFAULT_OBS_BIAS_FRAC,
                                     DEFAULT_PLANT_BIAS_FRAC);
  // Re-scale from boundary space to native image width so centring comparisons work
  best_col = best_col * MAX_IMAGE_WIDTH / boundary_len;

  // Map the best column to a direction and set heading_increment accordingly
  if (best_col < (MAX_IMAGE_WIDTH/2) - centerline_tolerance) {
    point_location    = LEFT;
    heading_increment = -heading_increment_degrees_setting;  /* rotate left (CCW) */
  } else if (best_col > (MAX_IMAGE_WIDTH/2) + centerline_tolerance) {
    point_location    = RIGHT;
    heading_increment = +heading_increment_degrees_setting;  /* rotate right (CW) */
  } else {
    point_location    = CENTERLINE;
    heading_increment = +heading_increment_degrees_setting; // dummy setting
  }

  // Increment the countdown each frame where obstacles exceed the threshold
  if (total_obstacle_width > obstacle_width_threshold) {
    obstacle_found_countdown += 1;
  }

  // After 5 consecutive blocked frames, transition to OBSTACLE_FOUND
  if (obstacle_found_countdown == 5 && nav_state != OUT_OF_BOUNDS) {
    nav_state = OBSTACLE_FOUND;
  } else {
    obstacle_found_countdown -= 1;
  }

  // state machine
  switch (nav_state) 
  {
  case ROTATE:
    // If a rotation budget is still active, keep rotating
    if (locked_rotate_cooldown != 0) {
      locked_rotate_cooldown -= 1;
      increase_nav_heading(heading_increment);
    }
    // Transition to GO once the safe direction is close enough to straight ahead
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
    // Update the trajectory waypoint every tick so the flight controller
    // always has a fresh lookahead position to compute speed against
    moveWaypointForward(WP_TRAJECTORY, 0.5 * maxDistance);  

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
    // GO budget exhausted — switch back to ROTATE to re-evaluate the scene
    if (locked_go_cooldown == 0) {
      nav_state = ROTATE;
      locked_rotate_cooldown = locked_rotate_cooldown_frames_setting;
      target_location = point_location;
      printf("GO finished. Setting new target.\n");
    }
    break;
  
  case OBSTACLE_FOUND:
    // Stop in place and wait for the obstacle count to drop below the
    // threshold for at least 2 frames before resuming rotation
    waypoint_move_here_2d(WP_GOAL);
    waypoint_move_here_2d(WP_TRAJECTORY);
    printf("Obstacle found.\n");
    if (obstacle_found_countdown < 2) {
    nav_state = ROTATE;
    locked_rotate_cooldown = locked_rotate_cooldown_frames_setting;
    }
    // obstacle_found_countdown = 0;
    break;

  case OUT_OF_BOUNDS:
    // Rotate slowly and probe WP_TRAJECTORY until it lands back inside the zone,
    // then give a longer rotation budget to reorient before driving forward
    printf("OUT OF BOUNDS.\n");
    increase_nav_heading(3);
    moveWaypointForward(WP_TRAJECTORY, 0.5 * maxDistance);

    if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(3);
        nav_state = ROTATE;
        locked_rotate_cooldown = 1.5 * locked_rotate_cooldown_frames_setting;
    }
    break;
    
  default: break;
  }
  return;
}

/* Rotate the drone's heading by incrementDegrees.
   Converts to radians, normalises to [-π, π], and sets nav.heading.
   Positive = clockwise, negative = counter-clockwise. */
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

/* Move a waypoint to a point distanceMeters ahead of the current position.
   Convenience wrapper around calculateForwards() and moveWaypoint(). */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/* Compute ENU coordinates for a point distanceMeters ahead along the current heading.
   Uses sin/cos of psi to decompose the displacement into East (x) and North (y). */
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

/* Move a named waypoint to the given ENU coordinates.
   Thin wrapper around Paparazzi's waypoint_move_xy_i(). */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/* Fallback: pick a random clockwise or counter-clockwise rotation direction.
   Not used in normal operation — chooseWiseIncrementAvoidance is preferred. */
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

/* Set heading_increment toward the side of the image where the safe corridor is.
   Columns above W/2 = drone's right → clockwise (+5°).
   Columns below W/2 = drone's left  → counter-clockwise (-5°). */
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

/* Clamp a float value to the range [lo, hi]. */
static float clampf_local(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

/*
 * Move a waypoint to a position forward_distance metres ahead and laterally
 * offset so that image column image_col maps to a physical lateral displacement
 * within [-lateral_range_m, +lateral_range_m].
 *
 * This allows the drone to steer toward a specific image column (e.g. the gate
 * centre or the safe corridor) rather than always heading straight forward.
 * The mapping is linear: column 0 → -lateral_range_m, column W/2 → 0 m,
 * column W → +lateral_range_m.
 */
static uint8_t moveWaypointToImageColumn(uint8_t waypoint, int image_col, float forward_distance, float lateral_range_m) {
  struct EnuCoor_i new_coor;
  float heading = stateGetNedToBodyEulers_f()->psi;
  float image_center = MAX_IMAGE_WIDTH / 2.0f;

  // Normalise column to [-1, +1]
  float normalized = ((float)image_col - image_center) / image_center;
  normalized = clampf_local(normalized, -1.0f, 1.0f);

  float lateral = normalized * lateral_range_m;
  float forward = forward_distance;

  // Decompose into world-frame East/North components accounting for current heading
  float dx = sinf(heading) * forward + cosf(heading) * lateral;
  float dy = cosf(heading) * forward - sinf(heading) * lateral;

  new_coor.x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(dx);
  new_coor.y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(dy);

  moveWaypoint(waypoint, &new_coor);
  return false;
}