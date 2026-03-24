/*
 * team10_custom_autopilot.c
 *
 * Gate-aware obstacle avoidance for the Bebop2.
 *
 * Changes vs. original:
 *   1. New GATE_APPROACH navigation state.
 *   2. ABI binding for TEAM10_GATE_DETECTION messages (gate_detected, gate_center_x).
 *   3. OBSTACLE_FOUND now checks gate_detected first:
 *        - gate present  → GATE_APPROACH (aim at gate centre, drive through)
 *        - no gate       → SEARCH_FOR_SAFE_HEADING  (original behaviour)
 *   4. chooseHeadingToGate() converts gate_center_x (pixel column, native res)
 *      to a heading increment so the drone faces the gate opening.
 *
 * Coordinate note:
 *   After the 90° CCW rotation applied in ground_detection, column 0 = RIGHT side
 *   and column W-1 = LEFT side of the drone's physical view.  gate_center_x is in
 *   the same native (rotated) space, so pixel > W/2 → gate is to the drone's right
 *   → positive heading increment (CW yaw).
 */

#include "modules/orange_avoider/team10_custom_autopilot.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/team10_logic.h"

#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <string.h>

#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* ── Forward declarations ──────────────────────────────────────────────────── */
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static uint8_t chooseWiseIncrementAvoidance(int safe_col);
static uint8_t chooseHeadingToGate(int gate_col);

float speed_multiplier = 0.5f;

/* ══════════════════════════════════════════════════════════════════════════════
 *  NAVIGATION STATE MACHINE
 * ══════════════════════════════════════════════════════════════════════════════ */
enum navigation_state_t {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    GATE_APPROACH,          /* NEW: obstacle is a gate — fly through it */
    OUT_OF_BOUNDS
};

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;
float   heading_increment        = 5.f;
float   maxDistance              = 2.25f;

/* ── Gate approach tuning ──────────────────────────────────────────────────── */
/* How many degrees per step to yaw toward the gate centre.                    */
#define GATE_HEADING_INCREMENT_DEG  5.f
/* How close the heading must be to "aligned" before we commit to driving
   forward through the gate (degrees, expressed as fraction of image width).   */
#define GATE_ALIGN_COLUMN_TOLERANCE 20   /* px: |gate_col - img_centre| < this */
/* Distance to push the TRAJECTORY waypoint when aligned with the gate.        */
#define GATE_APPROACH_DISTANCE_M    1.5f

/* ══════════════════════════════════════════════════════════════════════════════
 *  MODULE-LEVEL STATE
 * ══════════════════════════════════════════════════════════════════════════════ */
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];
static int16_t                  boundary_rows[MAX_IMAGE_HEIGHT];
static float                    boundary_rows_f[MAX_IMAGE_HEIGHT];
static uint8_t                  obstacle_count  = 0;
static uint8_t                  plant_count     = 0;
static uint16_t                 boundary_len    = 0;
uint16_t                        total_obstacle_width = 0;

/* Gate state — written by ABI callback, read by periodic */
static uint8_t  cur_gate_detected  = 0;
static int      cur_gate_center_x  = 0;

float obstacle_width_threshold = 0.2f;

const int16_t max_trajectory_confidence = 5;

/* ══════════════════════════════════════════════════════════════════════════════
 *  ABI — GROUND DETECTION
 * ══════════════════════════════════════════════════════════════════════════════ */
#ifndef TEAM10_GROUND_DETECTION_ID
#define TEAM10_GROUND_DETECTION_ID ABI_BROADCAST
#endif

static abi_event ground_detection_ev;

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

    boundary_len = in_bl;
    if (boundary_len > MAX_IMAGE_HEIGHT) boundary_len = MAX_IMAGE_HEIGHT;
    memcpy(boundary_rows, in_br, boundary_len * sizeof(int16_t));

    for (uint16_t i = 0; i < boundary_len; i++)
        boundary_rows_f[i] = (float)in_br[i];
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ABI — GATE DETECTION  (NEW)
 * ══════════════════════════════════════════════════════════════════════════════ */
#ifndef TEAM10_GATE_DETECTION_ID
#define TEAM10_GATE_DETECTION_ID ABI_BROADCAST
#endif

static abi_event gate_detection_ev;

static void gate_detection_callback(
    uint8_t __attribute__((unused)) sender_id,
    uint8_t  in_gate_detected,
    int      in_gate_center_x)
{
    cur_gate_detected = in_gate_detected;
    cur_gate_center_x = in_gate_center_x;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  INIT
 * ══════════════════════════════════════════════════════════════════════════════ */
void ground_obstacle_avoidance_init(void)
{
    srand(time(NULL));

    int safe_col = motion_logic_normalised(
            obstacles, obstacle_count,
            plants,    plant_count,
            boundary_rows_f,
            boundary_len, MAX_IMAGE_HEIGHT,
            DEFAULT_OBS_BIAS_FRAC,
            DEFAULT_PLANT_BIAS_FRAC);
    chooseWiseIncrementAvoidance(safe_col);

    AbiBindMsgTEAM10_GROUND_DETECTION(
            TEAM10_GROUND_DETECTION_ID, &ground_detection_ev,
            ground_detection_callback);

    /* NEW: bind gate detection ABI */
    AbiBindMsgTEAM10_GATE_DETECTION(
            TEAM10_GATE_DETECTION_ID, &gate_detection_ev,
            gate_detection_callback);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  PERIODIC — STATE MACHINE
 * ══════════════════════════════════════════════════════════════════════════════ */
void ground_obstacle_avoidance_periodic(void)
{
    if (!autopilot_in_flight()) return;

    printf("total obstacle width: %.2f\nthreshold (fraction): %.2f\nthreshold (total):    %.2f\n",
           total_obstacle_width / (float)MAX_IMAGE_WIDTH,
           obstacle_width_threshold,
           obstacle_width_threshold * MAX_IMAGE_WIDTH);

    /* Update confidence */
    if (total_obstacle_width < obstacle_width_threshold * MAX_IMAGE_WIDTH) {
        obstacle_free_confidence++;
    } else {
        obstacle_free_confidence -= 2;
    }
    Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

    float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

    switch (navigation_state) {

    /* ── SAFE ────────────────────────────────────────────────────────────── */
    case SAFE:
        moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
        if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
            navigation_state = OUT_OF_BOUNDS;
        } else if (obstacle_free_confidence == 0) {
            navigation_state = OBSTACLE_FOUND;
        } else {
            moveWaypointForward(WP_GOAL, moveDistance);
        }
        break;

    /* ── OBSTACLE_FOUND ──────────────────────────────────────────────────── */
    case OBSTACLE_FOUND:
        /* Stop in place */
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);

        if (cur_gate_detected) {
            /*
             * The obstacle in front is a gate.
             * Switch to gate approach: align heading with the gate opening,
             * then drive straight through.
             */
            VERBOSE_PRINT("Gate detected at col %d — entering GATE_APPROACH\n",
                          cur_gate_center_x);
            chooseHeadingToGate(cur_gate_center_x);
            navigation_state = GATE_APPROACH;
        } else {
            /*
             * Regular obstacle — find a safe direction and rotate toward it
             * (original behaviour).
             */
            int safe_col = motion_logic_normalised(
                    obstacles, obstacle_count,
                    plants,    plant_count,
                    boundary_rows_f,
                    boundary_len, MAX_IMAGE_HEIGHT,
                    DEFAULT_OBS_BIAS_FRAC,
                    DEFAULT_PLANT_BIAS_FRAC);
            chooseWiseIncrementAvoidance(safe_col);
            navigation_state = SEARCH_FOR_SAFE_HEADING;
        }
        break;

    /* ── GATE_APPROACH (NEW) ─────────────────────────────────────────────── */
    case GATE_APPROACH:
        /*
         * Keep turning toward the gate centre until aligned, then push the
         * waypoint forward through the gate opening.
         *
         * Alignment check: if the gate centre column is within
         * GATE_ALIGN_COLUMN_TOLERANCE pixels of the image centre, the drone
         * is facing the gate — commit to the approach.
         *
         * If the gate is lost (cur_gate_detected == 0) mid-approach we fall
         * back to OBSTACLE_FOUND to re-evaluate.
         */
        if (!cur_gate_detected) {
            VERBOSE_PRINT("Gate lost during approach — re-evaluating\n");
            navigation_state = OBSTACLE_FOUND;
            break;
        }

        {
            int img_centre = MAX_IMAGE_WIDTH / 2;
            int gate_err   = cur_gate_center_x - img_centre;   /* +ve = gate right */

            if (abs(gate_err) <= GATE_ALIGN_COLUMN_TOLERANCE) {
                /* Aligned: drive forward through the gate */
                VERBOSE_PRINT("Aligned with gate (err=%d px) — approaching\n", gate_err);
                obstacle_free_confidence = max_trajectory_confidence;   /* trust the path */
                moveWaypointForward(WP_TRAJECTORY, GATE_APPROACH_DISTANCE_M);
                moveWaypointForward(WP_GOAL,       GATE_APPROACH_DISTANCE_M);
                navigation_state = SAFE;
            } else {
                /* Still rotating toward the gate */
                VERBOSE_PRINT("Gate err=%d px — yawing\n", gate_err);
                chooseHeadingToGate(cur_gate_center_x);
                increase_nav_heading(heading_increment);
            }
        }
        break;

    /* ── SEARCH_FOR_SAFE_HEADING ─────────────────────────────────────────── */
    case SEARCH_FOR_SAFE_HEADING:
        increase_nav_heading(heading_increment);
        if (obstacle_free_confidence >= 2) {
            navigation_state = SAFE;
        }
        break;

    /* ── OUT_OF_BOUNDS ───────────────────────────────────────────────────── */
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

/* ══════════════════════════════════════════════════════════════════════════════
 *  HEADING HELPERS
 * ══════════════════════════════════════════════════════════════════════════════ */

/*
 * increase_nav_heading
 * Adds incrementDegrees to the current heading and normalises to [-π, π].
 */
uint8_t increase_nav_heading(float incrementDegrees)
{
    float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
    FLOAT_ANGLE_NORMALIZE(new_heading);
    nav.heading = new_heading;
    VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
    return false;
}

/*
 * chooseHeadingToGate  (NEW)
 *
 * Sets heading_increment so the drone yaws toward the gate centre.
 *
 * After the 90° CCW rotation applied by ground_detection:
 *   column 0       = physical RIGHT of the drone
 *   column W-1     = physical LEFT  of the drone
 *   column W/2     = straight ahead
 *
 * Therefore:
 *   gate_col > W/2  →  gate is to the RIGHT  →  positive yaw (CW, +deg)
 *   gate_col < W/2  →  gate is to the LEFT   →  negative yaw (CCW, -deg)
 */
static uint8_t chooseHeadingToGate(int gate_col)
{
    if (gate_col > MAX_IMAGE_WIDTH / 2) {
        heading_increment = GATE_HEADING_INCREMENT_DEG;
        VERBOSE_PRINT("Gate RIGHT: yaw +%.1f deg\n", heading_increment);
    } else if (gate_col < MAX_IMAGE_WIDTH / 2) {
        heading_increment = -GATE_HEADING_INCREMENT_DEG;
        VERBOSE_PRINT("Gate LEFT: yaw %.1f deg\n", heading_increment);
    }
    /* If gate_col == centre exactly, keep current heading_increment */
    return false;
}

/*
 * chooseWiseIncrementAvoidance
 * Sets heading_increment based on which side of the image has the safe column.
 */
uint8_t chooseWiseIncrementAvoidance(int safe_direction)
{
    if (safe_direction > MAX_IMAGE_WIDTH / 2) {
        heading_increment = 5.f;
        VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
    } else if (safe_direction < MAX_IMAGE_WIDTH / 2) {
        heading_increment = -5.f;
        VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
    }
    return false;
}

/*
 * chooseRandomIncrementAvoidance
 * Fallback: random CW / CCW direction.
 */
uint8_t chooseRandomIncrementAvoidance(void)
{
    if (rand() % 2 == 0) {
        heading_increment = 5.f;
    } else {
        heading_increment = -5.f;
    }
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
    return false;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  WAYPOINT HELPERS  (unchanged from original)
 * ══════════════════════════════════════════════════════════════════════════════ */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
    struct EnuCoor_i new_coor;
    calculateForwards(&new_coor, distanceMeters);
    moveWaypoint(waypoint, &new_coor);
    return false;
}

uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
    float heading = stateGetNedToBodyEulers_f()->psi;
    new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
    new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
    VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n",
                  distanceMeters,
                  POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                  stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
    return false;
}

uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
    VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint,
                  POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
    waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
    return false;
}