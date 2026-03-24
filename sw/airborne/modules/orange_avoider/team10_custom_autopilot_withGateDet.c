/*
 * team10_custom_autopilot.c — improved version
 *
 * Improvements applied vs. previous version:
 *
 *  #1  Corridor-width picker  — motion_logic_normalised (via team10_logic.c)
 *      now returns the centre of the widest clear corridor rather than the
 *      single deepest pixel.  No autopilot changes needed; inherited from logic.
 *
 *  #2  total_obstacle_width is now computed correctly inside
 *      ground_detection_callback from the actual per-frame obstacle data.
 *      Previously it was a module-level variable that was never written,
 *      so obstacle_free_confidence climbed every tick and OBSTACLE_FOUND
 *      was unreachable from SAFE via the confidence path.
 *
 *  #3  SEARCH_FOR_SAFE_HEADING re-evaluates the safe direction every tick
 *      instead of committing to the direction chosen in OBSTACLE_FOUND.
 *
 *  #4  Gate yaw uses proportional control — yaw rate scales with angular
 *      error so large offsets rotate fast and small offsets do fine correction.
 *
 *  #5  plant_bias_frac > obs_bias_frac — plants are narrower than walls so
 *      they need a larger relative safety margin.  Separate constants defined.
 *
 *  #6  Plateau centre tie-breaking — inherited from widest_corridor_centre
 *      in team10_logic.c; no autopilot change needed.
 *
 *  #7  Gate hysteresis — OBSTACLE_FOUND only commits to GATE_APPROACH after
 *      GATE_CONFIRM_STREAK consecutive frames with gate_detected == 1,
 *      preventing a single missed frame from sending the drone into avoidance.
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
#include <math.h>    /* fabsf */

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
static uint8_t chooseWiseIncrementAvoidance(int safe_col);
static uint8_t chooseRandomIncrementAvoidance(void);
static uint8_t chooseHeadingToGate(int gate_col);

float speed_multiplier = 0.5f;

/* ══════════════════════════════════════════════════════════════════════════════
 *  TUNING CONSTANTS
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Gate approach — kept as compile-time constants; not exposed as sliders */
#define GATE_HEADING_MAX_DEG     15.0f
#define GATE_ALIGN_COLUMN_TOL    20
#define GATE_APPROACH_DISTANCE_M 1.5f
#define GATE_CONFIRM_STREAK      2

/* IMPROVEMENT #7: number of consecutive gate-detected frames required before
 * committing to GATE_APPROACH.  Prevents a single noisy frame from overriding
 * the avoidance path.                                                          */
#define GATE_CONFIRM_STREAK      2

/* ══════════════════════════════════════════════════════════════════════════════
 *  NAVIGATION STATE MACHINE
 * ══════════════════════════════════════════════════════════════════════════════ */
enum navigation_state_t {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    GATE_APPROACH,
    OUT_OF_BOUNDS
};

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;
float   heading_increment        = 5.f;
float   maxDistance              = 2.25f;

const int16_t max_trajectory_confidence = 5;

/* ══════════════════════════════════════════════════════════════════════════════
 *  MODULE-LEVEL STATE
 * ══════════════════════════════════════════════════════════════════════════════ */
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];
static int16_t                  boundary_rows[MAX_IMAGE_HEIGHT];
static float                    boundary_rows_f[MAX_IMAGE_HEIGHT];
static uint8_t                  obstacle_count       = 0;
static uint8_t                  plant_count          = 0;
static uint16_t                 boundary_len         = 0;
uint16_t                        total_obstacle_width = 0;   /* #2: now written */

/* ══════════════════════════════════════════════════════════════════════════════
 *  RUNTIME-TUNABLE PARAMETERS
 *  All variables in this block are non-static globals so they can be adjusted
 *  via Paparazzi <dl_setting> sliders in the GCS without recompiling.
 *  Matching declarations must appear in team10_custom_autopilot.h and the
 *  module's settings XML (see team10_custom_autopilot.xml).
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Fraction of image width covered by ground-touching obstacles above which the
 * confidence counter decrements.  Lower = react to smaller obstacles sooner.
 * Range [0.05, 0.50].  Default 0.15.                                          */
float obstacle_width_threshold = 0.15f;

/* Safety margin added around each obstacle column span, as a fraction of the
 * obstacle's own width.  Range [0.1, 1.0].  Default 0.50.                     */
float obs_bias_frac   = 0.50f;

/* Safety margin around each plant column span.  Larger than obs_bias_frac
 * because plant pots are narrow and need proportionally more clearance.
 * Range [0.1, 1.5].  Default 0.75.                                            */
float plant_bias_frac = 0.75f;

/* How many periodic ticks between WP_GOAL advances in the SAFE state.
 * At 10 Hz: 1 = every 0.1 s, 5 = every 0.5 s, 10 = every 1 s.
 * Range [1, 20].  Default 5.                                                  */
int   wp_update_period_ticks = 5;

/* Corridor picker parameters — defined here (not in team10_logic.c) so that
 * there is exactly one definition in the build regardless of which logic file
 * version is compiled.  The logic file reads these as extern.
 * Range [0.5, 1.0].  Default 0.90.                                           */
float clear_frac            = 0.90f;

/* Minimum contiguous clear-column run to qualify as a usable corridor (px).
 * Range [10, 200].  Default 60.                                               */
int   min_corridor_width_px = 60;

/* Gate state */
static uint8_t  cur_gate_detected  = 0;
static int      cur_gate_center_x  = 0;
/* IMPROVEMENT #7: consecutive-detection streak counter                        */
static uint8_t  gate_seen_streak   = 0;

/* Waypoint update rate limiter counter — counts down from wp_update_period_ticks */
static uint8_t  wp_update_ticks    = 0;

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

    /* ── IMPROVEMENT #2 ────────────────────────────────────────────────────
     * Compute total_obstacle_width here, directly from the freshly received
     * obstacle data, counting only obstacles that touch the ground (those are
     * the ones that actually block the drone's path).
     * Previously this variable was declared but never written, so the
     * confidence counter always incremented and OBSTACLE_FOUND was never
     * reached from SAFE via the confidence threshold.                        */
    total_obstacle_width = 0;
    for (uint8_t i = 0; i < in_oc; i++) {
        if (in_obs[i].baseline_height >= MAX_IMAGE_HEIGHT)
            total_obstacle_width += in_obs[i].width;
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ABI — GATE DETECTION
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

    /* IMPROVEMENT #7: maintain a consecutive-detection streak so OBSTACLE_FOUND
     * requires GATE_CONFIRM_STREAK frames before trusting the gate flag.      */
    if (in_gate_detected)
        gate_seen_streak++;
    else
        gate_seen_streak = 0;
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
            boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
            obs_bias_frac,
            plant_bias_frac);
    chooseWiseIncrementAvoidance(safe_col);

    AbiBindMsgTEAM10_GROUND_DETECTION(
            TEAM10_GROUND_DETECTION_ID, &ground_detection_ev,
            ground_detection_callback);

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

    /* ── Confidence update (now meaningful — see improvement #2) ────────── */
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
            wp_update_ticks  = 0;   /* reset limiter so next SAFE entry is immediate */
            navigation_state = OBSTACLE_FOUND;
        } else {
            /* Rate-limit WP_GOAL updates: only move it every wp_update_period_ticks
             * ticks.  WP_TRAJECTORY still moves every tick as a lookahead probe so
             * the bounds check above stays responsive, but WP_GOAL — the target the
             * flight controller actually chases — advances at a slower, steadier
             * pace that the drone can physically track before it changes again.    */
            if (wp_update_ticks == 0) {
                moveWaypointForward(WP_GOAL, moveDistance);
                wp_update_ticks = (uint8_t)wp_update_period_ticks;
            } else {
                wp_update_ticks--;
            }
        }
        break;

    /* ── OBSTACLE_FOUND ──────────────────────────────────────────────────── */
    case OBSTACLE_FOUND:
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);

        /* IMPROVEMENT #7: require GATE_CONFIRM_STREAK consecutive detections
         * before trusting the gate flag and committing to GATE_APPROACH.
         * A single noisy frame can no longer override the avoidance path.    */
        if (cur_gate_detected && gate_seen_streak >= GATE_CONFIRM_STREAK) {
            VERBOSE_PRINT("Gate confirmed (streak=%d) at col %d — GATE_APPROACH\n",
                          gate_seen_streak, cur_gate_center_x);
            chooseHeadingToGate(cur_gate_center_x);
            navigation_state = GATE_APPROACH;
        } else {
            /* Regular obstacle — find safe corridor and rotate toward it */
            int safe_col = motion_logic_normalised(
                    obstacles, obstacle_count,
                    plants,    plant_count,
                    boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
                    obs_bias_frac,
                    plant_bias_frac);
            chooseWiseIncrementAvoidance(safe_col);
            navigation_state = SEARCH_FOR_SAFE_HEADING;
        }
        break;

    /* ── GATE_APPROACH ───────────────────────────────────────────────────── */
    case GATE_APPROACH:
        if (!cur_gate_detected) {
            VERBOSE_PRINT("Gate lost during approach — re-evaluating\n");
            navigation_state = OBSTACLE_FOUND;
            break;
        }

        {
            int img_centre = MAX_IMAGE_WIDTH / 2;
            int gate_err   = cur_gate_center_x - img_centre;

            if (abs(gate_err) <= GATE_ALIGN_COLUMN_TOL) {
                /* Aligned: drive through */
                VERBOSE_PRINT("Aligned with gate (err=%d px) — approaching\n", gate_err);
                obstacle_free_confidence = max_trajectory_confidence;
                moveWaypointForward(WP_TRAJECTORY, GATE_APPROACH_DISTANCE_M);
                moveWaypointForward(WP_GOAL,       GATE_APPROACH_DISTANCE_M);
                navigation_state = SAFE;
            } else {
                /* IMPROVEMENT #4: proportional yaw — larger error → faster turn.
                 * Normalise gate_err to [-1, +1] then scale to max yaw rate.
                 * This replaces the previous fixed ±5°/tick regardless of error. */
                float err_norm = (float)gate_err / (float)img_centre;   /* -1..+1 */
                float yaw_deg  = err_norm * GATE_HEADING_MAX_DEG;
                /* clamp (already bounded by construction, but be safe) */
                if (yaw_deg >  GATE_HEADING_MAX_DEG) yaw_deg =  GATE_HEADING_MAX_DEG;
                if (yaw_deg < -GATE_HEADING_MAX_DEG) yaw_deg = -GATE_HEADING_MAX_DEG;
                VERBOSE_PRINT("Gate err=%d px  yaw=%.1f deg\n", gate_err, yaw_deg);
                increase_nav_heading(yaw_deg);
            }
        }
        break;

    /* ── SEARCH_FOR_SAFE_HEADING ─────────────────────────────────────────── */
    case SEARCH_FOR_SAFE_HEADING:
        /* IMPROVEMENT #3: re-evaluate the safe direction every tick so that
         * a new obstacle entering from the side during rotation does not leave
         * the drone committed to a stale heading choice made in OBSTACLE_FOUND. */
        {
            int safe_col = motion_logic_normalised(
                    obstacles, obstacle_count,
                    plants,    plant_count,
                    boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
                    obs_bias_frac,
                    plant_bias_frac);
            chooseWiseIncrementAvoidance(safe_col);
        }
        increase_nav_heading(heading_increment);

        if (obstacle_free_confidence >= 2)
            navigation_state = SAFE;
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
uint8_t increase_nav_heading(float incrementDegrees)
{
    float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
    FLOAT_ANGLE_NORMALIZE(new_heading);
    nav.heading = new_heading;
    VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
    return false;
}

/*
 * chooseHeadingToGate
 *
 * Sets heading_increment for the initial coarse turn when first entering
 * GATE_APPROACH.  Once inside that state, improvement #4 takes over with
 * proportional control — this function only sets the sign for the first tick.
 *
 * Coordinate convention (90° CCW rotation):
 *   col 0   = physical RIGHT  →  gate_col > W/2  →  CW yaw  (+deg)
 *   col W-1 = physical LEFT   →  gate_col < W/2  →  CCW yaw (-deg)
 */
static uint8_t chooseHeadingToGate(int gate_col)
{
    if (gate_col > MAX_IMAGE_WIDTH / 2) {
        heading_increment = 5.f;
        VERBOSE_PRINT("Gate RIGHT: initial yaw +%.1f deg\n", heading_increment);
    } else if (gate_col < MAX_IMAGE_WIDTH / 2) {
        heading_increment = -5.f;
        VERBOSE_PRINT("Gate LEFT: initial yaw %.1f deg\n", heading_increment);
    }
    return false;
}

/*
 * chooseWiseIncrementAvoidance
 * Sets heading_increment based on which side of the image the safe corridor is.
 */
uint8_t chooseWiseIncrementAvoidance(int safe_direction)
{
    if (safe_direction > MAX_IMAGE_WIDTH / 2) {
        heading_increment = 5.f;
    } else if (safe_direction < MAX_IMAGE_WIDTH / 2) {
        heading_increment = -5.f;
    }
    VERBOSE_PRINT("Safe col=%d  heading_increment=%.1f\n", safe_direction, heading_increment);
    return false;
}

/*
 * chooseRandomIncrementAvoidance — fallback, kept for completeness.
 */
uint8_t chooseRandomIncrementAvoidance(void)
{
    heading_increment = (rand() % 2 == 0) ? 5.f : -5.f;
    VERBOSE_PRINT("Random avoidance increment: %.1f\n", heading_increment);
    return false;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  WAYPOINT HELPERS
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
    VERBOSE_PRINT("Calculated %f m forward: x=%f y=%f  pos=(%f,%f)  hdg=%f\n",
                  distanceMeters,
                  POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                  stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y,
                  DegOfRad(heading));
    return false;
}

uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
    VERBOSE_PRINT("Moving WP %d to x=%f y=%f\n", waypoint,
                  POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
    waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
    return false;
}