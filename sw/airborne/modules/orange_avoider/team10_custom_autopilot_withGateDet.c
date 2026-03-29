/*
 * team10_custom_autopilot_withGateDet.c
 *
 * Overview:
 *   The camera thread runs ground detection at up to 10 Hz and publishes
 *   obstacle regions, plant regions, a ground boundary array, and a gate
 *   detection result over the ABI message bus.  This module runs at 10 Hz,
 *   reads those results via ABI callbacks, and drives a five-state navigation
 *   state machine that moves two waypoints (WP_GOAL and WP_TRAJECTORY) to
 *   steer the drone through open corridors and through detected gates.
 *
 *   The camera image is rotated 90 degrees CCW before processing, so image
 *   column 0 corresponds to the drone's physical right and column W-1 to its
 *   physical left.  All column-based direction decisions use this convention.
 */

#include "modules/orange_avoider/team10_custom_autopilot_withGateDet.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/team10_logic_withGateDet.h"

#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

float speed_multiplier = 0.4f;

/* ══════════════════════════════════════════════════════════════════════════════
 *  GATE APPROACH CONSTANTS
 *
 *  These are not exposed as GCS sliders because they are structural parameters
 *  of the gate approach behaviour rather than tuning knobs.
 *
 *  GATE_HEADING_MAX_DEG  — maximum yaw rate per tick when aligning with a gate.
 *                          The actual rate is proportional to the angular error,
 *                          so this is only reached when the gate is far off-centre.
 *
 *  GATE_ALIGN_COLUMN_TOL — how close (in pixels) the gate centre must be to the
 *                          image centre before the drone is considered aligned
 *                          and commits to driving forward through the gate.
 *
 *  GATE_APPROACH_DISTANCE_M — how far forward the waypoints are pushed when the
 *                          drone is aligned and drives through the gate.
 *
 *  GATE_CONFIRM_STREAK   — how many consecutive camera frames must detect a gate
 *                          before OBSTACLE_FOUND commits to GATE_APPROACH.
 *                          Prevents a single noisy frame from triggering the gate
 *                          path when the obstacle is actually a solid wall.
 * ══════════════════════════════════════════════════════════════════════════════ */
#define GATE_HEADING_MAX_DEG     15.0f
#define GATE_ALIGN_COLUMN_TOL    20
#define GATE_APPROACH_DISTANCE_M 1.5f
#define GATE_CONFIRM_STREAK      3

/* ══════════════════════════════════════════════════════════════════════════════
 *  NAVIGATION STATES
 *
 *  SAFE                  — a valid corridor exists; drone moves forward.
 *  OBSTACLE_FOUND        — no safe path; drone stops and decides what to do.
 *  SEARCH_FOR_SAFE_HEADING — drone rotates in place until a corridor opens up.
 *  GATE_APPROACH         — obstacle is a gate; drone aligns and flies through.
 *  OUT_OF_BOUNDS         — trajectory waypoint left the arena; drone turns back.
 * ══════════════════════════════════════════════════════════════════════════════ */
enum navigation_state_t {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    GATE_APPROACH,
    OUT_OF_BOUNDS
};

/* Current navigation state — starts in SEARCH_FOR_SAFE_HEADING so the drone
 * verifies a clear path before moving on the first flight.                    */
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;

/* obstacle_free_confidence counts consecutive ticks where the path looks clear.
 * It increments by 1 each clear tick and decrements by 2 each blocked tick.
 * The asymmetry means a single bad frame outweighs two good ones, so the drone
 * reacts quickly to new obstacles.  Bounded to [0, max_trajectory_confidence]. */
int16_t obstacle_free_confidence = 0;

/* heading_increment is the yaw step applied each tick during rotation.
 * Positive = clockwise, negative = counter-clockwise.  Set by
 * chooseWiseIncrementAvoidance based on which side the safe corridor is on.   */
float heading_increment = 5.f;
float heading_increment_setting = 5.f;

/* Maximum distance the waypoint can be pushed forward in one step.            */
float maxDistance = 2.25f;

/* Maximum value of obstacle_free_confidence.                                  */
const int16_t max_trajectory_confidence = 5;

/* ══════════════════════════════════════════════════════════════════════════════
 *  INTERNAL STATE — written by ABI callbacks, read by the periodic function
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Latest obstacle and plant regions received from the ground detection module. */
static struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
static struct obstacle_region_t plants[MAX_PLANT_REGIONS];

/* Ground boundary array — one row index per image column indicating where the
 * top of the visible green ground is.  Low value = lots of clear ground ahead. */
static int16_t boundary_rows[MAX_IMAGE_HEIGHT];
static float   boundary_rows_f[MAX_IMAGE_HEIGHT];

static uint8_t  obstacle_count = 0;
static uint8_t  plant_count    = 0;
static uint16_t boundary_len   = 0;

/* Sum of widths of all ground-touching obstacles in the current frame.
 * Used to drive the confidence counter — if this exceeds the threshold the
 * counter decrements, signalling that the path ahead is blocked.              */
uint16_t total_obstacle_width = 0;

/* ══════════════════════════════════════════════════════════════════════════════
 *  RUNTIME-TUNABLE PARAMETERS
 *  All variables below are non-static globals exposed as GCS sliders via the
 *  module XML.  They can be adjusted in flight without recompiling.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Fraction of image width that ground-touching obstacles must cover before the
 * confidence counter starts decrementing.  Lower values make the drone react
 * to smaller or more distant obstacles sooner.                                */
float obstacle_width_threshold = 0.10f;

/* Safety margin added on each side of a detected obstacle, expressed as a
 * fraction of that obstacle's own column width.  A wall that is 40 columns
 * wide with obs_bias_frac = 0.5 will have 20 extra columns erased on each
 * side before the safe corridor search runs.                                  */
float obs_bias_frac = 0.65f;

/* Same as obs_bias_frac but applied to detected plant pots.  Set higher than
 * obs_bias_frac because plant pots are physically narrow, so a fraction of
 * their width produces a small absolute margin — they need proportionally
 * more clearance to be safe.                                                  */
float plant_bias_frac = 1.5f;

/* Number of 10 Hz periodic ticks between WP_GOAL advances while in SAFE.
 * WP_TRAJECTORY still moves every tick as a lookahead probe for the bounds
 * check, but WP_GOAL — the target the flight controller actually chases —
 * advances at this slower rate so the drone has time to physically reach each
 * waypoint before the next one is set.                                        */
int wp_update_period_ticks = 2;

/* Fraction of image height that a column's baseline value must be below for
 * that column to count as clear.  Higher = stricter — the drone demands a
 * deeper view of open ground before treating a column as passable.            */
float clear_frac = 0.90f;

/* Minimum width in pixels that a contiguous run of clear columns must have to
 * qualify as a usable corridor.  Narrower runs are discarded so the drone
 * never tries to pass through a gap too narrow for its body.                  */
int min_corridor_width_px = 60;

/* ── Gate detection state ──────────────────────────────────────────────────── */

/* Whether the gate is currently detected and where its centre column is.      */
static uint8_t cur_gate_detected = 0;
static int     cur_gate_center_x = 0;

/* Number of consecutive frames in which the gate has been detected.
 * OBSTACLE_FOUND only transitions to GATE_APPROACH once this reaches
 * GATE_CONFIRM_STREAK, preventing false positives from single noisy frames.  */
static uint8_t gate_seen_streak = 0;

/* Countdown timer for rate-limiting WP_GOAL updates in the SAFE state.       */
static uint8_t wp_update_ticks = 0;

/* ══════════════════════════════════════════════════════════════════════════════
 *  ABI — GROUND DETECTION CALLBACK
 *
 *  Called by the ABI bus each time the ground detection module publishes a new
 *  frame result.  Copies obstacle regions, plant regions, and the boundary row
 *  array into local buffers for use by the periodic state machine.
 *
 *  Also computes total_obstacle_width: the sum of column widths of all
 *  ground-touching obstacles (those whose baseline_height equals the full image
 *  height, meaning no ground was found beneath them).  This is the metric that
 *  drives the confidence counter in the periodic function.
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

    /* Sum widths of obstacles that physically block the path (touch the ground).
     * Floating obstacles whose baseline does not reach image height are ignored
     * because the drone can pass beneath or beside them without danger.        */
    total_obstacle_width = 0;
    for (uint8_t i = 0; i < in_oc; i++) {
        if (in_obs[i].baseline_height >= MAX_IMAGE_HEIGHT)
            total_obstacle_width += in_obs[i].width;
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ABI — GATE DETECTION CALLBACK
 *
 *  Called each time the gate detection pipeline publishes a result.
 *  Stores whether a gate was seen and where its centre column is.
 *  Maintains gate_seen_streak so OBSTACLE_FOUND can require multiple
 *  consecutive detections before committing to the gate approach path.
 * ══════════════════════════════════════════════════════════════════════════════ */
#ifndef TEAM10_GATE_DETECTION_ID
#define TEAM10_GATE_DETECTION_ID ABI_BROADCAST
#endif

static abi_event gate_detection_ev;

static void gate_detection_callback(
    uint8_t __attribute__((unused)) sender_id,
    float in_gate_detected,
    float in_gate_center_x)
{
    cur_gate_detected = (uint8_t)(in_gate_detected > 0.5f);
    cur_gate_center_x = (int)in_gate_center_x;

    if (in_gate_detected > 0.5f)
        gate_seen_streak++;
    else
        gate_seen_streak = 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  INIT
 *
 *  Seeds the random number generator, runs an initial corridor check to set a
 *  sensible starting heading_increment, then registers both ABI callbacks.
 * ══════════════════════════════════════════════════════════════════════════════ */
void ground_obstacle_avoidance_init(void)
{
    srand(time(NULL));

    /* Run the corridor logic once at startup.  All arrays are empty so this
     * will almost certainly return -1, in which case the default heading
     * increment of +5 degrees is kept.                                        */
    int safe_col = motion_logic_normalised(
            obstacles, obstacle_count,
            plants,    plant_count,
            boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
            obs_bias_frac,
            plant_bias_frac);
    if (safe_col >= 0)
        chooseWiseIncrementAvoidance(safe_col);

    AbiBindMsgTEAM10_GROUND_DETECTION(
            TEAM10_GROUND_DETECTION_ID, &ground_detection_ev,
            ground_detection_callback);

    AbiBindMsgTEAM10_GATE_DETECTION(
            TEAM10_GATE_DETECTION_ID, &gate_detection_ev,
            gate_detection_callback);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  PERIODIC — STATE MACHINE  (runs at 10 Hz)
 *
 *  Each tick:
 *    1. Update obstacle_free_confidence based on total_obstacle_width.
 *    2. Compute the distance the drone is allowed to move this tick.
 *    3. Run the appropriate state case.
 * ══════════════════════════════════════════════════════════════════════════════ */
void ground_obstacle_avoidance_periodic(void)
{
    if (!autopilot_in_flight()) return;

    printf("total obstacle width: %.2f\nthreshold (fraction): %.2f\nthreshold (total):    %.2f\n",
           total_obstacle_width / (float)MAX_IMAGE_WIDTH,
           obstacle_width_threshold,
           obstacle_width_threshold * MAX_IMAGE_WIDTH);

    /* Update confidence: +1 each tick the path looks clear, -2 each tick it
     * is blocked.  The asymmetry makes the drone react quickly to new obstacles
     * while recovering more cautiously when the path clears.                  */
    if (total_obstacle_width < obstacle_width_threshold * MAX_IMAGE_WIDTH) {
        obstacle_free_confidence++;
    } else {
        obstacle_free_confidence -= 2;
    }
    Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

    /* Scale move distance with confidence so the drone slows down as it
     * becomes less certain and stops completely at confidence zero.            */
    float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

    switch (navigation_state) {

    /* ── SAFE ────────────────────────────────────────────────────────────────
     * The drone moves forward by pushing both waypoints ahead.
     * WP_TRAJECTORY is the lookahead probe used for the bounds check.
     * WP_GOAL is the target the flight controller actually chases, and is
     * only updated every wp_update_period_ticks ticks to avoid moving the
     * target faster than the drone can physically follow.
     *
     * Before moving, a corridor check is run.  If no passable corridor exists
     * the drone hard-stops immediately — both waypoints are pinned to the
     * current position, confidence is reset, and the state jumps to
     * OBSTACLE_FOUND without waiting for confidence to drain naturally.        */
    case SAFE:
        {
            int safe_col = motion_logic_normalised(
                    obstacles, obstacle_count,
                    plants,    plant_count,
                    boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
                    obs_bias_frac,
                    plant_bias_frac);

            if (safe_col < 0) {
                VERBOSE_PRINT("No corridor in SAFE — hard stop\n");
                waypoint_move_here_2d(WP_GOAL);
                waypoint_move_here_2d(WP_TRAJECTORY);
                obstacle_free_confidence = 0;
                wp_update_ticks          = 0;
                navigation_state         = OBSTACLE_FOUND;
                break;
            }
        }

        moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
        if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
            navigation_state = OUT_OF_BOUNDS;
        } else if (obstacle_free_confidence == 0) {
            wp_update_ticks  = 0;
            navigation_state = OBSTACLE_FOUND;
        } else {
            if (wp_update_ticks == 0) {
                moveWaypointForward(WP_GOAL, moveDistance);
                wp_update_ticks = (uint8_t)wp_update_period_ticks;
            } else {
                wp_update_ticks--;
            }
        }
        break;

    /* ── OBSTACLE_FOUND ──────────────────────────────────────────────────────
     * The drone has stopped.  Both waypoints are pinned to the current position
     * so the flight controller holds the drone in place.
     *
     * Two paths forward:
     *   Gate detected for GATE_CONFIRM_STREAK consecutive frames → GATE_APPROACH
     *   Otherwise → run corridor logic to find which way to rotate, then
     *               SEARCH_FOR_SAFE_HEADING.
     *
     * If the corridor logic returns -1 (no passable gap anywhere) the heading
     * increment is left unchanged so the drone continues rotating in the same
     * direction it was already going when it gets to SEARCH_FOR_SAFE_HEADING.  */
    case OBSTACLE_FOUND:
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);

        if (cur_gate_detected && gate_seen_streak >= GATE_CONFIRM_STREAK) {
            VERBOSE_PRINT("Gate confirmed (streak=%d) at col %d — GATE_APPROACH\n",
                          gate_seen_streak, cur_gate_center_x);
            chooseHeadingToGate(cur_gate_center_x);
            navigation_state = GATE_APPROACH;
        } else {
            int safe_col = motion_logic_normalised(
                    obstacles, obstacle_count,
                    plants,    plant_count,
                    boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
                    obs_bias_frac,
                    plant_bias_frac);
            if (safe_col >= 0)
                chooseWiseIncrementAvoidance(safe_col);
            navigation_state = SEARCH_FOR_SAFE_HEADING;
        }
        break;

    /* ── GATE_APPROACH ───────────────────────────────────────────────────────
     * A gate has been confirmed in front of the drone.  The drone yaws toward
     * the gate centre column using proportional control: the yaw rate is
     * proportional to the angular error so large offsets rotate fast and
     * small offsets make fine corrections without overshooting.
     *
     * Once the gate centre is within GATE_ALIGN_COLUMN_TOL pixels of the image
     * centre the drone is considered aligned.  Confidence is set to maximum
     * and both waypoints are pushed forward through the gate.
     *
     * If the gate is lost mid-approach the drone falls back to OBSTACLE_FOUND
     * to re-evaluate rather than flying blind.                                 */
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
                VERBOSE_PRINT("Aligned with gate (err=%d px) — approaching\n", gate_err);
                obstacle_free_confidence = max_trajectory_confidence;
                moveWaypointForward(WP_TRAJECTORY, GATE_APPROACH_DISTANCE_M);
                moveWaypointForward(WP_GOAL,       GATE_APPROACH_DISTANCE_M);
                navigation_state = SAFE;
            } else {
                /* Proportional yaw: normalise error to [-1, +1] and scale to
                 * the maximum yaw rate.  Positive error means the gate is to
                 * the right (higher column index in the rotated image) so a
                 * positive heading increment turns clockwise toward it.        */
                float err_norm = (float)gate_err / (float)img_centre;
                float yaw_deg  = err_norm * GATE_HEADING_MAX_DEG;
                if (yaw_deg >  GATE_HEADING_MAX_DEG) yaw_deg =  GATE_HEADING_MAX_DEG;
                if (yaw_deg < -GATE_HEADING_MAX_DEG) yaw_deg = -GATE_HEADING_MAX_DEG;
                VERBOSE_PRINT("Gate err=%d px  yaw=%.1f deg\n", gate_err, yaw_deg);
                increase_nav_heading(yaw_deg);
            }
        }
        break;

    /* ── SEARCH_FOR_SAFE_HEADING ─────────────────────────────────────────────
     * The drone rotates in place until a passable corridor appears.
     * Both waypoints are pinned to the current position every tick to prevent
     * any forward drift caused by the flight controller tracking a waypoint
     * that was set before the stop.
     *
     * The corridor logic is re-run every tick so that if a new obstacle enters
     * from the side the rotation direction updates immediately rather than
     * continuing toward a direction that is no longer safe.
     *
     * The state only transitions back to SAFE once a valid corridor exists AND
     * confidence has reached 2, meaning two consecutive clear ticks have been
     * seen — confirming the heading is genuinely safe before moving forward.   */
    case SEARCH_FOR_SAFE_HEADING:
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);

        {
            int safe_col = motion_logic_normalised(
                    obstacles, obstacle_count,
                    plants,    plant_count,
                    boundary_rows_f, boundary_len, MAX_IMAGE_HEIGHT,
                    obs_bias_frac,
                    plant_bias_frac);
            if (safe_col >= 0) {
                chooseWiseIncrementAvoidance(safe_col);
                increase_nav_heading(heading_increment);
                if (obstacle_free_confidence >= 2)
                    navigation_state = SAFE;
            } else {
                /* No corridor found — keep rotating in the current direction  */
                increase_nav_heading(heading_increment);
            }
        }
        break;

    /* ── OUT_OF_BOUNDS ───────────────────────────────────────────────────────
     * The trajectory waypoint has left the allowed obstacle zone (arena boundary).
     * The drone rotates and probes WP_TRAJECTORY forward each tick.  Once the
     * probe lands back inside the arena the drone resets confidence to zero
     * and goes to SEARCH_FOR_SAFE_HEADING to verify the heading before moving. */
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

/* Adds incrementDegrees to the current body heading and normalises to [-π, π].
 * Sets nav.heading which the rotorcraft firmware uses as the heading setpoint. */
uint8_t increase_nav_heading(float incrementDegrees)
{
    float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
    FLOAT_ANGLE_NORMALIZE(new_heading);
    nav.heading = new_heading;
    VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
    return false;
}

/* Sets heading_increment so the drone yaws toward the gate centre column.
 *
 * After the 90 degree CCW rotation applied by the camera pipeline:
 *   column 0   = drone's physical RIGHT → gate_col > W/2 → clockwise yaw (+deg)
 *   column W-1 = drone's physical LEFT  → gate_col < W/2 → counter-clockwise (-deg)
 *
 * This only sets the sign for the first tick in GATE_APPROACH.  Subsequent ticks
 * use proportional control directly on the gate error, bypassing this function. */
static uint8_t chooseHeadingToGate(int gate_col)
{
    if (gate_col > MAX_IMAGE_WIDTH / 2) {
        heading_increment = heading_increment_setting;
        VERBOSE_PRINT("Gate RIGHT: initial yaw +%.1f deg\n", heading_increment);
    } else if (gate_col < MAX_IMAGE_WIDTH / 2) {
        heading_increment = -heading_increment_setting;
        VERBOSE_PRINT("Gate LEFT: initial yaw %.1f deg\n", heading_increment);
    }
    return false;
}

/* Sets heading_increment based on which side of the image the safe corridor is.
 * safe_direction is the column index returned by motion_logic_normalised.
 * Columns above W/2 are to the drone's right (clockwise turn needed),
 * columns below W/2 are to the drone's left (counter-clockwise turn needed).  */
uint8_t chooseWiseIncrementAvoidance(int safe_direction)
{
    if (safe_direction > MAX_IMAGE_WIDTH / 2) {
        heading_increment = heading_increment_setting;
    } else if (safe_direction < MAX_IMAGE_WIDTH / 2) {
        heading_increment = -heading_increment_setting;
    }
    VERBOSE_PRINT("Safe col=%d  heading_increment=%.1f\n", safe_direction, heading_increment);
    return false;
}

/* Fallback: pick a random rotation direction.  Not used in normal operation
 * but kept in case a caller needs a direction when no information is available. */
uint8_t chooseRandomIncrementAvoidance(void)
{
    heading_increment = (rand() % 2 == 0) ? 5.f : -5.f;
    VERBOSE_PRINT("Random avoidance increment: %.1f\n", heading_increment);
    return false;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  WAYPOINT HELPERS
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Pushes a waypoint distanceMeters ahead of the current position and heading. */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
    distanceMeters = distanceMeters * speed_multiplier;
    struct EnuCoor_i new_coor;
    calculateForwards(&new_coor, distanceMeters);
    moveWaypoint(waypoint, &new_coor);
    return false;
}

/* Computes the ENU coordinates of a point distanceMeters ahead of the drone
 * along its current heading.  Uses sin/cos of the psi (yaw) Euler angle.     */
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

/* Moves a named waypoint to the given ENU coordinates via the navigation API. */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
    VERBOSE_PRINT("Moving WP %d to x=%f y=%f\n", waypoint,
                  POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
    waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
    return false;
}