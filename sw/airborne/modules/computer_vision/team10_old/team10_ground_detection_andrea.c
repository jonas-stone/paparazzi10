/*
 * team10_ground_detection.c — master module (obstacles + plants + gate)
 * get_obstacle_info handles obstacle/plant detection.
 * detect_gate() runs on the same frame and adds gate presence + centre.
 */
#include "modules/computer_vision/team10_ground_detection.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0
#endif

static float    ground_baseline[MAX_IMAGE_WIDTH];
static int      baseline_inited       = 0;
static bool     obstacles_updated     = false;
static uint8_t  global_obstacle_count = 0;
static uint8_t  global_plant_count    = 0;
struct obstacle_region_t global_obstacles[MAX_OBSTACLE_REGIONS];
struct obstacle_region_t global_plants[MAX_PLANT_REGIONS];

/* ── Gate state (protected by mutex, written by camera thread) ─────────── */
uint8_t gate_detected  = 0;   /* 0 = no gate, 1 = gate present             */
int gate_center_x  = 0;   /* px from left edge of raw image to centre  */

static struct image_t *detect_obstacles_from_ground(struct image_t *img,
        uint8_t camera_id __attribute__((unused)))
{
    int                      boundary_rows[MAX_IMAGE_WIDTH];
    struct obstacle_region_t local_obstacles[MAX_OBSTACLE_REGIONS];
    struct obstacle_region_t local_plants[MAX_PLANT_REGIONS];
    uint8_t                  plant_count = 0;

    uint8_t obstacle_count = get_obstacle_info(
            img, ground_baseline, &baseline_inited,
            0.05f, 5, 20,
            local_obstacles, local_plants, &plant_count,
            boundary_rows, NULL, NULL);

    /* ── Gate detection ────────────────────────────────────────────────── */
    int gate_result[2];   /* [0] = detected flag, [1] = centre x pixel     */
    detect_gate(img, gate_result);

    pthread_mutex_lock(&mutex);
    memcpy(global_obstacles, local_obstacles, MAX_OBSTACLE_REGIONS * sizeof(struct obstacle_region_t));
    memcpy(global_plants, local_plants, MAX_PLANT_REGIONS * sizeof(struct obstacle_region_t));
    global_obstacle_count = obstacle_count;
    global_plant_count    = plant_count;
    gate_detected         = (uint8_t)gate_result[0];
    gate_center_x         = gate_result[1];
    obstacles_updated     = true;
    for (int i = 0; i < obstacle_count; i++)
        printf("Obstacle %d: left=%d width=%d\n", i, global_obstacles[i].start, global_obstacles[i].width);
    for (int i = 0; i < plant_count; i++)
        printf("Plant    %d: start=%d width=%d\n", i, global_plants[i].start, global_plants[i].width);
    if (gate_detected)
        printf("Gate: detected  centre_x=%d px\n", gate_center_x);
    else
        printf("Gate: not detected\n");
    pthread_mutex_unlock(&mutex);
    return img;
}

void ground_detection_init(void)
{
    memset(global_obstacles, 0, sizeof(global_obstacles));
    memset(global_plants, 0, sizeof(global_plants));
    pthread_mutex_init(&mutex, NULL);
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, detect_obstacles_from_ground, COLOR_OBJECT_DETECTOR_FPS1, 0);
}

void ground_detection_periodic(void)
{
    struct obstacle_region_t lo[MAX_OBSTACLE_REGIONS], lp[MAX_PLANT_REGIONS];
    uint8_t oc, pc, gd, gx;
    pthread_mutex_lock(&mutex);
    if (!obstacles_updated) { pthread_mutex_unlock(&mutex); return; }
    oc = global_obstacle_count; pc = global_plant_count;
    gd = gate_detected;
    gx = gate_center_x;
    memcpy(lo, global_obstacles, sizeof(lo));
    memcpy(lp, global_plants, sizeof(lp));
    obstacles_updated = false;
    pthread_mutex_unlock(&mutex);
    AbiSendMsgTEAM10_GROUND_DETECTION(TEAM10_GROUND_DETECTION_ID, lo, oc);
    AbiSendMsgTEAM10_GATE_DETECTION(TEAM10_GROUND_DETECTION_ID, gd, gx);
    (void)pc; (void)lp;
}