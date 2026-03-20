/*
 * team10_ground_detection.c — master module (obstacles + plants)
 *
 * Downscales camera frame to 0.8× before processing for performance,
 * then scales obstacle/plant coordinates back to native resolution
 * so the autopilot always works in native camera space.
 *
 * Coordinate convention (after standardization):
 *   obstacle_region_t.start = first row index (0-based)
 *   obstacle_region_t.width = number of consecutive rows
 *   Same format for both obstacles and plants.
 *
 *   After 90° CCW rotation:
 *     row 0        = RIGHT side of drone's view
 *     row H-1      = LEFT side of drone's view
 *     row H/2      = CENTER
 */
#include "modules/computer_vision/team10_ground_detection.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/team10_rtp_utilities.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <team10_rtp_utilities.h>
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

/* ── Downscale settings ───────────────────────────────────────────────────── */
/* Scale factor as fraction: 4/5 = 0.8×                                      */
/* Bebop2 camera: 240×520 → scaled: 192×416                                  */
#define SCALE_NUM  5
#define SCALE_DEN  5

#define MAX_SCALED_W  ((MAX_IMAGE_WIDTH  * SCALE_NUM / SCALE_DEN) + 2)
#define MAX_SCALED_H  ((MAX_IMAGE_HEIGHT * SCALE_NUM / SCALE_DEN) + 2)

static uint8_t  scaled_buf[MAX_SCALED_W * MAX_SCALED_H * 2];
static struct image_t scaled_img;
static int scaled_img_inited = 0;

/* ── Persistent state ─────────────────────────────────────────────────────── */
static float    ground_baseline[MAX_IMAGE_HEIGHT];
static int      baseline_inited       = 0;
static bool     obstacles_updated     = false;
static uint8_t  global_obstacle_count = 0;
static uint8_t  global_plant_count    = 0;
struct obstacle_region_t global_obstacles[MAX_OBSTACLE_REGIONS];
struct obstacle_region_t global_plants[MAX_PLANT_REGIONS];
static int16_t  global_boundary[MAX_IMAGE_HEIGHT];
static uint16_t global_boundary_len = 0;

/* ══════════════════════════════════════════════════════════════════════════════
 *  YUV422 NEAREST-NEIGHBOUR DOWNSAMPLE
 * ══════════════════════════════════════════════════════════════════════════════ */
static void downsample_yuv422(struct image_t *src, struct image_t *dst,
                               int dst_w, int dst_h)
{
    const uint8_t *sb = (const uint8_t *)src->buf;
    uint8_t *db = (uint8_t *)dst->buf;
    int sw = src->w;
    int sh = src->h;

    for (int dy = 0; dy < dst_h; dy++) {
        int sy = (dy * sh) / dst_h;
        if (sy >= sh) sy = sh - 1;

        for (int dx = 0; dx < dst_w; dx += 2) {
            int sx = (dx * sw) / dst_w;
            sx &= ~1;  /* align to macro-pixel boundary */
            if (sx >= sw) sx = sw - 2;

            int si = sy * sw * 2 + sx * 2;
            int di = dy * dst_w * 2 + dx * 2;
            db[di]     = sb[si];     /* U  */
            db[di + 1] = sb[si + 1]; /* Y0 */
            db[di + 2] = sb[si + 2]; /* V  */
            db[di + 3] = sb[si + 3]; /* Y1 */
        }
    }

    dst->type = IMAGE_YUV422;
    dst->w = (uint16_t)dst_w;
    dst->h = (uint16_t)dst_h;
    dst->buf_size = (uint32_t)(dst_w * dst_h * 2);
    dst->ts = src->ts;
    dst->eulers = src->eulers;
    dst->pprz_ts = src->pprz_ts;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  CAMERA CALLBACK
 * ══════════════════════════════════════════════════════════════════════════════ */
static struct image_t *detect_obstacles_from_ground(struct image_t *img,
        uint8_t camera_id __attribute__((unused)))
{
    /* ── Compute scaled dimensions ────────────────────────────────────────── */
    int dst_w = (img->w * SCALE_NUM) / SCALE_DEN;
    int dst_h = (img->h * SCALE_NUM) / SCALE_DEN;
    if (dst_w % 2) dst_w--;

    /* ── Init scaled image struct (once) ──────────────────────────────────── */
    if (!scaled_img_inited) {
        scaled_img.buf = scaled_buf;
        scaled_img_inited = 1;
    }

    /* ── Downsample ───────────────────────────────────────────────────────── */
    downsample_yuv422(img, &scaled_img, dst_w, dst_h);

    /* ── Run pipeline on scaled image ─────────────────────────────────────── */
    int                      boundary_rows[MAX_IMAGE_HEIGHT];
    struct obstacle_region_t local_obstacles[MAX_OBSTACLE_REGIONS];
    struct obstacle_region_t local_plants[MAX_PLANT_REGIONS];
    uint8_t                  plant_count = 0;

    obstacle_info_result_t result;
    uint8_t obstacle_count = get_obstacle_info(
            &scaled_img,
            ground_baseline,
            &baseline_inited,
            0.05f,   /* oa_color_count_frac */
            5,       /* median_ksize        */
            20,      /* min_width           */
            local_obstacles,
            local_plants,
            &plant_count,
            boundary_rows,
            NULL,
            NULL,
            &result
    );

    /* ── Scale coordinates back to native resolution ──────────────────────── */
    for (int i = 0; i < obstacle_count; i++) {
        local_obstacles[i].start           = (uint16_t)((local_obstacles[i].start * SCALE_DEN) / SCALE_NUM);
        local_obstacles[i].width           = (uint16_t)((local_obstacles[i].width * SCALE_DEN) / SCALE_NUM);
        local_obstacles[i].baseline_height = (uint16_t)((local_obstacles[i].baseline_height * SCALE_DEN) / SCALE_NUM);
    }
    for (int i = 0; i < plant_count; i++) {
        local_plants[i].start           = (uint16_t)((local_plants[i].start * SCALE_DEN) / SCALE_NUM);
        local_plants[i].width           = (uint16_t)((local_plants[i].width * SCALE_DEN) / SCALE_NUM);
        local_plants[i].baseline_height = 0;  /* not meaningful for plants */
    }

    /* ── Copy to globals ──────────────────────────────────────────────────── */
    uint16_t br_len = (uint16_t)dst_h;
    pthread_mutex_lock(&mutex);
    memcpy(global_obstacles, local_obstacles,
           MAX_OBSTACLE_REGIONS * sizeof(struct obstacle_region_t));
    memcpy(global_plants, local_plants,
           MAX_PLANT_REGIONS * sizeof(struct obstacle_region_t));
    global_obstacle_count = obstacle_count;
    global_plant_count    = plant_count;
    // copy boundary rows
    global_boundary_len = br_len;
    for (uint16_t i = 0; i < br_len; i++) {
        global_boundary[i] = (int16_t)((boundary_rows[i] * SCALE_DEN) / SCALE_NUM);
    }
    obstacles_updated = true;

    for (int i = 0; i < obstacle_count; i++)
        printf("Obstacle %d: start=%d width=%d\n",
               i, global_obstacles[i].start, global_obstacles[i].width);
    for (int i = 0; i < plant_count; i++)
        printf("Plant    %d: start=%d width=%d\n",
               i, global_plants[i].start, global_plants[i].width);
    pthread_mutex_unlock(&mutex);

    draw_mask_printer(img, result.mask, scaled_img.w, scaled_img.h);
    draw_toolbar_vertical(img, img->w, img->h, result.gf, obstacle_count, local_obstacles);
    draw_safe_direction_bar(img, img->w, img->h, obstacle_count, local_obstacles);
    draw_obstacle_detection_bar(img, img->w, img->h, obstacle_count, local_obstacles);

    return img;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  INIT + PERIODIC
 * ══════════════════════════════════════════════════════════════════════════════ */
void ground_detection_init(void)
{
    memset(global_obstacles, 0, sizeof(global_obstacles));
    memset(global_plants,    0, sizeof(global_plants));
    pthread_mutex_init(&mutex, NULL);
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1,
                     detect_obstacles_from_ground,
                     COLOR_OBJECT_DETECTOR_FPS1, 0);
}

void ground_detection_periodic(void)
{
    struct obstacle_region_t lo[MAX_OBSTACLE_REGIONS], lp[MAX_PLANT_REGIONS];
    int16_t  br[MAX_IMAGE_HEIGHT];
    uint8_t  oc, pc;
    uint16_t bl;

    pthread_mutex_lock(&mutex);
    if (!obstacles_updated) {
        pthread_mutex_unlock(&mutex);
        return;
    }
    oc = global_obstacle_count;
    pc = global_plant_count;
    bl = global_boundary_len;
    memcpy(lo, global_obstacles, sizeof(lo));
    memcpy(lp, global_plants,    sizeof(lp));
    memcpy(br, global_boundary,  bl * sizeof(int16_t));
    obstacles_updated = false;
    pthread_mutex_unlock(&mutex);

    // NOW send everything: obstacles, plants, and boundary
    AbiSendMsgTEAM10_GROUND_DETECTION(
        TEAM10_GROUND_DETECTION_ID,
        lo, oc,
        lp, pc,
        br, bl
    );
}
