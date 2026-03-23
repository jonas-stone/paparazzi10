/*
 * team10_get_obstacle_info.h
 *
 * Complete C port of the Python obstacle+plant detection pipeline
 * including is_smooth_blob (perimeter ratio + fractal dimension).
 */

#ifndef TEAM10_GET_OBSTACLE_INFO_H
#define TEAM10_GET_OBSTACLE_INFO_H

#include <stdint.h>
#include <stdbool.h>

/* ── image dimensions (both 520 to handle sideways camera) ─────────────────── */
#ifndef MAX_IMAGE_WIDTH
#define MAX_IMAGE_WIDTH   520
#endif
#ifndef MAX_IMAGE_HEIGHT
#define MAX_IMAGE_HEIGHT  520
#endif

/* buffer size includes +2 padding for smooth blob analysis */
#define MAX_PIXELS ((MAX_IMAGE_WIDTH + 2) * (MAX_IMAGE_HEIGHT + 2))

/* ── output limits ─────────────────────────────────────────────────────────── */
#ifndef MAX_OBSTACLE_REGIONS
#define MAX_OBSTACLE_REGIONS 20
#endif
#ifndef MAX_PLANT_REGIONS
#define MAX_PLANT_REGIONS 20
#endif

#ifndef MAX_CC_LABELS
#define MAX_CC_LABELS 512
#endif

/* ── detection type flags ──────────────────────────────────────────────────── */
#define DET_OBSTACLE  0
#define DET_PLANT     1

/* ── obstacle pipeline defaults (calibrated for full resolution) ───────────── */
#define DEFAULT_OA_COLOR_FRAC     0.05f
#define DEFAULT_MEDIAN_KSIZE      5
#define DEFAULT_MIN_WIDTH         20
#define DEFAULT_MAX_COL_GAP       5
#define DEFAULT_MIN_GROUND_PX     5
#define DEFAULT_MAX_GAP           10
#define DEFAULT_SMOOTH_KERNEL     5
#define DEFAULT_OBSTACLE_THRESH   50
#define DEFAULT_NO_GROUND_BASE    235
#define DEFAULT_BLOB_AREA_THRESH  200
#define DEFAULT_BASELINE_ALPHA    0.6f

/* ── smooth blob thresholds ────────────────────────────────────────────────── */
#define SMOOTH_PERIMETER_RATIO_THRESH  2.0f
#define SMOOTH_FRACTAL_DIM_THRESH      1.3f

/* ── plant pipeline defaults ───────────────────────────────────────────────── */
#define DEFAULT_PLANT_SCALE_NUM     3
#define DEFAULT_PLANT_SCALE_DEN    25
#define DEFAULT_PLANT_BLUR_KSIZE    3
#define DEFAULT_PLANT_MIN_WIDTH    20
#define DEFAULT_PLANT_MIN_PX_COL    2
#define DEFAULT_PLANT_MAX_COL_GAP   5

#define MAX_PLANT_WIDTH  ((MAX_IMAGE_WIDTH  * DEFAULT_PLANT_SCALE_NUM / DEFAULT_PLANT_SCALE_DEN) + 2)
#define MAX_PLANT_HEIGHT ((MAX_IMAGE_HEIGHT * DEFAULT_PLANT_SCALE_NUM / DEFAULT_PLANT_SCALE_DEN) + 2)

/* ── lax green thresholds (YUV, 0-255) ─────────────────────────────────────── */
#define LAX_GREEN_Y_MIN   29
#define LAX_GREEN_Y_MAX  140
#define LAX_GREEN_U_MIN    0
#define LAX_GREEN_U_MAX  116
#define LAX_GREEN_V_MIN    0
#define LAX_GREEN_V_MAX  141

/* ── region descriptor ─────────────────────────────────────────────────────── */
struct obstacle_region_t {
    uint16_t start;
    uint16_t width;
    uint16_t baseline_height;
};

/* ══════════════════════════════════════════════════════════════════════════════
 *  PUBLIC API
 * ══════════════════════════════════════════════════════════════════════════════ */
struct image_t;

uint8_t get_obstacle_info(
        struct image_t            *img,
        float                      ground_baseline[],
        int                       *baseline_inited,
        float                      oa_color_count_frac,
        int                        median_ksize,
        int                        min_width,
        struct obstacle_region_t   obstacles_out[],
        struct obstacle_region_t   plants_out[],
        uint8_t                   *plant_count_out,
        int                        boundary_rows_out[],
        int                       *ground_found_out,
        float                     *green_frac_out
);

/* ── exposed helpers for testing ───────────────────────────────────────────── */
uint8_t is_ground_pixel(uint8_t Y, uint8_t U, uint8_t V);
void detect_green_ground_ml(struct image_t *img, uint8_t mask_out[], int median_ksize, float *green_frac_out);
void isolate_ground_blob(uint8_t mask[], int w, int h, int blob_area_threshold);
void fill_holes_mask(uint8_t mask[], int w, int h);
void find_ground_boundary(const uint8_t mask_flipped[], int w, int h, int boundary_out[], int min_ground_pixels, int max_gap, int smooth_kernel);
uint8_t update_and_detect(const int boundary_row[], int h, int w, float ground_baseline[], int *baseline_inited, int min_width, int obstacle_threshold, int no_ground_baseline, int max_col_gap, struct obstacle_region_t obstacles_out[],
                          const uint8_t *img_buf, int img_w, int img_h);
void detect_all_green_lax(struct image_t *img, int scale_num, int scale_den, int blur_ksize, uint8_t plant_mask_out[], int *pw_out, int *ph_out);
uint8_t detect_plant_regions(const uint8_t plant_mask[], int w, int h, int min_width, int min_pixels_per_col, int max_col_gap, struct obstacle_region_t plants_out[]);

/* smooth blob analysis */
int  compute_blob_perimeter(const int16_t *labels, int w, int h, int16_t lbl);
float compute_fractal_dimension(const int16_t *labels, int w, int h, int16_t lbl);
int  is_smooth_blob(const int16_t *labels, int w, int h, int16_t lbl, int blob_area, int bb_height);

/* debug */
int dump_mask_to_file(const char *path, const uint8_t mask[], int w, int h);
int dump_obstacles_to_file(const char *path, const struct obstacle_region_t obs[], int count);

#endif