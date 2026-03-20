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
#define DEFAULT_NO_GROUND_BASE    220
#define DEFAULT_BLOB_AREA_THRESH  1000
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
uint8_t update_and_detect(const int boundary_row[], int h, int w, float ground_baseline[], int *baseline_inited, int min_width, int obstacle_threshold, int no_ground_baseline, int max_col_gap, struct obstacle_region_t obstacles_out[]);
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

/* ══════════════════════════════════════════════════════════════════════════════
 *  GATE DETECTION API
 * ══════════════════════════════════════════════════════════════════════════════
 *
 * detect_gate() scans the image for two parallel, similarly-sized blue blobs
 * that correspond to the TU Delft gate pillars, confirms each with a checker
 * pattern check, and returns the gate centre.
 *
 * Output array layout:
 *   result[0] : 0 = no gate, 1 = gate detected
 *   result[1] : pixel distance from the LEFT edge of the image to the gate
 *               centre X coordinate (= distance from the TOP edge when the
 *               image is displayed upright after 90° CCW rotation)
 *
 * The image is assumed to be in the same YUV422 (UYVY) format used throughout
 * this module, and to be portrait/rotated (floor on the LEFT side).
 */

/* ── YUV thresholds for TU Delft blue ─────────────────────────────────────── */
#ifndef GATE_BLUE_Y_MIN
#define GATE_BLUE_Y_MIN     0
#endif
#ifndef GATE_BLUE_Y_MAX
#define GATE_BLUE_Y_MAX   220
#endif
#ifndef GATE_BLUE_U_MIN
#define GATE_BLUE_U_MIN   122   /* blue drives Cb well above 128 */
#endif
#ifndef GATE_BLUE_U_MAX
#define GATE_BLUE_U_MAX   255
#endif
#ifndef GATE_BLUE_V_MIN
#define GATE_BLUE_V_MIN     0
#endif
#ifndef GATE_BLUE_V_MAX
#define GATE_BLUE_V_MAX   123   /* blue keeps Cr well below 128  */
#endif

/* ── Blob filtering ─────────────────────────────────────────────────────────── */
/* Minimum blob area as a fraction of total image pixels, in units of 1/100000 */
/* Default 0.003 * 100000 = 300 → area >= w*h*300/100000                       */
#ifndef GATE_BLOB_MIN_AREA_NUM
#define GATE_BLOB_MIN_AREA_NUM    3
#endif
#ifndef GATE_BLOB_MIN_AREA_DEN
#define GATE_BLOB_MIN_AREA_DEN 1000
#endif
#ifndef GATE_BLOB_MIN_DIM
#define GATE_BLOB_MIN_DIM        10   /* px — kills thin noise lines           */
#endif

/* ── Parallelism thresholds ─────────────────────────────────────────────────── */
/* Max relative aspect-ratio difference, in units of 1/100 (50 = 50%)          */
#ifndef GATE_MAX_ASPECT_DIFF_PCT
#define GATE_MAX_ASPECT_DIFF_PCT  50
#endif
/* Max relative area difference, in units of 1/100 (60 = 60%)                  */
#ifndef GATE_MAX_AREA_DIFF_PCT
#define GATE_MAX_AREA_DIFF_PCT    60
#endif
/* Max skew of centroid-join from perpendicular, in degrees * 10 (100 = 10°)   */
#ifndef GATE_MAX_SKEW_DEG10
#define GATE_MAX_SKEW_DEG10      100
#endif

/* ── Checker confirmation ─────────────────────────────────────────────────── */
/* Search band width = blob_width * GATE_CHECKER_BAND_NUM / GATE_CHECKER_BAND_DEN */
#ifndef GATE_CHECKER_BAND_NUM
#define GATE_CHECKER_BAND_NUM   4
#endif
#ifndef GATE_CHECKER_BAND_DEN
#define GATE_CHECKER_BAND_DEN   5
#endif
/* Minimum grayscale range (max-min) in a column to count as "high contrast"   */
#ifndef GATE_CHECKER_CONTRAST
#define GATE_CHECKER_CONTRAST  120
#endif
/* Min fraction of columns passing contrast test, in units of 1/100 (35 = 35%) */
#ifndef GATE_CHECKER_MIN_FRAC_PCT
#define GATE_CHECKER_MIN_FRAC_PCT  35
#endif

/* Maximum number of blue blobs tracked during gate detection                  */
#ifndef GATE_MAX_BLOBS
#define GATE_MAX_BLOBS  16
#endif

/* ── Public function ─────────────────────────────────────────────────────── */
void detect_gate(struct image_t *img, int result[2]);