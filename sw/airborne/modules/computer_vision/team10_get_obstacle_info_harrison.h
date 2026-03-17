/*
 * team10_get_obstacle_info.h
 *
 * Ground detection and obstacle finding pipeline.
 *
 * Changelog vs. previous version
 * --------------------------------
 * 1. Added MAX_IMAGE_HEIGHT constant (matches MAX_IMAGE_WIDTH = 520).
 * 2. No changes to any public function signatures.
 */

#ifndef TEAM10_GET_OBSTACLES_INFO_H
#define TEAM10_GET_OBSTACLES_INFO_H

#include "modules/computer_vision/lib/vision/image.h"
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

#define OBSTACLE_ALPHA       0.6f
#define OBSTACLE_THRESHOLD   50.0f
#define NO_GROUND_BASELINE   220.0f
#define MAX_OBSTACLE_REGIONS 8
#define MAX_IMAGE_WIDTH      520
#define MAX_IMAGE_HEIGHT     520   /* added: needed for static buffer sizing */

/* ------------------------------------------------------------------ */
/* Shared struct                                                        */
/* ------------------------------------------------------------------ */

/**
 * A contiguous obstacle region expressed in original (un-flipped)
 * image column coordinates.
 *
 *  start : leftmost column  (inclusive)
 *  end   : rightmost column (inclusive)
 *  width : end - start + 1
 */
struct obstacle_region_t {
  uint16_t start;
  uint16_t end;
  uint16_t width;
};

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

/**
 * detect_green_ground_ml
 *
 * Classifies every pixel of a YUV422 image with the decision-tree
 * ground classifier, writes a binary grayscale mask (255 = ground,
 * 0 = not ground), optionally applies a 3x3 median blur, and returns
 * the ground fraction plus a found/not-found flag.
 *
 * @param input          Source IMAGE_YUV422
 * @param mask_out       Pre-created IMAGE_GRAYSCALE, same dimensions
 * @param threshold      Fraction in [0,1]; above this -> ground found
 * @param apply_median   Non-zero -> apply 3x3 median blur on the mask
 * @param use_sim        0 = real flight (is_ground), 1 = simulator (is_ground_sim)
 * @param green_fraction OUTPUT: fraction of pixels classified as ground
 * @return               1 = GROUND FOUND, 0 = NO GROUND
 */
int detect_green_ground_ml(struct image_t *input,
                           struct image_t *mask_out,
                           float           threshold,
                           int             apply_median,
                           int             use_sim,
                           float          *green_fraction);

/**
 * find_ground_boundary
 *
 * For each column of a horizontally-flipped grayscale ground mask,
 * finds the row index of the ground boundary (furthest safe ground
 * pixel), then smooths the result with a 1-D median filter.
 *
 * @param mask_flipped       IMAGE_GRAYSCALE, dimensions W x H
 * @param boundary_rows_out  Caller-supplied int array of length W.
 *                           Set to H when no ground found in that column.
 * @param min_ground_pixels  Minimum ground pixels required at far edge
 * @param max_gap            Maximum gap allowed inside a ground run
 * @param smooth_kernel      Odd window size for 1-D median smoothing
 */
void find_ground_boundary(const struct image_t *mask_flipped,
                          int                  *boundary_rows_out,
                          int                   min_ground_pixels,
                          int                   max_gap,
                          int                   smooth_kernel);

/**
 * get_obstacle_regions
 *
 * Groups a sorted list of obstacle column indices into contiguous
 * regions, bridging gaps of at most max_col_gap columns and keeping
 * only regions with width >= min_width.
 *
 * @param obstacle_cols  Sorted array of obstacle column indices
 * @param n_cols         Length of obstacle_cols
 * @param min_width      Minimum region width to keep
 * @param max_col_gap    Maximum gap to bridge
 * @param regions_out    Caller-supplied array, size MAX_OBSTACLE_REGIONS
 * @return               Number of regions written
 */
uint8_t get_obstacle_regions(const uint16_t           *obstacle_cols,
                            int                       n_cols,
                            int                       min_width,
                            int                       max_col_gap,
                            struct obstacle_region_t *regions_out);

/**
 * update_and_detect
 *
 * Updates the rolling ground baseline with an EMA and detects obstacle
 * regions by comparing the current boundary against the baseline.
 *
 * @param boundary_row    int array of length `width` (current boundaries)
 * @param width           Image width
 * @param h               Image height; boundary_row[x] >= h -> no ground
 * @param ground_baseline Float array of length `width` (persistent state)
 * @param baseline_inited Flag; set to 0 before the very first call
 * @param min_width       Minimum obstacle region width to report
 * @param max_col_gap     Maximum column gap to bridge when grouping regions
 * @param regions_out     Caller-supplied array, size MAX_OBSTACLE_REGIONS
 * @return                Number of obstacle regions (0 on first call)
 */
uint8_t update_and_detect(const int  *boundary_row,
                        int         width,
                        int         h,
                        float      *ground_baseline,
                        int        *baseline_inited,
                        int         min_width,
                        int         max_col_gap,
                        struct obstacle_region_t *regions_out);

/**
 * get_obstacle_info
 *
 * Full pipeline: ground mask -> blob isolation -> hole filling ->
 *                boundary finding -> obstacle detection.
 * Results are expressed in original (un-flipped) image coordinates.
 *
 * NEW vs. previous version: the raw mask is now cleaned by
 * isolate_ground_blob() + fill_holes() before boundary detection,
 * matching the Python pipeline. Function signature is UNCHANGED.
 *
 * @param input               Source IMAGE_YUV422
 * @param ground_baseline     Float array of length input->w (persistent)
 * @param baseline_inited     Flag pointer; set to 0 before first call
 * @param oa_color_count_frac Ground fraction threshold
 * @param median_ksize        Odd kernel for median blur (0/1 = skip)
 * @param min_width           Minimum obstacle region width (columns)
 * @param min_ground_pixels   Minimum ground pixels at the far edge per col
 * @param max_gap             Maximum gap allowed inside a ground run
 * @param smooth_kernel       Odd kernel size for 1-D boundary smoothing
 * @param max_col_gap         Maximum column gap when grouping obstacle cols
 * @param use_sim             0 = real flight (is_ground), 1 = simulator (is_ground_sim)
 * @param obstacles_out       Caller array of obstacle_region_t,
 *                            size MAX_OBSTACLE_REGIONS
 * @param boundary_rows_out   Caller int array of length input->w
 * @param ground_found_out    OUTPUT: 1 if ground was detected, else 0
 * @param green_frac_out      OUTPUT: ground pixel fraction (may be NULL)
 * @return                    Number of obstacles found (0 when no ground)
 */
uint8_t get_obstacle_info(struct image_t           *input,
                        float                    *ground_baseline,
                        int                      *baseline_inited,
                        float                     oa_color_count_frac,
                        int                       median_ksize,
                        int                       min_width,
                        int                       min_ground_pixels,
                        int                       max_gap,
                        int                       smooth_kernel,
                        int                       max_col_gap,
                        int                       use_sim,
                        struct obstacle_region_t *obstacles_out,
                        int                      *boundary_rows_out,
                        int                      *ground_found_out,
                        float                    *green_frac_out);

#endif /* TEAM10_GET_OBSTACLES_INFO_H */
