/*
 * team10_RANSAC_adaptation.h
 */

#ifndef TEAM10_RANSAC_ADAPTATION_H
#define TEAM10_RANSAC_ADAPTATION_H

#include <stdint.h>
#include <stdbool.h>
#include "modules/computer_vision/lib/vision/image.h"

/* ══════════════════════════════════════════════════════════════════════════════
 *  STRUCTS
 * ══════════════════════════════════════════════════════════════════════════════ */
struct column_bucket_t {
    uint16_t edge_pixel_count; // for plant detection
    int black_below_ground_line;
    int white_below_ground_line;
    int black_above_ground_line;
    int white_above_ground_line;
};

/* ══════════════════════════════════════════════════════════════════════════════
 *  RANSAC CORE
 * ══════════════════════════════════════════════════════════════════════════════ */

void RANSAC_linear_model(int n_samples, int n_iterations, float error_threshold, float *targets, int D,
                         float (*samples)[1], uint16_t count, bool use_bias, int degree, float *params,
                         float *fit_error);

void get_indices_without_replacement(int *indices_subset, int n_samples, int count);

float predict_value(float *sample, float *weights, int D, int degree, bool use_bias);

/* ══════════════════════════════════════════════════════════════════════════════
 *  EDGE + COORDINATE EXTRACTION
 * ══════════════════════════════════════════════════════════════════════════════ */

int get_blob_edge_mask(uint8_t *binary_mask, int w, int h, uint8_t *edge_out);

void get_edge_pixel_coordinates(int w, int h, uint8_t *edge_mask, float (*edge_coordinates)[2]);

/* ══════════════════════════════════════════════════════════════════════════════
 *  PARABOLA
 * ══════════════════════════════════════════════════════════════════════════════ */

int eval_parabola(float *coeffs, float x);


// for pixel color stuff
typedef enum {
  GREEN,
  RED,
  BLUE,
  WHITE,
  BLACK,
} yuv_color_e;

typedef struct {
  uint8_t U;
  uint8_t Y;
  uint8_t V;
} yuv_color_t;

static const yuv_color_t yuv_colors[] = {
  [GREEN] = {44,  150,  21},
  [RED]   = {90,   76, 255},
  [BLUE]  = {255,  29, 107},
  [WHITE] = {128, 255, 128},
  [BLACK] = {128,   0, 128},
};

void draw_neon_green_pixel(struct image_t *img, int row, int col);
void draw_colored_pixel(struct image_t *img, int row, int col, yuv_color_e color);

#endif /* TEAM10_RANSAC_ADAPTATION_H */