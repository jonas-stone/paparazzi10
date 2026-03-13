/*
 * @file modules/computer_vision/solidity_detection.c
 * @brief Blob shape analysis for ground vs obstacle classification
 *
 * Translated from solidity_detection.py
 * Uses perimeter ratio and fractal dimension to determine if a blob
 * is smooth (ground) or irregular (plant/obstacle).
 */

#include "solidity_detection.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// -----------------------------------------------------------------------
// get_blob_edge
// Marks pixels where a transition between 0 and non-zero occurs.
// Equivalent to get_blob_edge() in solidity_detection.py
//
// edge_out must be same size as binary_img (w * h bytes)
// -----------------------------------------------------------------------
void get_blob_edge(uint8_t *binary_img, uint8_t *edge_out,
                   uint16_t w, uint16_t h)
{
  memset(edge_out, 0, w * h);

  // Horizontal transitions
  for (uint16_t row = 0; row < h; row++) {
    for (uint16_t col = 0; col < w - 1; col++) {
      uint8_t curr  = binary_img[row * w + col]     > 0 ? 1 : 0;
      uint8_t right = binary_img[row * w + col + 1] > 0 ? 1 : 0;
      if (curr != right) {
        edge_out[row * w + col] = 1;
      }
    }
  }

  // Vertical transitions
  for (uint16_t row = 0; row < h - 1; row++) {
    for (uint16_t col = 0; col < w; col++) {
      uint8_t curr = binary_img[row * w + col]       > 0 ? 1 : 0;
      uint8_t down = binary_img[(row + 1) * w + col] > 0 ? 1 : 0;
      if (curr != down) {
        edge_out[row * w + col] = 1;
      }
    }
  }
}

// -----------------------------------------------------------------------
// blob_perimeter
// Counts transitions between 0 and non-zero pixels horizontally
// and vertically.
// Equivalent to blob_perimeter() in solidity_detection.py
//
// NOTE: mask must be padded by 1 pixel on all sides (w and h include padding)
// -----------------------------------------------------------------------
uint32_t blob_perimeter(uint8_t *padded_mask, uint16_t w, uint16_t h)
{
  uint32_t perimeter = 0;

  // Horizontal transitions
  for (uint16_t row = 0; row < h; row++) {
    for (uint16_t col = 0; col < w - 1; col++) {
      uint8_t curr  = padded_mask[row * w + col]     > 0 ? 1 : 0;
      uint8_t right = padded_mask[row * w + col + 1] > 0 ? 1 : 0;
      if (curr != right) perimeter++;
    }
  }

  // Vertical transitions
  for (uint16_t row = 0; row < h - 1; row++) {
    for (uint16_t col = 0; col < w; col++) {
      uint8_t curr = padded_mask[row * w + col]       > 0 ? 1 : 0;
      uint8_t down = padded_mask[(row + 1) * w + col] > 0 ? 1 : 0;
      if (curr != down) perimeter++;
    }
  }

  return perimeter;
}

// -----------------------------------------------------------------------
// compute_average_blob_height
// Equivalent to compute_average_blob_height() in solidity_detection.py
// -----------------------------------------------------------------------
float compute_average_blob_height(float blob_area, uint16_t bounding_box_width)
{
  if (bounding_box_width == 0) return 0.0f;
  return blob_area / (float)bounding_box_width;
}

// -----------------------------------------------------------------------
// compute_fractal_dimension
// Box-counting fractal dimension of the blob edge.
// Equivalent to compute_fractal_dimension() in solidity_detection.py
// -----------------------------------------------------------------------
float compute_fractal_dimension(uint8_t *binary_img, uint16_t w, uint16_t h)
{
  // Get edge pixels
  uint8_t edge[IMAGE_WIDTH * IMAGE_HEIGHT];
  get_blob_edge(binary_img, edge, w, h);

  // Box sizes as powers of 2 up to min(w, h)
  uint16_t min_dim = (w < h) ? w : h;

  // Max 8 scales (2^1 to 2^8 = 256)
  float eps_axis[8];
  float box_axis[8];
  uint8_t n_scales = 0;

  for (uint8_t i = 1; (1 << i) <= min_dim; i++) {
    uint16_t s = 1 << i; // box size = 2^i

    uint16_t trimmed_h = (h / s) * s;
    uint16_t trimmed_w = (w / s) * s;

    uint32_t count = 0;

    // Count non-empty s x s boxes
    for (uint16_t br = 0; br < trimmed_h; br += s) {
      for (uint16_t bc = 0; bc < trimmed_w; bc += s) {
        bool any = false;
        for (uint16_t r = br; r < br + s && !any; r++) {
          for (uint16_t c = bc; c < bc + s && !any; c++) {
            if (edge[r * w + c] > 0) any = true;
          }
        }
        if (any) count++;
      }
    }

    if (count > 0) {
      eps_axis[n_scales] = logf((float)s);
      box_axis[n_scales] = logf((float)count);
      n_scales++;
    }
  }

  if (n_scales < 2) return 0.0f;

  // Linear regression (polyfit degree 1) on log-log data
  float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
  for (uint8_t i = 0; i < n_scales; i++) {
    sum_x  += eps_axis[i];
    sum_y  += box_axis[i];
    sum_xy += eps_axis[i] * box_axis[i];
    sum_x2 += eps_axis[i] * eps_axis[i];
  }

  float denom = (float)n_scales * sum_x2 - sum_x * sum_x;
  if (fabsf(denom) < 1e-10f) return 0.0f;

  float slope = ((float)n_scales * sum_xy - sum_x * sum_y) / denom;

  // Slope is negative, negate to get positive dimension
  return -slope;
}

// -----------------------------------------------------------------------
// is_smooth_blob
// Returns true if blob is smooth (ground), false if irregular (obstacle).
// Equivalent to is_smooth_blob() in solidity_detection.py
// -----------------------------------------------------------------------
bool is_smooth_blob(uint8_t *single_blob_mask, uint16_t w, uint16_t h,
                    float area, uint16_t bounding_box_width)
{
  // Pad mask by 1 pixel on all sides
  uint16_t pw = w + 2;
  uint16_t ph = h + 2;
  uint8_t padded[pw * ph];
  memset(padded, 0, pw * ph);

  for (uint16_t row = 0; row < h; row++) {
    for (uint16_t col = 0; col < w; col++) {
      padded[(row + 1) * pw + (col + 1)] = single_blob_mask[row * w + col];
    }
  }

  // Compute perimeter of padded blob
  uint32_t perimeter = blob_perimeter(padded, pw, ph);

  // Compute average height and equivalent rectangle perimeter
  float avg_height = compute_average_blob_height(area, bounding_box_width);
  float equiv_rect_perimeter = 2.0f * (avg_height + (float)bounding_box_width);

  // Compute fractal dimension
  float fractal_dim = compute_fractal_dimension(single_blob_mask, w, h);

  // Check smoothness conditions
  // If either condition is met -> blob is smooth -> return true
  if (equiv_rect_perimeter > 0.0f) {
    float perimeter_ratio = (float)perimeter / equiv_rect_perimeter;
    if (perimeter_ratio < THRESHOLD_PERIMETER_RATIO
        || fractal_dim < THRESHOLD_FRACTAL_DIM) {
      return true;
    }
  }

  return false;
}