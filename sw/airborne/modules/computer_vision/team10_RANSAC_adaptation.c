/*
 * Copyright (C) 2018 Guido de Croon
 *
 * This file is part of paparazzi.
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

/**
 * @file RANSAC.c
 * @brief Perform Random Sample Consensus (RANSAC), a robust fitting method.
 *
 * Supports linear (degree=1) and quadratic (degree=2) fits.
 * For degree=2, the feature vector is expanded to include all quadratic terms
 * x_i^2 and cross terms x_i*x_j (i <= j), giving an expanded dimension of
 * D + D*(D+1)/2 features (plus optional bias).
 *
 * The `params` output array must be allocated by the caller with size:
 *   degree=1: D   + (use_bias ? 1 : 0)
 *   degree=2: D + D*(D+1)/2 + (use_bias ? 1 : 0)
 *
 * Read: Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus:
 * a paradigm for model fitting with applications to image analysis and
 * automated cartography. Communications of the ACM, 24(6), 381-395.
 */

#include "modules/computer_vision/team10_RANSAC_adaptation.h"
#include "modules/computer_vision/team10_get_obstacle_info.h"

// for fit_linear_model
#include "math/pprz_matrix_decomp_float.h"
#include "math/pprz_algebra_float.h"

#include "modules/computer_vision/lib/vision/image.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "stdio.h"

/**
 * @brief Compute the number of features after polynomial expansion.
 *
 * degree=1: D features (just the original)
 * degree=2: D + D*(D+1)/2 features (original + all squared/cross terms)
 *
 * @param D    Original feature dimensionality
 * @param degree  Polynomial degree (1 or 2)
 * @return Number of expanded features (NOT counting the bias term)
 */
static int expanded_dim(int D, int degree)
{
  if (degree == 2) {
    return D + D * (D + 1) / 2;
  }
  return D;
}

/**
 * @brief Expand a sample vector into polynomial features in-place into `out`.
 *
 * For degree=1: out = [x_0, x_1, ..., x_{D-1}]
 * For degree=2: out = [x_0, ..., x_{D-1},
 *                      x_0*x_0, x_0*x_1, ..., x_0*x_{D-1},
 *                               x_1*x_1, ..., x_1*x_{D-1},
 *                                        ...  x_{D-1}*x_{D-1}]
 *
 * @param sample  Input sample of size D
 * @param out     Output buffer of size expanded_dim(D, degree)
 * @param D       Original feature dimensionality
 * @param degree  Polynomial degree (1 or 2)
 */
static void expand_features(const float *sample, float *out, int D, int degree)
{
  // Copy linear terms
  for (int i = 0; i < D; i++) {
    out[i] = sample[i];
  }

  if (degree == 2) {
    int idx = D;
    for (int i = 0; i < D; i++) {
      for (int j = i; j < D; j++) {
        out[idx++] = sample[i] * sample[j];
      }
    }
  }
}

/**
 * @brief Predict the value of a sample with learned weights.
 *
 * Internally expands the sample to the correct polynomial degree before
 * computing the dot product with the weights.
 *
 * @param sample   The raw sample vector of size D
 * @param weights  The weight vector of size expanded_dim(D, degree) + (use_bias ? 1 : 0)
 * @param D        The dimension of the raw sample
 * @param degree   Polynomial degree used during fitting (1 or 2)
 * @param use_bias Whether a bias term was included during fitting
 * @return The predicted value
 */
float predict_value(float *sample, float *weights, int D, int degree, bool use_bias)
{
  int exp_D = expanded_dim(D, degree);
  float expanded[exp_D];
  expand_features(sample, expanded, D, degree);

  float sum = 0.0f;
  for (int w = 0; w < exp_D; w++) {
    sum += weights[w] * expanded[w];
  }
  if (use_bias) {
    sum += weights[exp_D];
  }

  return sum;
}

/**
 * @brief Perform RANSAC to fit a polynomial model (degree 1 or 2).
 *
 * @param n_samples        Number of samples used per RANSAC iteration
 * @param n_iterations     Number of RANSAC iterations
 * @param error_threshold  Per-sample error cap for robust fitting
 * @param targets          Target values, size [count]
 * @param D                Dimensionality of raw samples
 * @param samples          Sample matrix, size [count][D]
 * @param count            Total number of samples
 * @param use_bias         Whether to include a bias (intercept) term
 * @param degree           Polynomial degree: 1 (linear) or 2 (quadratic)
 * @param params           Output parameters, caller must allocate:
 *                           degree=1: D   + (use_bias ? 1 : 0)
 *                           degree=2: D + D*(D+1)/2 + (use_bias ? 1 : 0)
 * @param fit_error        Unused (kept for API compatibility)
 */
void RANSAC_linear_model(int n_samples, int n_iterations, float error_threshold, float *targets, int D,
                         float (*samples)[1], uint16_t count, bool use_bias, int degree, float *params,
                         float *fit_error __attribute__((unused)))
{
  int exp_D = expanded_dim(D, degree);
  int D_1 = use_bias ? exp_D + 1 : exp_D;

  float err;
  float errors[n_iterations];
  int indices_subset[n_samples];
  float subset_targets[n_samples];

  // Expanded feature matrix for the subset
  float subset_samples_exp[n_samples][exp_D];

  // One set of params per iteration
  float subset_params[n_iterations][D_1];

  // Clamp n_samples to valid range
  n_samples = (n_samples < D_1)    ? D_1    : n_samples;
  n_samples = (n_samples < (int)count) ? n_samples : (int)count;

  // Expanded features for the full dataset, used for error evaluation
  float all_expanded[count][exp_D];
  for (int i = 0; i < (int)count; i++) {
    expand_features(samples[i], all_expanded[i], D, degree);
  }

  for (int i = 0; i < n_iterations; i++) {

    // Pick a random subset of indices without replacement
    get_indices_without_replacement(indices_subset, n_samples, count);

    // Build the subset using expanded features
    for (int j = 0; j < n_samples; j++) {
      subset_targets[j] = targets[indices_subset[j]];
      for (int k = 0; k < exp_D; k++) {
        subset_samples_exp[j][k] = all_expanded[indices_subset[j]][k];
      }
    }

    // Fit a linear model on the expanded feature subset
    // (fit_linear_model works on any dimensionality, so passing exp_D is correct)
    fit_linear_model(subset_targets, exp_D, subset_samples_exp, n_samples, use_bias,
                     subset_params[i], &err);

    // Evaluate fit on the full dataset with capped errors
    float err_sum = 0.0f;
    for (int j = 0; j < (int)count; j++) {
      float prediction = 0.0f;
      for (int k = 0; k < exp_D; k++) {
        prediction += subset_params[i][k] * all_expanded[j][k];
      }
      if (use_bias) {
        prediction += subset_params[i][exp_D];
      }
      err = fabsf(prediction - targets[j]);
      err = (err > error_threshold) ? error_threshold : err;
      err_sum += err;
    }
    errors[i] = err_sum;
  }

  // Find the iteration with the lowest total error
  float min_err = errors[0];
  int min_ind = 0;
  for (int i = 1; i < n_iterations; i++) {
    if (errors[i] < min_err) {
      min_err = errors[i];
      min_ind = i;
    }
  }

  // Copy best parameters to output
  for (int d = 0; d < D_1; d++) {
    params[d] = subset_params[min_ind][d];
  }
}

/**
 * @brief Get indices without replacement.
 *
 * @param indices_subset  Output: filled with n_samples unique indices in [0, count)
 * @param n_samples       Number of indices to sample
 * @param count           Range of indices to sample from
 */
void get_indices_without_replacement(int *indices_subset, int n_samples, int count)
{
  int index;

  for (int j = 0; j < n_samples; j++) {
    bool picked_number = false;
    while (!picked_number) {
      index = rand() % count;
      bool new_val = true;
      for (int k = 0; k < j; k++) {
        if (indices_subset[k] == index) {
          new_val = false;
          break;
        }
      }
      if (new_val) {
        indices_subset[j] = index;
        picked_number = true;
      }
    }
  }
}


int get_blob_edge_mask(uint8_t *binary_mask, int w, int h, uint8_t *edge_out) {

  // do NOT pad image (only wastes computational resources)
  // and edge_out can be passed as a zero array of equal size as img->buf so doesnt need to be padded anyway

  /* ============================ PYTHON FUNCTION ==============================
    edge1 = (padded_binary_img[:,1:] != padded_binary_img[:,:-1])
    edge1 = np.hstack([edge1, np.zeros((edge1.shape[0], 1), dtype=edge1.dtype)])
    
    return edge1
    ============================= PYTHON FUNCTION =========================== */

  int edge_pixel_count = 0;

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w-1; col++) {

      // flipped: read from right to left
      int flipped_col      = w - 1 - col;
      int flipped_col_next = w - 1 - (col + 1) - 1;

      // horizontal edge: differs from right neighbour
      uint8_t current  = binary_mask[row * w + col]; 
      uint8_t next     = binary_mask[row * w + col + 1];

      if (current != next) {
        edge_out[row * w + col] = 1;
        edge_pixel_count++;
        break;
      } else {
        edge_out[row * w + col] = 0;
      }
    }
    // // add padding zeros to last column (no right neighbour)
    // edge_out[row * width + (width - 1)] = 0;
  }

  return edge_pixel_count;
}


void get_edge_pixel_coordinates(int w, int h, uint8_t *edge_mask, float (*edge_coordinates)[2]) {

  
  // used for iterating through edge_coordinates
  int i = 0;

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      
      if (edge_mask[row * w + col] == 1) {
        edge_coordinates[i][0] = row;
        edge_coordinates[i][1] = col;
        i++;
      } 
    }
  }
}


int eval_parabola(float *coeffs, float x) {
  return (int)(coeffs[0] * x*x + coeffs[1] * x + coeffs[2]);
}


// ======================== PLACEHOLDER LOCATION =========================

/**
 * @brief Draw a single neon green pixel at (row, col) on a YUV422 image buffer.
 *
 * @param img  Pointer to the image
 * @param row  Row index
 * @param col  Column index
 */
void draw_neon_green_pixel(struct image_t *img, int row, int col)
{
  int even_col = col & ~1;
  if (even_col < 0 || even_col >= img->w - 1 || row < 0 || row >= img->h) {
    return;
  }

  uint8_t *buf = (uint8_t*)img->buf;
  int idx = 2 * img->w * row + 2 * even_col;

  buf[idx]     = 44;   // U  (neon green)
  buf[idx + 1] = 150;  // Y1
  buf[idx + 2] = 21;   // V  (neon green)
  buf[idx + 3] = 150;  // Y2
}

/**
 * @brief Draw a single colored pixel at (row, col) on a YUV422 image buffer.
 *
 * @param img  Pointer to the image
 * @param row  Row index
 * @param col  Column index
 */
void draw_colored_pixel(struct image_t *img, int row, int col, yuv_color_e color)
{
  int even_col = col & ~1;
  if (even_col < 0 || even_col >= img->w - 1 || row < 0 || row >= img->h) {
    return;
  }

  uint8_t *buf = (uint8_t*)img->buf;
  int idx = 2 * img->w * row + 2 * even_col;
  yuv_color_t c = yuv_colors[color];

  buf[idx]     = c.U;
  buf[idx + 1] = c.Y;
  buf[idx + 2] = c.V;
  buf[idx + 3] = c.Y;
}

/*
  FUNCTION: get_obstacles_ransac

  Hses RANSAC algorithm to detect obstacles.
  Main pipeline of this obstacle detection logic.

*/
// void get_obstacles_RANSAC(struct image_t *img, struct image_column *buckets) {

//   int w = img->w;
//   int h = img->h;

//   /* ── 1. GET GROUND BLOB ──────────────────────────────────────────────────
//     * detect_green_ground_ml  → raw binary mask in work_mask
//     * isolate_ground_blob     → removes non-ground blobs in-place
//     * Both functions live in team10_get_obstacle_info.c
//     * work_mask is a static buffer declared there, so we need a local copy
//     * we can own. We use edge_out (declared below) as scratch first, then
//     * we allocate a local mask on the stack (or static).
//   */
//   static uint8_t ground_mask[MAX_PIXELS];  // our local copy of the mask

//   float green_frac = 0.0f;
//   detect_green_ground_ml(img, ground_mask, 5, &green_frac);

//   // if barely any green, nothing to do
//   if (green_frac < 0.05f) return;

//   isolate_ground_blob(ground_mask, w, h, DEFAULT_BLOB_AREA_THRESH);

//   /* ── 2. FILL HOLES IN GROUND BLOB ───────────────────────────────────────
//     * fill_holes_mask modifies the mask in-place using BFS from the border.
//     * Any black region not reachable from the border gets filled white.
//   */
//   fill_holes_mask(ground_mask, w, h);

//   /* ── 3. GET EDGE ─────────────────────────────────────────────────────────
//     * get_blob_edge_mask reads the image buffer (Y channel) and writes a
//     * binary edge map into edge_out (1 = edge pixel, 0 = not).
//     * Returns the total count of edge pixels found.
//     *
//     * NOTE: fix the bug in get_blob_edge_mask before using:
//     *   'cur' must be renamed to 'current' to match the declaration above it.
//   */
//   static uint8_t edge_mask[MAX_PIXELS];
//   int n_edge_pixels = get_blob_edge_mask(img, edge_mask);

//   if (n_edge_pixels == 0) return;  // no edge found, nothing to fit

//   /* ── 4. GET EDGE PIXEL COORDINATES ──────────────────────────────────────
//     * get_edge_pixel_coordinates walks the edge_mask and fills a [N][2]
//     * array with (x, y) of every edge pixel.
//     * We need to allocate for the worst case (all pixels are edges).
//   */
//   static float edge_coords[MAX_PIXELS][2];
//   get_edge_pixel_coordinates(img, edge_mask, edge_coords);

//   /* ── 5. USE RANSAC, GET PARABOLA COEFFS ──────────────────────────────────
//     * We want to fit:  y = a*x^2 + b*x + c
//     * i.e. target = y,  sample = [x]  with degree=2 and use_bias=true
//     *
//     * RANSAC_linear_model signature:
//     *   RANSAC_linear_model(n_samples, n_iterations, error_threshold,
//     *                       targets, D, samples, count,
//     *                       use_bias, degree, params, fit_error)
//     *
//     * With D=1, degree=2:
//     *   expanded_dim = 1 + 1*(1+1)/2 = 2  → features are [x, x^2]
//     *   params size  = 2 + 1 (bias) = 3   → [w_x, w_x2, bias]
//     *   prediction   = w_x*x + w_x2*x^2 + bias
//     *                = w_x2*x^2 + w_x*x + bias
//     *   so coeffs for eval_parabola(coeffs, x) = a*x^2 + b*x + c are:
//     *     coeffs[0] = params[1]   (x^2 term)
//     *     coeffs[1] = params[0]   (x   term)
//     *     coeffs[2] = params[2]   (bias / constant)
//   */

//   // Build targets (y values) and samples (x values) from edge_coords
//   static float targets[MAX_PIXELS];
//   static float samples[MAX_PIXELS][1];  // D=1: only x

//   for (int i = 0; i < n_edge_pixels; i++) {
//       samples[i][0] = edge_coords[i][0];  // x
//       targets[i]    = edge_coords[i][1];  // y
//   }

//   float params[3];  // [w_x, w_x2, bias]  (size = expanded_dim(1,2) + 1 = 3)
//   float fit_error = 0.0f;

//   RANSAC_linear_model(
//       10,               // n_samples per iteration
//       100,              // n_iterations
//       5.0f,             // error_threshold in pixels
//       targets,          // y values
//       1,                // D = 1 (only x as input feature)
//       samples,          // x values
//       (uint16_t)n_edge_pixels,
//       true,             // use_bias → adds constant term c
//       2,                // degree=2 → quadratic fit
//       params,
//       &fit_error
//   );

//   // remap params to standard parabola convention for eval_parabola()
//   float coeffs[3];
//   coeffs[0] = params[1];  // a: coefficient of x^2
//   coeffs[1] = params[0];  // b: coefficient of x
//   coeffs[2] = params[2];  // c: bias / constant

//   /* ── 6. EVAL PARABOLA ON IMAGE ───────────────────────────────────────────
//     * For each row r (= x in parabola space, since image is rotated),
//     * compute the fitted column position y = a*r^2 + b*r + c.
//     * Output is a vector of length h (number of rows).
//   */
//   static int parabola_boundary[MAX_IMAGE_HEIGHT];

//   for (int r = 0; r < h; r++) {
//       parabola_boundary[r] = eval_parabola(coeffs, (float)r); 
//   }

//   // NOW: create 7 image columns.
//   uint8_t number_of_image_columns = 7;
//   struct image_column img_columns[number_of_image_columns]
//   for (int c = 0; c < 7; c++) {
//     img_columns[c].width = MAX_IMAGE_HEIGHT / number_of_image_columns;
//   }

//   // update them with their respective counters

//   // initialize their confidence level to 0
// }