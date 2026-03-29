/**
 * @file RANSAC.c
 * @brief Perform Random Sample Consensus (RANSAC), a robust fitting method.
 *
 * RANSAC fits a curve to a noisy dataset by repeatedly trying random small
 * subsets of the data, fitting a model to each subset, and keeping whichever
 * model explains the most data points within a tolerable error.
 *
 * This version supports fitting a straight line (degree=1) or a parabola
 * (degree=2). We use it to fit a parabola to the edge of the ground blob
 * so we know where the floor is in each image column.
 *
 * For degree=2, the raw feature vector [x] is expanded to [x, x^2].
 * With a bias term this gives three parameters: a (x^2 coeff), b (x coeff),
 * and c (constant), matching the standard parabola y = a*x^2 + b*x + c.
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
 * @brief How many features does a sample have after polynomial expansion?
 *
 * degree=1 (linear fit): the feature vector stays the same size as the
 *   raw input — just [x_0, x_1, ..., x_{D-1}].
 *
 * degree=2 (quadratic fit): we add all squared and cross terms on top of
 *   the original features, giving D + D*(D+1)/2 total features.
 *   For D=1 (our case): 1 + 1 = 2 features → [x, x^2].
 *
 * The bias term (constant c) is NOT counted here — it is appended separately.
 *
 * @param D       Number of raw input features (1 in our pipeline — just x)
 * @param degree  Polynomial degree (1 or 2)
 * @return        Number of features BEFORE adding the bias term
 */
static int expanded_dim(int D, int degree)
{
  if (degree == 2) {
    return D + D * (D + 1) / 2;
  }
  return D;
}

/**
 * @brief Expand a raw sample vector into its full polynomial feature vector.
 *
 * For a linear fit (degree=1) nothing changes: out = [x_0, ..., x_{D-1}].
 * For a quadratic fit (degree=2) the squared and cross terms are appended:
 *   out = [x_0, ..., x_{D-1},  x_0^2, x_0*x_1, ..., x_{D-1}^2]
 *
 * For our 1-D ground edge case (D=1, degree=2):
 *   out = [x, x^2]  — two features total.
 *
 * @param sample  Raw input sample of size D
 * @param out     Output buffer of size expanded_dim(D, degree) — caller allocates
 * @param D       Number of raw features
 * @param degree  Polynomial degree (1 or 2)
 */
static void expand_features(const float *sample, float *out, int D, int degree)
{
  // Copy the original features first
  for (int i = 0; i < D; i++) {
    out[i] = sample[i];
  }

  // Append squared and cross terms for quadratic fitting
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
 * @brief Predict the output value for a sample using previously learned weights.
 *
 * Expands the raw sample to the full polynomial feature vector, then computes
 * the dot product with the weight vector. If a bias term was used during
 * training, it is added as the last weight.
 *
 * @param sample   Raw input sample of size D
 * @param weights  Learned weights of size expanded_dim(D,degree) + (use_bias?1:0)
 * @param D        Number of raw input features
 * @param degree   Polynomial degree (must match what was used to fit)
 * @param use_bias Whether a bias (constant) term is included in weights
 * @return         Predicted output value
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
 * @brief Fit a polynomial model to noisy data using RANSAC.
 *
 * The core idea: instead of fitting to all data at once (which noise and
 * outliers would corrupt), we repeatedly pick a tiny random subset, fit the
 * model to just that subset, then score it against the full dataset. After
 * many iterations we keep the model that scored best (lowest total error).
 *
 * Each sample is first expanded into polynomial features (linear or quadratic)
 * before being passed to the underlying least-squares solver.
 *
 * Error is capped at error_threshold per point so that extreme outliers don't
 * dominate the scoring. This is the key difference from plain least squares.
 *
 * @param n_samples        How many data points to draw per RANSAC trial
 * @param n_iterations     How many random trials to run
 * @param error_threshold  Per-point error above this is capped (not penalised more)
 * @param targets          The y-values we are trying to predict, length [count]
 * @param D                Number of input features per sample (1 in our pipeline)
 * @param samples          The x-values, a [count][D] matrix
 * @param count            Total number of data points
 * @param use_bias         If true, a constant intercept term is added
 * @param degree           1 = straight line, 2 = parabola
 * @param params           OUTPUT: best-fit parameters — caller must allocate
 * @param fit_error        Unused, kept for API compatibility
 */
void RANSAC_linear_model(int n_samples, int n_iterations, float error_threshold, float *targets, int D,
                         float (*samples)[1], uint16_t count, bool use_bias, int degree, float *params,
                         float *fit_error __attribute__((unused)))
{
  int exp_D = expanded_dim(D, degree);
  int D_1 = use_bias ? exp_D + 1 : exp_D;  // total number of parameters to solve for

  float err;
  float errors[n_iterations];          // total (capped) error for each trial
  int indices_subset[n_samples];       // which data points were chosen this trial
  float subset_targets[n_samples];     // their y-values

  // Expanded feature matrix for the randomly chosen subset
  float subset_samples_exp[n_samples][exp_D];

  // Store the fitted parameters from every trial so we can compare them at the end
  float subset_params[n_iterations][D_1];

  // Make sure we draw at least as many points as we have parameters to solve
  // (otherwise the system is under-determined and has no unique solution)
  n_samples = (n_samples < D_1)    ? D_1    : n_samples;
  n_samples = (n_samples < (int)count) ? n_samples : (int)count;

  // Pre-expand all data points so we don't repeat this work inside the loop
  float all_expanded[count][exp_D];
  for (int i = 0; i < (int)count; i++) {
    expand_features(samples[i], all_expanded[i], D, degree);
  }

  // ── Main RANSAC loop ─────────────────────────────────────────────────────
  for (int i = 0; i < n_iterations; i++) {

    // Pick a random subset (no repeated indices)
    get_indices_without_replacement(indices_subset, n_samples, count);

    // Build the subset using expanded features
    for (int j = 0; j < n_samples; j++) {
      subset_targets[j] = targets[indices_subset[j]];
      for (int k = 0; k < exp_D; k++) {
        subset_samples_exp[j][k] = all_expanded[indices_subset[j]][k];
      }
    }

    // Fit a linear model on the expanded subset
    // (This is a standard least-squares solve; passing exp_D is correct because
    //  the quadratic features are already baked into the expanded matrix.)
    fit_linear_model(subset_targets, exp_D, subset_samples_exp, n_samples, use_bias,
                     subset_params[i], &err);

    // Score this model on ALL data points (not just the subset)
    // Errors beyond error_threshold are capped so outliers don't ruin good models
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
      err = (err > error_threshold) ? error_threshold : err;  // cap the error
      err_sum += err;
    }
    errors[i] = err_sum;
  }

  // Find which trial gave the lowest total error
  float min_err = errors[0];
  int min_ind = 0;
  for (int i = 1; i < n_iterations; i++) {
    if (errors[i] < min_err) {
      min_err = errors[i];
      min_ind = i;
    }
  }

  // Copy the winning parameters to the output array
  for (int d = 0; d < D_1; d++) {
    params[d] = subset_params[min_ind][d];
  }
}

/**
 * @brief Draw n_samples unique random indices from the range [0, count).
 *
 * "Without replacement" means the same index cannot appear twice in the same
 * draw. This is important for RANSAC because if we accidentally picked the
 * same point twice, we would effectively have one fewer data point to fit.
 *
 * @param indices_subset  OUTPUT: array of n_samples unique random indices
 * @param n_samples       How many unique indices to draw
 * @param count           The upper bound (exclusive) for valid indices
 */
void get_indices_without_replacement(int *indices_subset, int n_samples, int count)
{
  int index;

  for (int j = 0; j < n_samples; j++) {
    bool picked_number = false;
    while (!picked_number) {
      index = rand() % count;
      // Reject this index if we already picked it in this round
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


/**
 * @brief Find the first left-to-right transition in each row of a binary mask.
 *
 * This is used to locate the edge of the ground blob — the column where the
 * ground mask switches from background (0) to ground (255) or vice versa.
 * We record only the FIRST such transition per row (via the `break` statement),
 * giving one edge point per image row.
 *
 * The image is rotated 90° CW by the camera, so "rows" here correspond to
 * vertical slices in the drone's actual field of view.
 *
 * Internally a Python version of this algorithm was:
 *   edge = (padded_img[:,1:] != padded_img[:,:-1])
 * This is the C equivalent, column by column.
 *
 * @param binary_mask  Input binary mask (0 or 255), size [h × w]
 * @param w            Image width in pixels
 * @param h            Image height in pixels
 * @param edge_out     OUTPUT: edge mask (0 or 1), same size as binary_mask.
 *                     Caller must zero-initialise this before calling.
 * @return             Total number of edge pixels found
 */
int get_blob_edge_mask(uint8_t *binary_mask, int w, int h, uint8_t *edge_out) {

  // We do NOT pad the image — edge_out is already the same size as the mask
  // and the caller passes it pre-zeroed, so unset pixels stay 0.

  int edge_pixel_count = 0;

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w-1; col++) {

      // These flipped indices were kept from a previous version that scanned
      // right-to-left; they are computed but the standard left/right neighbours
      // below are used for the actual comparison.
      int flipped_col      = w - 1 - col;
      int flipped_col_next = w - 1 - (col + 1) - 1;

      uint8_t current  = binary_mask[row * w + col];
      uint8_t next     = binary_mask[row * w + col + 1];

      if (current != next) {
        // This pixel is where the mask changes value — it is an edge pixel
        edge_out[row * w + col] = 1;
        edge_pixel_count++;
        break;  // Only record the FIRST transition in each row
      } else {
        edge_out[row * w + col] = 0;
      }
    }
  }

  return edge_pixel_count;
}


/**
 * @brief Collect the (row, col) coordinates of every edge pixel into an array.
 *
 * After get_blob_edge_mask() marks which pixels are on the edge, this function
 * walks the edge mask and records each marked pixel's position. The resulting
 * array is what we pass into RANSAC_linear_model() as the sample data.
 *
 * In our pipeline the image is rotated, so:
 *   edge_coordinates[i][0] = row  (acts as x in the parabola equation)
 *   edge_coordinates[i][1] = col  (acts as y — what the parabola predicts)
 *
 * @param w                 Image width
 * @param h                 Image height
 * @param edge_mask         Binary edge mask from get_blob_edge_mask() (0 or 1)
 * @param edge_coordinates  OUTPUT: [N][2] array of (row, col) for each edge pixel
 */
void get_edge_pixel_coordinates(int w, int h, uint8_t *edge_mask, float (*edge_coordinates)[2]) {

  int i = 0;  // index into edge_coordinates

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      if (edge_mask[row * w + col] == 1) {
        edge_coordinates[i][0] = row;   // x for the parabola
        edge_coordinates[i][1] = col;   // y for the parabola
        i++;
      }
    }
  }
}


/**
 * @brief Evaluate the fitted parabola at a given x position.
 *
 * Given coefficients [a, b, c] (a for x^2, b for x, c for constant), returns
 * the integer column position y = a*x^2 + b*x + c.
 *
 * This is called once per image row to get the estimated ground boundary
 * column for that row.
 *
 * @param coeffs  [a, b, c] — coefficients of the parabola
 * @param x       The row (x position) to evaluate at
 * @return        Predicted column position (integer), i.e. where the ground boundary is
 */
int eval_parabola(float *coeffs, float x) {
  return (int)(coeffs[0] * x*x + coeffs[1] * x + coeffs[2]);
}


// ======================== PIXEL DRAWING UTILITIES =========================

/**
 * @brief Paint a single neon-green pixel directly into a YUV422 image buffer.
 *
 * YUV422 (UYVY format) stores two pixels in every 4 bytes:
 *   [U, Y0, V, Y1]
 * where U and V are shared between both pixels, and Y0/Y1 are individual
 * luma values. To write a colour we must align to the even-numbered column
 * of the pixel pair (col & ~1) and overwrite all four bytes.
 *
 * The values below produce a visually bright neon green in YUV:
 *   U=44, Y=150, V=21
 *
 * @param img  Pointer to the YUV422 image to draw into
 * @param row  Row index of the target pixel
 * @param col  Column index of the target pixel (snapped to even pair)
 */
void draw_neon_green_pixel(struct image_t *img, int row, int col)
{
  int even_col = col & ~1;
  if (even_col < 0 || even_col >= img->w - 1 || row < 0 || row >= img->h) {
    return;  // Ignore out-of-bounds requests silently
  }

  uint8_t *buf = (uint8_t*)img->buf;
  int idx = 2 * img->w * row + 2 * even_col;

  buf[idx]     = 44;   // U  (neon green)
  buf[idx + 1] = 150;  // Y1
  buf[idx + 2] = 21;   // V  (neon green)
  buf[idx + 3] = 150;  // Y2
}

/**
 * @brief Paint a single pixel with a chosen colour into a YUV422 image buffer.
 *
 * Works the same way as draw_neon_green_pixel() but looks up the YUV values
 * from the yuv_colors[] table using the yuv_color_e enum. This lets the caller
 * pick green, red, blue, white, or black with a readable name rather than raw
 * YUV byte values.
 *
 * @param img    Pointer to the YUV422 image to draw into
 * @param row    Row index of the target pixel
 * @param col    Column index (snapped to even pair)
 * @param color  One of: GREEN, RED, BLUE, WHITE, BLACK
 */
void draw_colored_pixel(struct image_t *img, int row, int col, yuv_color_e color)
{
  int even_col = col & ~1;
  if (even_col < 0 || even_col >= img->w - 1 || row < 0 || row >= img->h) {
    return;  // Ignore out-of-bounds requests silently
  }

  uint8_t *buf = (uint8_t*)img->buf;
  int idx = 2 * img->w * row + 2 * even_col;
  yuv_color_t c = yuv_colors[color];

  buf[idx]     = c.U;
  buf[idx + 1] = c.Y;
  buf[idx + 2] = c.V;
  buf[idx + 3] = c.Y;
}