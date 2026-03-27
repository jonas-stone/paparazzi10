/*
 * team10_ground_detection.c — master module (obstacles + plants)
 * Callback structure unchanged. get_obstacle_info now includes
 * is_smooth_blob filtering and plant detection.
 */

 // own header
#include "team10_RANSAC_adaptation.h"
#include "team10_ground_detection_RANSAC.h"

// for fit_linear_model
#include "math/pprz_matrix_decomp_float.h"
#include "math/pprz_algebra_float.h"

#include "team10_rtp_utilities.h"
#include "team10_get_obstacle_info.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/computer_vision/cv.h"

#include "modules/core/abi.h"

#include "std.h"
#include <string.h>
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

// ========== FORWARD DECLARATION OF FUNCTIONS HERE ========= //
float compute_bucket_confidence(struct column_bucket_t *bucket);
// ========== FORWARD DECLARATION OF FUNCTIONS HERE ========= //

// create 7 image columns.
const uint8_t bucket_width = (uint8_t)(MAX_IMAGE_WIDTH / NUMBER_VERTICAL_BUCKETS);
struct column_bucket_t buckets[NUMBER_VERTICAL_BUCKETS] = {0};

// ground detection variables
static bool obstacles_updated = false;
uint8_t ground_mask[MAX_PIXELS];
uint8_t edge_mask[MAX_PIXELS];
float   edge_coords[MAX_PIXELS][2];
int     parabola_boundary[MAX_IMAGE_HEIGHT];

// obstacle detection variables
float   max_safety = 0.0f;
float   safest_bucket = NUMBER_VERTICAL_BUCKETS / 2;
float   safety_level[NUMBER_VERTICAL_BUCKETS];
int     total_edge_pixel_count = 0;

// stuff
static float smoothed_safety[NUMBER_VERTICAL_BUCKETS] = {0};
static float local_safest_bucket = NUMBER_VERTICAL_BUCKETS / 2;
static float local_max_safety = 0.0f;

// controllable sliders on Paparazzi GCS
int     area_threshold    = DEFAULT_BLOB_AREA_THRESH;
uint8_t draw_parabola     = 1;
uint8_t draw_ground_mask  = 1;
float   score_multiplier  = 2;

// color selectors for various objects
yuv_color_e parabola_color = BLUE;
yuv_color_e bucket_color   = BLUE;
yuv_color_e safe_color     = GREEN;

// exponential smoothing parameter -> higher, less smoothing
float alpha_parabola = 0.7;
float alpha_safety   = 0.2;
/*
  FUNCTION: get_obstacles_ransac

  Hses RANSAC algorithm to detect obstacles.
  Main pipeline of this obstacle detection logic.
*/
static struct image_t *get_obstacles_RANSAC(struct image_t *img, uint8_t camera_id __attribute__((unused))) 
{

  // just so we don't modify the contended globals all the time.
  local_safest_bucket = safest_bucket; // persists between calls
  local_max_safety = max_safety;

  int w = img->w;
  int h = img->h;

  /* ── 1. GET GROUND BLOB ──────────────────────────────────────────────────
    * detect_green_ground_ml  → raw binary mask in work_mask
    * isolate_ground_blob     → removes non-ground blobs in-place
    * Both functions live in team10_get_obstacle_info.c
    * work_mask is a static buffer declared there, so we need a local copy
    * we can own. We use edge_out (declared below) as scratch first, then
    * we allocate a local mask on the stack (or static).
  */

  float green_frac = 0.0f;
  detect_green_ground_ml(img, ground_mask, 5, &green_frac);

  // if barely any green, nothing to do
  if (green_frac < 0.05f) return img;

  isolate_ground_blob(ground_mask, w, h, area_threshold);

  /* ── 2. FILL HOLES IN GROUND BLOB ───────────────────────────────────────
    * fill_holes_mask modifies the mask in-place using BFS from the border.
    * Any black region not reachable from the border gets filled white.
  */
  fill_holes_mask(ground_mask, w, h);

  // DRAW GROUND MASK ON img
  if (draw_ground_mask) draw_mask_printer(img, ground_mask, w, h);

  /* ── 3. GET EDGE ─────────────────────────────────────────────────────────
    * get_blob_edge_mask reads the mask and writes a
    * binary edge map into edge_out (1 = edge pixel, 0 = not).
    * Returns the total count of edge pixels found.
  */
  total_edge_pixel_count = get_blob_edge_mask(ground_mask, w, h, edge_mask);

  if (total_edge_pixel_count == 0) return img;  // no edge found, nothing to fit

  /* ── 4. GET EDGE PIXEL COORDINATES ──────────────────────────────────────
    * get_edge_pixel_coordinates walks the edge_mask and fills a [N][2]
    * array with (x, y) of every edge pixel.
    * We need to allocate for the worst case (all pixels are edges).
  */
  get_edge_pixel_coordinates(w, h, edge_mask, edge_coords);

  /* ── 5. USE RANSAC, GET PARABOLA COEFFS ──────────────────────────────────
    * We want to fit:  y = a*x^2 + b*x + c
    * i.e. target = y,  sample = [x]  with degree=2 and use_bias=true
    *
    * RANSAC_linear_model signature:
    *   RANSAC_linear_model(n_samples, n_iterations, error_threshold,
    *                       targets, D, samples, count,
    *                       use_bias, degree, params, fit_error)
    *
    * With D=1, degree=2:
    *   expanded_dim = 1 + 1*(1+1)/2 = 2  → features are [x, x^2]
    *   params size  = 2 + 1 (bias) = 3   → [w_x, w_x2, bias]
    *   prediction   = w_x*x + w_x2*x^2 + bias
    *                = w_x2*x^2 + w_x*x + bias
    *   so coeffs for eval_parabola(coeffs, x) = a*x^2 + b*x + c are:
    *     coeffs[0] = params[1]   (x^2 term)
    *     coeffs[1] = params[0]   (x   term)
    *     coeffs[2] = params[2]   (bias / constant)
  */

  // Build targets (y values) and samples (x values) from edge_coords
  static float targets[MAX_EDGE_PIXELS];
  static float samples[MAX_EDGE_PIXELS][1];  // D=1: only x

  for (int i = 0; i < total_edge_pixel_count; i++) {
      samples[i][0] = edge_coords[i][0];  // x: rotated image row
      targets[i]    = edge_coords[i][1];  // y: rotated image col
  }

  float params[3];  // [w_x, w_x2, bias]  (size = expanded_dim(1,2) + 1 = 3)
  float fit_error = 0.0f;

  RANSAC_linear_model(
      10,               // n_samples per iteration
      100,              // n_iterations
      5.0f,             // error_threshold in pixels
      targets,          // y values
      1,                // D = 1 (only x as input feature)
      samples,          // x values
      (uint16_t)total_edge_pixel_count,
      true,             // use_bias → adds constant term c
      2,                // degree=2 → quadratic fit
      params,
      &fit_error
  );

  // remap params to standard parabola convention for eval_parabola()
  static float coeffs[3];
  coeffs[0] = params[1] * alpha_parabola + coeffs[0] * (1 - alpha_parabola);  // a: coefficient of x^2
  coeffs[1] = params[0];  // b: coefficient of x
  coeffs[2] = params[2];  // c: bias / constant

  /* ── 6. EVAL PARABOLA ON IMAGE ───────────────────────────────────────────
    * For each row r (= x in parabola space, since image is rotated),
    * compute the fitted column position y = a*r^2 + b*r + c.
    * Output is a vector of length h (number of rows).
  */
  pthread_mutex_lock(&mutex);
  // RESET RETURN VALUE; if not, pixels will accumulate forever
  memset(buckets, 0, sizeof(buckets));

  // rows are the "x" for our parabola evaluation
  for (int row = 0; row < h; row++) {

    parabola_boundary[row] = eval_parabola(coeffs, (float)row); 
    Bound(parabola_boundary[row], 0, w);

    bool bucket_boundary_row = (row % bucket_width == 0) ? true : false;
    uint8_t current_bucket = (uint8_t)(row * NUMBER_VERTICAL_BUCKETS / h);
    Bound(current_bucket, 0, NUMBER_VERTICAL_BUCKETS);

    if (draw_parabola) draw_colored_pixel(img, row, parabola_boundary[row], parabola_color);

    // columns are the "y" for our parabola evaluation
    for (uint8_t col = 0; col < w; col++) {
      
      if (bucket_boundary_row) draw_colored_pixel(img, row, col, bucket_color);

      // for plant detection
      uint8_t is_edge_pixel = edge_mask[w * row + col];
      if (is_edge_pixel) buckets[current_bucket].edge_pixel_count++;
      
      // for pixel count
      uint8_t current_pixel = ground_mask[w * row + col];
      switch (current_pixel) {
        case 0:
          if (col > parabola_boundary[row]) buckets[current_bucket].black_above_ground_line++;
          else buckets[current_bucket].black_below_ground_line++;
          break;

        case 255:
          if (col > parabola_boundary[row]) buckets[current_bucket].white_above_ground_line++;
          else buckets[current_bucket].white_below_ground_line++;
          break;
      }
    }
  }
  
  // update our bucket safety levels
  // uint8_t local_safest_bucket_new = 0;
  // for (uint8_t b = 0; b < NUMBER_VERTICAL_BUCKETS; b++) {
  //   safety_level[b] = compute_bucket_confidence(&buckets[b]);
  //   smoothed_safety[b] = alpha * safety_level[b] + (1.0f - alpha) * smoothed_safety[b];
  //   if (smoothed_safety[b] > local_max_safety) {
  //     local_max_safety = smoothed_safety[b];
  //     local_safest_bucket_new = b;
  //   }
  // }

  float local_max_safety_new = 0;
  uint8_t local_safest_bucket_new = 0;
  float local_max_smoothed = -0.0f;
  for (uint8_t b = 0; b < NUMBER_VERTICAL_BUCKETS; b++) {
      safety_level[b] = compute_bucket_confidence(&buckets[b]);
      smoothed_safety[b] = alpha_safety * safety_level[b] + (1.0f - alpha_safety) * smoothed_safety[b];
      if (smoothed_safety[b] > local_max_smoothed) {
          local_max_smoothed = smoothed_safety[b];
          local_safest_bucket_new = b;
      }
  }
  local_max_safety = local_max_smoothed;
  local_safest_bucket = local_safest_bucket_new;

  // do a running time average of the safest bucket
  // float alpha = 0.5; // <-- higher alpha, faster adjustment
  // local_safest_bucket = (float)(alpha * local_safest_bucket_new + (1 - alpha) * local_safest_bucket);

  // DRAW safest bucket on image buffer (top of image -> right of raw image)
  for (uint8_t row = 0; row < bucket_width; row++) {
    draw_colored_pixel(img, (int) local_safest_bucket * bucket_width + row, w-1, safe_color);
    draw_colored_pixel(img, (int) local_safest_bucket * bucket_width + row, w-2, safe_color);
    draw_colored_pixel(img, (int) local_safest_bucket * bucket_width + row, w-3, safe_color);
  }

  // global variable allocation only here.
  // pthread_mutex_lock(&mutex);
  safest_bucket     = local_safest_bucket;
  max_safety        = local_max_safety;
  obstacles_updated = true;
  pthread_mutex_unlock(&mutex);

  return img;
}

/*
COMPUTES the confidence level of ONE (1) vertical bucket.
*/
float compute_bucket_confidence(struct column_bucket_t *bucket) {
    int white = bucket->white_below_ground_line;
    int black = bucket->black_below_ground_line;
    int total_below = white + black;

    if (total_below == 0) return 0.0f;

    // 1. QUALITY: What percentage of the ground is white?
    // We use (white / total) but penalize black pixels more heavily.
    float ratio = (float)(white - (2 * black)) / total_below;
    if (ratio < 0.0f) ratio = 0.0f; 

    // 2. QUANTITY: "Saturation" logic. 
    // If a bucket has 200+ pixels, we trust it 100%. 
    // If it only has 10 pixels, we trust it very little.
    const float TRUST_THRESHOLD = 5 * area_threshold; // Adjust from PAPARAZZI based on resolution
    float abundance = fminf((float)white / TRUST_THRESHOLD, 1.0f);

    // 3. COMBINE: High score only if ratio is good AND we have plenty of pixels.
    float score = ratio * abundance;

    // 4. EDGE PENALTY (Keep your existing logic, but clamp the result)
    float average_edges = (float)total_edge_pixel_count / NUMBER_VERTICAL_BUCKETS;
    if (bucket->edge_pixel_count > average_edges) {
        float excess  = (bucket->edge_pixel_count - average_edges) / (average_edges + 1.0f);
        score -= 0.3f * fminf(excess, 1.0f);
    }

    // Ensure we don't return negative confidence
    return fmaxf(score, 0.0f); 
}

void ground_detection_init(void)
{
  // maybe set memory to stuff
  pthread_mutex_init(&mutex, NULL);
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, get_obstacles_RANSAC, COLOR_OBJECT_DETECTOR_FPS1, 0);
}

void ground_detection_periodic(void)
{
  float local_safety[NUMBER_VERTICAL_BUCKETS];
  struct column_bucket_t local_img_cols[NUMBER_VERTICAL_BUCKETS];
  float local_max_safety;
  float local_safest_bucket;

  pthread_mutex_lock(&mutex);
  if (!obstacles_updated) { 
    pthread_mutex_unlock(&mutex); return; 
  }

  memcpy(local_safety, smoothed_safety, NUMBER_VERTICAL_BUCKETS * sizeof(float));
  memcpy(local_img_cols, buckets, NUMBER_VERTICAL_BUCKETS * sizeof(struct column_bucket_t));
  local_max_safety    = max_safety;
  local_safest_bucket = safest_bucket;
  obstacles_updated   = false;
  pthread_mutex_unlock(&mutex);

  // remember to change this function isnside mr. ABI stuff
  AbiSendMsgTEAM10_RANSAC_DETECTION(TEAM10_RANSAC_DETECTION_ID, local_max_safety, local_safest_bucket, local_img_cols, local_safety);
}
