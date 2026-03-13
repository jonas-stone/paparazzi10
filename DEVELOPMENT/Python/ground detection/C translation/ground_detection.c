/*
 * @file modules/computer_vision/ground_detection.c
 * @brief Green ground detection and obstacle avoidance for Parrot Bebop 1
 *
 * Translates the Python pipeline from green_ground_detection.py into C
 * using the Paparazzi computer vision framework.
 *
 * Pipeline:
 *   1. Apply ML decision tree pixel by pixel to build ground mask
 *   2. Compute green fraction -> decide if ground is visible
 *   3. Isolate valid ground blobs and fill holes
 *   4. Find ground boundary per column
 *   5. Compare boundary to baseline -> detect obstacles
 */

#include "ground_detection.h"
#include "colored_blob_separator.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "boards/bebop.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// -----------------------------------------------------------------------
// Exported results
// -----------------------------------------------------------------------
volatile int     green_pixel_count = 0;
volatile float   green_fraction    = 0.0f;
volatile bool    ground_found      = false;
volatile uint8_t obstacle_count    = 0;
struct obstacle_region_t obstacle_regions[20];

// -----------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------
static float    ground_baseline[IMAGE_WIDTH];
static bool     baseline_initialized = false;
static uint16_t boundary_rows[IMAGE_WIDTH];

// -----------------------------------------------------------------------
// is_ground
// Hardcoded decision tree trained on YUV pixel data.
// Returns 255 if pixel is ground, 0 otherwise.
// Equivalent to is_ground() in green_ground_detection.py (ML version)
// -----------------------------------------------------------------------
uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V)
{
  if (U <= 115) {
    if (V <= 145) {
      if (Y <= 85) {
        if (Y <= 78) return 0;
        else         return 0;
      } else {
        if (U <= 92) return 0;
        else         return 255;
      }
    } else {
      if (V <= 152) {
        if (Y <= 177) return 255;
        else          return 0;
      } else {
        return 0;
      }
    }
  } else {
    if (U <= 121) {
      if (V <= 137) {
        if (Y <= 87) return 0;
        else         return 255;
      } else {
        if (U <= 116) return 0;
        else          return 0;
      }
    } else {
      if (U <= 122) {
        if (V <= 126) return 0;
        else          return 0;
      } else {
        if (Y <= 62) return 0;
        else         return 0;
      }
    }
  }
}

// -----------------------------------------------------------------------
// detect_green_ground_ml
// Applies the decision tree pixel by pixel to build a ground mask.
// Equivalent to detect_green_ground_ml() in green_ground_detection.py
// -----------------------------------------------------------------------
static void detect_green_ground_ml(struct image_t *img, uint8_t *mask,
                                    float *green_frac)
{
  uint8_t *buf = (uint8_t *)img->buf;
  uint32_t green_count = 0;
  uint32_t total = img->w * img->h;

  for (uint16_t row = 0; row < img->h; row++) {
    for (uint16_t col = 0; col < img->w; col++) {

      // Extract YUV values from YUV422 (UYVY format)
      // Each group of 4 bytes = U Y0 V Y1
      uint32_t block = (row * img->w + col) / 2 * 4;
      uint8_t U = buf[block];
      uint8_t V = buf[block + 2];
      uint8_t Y = (col % 2 == 0) ? buf[block + 1] : buf[block + 3];

      // Apply decision tree
      uint8_t result = is_ground(Y, U, V);
      mask[row * img->w + col] = result;

      if (result > 0) green_count++;
    }
  }

  *green_frac = (total > 0) ? (float)green_count / (float)total : 0.0f;
}

// -----------------------------------------------------------------------
// find_ground_boundary
// Scans each column of the flipped mask (ground on left side) and finds
// the topmost green pixel, representing the ground boundary.
// Equivalent to find_ground_boundary() in green_ground_detection.py
// -----------------------------------------------------------------------
static void find_ground_boundary(uint8_t *mask, uint16_t w, uint16_t h,
                                  uint16_t *boundary)
{
  for (uint16_t col = 0; col < w; col++) {

    boundary[col] = h; // sentinel: no ground found in this column

    // Count green pixels in the last MIN_GROUND_PIXELS rows of this column
    uint8_t ground_valid = 0;
    uint16_t check_start = (h > MIN_GROUND_PIXELS) ? h - MIN_GROUND_PIXELS : 0;
    for (uint16_t row = check_start; row < h; row++) {
      if (mask[row * w + col] > 0) {
        ground_valid++;
      }
    }

    if (ground_valid < MIN_GROUND_PIXELS) {
      continue; // not enough green at ground side
    }

    // Find the topmost green pixel scanning from top
    for (uint16_t row = 0; row < h; row++) {
      if (mask[row * w + col] > 0) {
        boundary[col] = row;
        break;
      }
    }
  }
}

// -----------------------------------------------------------------------
// get_obstacle_regions
// Groups obstacle columns into contiguous regions and filters by width.
// Equivalent to get_obstacle_regions() in green_ground_detection.py
// -----------------------------------------------------------------------
static uint8_t get_obstacle_regions(uint8_t *obstacle_mask, uint16_t w,
                                     struct obstacle_region_t *regions,
                                     uint8_t max_regions)
{
  uint8_t count = 0;
  bool in_region = false;
  uint16_t start = 0;
  uint16_t end   = 0;

  for (uint16_t col = 0; col < w; col++) {
    if (obstacle_mask[col]) {
      if (!in_region) {
        start = col;
        end   = col;
        in_region = true;
      } else {
        if (col - end <= MAX_COL_GAP) {
          end = col;
        } else {
          if ((end - start + 1) >= MIN_OBSTACLE_WIDTH && count < max_regions) {
            regions[count].start_col = start;
            regions[count].end_col   = end;
            regions[count].width     = end - start + 1;
            count++;
          }
          start = col;
          end   = col;
        }
      }
    }
  }

  // Save last region
  if (in_region && (end - start + 1) >= MIN_OBSTACLE_WIDTH
      && count < max_regions) {
    regions[count].start_col = start;
    regions[count].end_col   = end;
    regions[count].width     = end - start + 1;
    count++;
  }

  return count;
}

// -----------------------------------------------------------------------
// update_and_detect
// Updates the ground baseline and detects obstacles by comparing the
// current boundary to the baseline.
// Equivalent to update_and_detect() in green_ground_detection.py
// -----------------------------------------------------------------------
static uint8_t update_and_detect(uint16_t *boundary, uint16_t w, uint16_t h,
                                  struct obstacle_region_t *regions,
                                  uint8_t max_regions)
{
  uint8_t obstacle_mask[IMAGE_WIDTH];
  memset(obstacle_mask, 0, sizeof(obstacle_mask));

  // Initialize baseline on first call
  if (!baseline_initialized) {
    for (uint16_t col = 0; col < w; col++) {
      ground_baseline[col] = (boundary[col] < h)
                             ? (float)boundary[col]
                             : NO_GROUND_BASELINE;
    }
    baseline_initialized = true;
    return 0;
  }

  // Compare boundary to baseline, build obstacle mask
  for (uint16_t col = 0; col < w; col++) {
    bool valid     = boundary[col] < h;
    bool no_ground = !valid;

    // Condition 1: ground visible but too far from baseline
    bool deviation_obstacle = valid &&
        ((int)boundary[col] - (int)ground_baseline[col] > OBSTACLE_THRESHOLD);

    // Condition 2: no green visible at all in this column
    obstacle_mask[col] = (deviation_obstacle || no_ground) ? 1 : 0;
  }

  // Update baseline only on clean columns
  float last_good = NO_GROUND_BASELINE;
  for (uint16_t col = 0; col < w; col++) {
    bool valid  = boundary[col] < h;
    bool no_obs = valid && !obstacle_mask[col];

    if (no_obs) {
      // Exponential moving average update
      ground_baseline[col] = (1.0f - ALPHA) * ground_baseline[col]
                             + ALPHA * (float)boundary[col];
      last_good = ground_baseline[col];
    } else {
      // Fill with last good baseline value
      ground_baseline[col] = last_good;
    }
  }

  return get_obstacle_regions(obstacle_mask, w, regions, max_regions);
}

// -----------------------------------------------------------------------
// ground_detection_func
// Main callback called by Paparazzi on every camera frame.
// Equivalent to the main loop body in green_ground_detection.py
// -----------------------------------------------------------------------
static struct image_t *ground_detection_func(struct image_t *img,
    uint8_t camera_id __attribute__((unused)))
{
  // Step 1 - Apply ML decision tree pixel by pixel
  // Equivalent to: mask, result, actual_frac, status = detect_green_ground_ml(...)
  uint8_t mask[IMAGE_WIDTH * IMAGE_HEIGHT];
  detect_green_ground_ml(img, mask, (float *)&green_fraction);

  // Step 2 - Decide if ground is visible
  // Equivalent to: status = "GROUND FOUND" if green_fraction > threshold
  ground_found = (green_fraction > OA_COLOR_COUNT_FRAC);
  green_pixel_count = (int)(green_fraction * img->w * img->h);

  if (!ground_found) {
    obstacle_count = 0;
    return img;
  }

  // Step 3 - Isolate valid ground blobs and fill holes
  // Equivalent to:
  //   mask = cds.isolate_ground_blob(mask, blob_area_threshold=1000)
  //   mask = cds.fill_holes(mask)
  isolate_ground_blob(mask, img->w, img->h, BLOB_AREA_THRESHOLD);
  fill_holes(mask, img->w, img->h);

  // Step 4 - Flip mask horizontally (ground on left side)
  // Equivalent to: mask_flipped = mask[:, ::-1]
  uint8_t mask_flipped[IMAGE_WIDTH * IMAGE_HEIGHT];
  for (uint16_t row = 0; row < img->h; row++) {
    for (uint16_t col = 0; col < img->w; col++) {
      mask_flipped[row * img->w + col] =
          mask[row * img->w + (img->w - 1 - col)];
    }
  }

  // Step 5 - Find ground boundary per column
  // Equivalent to: boundary_rows = find_ground_boundary(mask_flipped)
  find_ground_boundary(mask_flipped, img->w, img->h, boundary_rows);

  // Step 6 - Detect obstacles by comparing boundary to baseline
  // Equivalent to: obstacle_regions, ground_baseline = update_and_detect(...)
  obstacle_count = update_and_detect(boundary_rows, img->w, img->h,
                                      obstacle_regions, 20);

  // Debug print
  if (obstacle_count > 0) {
    printf("[ground_detection] %d obstacle(s) | green: %.2f%%\n",
           obstacle_count, green_fraction * 100.0f);
    for (uint8_t i = 0; i < obstacle_count; i++) {
      printf("  obstacle %d: col %d-%d (width=%d)\n", i,
             obstacle_regions[i].start_col,
             obstacle_regions[i].end_col,
             obstacle_regions[i].width);
    }
  }

  return img;
}

// -----------------------------------------------------------------------
// ground_detection_init
// Registers the module on the Bebop front camera.
// Equivalent to colorfilter_init() in colorfilter.c
// -----------------------------------------------------------------------
void ground_detection_init(void)
{
  // Reset internal state
  baseline_initialized = false;
  memset(ground_baseline, 0, sizeof(ground_baseline));
  memset(boundary_rows,   0, sizeof(boundary_rows));
  obstacle_count = 0;

  // Register callback on front camera
  cv_add_to_device(&front_camera, ground_detection_func,
                   GROUND_DETECTION_FPS, 0);
}