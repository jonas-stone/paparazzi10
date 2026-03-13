/*
 * @file modules/computer_vision/colored_blob_separator.c
 * @brief Connected component labeling and blob filtering in pure C
 *
 * Translated from colored_blob_separator.py
 * Replaces OpenCV connectedComponentsWithStats and findContours
 * with C implementations using flood-fill.
 */

#include "colored_blob_separator.h"
#include "solidity_detection.h"
#include <string.h>
#include <stdlib.h>

// -----------------------------------------------------------------------
// Internal flood-fill stack (avoids recursion to prevent stack overflow)
// -----------------------------------------------------------------------
#define FLOOD_STACK_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

static uint32_t flood_stack[FLOOD_STACK_SIZE];
static uint16_t label_map[IMAGE_WIDTH * IMAGE_HEIGHT];

// -----------------------------------------------------------------------
// flood_fill
// Labels all connected pixels starting from (row, col) with given label.
// Uses iterative approach with explicit stack to avoid recursion overflow.
// -----------------------------------------------------------------------
static void flood_fill(uint8_t *binary_img, uint16_t *labels,
                       uint16_t w, uint16_t h,
                       uint16_t start_row, uint16_t start_col,
                       uint16_t label)
{
  uint32_t stack_top = 0;
  flood_stack[stack_top++] = start_row * w + start_col;
  labels[start_row * w + start_col] = label;

  while (stack_top > 0) {
    uint32_t idx = flood_stack[--stack_top];
    uint16_t row = idx / w;
    uint16_t col = idx % w;

    // Check all 8 neighbours (8-connectivity)
    for (int8_t dr = -1; dr <= 1; dr++) {
      for (int8_t dc = -1; dc <= 1; dc++) {
        if (dr == 0 && dc == 0) continue;

        int32_t nr = (int32_t)row + dr;
        int32_t nc = (int32_t)col + dc;

        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;

        uint32_t nidx = nr * w + nc;
        if (binary_img[nidx] > 0 && labels[nidx] == 0) {
          labels[nidx] = label;
          if (stack_top < FLOOD_STACK_SIZE) {
            flood_stack[stack_top++] = nidx;
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------
// isolate_ground_blob
// Labels connected components, filters by area and smoothness,
// keeps only valid blobs (set to 255), removes others (set to 0).
// Equivalent to isolate_ground_blob() in colored_blob_separator.py
// -----------------------------------------------------------------------
void isolate_ground_blob(uint8_t *binary_img, uint16_t w, uint16_t h,
                         uint32_t area_threshold)
{
  // Clear label map
  memset(label_map, 0, w * h * sizeof(uint16_t));

  // Blob info storage
  struct blob_info_t blobs[MAX_BLOBS];
  uint8_t blob_count = 0;

  // Step 1 - Label all connected components
  uint16_t current_label = 1;
  for (uint16_t row = 0; row < h; row++) {
    for (uint16_t col = 0; col < w; col++) {
      uint32_t idx = row * w + col;
      if (binary_img[idx] > 0 && label_map[idx] == 0
          && blob_count < MAX_BLOBS) {

        // Initialize blob info
        blobs[blob_count].label  = current_label;
        blobs[blob_count].x_min  = col;
        blobs[blob_count].y_min  = row;
        blobs[blob_count].width  = 0;
        blobs[blob_count].height = 0;
        blobs[blob_count].area   = 0;

        // Flood fill from this pixel
        flood_fill(binary_img, label_map, w, h, row, col, current_label);

        // Step 2 - Compute stats for this blob
        uint16_t x_max = 0, y_max = 0;
        uint16_t x_min = w, y_min = h;

        for (uint16_t r = 0; r < h; r++) {
          for (uint16_t c = 0; c < w; c++) {
            if (label_map[r * w + c] == current_label) {
              blobs[blob_count].area++;
              if (c < x_min) x_min = c;
              if (c > x_max) x_max = c;
              if (r < y_min) y_min = r;
              if (r > y_max) y_max = r;
            }
          }
        }

        blobs[blob_count].x_min  = x_min;
        blobs[blob_count].y_min  = y_min;
        blobs[blob_count].width  = x_max - x_min + 1;
        blobs[blob_count].height = y_max - y_min + 1;

        blob_count++;
        current_label++;
      }
    }
  }

  // Step 3 - Filter blobs: remove small or irregular ones
  for (uint8_t b = 0; b < blob_count; b++) {
    bool keep = false;

    if (blobs[b].area < area_threshold) {
      keep = false;
    } else {
      // Extract single blob mask for smoothness check
      uint8_t single_blob[IMAGE_WIDTH * IMAGE_HEIGHT];
      memset(single_blob, 0, w * h);

      for (uint16_t r = 0; r < h; r++) {
        for (uint16_t c = 0; c < w; c++) {
          if (label_map[r * w + c] == blobs[b].label) {
            single_blob[r * w + c] = 255;
          }
        }
      }

      // Check smoothness using solidity_detection
      keep = is_smooth_blob(single_blob, w, h,
                            (float)blobs[b].area,
                            blobs[b].width);
    }

    // Remove blob from image if not valid
    if (!keep) {
      for (uint16_t r = 0; r < h; r++) {
        for (uint16_t c = 0; c < w; c++) {
          if (label_map[r * w + c] == blobs[b].label) {
            binary_img[r * w + c] = 0;
          }
        }
      }
    }
  }

  // Step 4 - Set remaining valid blobs to white (255)
  for (uint32_t i = 0; i < (uint32_t)(w * h); i++) {
    if (binary_img[i] > 0) {
      binary_img[i] = 255;
    }
  }
}

// -----------------------------------------------------------------------
// fill_holes
// Finds background pixels connected to the image border using flood fill,
// then marks everything else inside blobs as foreground (255).
// Equivalent to fill_holes() in colored_blob_separator.py
// -----------------------------------------------------------------------
void fill_holes(uint8_t *binary_img, uint16_t w, uint16_t h)
{
  // Use label_map to mark background pixels reachable from border
  memset(label_map, 0, w * h * sizeof(uint16_t));

  // Flood fill from all border pixels that are background (0)
  for (uint16_t col = 0; col < w; col++) {
    if (binary_img[0 * w + col] == 0 && label_map[0 * w + col] == 0)
      flood_fill(binary_img, label_map, w, h, 0, col, 1);
    if (binary_img[(h-1) * w + col] == 0 && label_map[(h-1) * w + col] == 0)
      flood_fill(binary_img, label_map, w, h, h - 1, col, 1);
  }
  for (uint16_t row = 0; row < h; row++) {
    if (binary_img[row * w + 0] == 0 && label_map[row * w + 0] == 0)
      flood_fill(binary_img, label_map, w, h, row, 0, 1);
    if (binary_img[row * w + (w-1)] == 0 && label_map[row * w + (w-1)] == 0)
      flood_fill(binary_img, label_map, w, h, row, w - 1, 1);
  }

  // Any 0 pixel NOT reachable from border is a hole -> fill it with 255
  for (uint32_t i = 0; i < (uint32_t)(w * h); i++) {
    if (binary_img[i] == 0 && label_map[i] == 0) {
      binary_img[i] = 255;
    }
  }
}