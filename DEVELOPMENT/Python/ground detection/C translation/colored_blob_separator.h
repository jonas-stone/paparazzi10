#ifndef COLORED_BLOB_SEPARATOR_H
#define COLORED_BLOB_SEPARATOR_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

// Max number of blobs to analyze per frame
#define MAX_BLOBS 32

// Blob info structure (equivalent to connectedComponentsWithStats output)
struct blob_info_t {
  uint16_t x_min;    // leftmost column
  uint16_t y_min;    // topmost row
  uint16_t width;    // bounding box width
  uint16_t height;   // bounding box height
  uint32_t area;     // pixel count
  uint8_t  label;    // label assigned to this blob
};

// Isolates valid ground blobs by filtering small and irregular ones.
// Modifies binary_img in place (non-valid blobs set to 0).
// Equivalent to isolate_ground_blob() in colored_blob_separator.py
void isolate_ground_blob(uint8_t *binary_img, uint16_t w, uint16_t h,
                         uint32_t area_threshold);

// Fills black holes inside white ground blobs.
// Modifies binary_img in place.
// Equivalent to fill_holes() in colored_blob_separator.py
void fill_holes(uint8_t *binary_img, uint16_t w, uint16_t h);

#endif