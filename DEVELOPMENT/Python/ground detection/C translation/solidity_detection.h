#ifndef SOLIDITY_DETECTION_H
#define SOLIDITY_DETECTION_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

// Thresholds for blob smoothness
#define THRESHOLD_PERIMETER_RATIO  2.0f
#define THRESHOLD_FRACTAL_DIM      1.3f

// Returns true if the blob is smooth (likely ground),
// false if irregular (likely plant or obstacle)
bool is_smooth_blob(uint8_t *single_blob_mask, uint16_t w, uint16_t h,
                    float area, uint16_t bounding_box_width);

// Computes the perimeter of a binary blob
uint32_t blob_perimeter(uint8_t *padded_mask, uint16_t w, uint16_t h);

// Computes the average blob height given area and bounding box width
float compute_average_blob_height(float blob_area, uint16_t bounding_box_width);

// Extracts edge pixels of a blob into an output mask
void get_blob_edge(uint8_t *binary_img, uint8_t *edge_out, uint16_t w, uint16_t h);

// Computes the fractal dimension of a blob edge
float compute_fractal_dimension(uint8_t *binary_img, uint16_t w, uint16_t h);

#endif