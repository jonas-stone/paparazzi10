/*
 * blob_shape.h
 *
 * Blob smoothness / shape analysis translated from Python to C.
 * Compatible with the Paparazzi image_t / IMAGE_GRAYSCALE format.
 *
 * The input mask is always an IMAGE_GRAYSCALE image_t where
 * non-zero pixels belong to the blob and zero pixels are background.
 */

#ifndef SOLIDITY_DETECTION_H
#define SOLIDITY_DETECTION_H

#include <stdbool.h>
#include <stdint.h>
#include "modules/computer_vision/lib/vision/image.h"

/**
 * Compute the perimeter of a blob in a padded binary image.
 * Counts transitions between zero and non-zero pixels along
 * every row (horizontal edges) and every column (vertical edges).
 *
 * @param padded  Grayscale image_t that has already been padded by 1 pixel
 *                (use image_add_border before calling).
 * @return        Perimeter in pixels.
 */
uint32_t blob_perimeter(const struct image_t *padded);

/**
 * Fill edge_out (same dimensions as padded) with 1 where a blob-edge
 * pixel exists (horizontal OR vertical transition), 0 elsewhere.
 * edge_out must already be allocated with the same w/h as padded.
 *
 * @param padded    Padded grayscale image_t (blob pixels != 0).
 * @param edge_out  Pre-allocated IMAGE_GRAYSCALE image_t of the same size.
 */
void get_blob_edge(const struct image_t *padded, struct image_t *edge_out);

/**
 * Compute the average blob height as area / bounding_box_width.
 *
 * @param blob_area         Number of blob pixels.
 * @param bounding_box_width  Width of the bounding box (roof-to-ground direction).
 * @return Average height (integer division).
 */
uint32_t compute_average_blob_height(uint32_t blob_area, uint16_t bounding_box_width);

/**
 * Estimate the fractal (box-counting) dimension of a blob's edge.
 * Uses box sizes 2, 4, 8, … up to the largest power of 2 that fits
 * in the smaller image dimension, then fits log(boxes) vs log(scale)
 * with a least-squares line and returns the negated slope.
 *
 * @param padded  Padded grayscale image_t containing ONE blob (blob pixels != 0).
 * @return        Fractal dimension (positive float, typically 1.0 – 2.0).
 */
float compute_fractal_dimension(const struct image_t *padded);

/**
 * Decide whether a blob is smooth (ground) or spiky (plant).
 *
 * Mirrors the Python is_smooth_blob() logic exactly:
 *   - perimeter / equivalent_rectangle_perimeter < 2  → smooth
 *   - fractal_dimension < 1.3                         → smooth
 *
 * @param single_blob_mask   IMAGE_GRAYSCALE image_t with exactly one blob
 *                           (non-zero pixels = blob, zero = background).
 * @param area               Number of blob pixels.
 * @param bounding_box_width Bounding-box width in the roof-to-ground direction.
 * @return true if smooth (likely ground), false if spiky (likely plant).
 */
bool is_smooth_blob(const struct image_t *single_blob_mask, uint32_t area, uint16_t bounding_box_width);

#endif /* BLOB_SHAPE_H */