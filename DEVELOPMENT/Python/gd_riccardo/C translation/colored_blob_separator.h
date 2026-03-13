/*
 * ground_blob.h
 *
 * C translations of isolate_ground_blob() and fill_holes() for use
 * within the Paparazzi computer-vision framework.
 *
 * All functions operate on IMAGE_GRAYSCALE image_t buffers (uint8, one
 * byte per pixel) so they slot cleanly into the existing pipeline.
 */
 
#ifndef COLORED_BLOB_SEPARATOR
#define COLORED_BLOB_SEPARATOR
 
#include "image.h"   /* struct image_t, image_create, image_free … */
#include <stdint.h>
#include <stdbool.h>
 
/*
 * isolate_ground_blob
 *
 * Performs 8-connected component labelling on a binary grayscale image
 * (non-zero pixels are treated as foreground).  Blobs whose pixel count
 * is strictly less than MIN_BLOB_AREA are erased.  Every surviving
 * foreground pixel is set to 255; background pixels are set to 0.
 *
 * The result is written back into `img` in-place.
 *
 * @param img   IMAGE_GRAYSCALE image_t (modified in-place)
 */
void isolate_ground_blob(struct image_t *img);
 
/*
 * fill_holes
 *
 * Fills interior holes (connected regions of zero pixels that are not
 * reachable from any image border) with 255.
 *
 * The result is written back into `img` in-place.
 *
 * @param img   IMAGE_GRAYSCALE image_t (modified in-place)
 */
void fill_holes(struct image_t *img);
 
#endif /* GROUND_BLOB_H */