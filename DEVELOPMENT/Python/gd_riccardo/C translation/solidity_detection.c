/*
 * blob_shape.c
 *
 * Blob smoothness / shape analysis translated from Python to C.
 * Compatible with the Paparazzi image_t / IMAGE_GRAYSCALE format.
 */

#include "solidity_detection.h"

#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "modules/computer_vision/lib/vision/image.h"

/* -------------------------------------------------------------------------
 * Helper: inline pixel accessor for a GRAYSCALE image_t.
 * Returns the byte value at (col, row).  No bounds checking – callers are
 * responsible for staying inside the image.
 * ---------------------------------------------------------------------- */
static inline uint8_t px(const struct image_t *img, uint16_t col, uint16_t row)
{
  return ((const uint8_t *)img->buf)[row * img->w + col];
}

/* =========================================================================
 * blob_perimeter
 *
 * Python equivalent:
 *   perimeter = np.sum(padded[:,1:] != padded[:,:-1]) +   # horizontal transitions
 *               np.sum(padded[1:,:] != padded[:-1,:])     # vertical   transitions
 * ========================================================================= */
uint32_t blob_perimeter(const struct image_t *padded)
{
  uint32_t count = 0;
  uint16_t w = padded->w;
  uint16_t h = padded->h;

  /* Horizontal transitions: compare pixel (col, row) with (col+1, row) */
  for (uint16_t row = 0; row < h; row++) {
    for (uint16_t col = 0; col < w - 1; col++) {
      if ((px(padded, col, row) != 0) != (px(padded, col + 1, row) != 0)) {
        count++;
      }
    }
  }

  /* Vertical transitions: compare pixel (col, row) with (col, row+1) */
  for (uint16_t row = 0; row < h - 1; row++) {
    for (uint16_t col = 0; col < w; col++) {
      if ((px(padded, col, row) != 0) != (px(padded, col, row + 1) != 0)) {
        count++;
      }
    }
  }

  return count;
}

/* =========================================================================
 * get_blob_edge
 *
 * Python equivalent:
 *   edge1 = (padded[:,1:] != padded[:,:-1])   padded to original width
 *   edge2 = (padded[1:,:] != padded[:-1,:])   padded to original height
 *   return edge1 | edge2
 *
 * edge_out must be IMAGE_GRAYSCALE with the same w/h as padded and its
 * buffer already allocated.
 * ========================================================================= */
void get_blob_edge(const struct image_t *padded, struct image_t *edge_out)
{
  uint16_t w = padded->w;
  uint16_t h = padded->h;
  uint8_t *out = (uint8_t *)edge_out->buf;

  /* Zero the output buffer first */
  for (uint32_t i = 0; i < (uint32_t)w * h; i++) {
    out[i] = 0;
  }

  /* Horizontal transitions → set pixel at the left-side column */
  for (uint16_t row = 0; row < h; row++) {
    for (uint16_t col = 0; col < w - 1; col++) {
      if ((px(padded, col, row) != 0) != (px(padded, col + 1, row) != 0)) {
        out[row * w + col] = 1;
      }
    }
  }

  /* Vertical transitions → OR into the upper-side row */
  for (uint16_t row = 0; row < h - 1; row++) {
    for (uint16_t col = 0; col < w; col++) {
      if ((px(padded, col, row) != 0) != (px(padded, col, row + 1) != 0)) {
        out[row * w + col] = 1;
      }
    }
  }
}

/* =========================================================================
 * compute_average_blob_height
 *
 * Python equivalent:
 *   avg_height = blob_area // bounding_box_width
 * ========================================================================= */
uint32_t compute_average_blob_height(uint32_t blob_area, uint16_t bounding_box_width)
{
  if (bounding_box_width == 0) {
    return 0;
  }
  return blob_area / (uint32_t)bounding_box_width;
}

/* =========================================================================
 * compute_fractal_dimension
 *
 * Python equivalent (box-counting fractal dimension):
 *
 *   img    = get_blob_edge(padded)
 *   scales = [2**i for i in range(1, int(log2(min_dim)))]
 *   for s in scales:
 *       count non-empty s×s tiles
 *   coeff = np.polyfit(log(eps), log(boxes), 1)   # degree-1 LSQ fit
 *   return -coeff[0]
 *
 * np.polyfit degree-1 closed form:
 *   slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
 * ========================================================================= */
float compute_fractal_dimension(const struct image_t *padded)
{
  /* ---- Step 1: build the edge image ------------------------------------ */
  struct image_t edge;
  image_create(&edge, padded->w, padded->h, IMAGE_GRAYSCALE);
  get_blob_edge(padded, &edge);

  uint16_t w = edge.w;
  uint16_t h = edge.h;
  const uint8_t *edge_buf = (const uint8_t *)edge.buf;

  /* ---- Step 2: determine scale range ----------------------------------- */
  uint16_t min_dim = (w < h) ? w : h;

  /* Number of valid scales: s = 2^1, 2^2, ..., 2^(floor(log2(min_dim))-1)
   * Python: range(1, int(np.log2(min_dim)))  → exponents 1 … floor(log2)-1 */
  int max_exp = 0;
  while ((1 << (max_exp + 1)) < (int)min_dim) {
    max_exp++;
  }
  /* max_exp = floor(log2(min_dim)) - 1  (same as Python's exclusive upper bound) */

  int n_scales = max_exp; /* exponents 1 .. max_exp */

  if (n_scales < 2) {
    /* Not enough scales for a meaningful fit – return a neutral value */
    image_free(&edge);
    return 1.0f;
  }

  /* ---- Step 3: box counting for each scale ----------------------------- */
  /* Use VLAs – scales are small (≤ ~10 entries for typical image sizes) */
  float eps_log[n_scales];
  float box_log[n_scales];

  for (int idx = 0; idx < n_scales; idx++) {
    int s = 1 << (idx + 1); /* s = 2, 4, 8, … */

    int trimmed_h = (h / s) * s;
    int trimmed_w = (w / s) * s;

    uint32_t box_count = 0;

    for (int by = 0; by < trimmed_h / s; by++) {
      for (int bx = 0; bx < trimmed_w / s; bx++) {
        /* Check whether any pixel in the s×s tile is non-zero */
        bool any_set = false;
        for (int dy = 0; dy < s && !any_set; dy++) {
          for (int dx = 0; dx < s && !any_set; dx++) {
            int row = by * s + dy;
            int col = bx * s + dx;
            if (edge_buf[row * w + col] != 0) {
              any_set = true;
            }
          }
        }
        if (any_set) {
          box_count++;
        }
      }
    }

    eps_log[idx] = logf((float)s);
    /* Protect against log(0) – should not happen on a real blob */
    box_log[idx] = (box_count > 0) ? logf((float)box_count) : 0.0f;
  }

  image_free(&edge);

  /* ---- Step 4: least-squares slope (degree-1 polyfit) ------------------ */
  /* slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)                           */
  float sum_x  = 0.0f, sum_y  = 0.0f;
  float sum_xy = 0.0f, sum_xx = 0.0f;
  int   n      = n_scales;

  for (int i = 0; i < n; i++) {
    sum_x  += eps_log[i];
    sum_y  += box_log[i];
    sum_xy += eps_log[i] * box_log[i];
    sum_xx += eps_log[i] * eps_log[i];
  }

  float denom = (float)n * sum_xx - sum_x * sum_x;
  if (fabsf(denom) < 1e-9f) {
    return 1.0f; /* degenerate case */
  }

  float slope = ((float)n * sum_xy - sum_x * sum_y) / denom;

  /* Python returns -coeff[0]; slope of log(boxes)/log(eps) is negative
   * for a fractal, so we negate to get a positive dimension. */
  return -slope;
}

/* =========================================================================
 * is_smooth_blob
 *
 * Python equivalent:
 *   padded = np.pad(pixels, 1, mode='constant', constant_values=0)
 *   perimeter     = blob_perimeter(padded)
 *   avg_height    = compute_average_blob_height(area, bb_width)
 *   eq_rect_perim = 2 * (avg_height + bb_width)
 *   fractal_dim   = compute_fractal_dimension(padded)
 *   if perimeter / eq_rect_perim < 2 or fractal_dim < 1.3:
 *       return True
 *   return False
 * ========================================================================= */
bool is_smooth_blob(const struct image_t *single_blob_mask,
                    uint32_t area,
                    uint16_t bounding_box_width)
{
  const float THRESHOLD_PERIMETER   = 2.0f;
  const float THRESHOLD_FRACTAL_DIM = 1.3f;

  /* ---- Pad the mask by 1 pixel with zeros (mirrors np.pad constant=0) -- */
  struct image_t padded;
  /* image_add_border creates a new image with border_size=1 padding by
   * mirroring edges – but we need zero-padding, not mirror-padding.
   * We therefore create the padded image manually.                         */
  uint16_t pw = single_blob_mask->w + 2;
  uint16_t ph = single_blob_mask->h + 2;
  image_create(&padded, pw, ph, IMAGE_GRAYSCALE);

  uint8_t *pad_buf  = (uint8_t *)padded.buf;
  const uint8_t *src_buf = (const uint8_t *)single_blob_mask->buf;

  /* Zero the entire padded buffer first */
  for (uint32_t i = 0; i < (uint32_t)pw * ph; i++) {
    pad_buf[i] = 0;
  }
  /* Copy original rows into the centre of the padded buffer */
  for (uint16_t row = 0; row < single_blob_mask->h; row++) {
    for (uint16_t col = 0; col < single_blob_mask->w; col++) {
      pad_buf[(row + 1) * pw + (col + 1)] = src_buf[row * single_blob_mask->w + col];
    }
  }

  /* ---- Perimeter ------------------------------------------------------ */
  uint32_t perimeter = blob_perimeter(&padded);

  /* ---- Equivalent rectangle perimeter --------------------------------- */
  uint32_t avg_height = compute_average_blob_height(area, bounding_box_width);
  float eq_rect_perim = 2.0f * ((float)avg_height + (float)bounding_box_width);

  /* ---- Fractal dimension ---------------------------------------------- */
  float fractal_dim = compute_fractal_dimension(&padded);

  image_free(&padded);

  /* ---- Decision -------------------------------------------------------- */
  if (eq_rect_perim < 1e-6f) {
    /* Degenerate bounding box – treat as smooth */
    return true;
  }

  if (((float)perimeter / eq_rect_perim) < THRESHOLD_PERIMETER
      || fractal_dim < THRESHOLD_FRACTAL_DIM) {
    return true;
  }

  return false;
}