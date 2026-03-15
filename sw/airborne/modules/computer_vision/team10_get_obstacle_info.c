/*
 * HELPER_FUNCTIONS.c
 *
 * Ground detection and obstacle finding pipeline.
 */

#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Forward declarations for file-private helpers                        */
/* ------------------------------------------------------------------ */

static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V);
static void apply_ground_mask(struct image_t *input, struct image_t *mask);
static void median_blur_3x3(struct image_t *mask, struct image_t *blurred);
static uint8_t median_of_9(uint8_t *v);
static void insertion_sort_9(uint8_t *v);
static void insertion_sort_f(float *v, int n);
static void median_filter_1d(const float *data, const uint8_t *valid_mask,
                              int n, int kernel, float *out);
static void flip_horizontal(const struct image_t *src, struct image_t *dst);

/* ================================================================== */
/* SECTION 1 – Per-pixel ground classifier (decision tree)             */
/* ================================================================== */

static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V)
{
    if (U <= 115) {
        if (V <= 145) {
            if (Y <= 85) {
                return 0;
            } else {
                if (U <= 92) {
                    return 0;
                } else {
                    return 255;
                }
            }
        } else {
            if (V <= 152) {
                if (Y <= 177) {
                    return 255;
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        }
    } else {
        if (U <= 121) {
            if (V <= 137) {
                if (Y <= 87) {
                    return 0;
                } else {
                    return 255;
                }
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }
}

/* ================================================================== */
/* SECTION 2 – Build grayscale mask from YUV422 image                  */
/* ================================================================== */

/*
 * UYVY macro-pixel layout (4 bytes → 2 pixels):
 *   byte 0 : U  (shared)
 *   byte 1 : Y0 (pixel 0)
 *   byte 2 : V  (shared)
 *   byte 3 : Y1 (pixel 1)
 */
static void apply_ground_mask(struct image_t *input, struct image_t *mask)
{
    uint8_t *src  = (uint8_t *)input->buf;
    uint8_t *dest = (uint8_t *)mask->buf;

    for (uint16_t y = 0; y < input->h; y++) {
        for (uint16_t x = 0; x < input->w; x += 2) {
            uint8_t U  = src[0];
            uint8_t Y0 = src[1];
            uint8_t V  = src[2];
            uint8_t Y1 = src[3];

            dest[0] = is_ground(Y0, U, V);
            dest[1] = is_ground(Y1, U, V);

            src  += 4;
            dest += 2;
        }
    }
}

/* ================================================================== */
/* SECTION 3 – 3×3 median blur on a grayscale mask                    */
/* ================================================================== */

static void insertion_sort_9(uint8_t *v)
{
    for (int i = 1; i < 9; i++) {
        uint8_t key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}

static uint8_t median_of_9(uint8_t *v)
{
    insertion_sort_9(v);
    return v[4];
}

static void median_blur_3x3(struct image_t *mask, struct image_t *blurred)
{
    uint8_t *src  = (uint8_t *)mask->buf;
    uint8_t *dest = (uint8_t *)blurred->buf;
    uint16_t w    = mask->w;
    uint16_t h    = mask->h;

    /* Border pixels copied as-is */
    memcpy(dest, src, (uint32_t)w * h);

    for (uint16_t y = 1; y < h - 1; y++) {
        for (uint16_t x = 1; x < w - 1; x++) {
            uint8_t window[9];
            int idx = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    window[idx++] = src[(y + dy) * w + (x + dx)];
                }
            }
            dest[y * w + x] = median_of_9(window);
        }
    }
}

/* ================================================================== */
/* SECTION 4 – detect_green_ground_ml (public)                         */
/* ================================================================== */

int detect_green_ground_ml(struct image_t *input,
                           struct image_t *mask_out,
                           float           threshold,
                           int             apply_median,
                           float          *green_fraction)
{
    /* Step 1 – classify every pixel into the mask */
    apply_ground_mask(input, mask_out);

    /* Step 2 – optional 3×3 median blur
     *
     * image_switch() swaps only the struct metadata (buf pointer, sizes),
     * which is O(1) and avoids the full pixel copy that image_copy() would
     * do.  After the swap, tmp holds the old mask_out buffer and is freed.
     */
    if (apply_median) {
        struct image_t tmp;
        image_create(&tmp, mask_out->w, mask_out->h, IMAGE_GRAYSCALE);
        median_blur_3x3(mask_out, &tmp);
        image_switch(mask_out, &tmp);   /* O(1) pointer swap */
        image_free(&tmp);               /* frees the pre-blur buffer */
    }

    /* Step 3 – count ground pixels */
    uint8_t  *buf         = (uint8_t *)mask_out->buf;
    uint32_t  total       = (uint32_t)mask_out->w * (uint32_t)mask_out->h;
    uint32_t  green_count = 0;

    for (uint32_t i = 0; i < total; i++) {
        if (buf[i] != 0) {
            green_count++;
        }
    }

    /* Step 4 – fraction and threshold */
    float frac = (total > 0) ? ((float)green_count / (float)total) : 0.0f;
    if (green_fraction != NULL) {
        *green_fraction = frac;
    }

    return (frac > threshold) ? 1 : 0;
}

/* ================================================================== */
/* SECTION 5 – 1-D median filter (for boundary smoothing)              */
/* ================================================================== */

static void insertion_sort_f(float *v, int n)
{
    for (int i = 1; i < n; i++) {
        float key = v[i];
        int   j   = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}

/*
 * 1-D median filter with reflect-padding (matches scipy default).
 * Only positions where valid_mask[i] != 0 are updated; others keep
 * their original value from data[].
 */
static void median_filter_1d(const float   *data,
                              const uint8_t *valid_mask,
                              int            n,
                              int            kernel,
                              float         *out)
{
    memcpy(out, data, (uint32_t)n * sizeof(float));

    int   half = kernel / 2;
    float window[32]; /* kernel ≤ 31 covers all sane usage */

    for (int i = 0; i < n; i++) {
        if (!valid_mask[i]) {
            continue;
        }

        for (int k = -half; k <= half; k++) {
            int idx = i + k;

            /* reflect padding */
            if (idx < 0)  { idx = -idx; }
            if (idx >= n) { idx = 2 * (n - 1) - idx; }
            /* clamp for very small n */
            if (idx < 0)  { idx = 0; }
            if (idx >= n) { idx = n - 1; }

            window[k + half] = data[idx];
        }

        insertion_sort_f(window, kernel);
        out[i] = window[half];
    }
}

/* ================================================================== */
/* SECTION 6 – find_ground_boundary (public)                           */
/* ================================================================== */

void find_ground_boundary(const struct image_t *mask_flipped,
                          int                  *boundary_rows_out,
                          int                   min_ground_pixels,
                          int                   max_gap,
                          int                   smooth_kernel)
{
    int            image_height = mask_flipped->w;
    int            n_rows       = mask_flipped->h;
    const uint8_t *buf          = (const uint8_t *)mask_flipped->buf;

    uint16_t green_pos[MAX_IMAGE_WIDTH];

    /* Default: no ground found in any row */
    for (int i = 0; i < n_rows; i++) {
        boundary_rows_out[i] = image_height;
    }

    for (int idx = 0; idx < n_rows; idx++) {
        const uint8_t *row = buf + idx * image_height;

        /* Collect green pixel positions in this row */
        int n_green = 0;
        for (int c = 0; c < image_height; c++) {
            if (row[c] > 0) {
                green_pos[n_green++] = (uint16_t)c;
            }
        }

        if (n_green == 0) {
            continue;
        }

        /* Check ground_valid: last min_ground_pixels columns all green */
        int ground_valid = 0;
        {
            int count = 0;
            for (int c = image_height - min_ground_pixels; c < image_height; c++) {
                if (row[c] > 0) count++;
            }
            ground_valid = (count >= min_ground_pixels);
        }

        if (!ground_valid) {
            /*
             * Find the longest contiguous run of green pixels.
             * Skip row if longest run < max_gap.
             */
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < n_green; i++) {
                if (green_pos[i] - green_pos[i - 1] == 1) {
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 1;
                }
            }
            if (max_run < max_gap) {
                continue;
            }
        }

        /*
         * Work on the reversed green_pos array (called "gp" in Python).
         * GP(i) = green_pos[n_green - 1 - i]  (descending order)
         */
#define GP(i) green_pos[n_green - 1 - (i)]

        if (n_green == 1) {
            boundary_rows_out[idx] = GP(0);
            continue;
        }

        /*
         * gaps[i] = GP(i) - GP(i+1) - 1  (positive, gp is descending)
         * big gap when gaps[i] > max_gap
         */
        int first_big_gap = -1;
        for (int i = 0; i < n_green - 1; i++) {
            int gap = (int)GP(i) - (int)GP(i + 1) - 1;
            if (gap > max_gap) {
                first_big_gap = i;
                break;
            }
        }

        if (first_big_gap < 0) {
            /* No big gap → boundary is the smallest green position */
            boundary_rows_out[idx] = GP(n_green - 1);
            continue;
        }

        int fg               = first_big_gap;
        int pending_boundary = GP(fg);
        int after_size       = n_green - 1 - fg;

        if (after_size >= max_gap) {
            /*
             * Check if after_gap contains a run of length >= max_gap.
             * after_gap[i] = GP(fg + 1 + i), also descending.
             * Consecutive when GP(fg+1+i-1) - GP(fg+1+i) - 1 == 0.
             */
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < after_size; i++) {
                int g = (int)GP(fg + 1 + i - 1) - (int)GP(fg + 1 + i) - 1;
                if (g == 0) {
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 1;
                }
            }
            if (max_run >= max_gap) {
                boundary_rows_out[idx] = GP(n_green - 1);
                continue;
            }
        }

        boundary_rows_out[idx] = pending_boundary;

#undef GP
    }

    /* 1-D median smoothing over valid boundary positions */
    if (smooth_kernel >= 3 && smooth_kernel % 2 == 1) {
        uint8_t valid_mask[MAX_IMAGE_WIDTH];
        int     valid_count = 0;

        for (int i = 0; i < n_rows; i++) {
            valid_mask[i] = (boundary_rows_out[i] < image_height) ? 1 : 0;
            if (valid_mask[i]) valid_count++;
        }

        if (valid_count > smooth_kernel) {
            float data[MAX_IMAGE_WIDTH];
            float smoothed[MAX_IMAGE_WIDTH];

            for (int i = 0; i < n_rows; i++) {
                data[i] = (float)boundary_rows_out[i];
            }

            median_filter_1d(data, valid_mask, n_rows, smooth_kernel, smoothed);

            for (int i = 0; i < n_rows; i++) {
                if (valid_mask[i]) {
                    boundary_rows_out[i] = (int)(smoothed[i] + 0.5f);
                }
            }
        }
    }
}

/* ================================================================== */
/* SECTION 7 – get_obstacle_regions (public)                           */
/* ================================================================== */

int get_obstacle_regions(const uint16_t           *obstacle_cols,
                         int                       n_cols,
                         int                       min_width,
                         int                       max_col_gap,
                         struct obstacle_region_t *regions_out)
{
    if (n_cols == 0) {
        return 0;
    }

    int      n_regions = 0;
    uint16_t start     = obstacle_cols[0];
    uint16_t end       = obstacle_cols[0];

    for (int i = 1; i < n_cols; i++) {
        uint16_t col = obstacle_cols[i];

        if ((int)(col - end) <= max_col_gap) {
            end = col;
        } else {
            uint16_t w = end - start + 1;
            if (w >= (uint16_t)min_width && n_regions < MAX_OBSTACLE_REGIONS) {
                regions_out[n_regions].start = start;
                regions_out[n_regions].end   = end;
                regions_out[n_regions].width = w;
                n_regions++;
            }
            start = col;
            end   = col;
        }
    }

    /* Close final region */
    uint16_t w = end - start + 1;
    if (w >= (uint16_t)min_width && n_regions < MAX_OBSTACLE_REGIONS) {
        regions_out[n_regions].start = start;
        regions_out[n_regions].end   = end;
        regions_out[n_regions].width = w;
        n_regions++;
    }

    return n_regions;
}

/* ================================================================== */
/* SECTION 8 – update_and_detect (public)                              */
/* ================================================================== */

/*
 * boundary_row is now int* throughout — fixes the int / uint16_t type
 * mismatch that existed in the original get_obstacle_info call site.
 */
int update_and_detect(const int  *boundary_row,
                      int         width,
                      int         h,
                      float      *ground_baseline,
                      int        *baseline_inited,
                      int         min_width,
                      struct obstacle_region_t *regions_out)
{
    /* First call: initialise baseline, skip detection */
    if (!(*baseline_inited)) {
        for (int x = 0; x < width; x++) {
            ground_baseline[x] = (boundary_row[x] >= h)
                                 ? NO_GROUND_BASELINE
                                 : (float)boundary_row[x];
        }
        *baseline_inited = 1;
        return 0;
    }

    uint16_t obstacle_cols[MAX_IMAGE_WIDTH];
    int      n_obstacle_cols = 0;

    uint8_t valid[MAX_IMAGE_WIDTH];
    uint8_t deviation_obstacle[MAX_IMAGE_WIDTH];
    uint8_t no_obstacle_mask[MAX_IMAGE_WIDTH];

    for (int x = 0; x < width; x++) {
        valid[x] = (boundary_row[x] < h) ? 1 : 0;

        float deviation       = (float)boundary_row[x] - ground_baseline[x];
        deviation_obstacle[x] = (valid[x] && deviation > OBSTACLE_THRESHOLD) ? 1 : 0;

        /* Obstacle if deviation is too large OR no ground at all */
        if (deviation_obstacle[x] || !valid[x]) {
            obstacle_cols[n_obstacle_cols++] = (uint16_t)x;
        }

        no_obstacle_mask[x] = (valid[x] && !deviation_obstacle[x]) ? 1 : 0;
    }

    /* Detect obstacle regions */
    int n_regions = get_obstacle_regions(obstacle_cols, n_obstacle_cols,
                                         min_width, 5, regions_out);

    /* EMA update for non-obstacle columns */
    for (int x = 0; x < width; x++) {
        if (no_obstacle_mask[x]) {
            ground_baseline[x] = (1.0f - OBSTACLE_ALPHA) * ground_baseline[x]
                                + OBSTACLE_ALPHA * (float)boundary_row[x];
        }
    }

    /* Forward-fill baseline for obstacle / no-ground columns */
    float last_good = NO_GROUND_BASELINE;
    for (int x = 0; x < width; x++) {
        if (no_obstacle_mask[x]) {
            last_good = ground_baseline[x];
        } else {
            ground_baseline[x] = last_good;
        }
    }

    return n_regions;
}

/* ================================================================== */
/* SECTION 9 – flip_horizontal (file-private helper)                   */
/* ================================================================== */

static void flip_horizontal(const struct image_t *src, struct image_t *dst)
{
    const uint8_t *s = (const uint8_t *)src->buf;
    uint8_t       *d = (uint8_t *)dst->buf;
    int w = src->w;
    int h = src->h;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            d[y * w + (w - 1 - x)] = s[y * w + x];
        }
    }
}

/* ================================================================== */
/* SECTION 10 – get_obstacle_info (public)                             */
/* ================================================================== */

int get_obstacle_info(struct image_t           *input,
                      float                    *ground_baseline,
                      int                      *baseline_inited,
                      float                     oa_color_count_frac,
                      int                       median_ksize,
                      int                       min_width,
                      struct obstacle_region_t *obstacles_out,
                      int                      *boundary_rows_out,
                      int                      *ground_found_out,
                      float                    *green_frac_out)
{
    int W = input->w;
    int H = input->h;

    /* Step 1 – ground mask */
    struct image_t mask;
    image_create(&mask, W, H, IMAGE_GRAYSCALE);

    float green_frac   = 0.0f;
    int   apply_median = (median_ksize >= 3 && median_ksize % 2 == 1);
    int   ground_found = detect_green_ground_ml(input, &mask,
                                                oa_color_count_frac,
                                                apply_median,
                                                &green_frac);

    if (ground_found_out != NULL) { *ground_found_out = ground_found; }
    if (green_frac_out   != NULL) { *green_frac_out   = green_frac;   }

    if (!ground_found) {
        image_free(&mask);
        return 0;
    }

    /* Step 2 – flip mask horizontally */
    struct image_t mask_flipped;
    image_create(&mask_flipped, W, H, IMAGE_GRAYSCALE);
    flip_horizontal(&mask, &mask_flipped);
    image_free(&mask);

    /* Step 3 – find ground boundary per column */
    find_ground_boundary(&mask_flipped,
                         boundary_rows_out,
                         /* min_ground_pixels= */ 5,
                         /* max_gap=           */ 10,
                         /* smooth_kernel=     */ 5);
    image_free(&mask_flipped);

    /* Step 4 – update baseline and detect obstacle regions
     *
     * update_and_detect now takes int* directly, no cast needed.
     * regions are still in flipped coordinates here.
     */
    struct obstacle_region_t raw_regions[MAX_OBSTACLE_REGIONS];
    int n_regions = update_and_detect(boundary_rows_out,
                                      W, H,
                                      ground_baseline,
                                      baseline_inited,
                                      min_width,
                                      raw_regions);

    /*
     * Step 5 – convert from flipped to original image coordinates.
     *
     * Flipped column c  →  original column (W - 1 - c).
     * Region [s, e] in flipped space maps to:
     *   original left  = W - 1 - e   (smallest original index)
     *   original right = W - 1 - s
     *   width          = e - s + 1   (unchanged)
     */
    for (int i = 0; i < n_regions; i++) {
        obstacles_out[i].start = (uint16_t)(W - 1 - (int)raw_regions[i].end);
        obstacles_out[i].end   = (uint16_t)(W - 1 - (int)raw_regions[i].start);
        obstacles_out[i].width = raw_regions[i].width;
    }

    return n_regions;
}