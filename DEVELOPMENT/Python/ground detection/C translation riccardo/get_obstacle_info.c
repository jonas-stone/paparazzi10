#include "modules/computer_vision/image.h"
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Forward declarations for helpers defined below                       */
/* ------------------------------------------------------------------ */
static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V);
static void apply_ground_mask(struct image_t *input, struct image_t *mask);
static void median_blur_3x3(struct image_t *mask, struct image_t *blurred);
static uint8_t median_of_9(uint8_t *v);
static void insertion_sort_9(uint8_t *v);

/* ------------------------------------------------------------------ */
/* Main detection function                                              */
/* ------------------------------------------------------------------ */

/**
 * detect_green_ground_ml
 *
 * Applies the decision-tree ground classifier to every pixel of a YUV422
 * image, optionally smooths the result with a 3x3 median filter, then
 * returns the ground fraction and a status flag.
 *
 * @param input           Source IMAGE_YUV422 image
 * @param mask_out        OUTPUT: IMAGE_GRAYSCALE mask (pre-created, same size)
 *                        255 = ground pixel, 0 = not ground
 * @param threshold       Fraction threshold in [0,1]; above → ground found
 * @param apply_median    Non-zero to apply 3x3 median blur on the mask
 * @param green_fraction  OUTPUT: fraction of pixels classified as ground
 * @return                1 if "GROUND FOUND", 0 if "NO GROUND"
 */
int detect_green_ground_ml(struct image_t *input,
                            struct image_t *mask_out,
                            float threshold,
                            int apply_median,
                            float *green_fraction)
{
    /* Step 1 - build the raw ground mask from the YUV422 input */
    apply_ground_mask(input, mask_out);

    /* Step 2 - optional 3x3 median blur to reduce speckles */
    if (apply_median) {
        /* Work in-place: write into a temporary, then copy back */
        struct image_t tmp;
        image_create(&tmp, mask_out->w, mask_out->h, IMAGE_GRAYSCALE);
        median_blur_3x3(mask_out, &tmp);
        image_copy(&tmp, mask_out);
        image_free(&tmp);
    }

    /* Step 3 - count non-zero (ground) pixels */
    uint8_t *buf         = (uint8_t *)mask_out->buf;
    uint32_t total       = (uint32_t)mask_out->w * (uint32_t)mask_out->h;
    uint32_t green_count = 0;

    for (uint32_t i = 0; i < total; i++) {
        if (buf[i] != 0) {
            green_count++;
        }
    }

    /* Step 4 - compute fraction and threshold */
    float frac = (total > 0) ? ((float)green_count / (float)total) : 0.0f;
    if (green_fraction != NULL) {
        *green_fraction = frac;
    }

    return (frac > threshold) ? 1 : 0;   /* 1 = GROUND FOUND, 0 = NO GROUND */
}

/* ------------------------------------------------------------------ */
/* Decision tree (from previous translation)                            */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Build grayscale mask from a YUV422 image                             */
/* ------------------------------------------------------------------ */

/**
 * Iterate over every pixel of a UYVY image and write 255/0 into the
 * grayscale mask according to is_ground().
 */
static void apply_ground_mask(struct image_t *input, struct image_t *mask)
{
    uint8_t *src  = (uint8_t *)input->buf;
    uint8_t *dest = (uint8_t *)mask->buf;

    /*
     * UYVY macro-pixel layout (4 bytes → 2 pixels):
     *   [0] U   shared
     *   [1] Y0  pixel 0
     *   [2] V   shared
     *   [3] Y1  pixel 1
     */
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

/* ------------------------------------------------------------------ */
/* 3x3 median blur on a grayscale image                                 */
/* ------------------------------------------------------------------ */

/**
 * Sort 9 values in-place with insertion sort (tiny array, no overhead).
 */
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
    return v[4];   /* middle element */
}

/**
 * Apply a 3x3 median filter to a grayscale mask.
 * Border pixels are left as-is (clamped to original value).
 *
 * @param mask     Input  IMAGE_GRAYSCALE
 * @param blurred  Output IMAGE_GRAYSCALE (pre-created, same size)
 */
static void median_blur_3x3(struct image_t *mask, struct image_t *blurred)
{
    uint8_t *src  = (uint8_t *)mask->buf;
    uint8_t *dest = (uint8_t *)blurred->buf;
    uint16_t w    = mask->w;
    uint16_t h    = mask->h;

    /* Copy everything first so border pixels are already correct */
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


/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

#define OBSTACLE_ALPHA          0.6f
#define OBSTACLE_THRESHOLD      50.0f
#define NO_GROUND_BASELINE      220.0f
#define MAX_OBSTACLE_REGIONS    64    /* max regions we can return    */
#define MAX_IMAGE_WIDTH         520   /* adjust to your camera width  */

/* ------------------------------------------------------------------ */
/* Structs                                                              */
/* ------------------------------------------------------------------ */

/* Mirrors the Python (start, end, width) tuple */
struct obstacle_region_t {
    uint16_t start;   ///< first column of the region (inclusive)
    uint16_t end;     ///< last  column of the region (inclusive)
    uint16_t width;   ///< end - start + 1
};

/* ------------------------------------------------------------------ */
/* get_obstacle_regions                                                 */
/* ------------------------------------------------------------------ */

/**
 * Groups a sorted list of obstacle column indices into contiguous regions.
 * Gaps of at most max_col_gap columns are bridged.
 * Only regions with width >= min_width are kept.
 *
 * @param obstacle_cols   Sorted array of column indices flagged as obstacle
 * @param n_cols          Number of entries in obstacle_cols
 * @param min_width       Minimum region width to keep
 * @param max_col_gap     Maximum gap between columns before a new region starts
 * @param regions_out     Output array (caller-supplied, size MAX_OBSTACLE_REGIONS)
 * @return                Number of valid regions written to regions_out
 */
int get_obstacle_regions(const uint16_t *obstacle_cols,
                         int             n_cols,
                         int             min_width,
                         int             max_col_gap,
                         struct obstacle_region_t *regions_out)
{
    if (n_cols == 0) {
        return 0;
    }

    int n_regions = 0;

    uint16_t start = obstacle_cols[0];
    uint16_t end   = obstacle_cols[0];

    for (int i = 1; i < n_cols; i++) {
        uint16_t col = obstacle_cols[i];

        if ((int)(col - end) <= max_col_gap) {
            /* extend current region */
            end = col;
        } else {
            /* close current region and start a new one */
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

    /* close the final region */
    uint16_t w = end - start + 1;
    if (w >= (uint16_t)min_width && n_regions < MAX_OBSTACLE_REGIONS) {
        regions_out[n_regions].start = start;
        regions_out[n_regions].end   = end;
        regions_out[n_regions].width = w;
        n_regions++;
    }

    return n_regions;
}

/* ------------------------------------------------------------------ */
/* update_and_detect                                                    */
/* ------------------------------------------------------------------ */

/**
 * Updates the rolling ground baseline and detects obstacle regions.
 *
 * @param boundary_row    Array of length `width`: for each column, the row
 *                        index of the ground boundary found by the vision
 *                        pipeline, or >= h if no ground was found.
 * @param width           Number of columns (image width)
 * @param h               Image height; boundary_row[x] >= h means no ground
 * @param ground_baseline Float array of length `width` holding the running
 *                        baseline.  Pass NULL on the very first call — the
 *                        array is then initialised from boundary_row and no
 *                        obstacle detection is performed.
 * @param baseline_inited Non-zero once ground_baseline has been initialised
 *                        (set to 1 by this function after the first call)
 * @param min_width       Minimum obstacle region width to report
 * @param regions_out     Caller-supplied output array (size MAX_OBSTACLE_REGIONS)
 * @return                Number of obstacle regions found (0 on first call)
 */
int update_and_detect(const uint16_t *boundary_row,
                      int             width,
                      int             h,
                      float          *ground_baseline,
                      int            *baseline_inited,
                      int             min_width,
                      struct obstacle_region_t *regions_out)
{
    /* ---- First call: initialise the baseline, skip detection ---- */
    if (!(*baseline_inited)) {
        for (int x = 0; x < width; x++) {
            if (boundary_row[x] >= (uint16_t)h) {
                ground_baseline[x] = NO_GROUND_BASELINE;
            } else {
                ground_baseline[x] = (float)boundary_row[x];
            }
        }
        *baseline_inited = 1;
        return 0;
    }

    /* ---- Subsequent calls ---- */

    /* Build obstacle column list */
    uint16_t obstacle_cols[MAX_IMAGE_WIDTH];
    int      n_obstacle_cols = 0;

    /* Boolean scratch arrays (stack, cheap for typical image widths) */
    uint8_t valid[MAX_IMAGE_WIDTH];
    uint8_t no_ground[MAX_IMAGE_WIDTH];
    uint8_t deviation_obstacle[MAX_IMAGE_WIDTH];
    uint8_t no_obstacle_mask[MAX_IMAGE_WIDTH];

    for (int x = 0; x < width; x++) {
        valid[x]    = (boundary_row[x] < (uint16_t)h) ? 1 : 0;
        no_ground[x] = !valid[x];

        float deviation = (float)boundary_row[x] - ground_baseline[x];
        deviation_obstacle[x] = (valid[x] && deviation > OBSTACLE_THRESHOLD) ? 1 : 0;

        /* obstacle if deviation is too large OR if there is no ground */
        if (deviation_obstacle[x] || no_ground[x]) {
            obstacle_cols[n_obstacle_cols++] = (uint16_t)x;
        }

        /* columns that are valid AND not a deviation obstacle get baseline update */
        no_obstacle_mask[x] = (valid[x] && !deviation_obstacle[x]) ? 1 : 0;
    }

    /* ---- Detect obstacle regions ---- */
    int n_regions = get_obstacle_regions(obstacle_cols, n_obstacle_cols,
                                         min_width, 5, regions_out);

    /* ---- Update baseline for non-obstacle columns (EMA) ---- */
    for (int x = 0; x < width; x++) {
        if (no_obstacle_mask[x]) {
            ground_baseline[x] = (1.0f - OBSTACLE_ALPHA) * ground_baseline[x]
                                + OBSTACLE_ALPHA * (float)boundary_row[x];
        }
    }

    /* ---- Forward-fill baseline for obstacle/no-ground columns ---- */
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

/* ------------------------------------------------------------------ */
/* 1-D median filter helper (mirrors scipy.ndimage.median_filter)      */
/* ------------------------------------------------------------------ */

/**
 * Sort `n` floats in-place with insertion sort.
 * Only called with small n (== smooth_kernel, typically 5).
 */
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

/**
 * 1-D median filter with "reflect" padding (scipy default).
 * Only filters positions where valid_mask[i] != 0.
 *
 * @param data        Input float array of length n
 * @param valid_mask  uint8_t array; only indices where this is non-zero are updated
 * @param n           Length of data / valid_mask
 * @param kernel      Odd window size (e.g. 5)
 * @param out         Output float array of length n (may equal data for in-place)
 */
static void median_filter_1d(const float   *data,
                              const uint8_t *valid_mask,
                              int            n,
                              int            kernel,
                              float         *out)
{
    /* copy first so unfiltered positions keep their original value */
    memcpy(out, data, n * sizeof(float));

    int half = kernel / 2;
    float window[32]; /* kernel <= 31 in any sane usage */

    for (int i = 0; i < n; i++) {
        if (!valid_mask[i]) {
            continue;
        }

        /* fill window with reflect-padded neighbours */
        for (int k = -half; k <= half; k++) {
            int idx = i + k;

            /* reflect padding: mirror at both edges */
            if (idx < 0)  { idx = -idx; }
            if (idx >= n) { idx = 2 * (n - 1) - idx; }
            /* clamp in case n is very small */
            if (idx < 0)  { idx = 0; }
            if (idx >= n) { idx = n - 1; }

            window[k + half] = data[idx];
        }

        insertion_sort_f(window, kernel);
        out[i] = window[half]; /* median */
    }
}

/* ------------------------------------------------------------------ */
/* find_ground_boundary                                                 */
/* ------------------------------------------------------------------ */

/**
 * For each row of a rotated ground mask, finds the column index of the
 * ground boundary (the furthest "safe" ground pixel from the drone's
 * perspective), then smooths the result with a 1-D median filter.
 *
 * The mask is IMAGE_GRAYSCALE and is assumed to be stored in the
 * "rotated" orientation used by the Python script:
 *   - mask_rotated.shape[0]  == n_rows  (number of scan lines)
 *   - mask_rotated.shape[1]  == image_height (pixels per scan line)
 * i.e. rows are contiguous in memory: buf[row * image_height + col].
 *
 * @param mask_rotated       Grayscale mask (IMAGE_GRAYSCALE),
 *                           dimensions: n_rows × image_height
 * @param boundary_rows_out  Caller-supplied int array of length n_rows.
 *                           Each entry is set to the boundary column index,
 *                           or image_height if no ground was found in that row.
 * @param min_ground_pixels  Minimum ground pixels needed at the far edge
 * @param max_gap            Maximum allowed gap in a ground run
 * @param smooth_kernel      Odd window size for the final 1-D median filter
 */
void find_ground_boundary(const struct image_t *mask_rotated,
                          int                  *boundary_rows_out,
                          int                   min_ground_pixels,
                          int                   max_gap,
                          int                   smooth_kernel)
{
    int image_height = mask_rotated->w; /* columns per row in rotated frame */
    int n_rows       = mask_rotated->h; /* number of scan rows              */
    const uint8_t *buf = (const uint8_t *)mask_rotated->buf;

    /* Scratch buffer for green pixel positions in one row */
    uint16_t green_pos[MAX_IMAGE_WIDTH];

    /* initialise all boundaries to image_height (= "no ground") */
    for (int i = 0; i < n_rows; i++) {
        boundary_rows_out[i] = image_height;
    }

    /* ---------------------------------------------------------------- */
    /* Per-row boundary finding                                          */
    /* ---------------------------------------------------------------- */
    for (int idx = 0; idx < n_rows; idx++) {
        const uint8_t *row = buf + idx * image_height;

        /* collect positions of green (non-zero) pixels */
        int n_green = 0;
        for (int c = 0; c < image_height; c++) {
            if (row[c] > 0) {
                green_pos[n_green++] = (uint16_t)c;
            }
        }

        if (n_green == 0) {
            continue; /* no ground in this row */
        }

        /* ---- check ground_valid: last min_ground_pixels cols all green ---- */
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
             * If the longest run < max_gap, skip this row entirely.
             */
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < n_green; i++) {
                /* diffs[i-1] = green_pos[i] - green_pos[i-1] */
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

        /* ---- work on the reversed green_pos array (gp in Python) ---- */
        /* We reverse in-place into a second pass using index arithmetic  */
        /* gp[i] = green_pos[n_green - 1 - i]                            */
#define GP(i) green_pos[n_green - 1 - (i)]

        if (n_green == 1) {
            boundary_rows_out[idx] = GP(0);
            continue;
        }

        /*
         * gaps[i] = -diff(gp)[i] - 1
         *         = -(gp[i+1] - gp[i]) - 1
         *         = gp[i] - gp[i+1] - 1    (positive because gp is descending)
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
            /* no big gap → boundary is the smallest green position */
            boundary_rows_out[idx] = GP(n_green - 1);
            continue;
        }

        int fg = first_big_gap;
        int pending_boundary = GP(fg);

        /*
         * after_gap = gp[fg+1 :]   (length = n_green - 1 - fg)
         * after_gap[i] = GP(fg + 1 + i)
         */
        int after_size = n_green - 1 - fg;

        if (after_size >= max_gap) {
            /*
             * Check whether after_gap contains a run of length >= max_gap.
             * after_gaps[i] = after_gap[i] - after_gap[i+1] - 1
             * split on after_gaps > 0 (any gap), then check run lengths.
             */
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < after_size; i++) {
                int g = (int)GP(fg + 1 + i - 1) - (int)GP(fg + 1 + i) - 1;
                if (g == 0) { /* consecutive → same run */
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

    /* ---------------------------------------------------------------- */
    /* 1-D median smoothing over valid boundary positions               */
    /* ---------------------------------------------------------------- */
    if (smooth_kernel >= 3 && smooth_kernel % 2 == 1) {
        /* count valid entries */
        uint8_t valid_mask[MAX_IMAGE_WIDTH];
        int valid_count = 0;
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
                    boundary_rows_out[i] = (int)(smoothed[i] + 0.5f); /* round */
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Output structs                                                        */
/* ------------------------------------------------------------------ */

/* Single obstacle: left column (image coordinates) and width */
struct obstacle_t {
    int left;   ///< leftmost column in original (unflipped) image coordinates
    int width;  ///< width in columns
};

/* Debug / diagnostic info returned alongside obstacles */
struct ground_debug_t {
    int     ground_found;   ///< 1 = "GROUND FOUND", 0 = "NO GROUND"
    float   green_frac;     ///< fraction of pixels classified as ground
    /* boundary_rows and obstacle_regions_raw are owned by the caller --
       pass the same arrays you supply to get_obstacle_info()            */
};

/* ------------------------------------------------------------------ */
/* Flip mask horizontally into a pre-allocated output image             */
/* ------------------------------------------------------------------ */

/**
 * Mirrors a grayscale image left-right (equivalent to mask[:, ::-1]).
 * @param src  Input  IMAGE_GRAYSCALE
 * @param dst  Output IMAGE_GRAYSCALE, same dimensions, pre-created
 */
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

/* ------------------------------------------------------------------ */
/* get_obstacle_info                                                     */
/* ------------------------------------------------------------------ */

/**
 * Full pipeline: ground detection → boundary finding → obstacle detection.
 *
 * @param input              Source IMAGE_YUV422
 * @param ground_baseline    Float array of length input->w (persistent across
 *                           calls). The caller owns this buffer.
 * @param baseline_inited    Pointer to int flag; set to 0 before the very
 *                           first call, then left alone.
 * @param oa_color_count_frac  Green fraction threshold for "GROUND FOUND"
 * @param median_ksize       Odd kernel size for median blur in ground detector
 *                           (pass 0 or 1 to skip)
 * @param min_width          Minimum obstacle region width (columns)
 * @param obstacles_out      Caller-supplied array of struct obstacle_t,
 *                           size MAX_OBSTACLE_REGIONS
 * @param boundary_rows_out  Caller-supplied int array of length input->w,
 *                           filled with the ground boundary per column
 * @param debug_out          Filled with diagnostic info
 * @return                   Number of obstacles found (0 when no ground)
 */
int get_obstacle_info(struct image_t           *input,
                      float                    *ground_baseline,
                      int                      *baseline_inited,
                      float                     oa_color_count_frac,
                      int                       median_ksize,
                      int                       min_width,
                      struct obstacle_t        *obstacles_out,
                      int                      *boundary_rows_out,
                      struct ground_debug_t    *debug_out)
{
    int W = input->w;
    int H = input->h;

    /* ---- Step 1: ground mask ---- */
    struct image_t mask;
    image_create(&mask, W, H, IMAGE_GRAYSCALE);

    float green_frac  = 0.0f;
    int   apply_median = (median_ksize >= 3 && median_ksize % 2 == 1);
    int   ground_found = detect_green_ground_ml(input, &mask,
                                                oa_color_count_frac,
                                                apply_median,
                                                &green_frac);

    if (debug_out != NULL) {
        debug_out->ground_found = ground_found;
        debug_out->green_frac   = green_frac;
    }

    if (!ground_found) {
        /* no ground: leave baseline unchanged, return zero obstacles */
        image_free(&mask);
        return 0;
    }

    /* ---- Step 2: flip mask horizontally (mask[:, ::-1]) ---- */
    struct image_t mask_flipped;
    image_create(&mask_flipped, W, H, IMAGE_GRAYSCALE);
    flip_horizontal(&mask, &mask_flipped);
    image_free(&mask); /* no longer needed */

    /* ---- Step 3: find ground boundary ---- */
    find_ground_boundary(&mask_flipped,
                         boundary_rows_out,
                         /* min_ground_pixels= */ 5,
                         /* max_gap=           */ 10,
                         /* smooth_kernel=     */ 5);
    image_free(&mask_flipped);

    /* ---- Step 4: update baseline and detect obstacle regions ---- */
    struct obstacle_region_t raw_regions[MAX_OBSTACLE_REGIONS];
    int n_regions = update_and_detect(
                        (const uint16_t *)boundary_rows_out,
                        W, H,
                        ground_baseline,
                        baseline_inited,
                        min_width,
                        raw_regions);

    /* ---- Step 5: convert from flipped coords to image coords ----
     *
     * In the flipped mask a column index c corresponds to column
     * (W - 1 - c) in the original image.  An obstacle region [s, e]
     * in flipped coords therefore spans:
     *
     *   original right edge : W - 1 - s
     *   original left  edge : W - 1 - e   ← this is "left" in left→right order
     *   width               : e - s + 1   (unchanged)
     */
    for (int i = 0; i < n_regions; i++) {
        obstacles_out[i].left  = W - 1 - (int)raw_regions[i].end;
        obstacles_out[i].width = (int)raw_regions[i].width;
    }

    return n_regions;
}

/* --- module-level state (persistent across frames) --- */
static float ground_baseline[MAX_IMAGE_WIDTH];
static int   baseline_inited = 0;

/* --- called each frame --- */
void vision_periodic(struct image_t *camera_img)
{
    struct obstacle_t        obstacles[MAX_OBSTACLE_REGIONS];
    int                      boundary_rows[MAX_IMAGE_WIDTH];
    struct ground_debug_t    debug;

    int n = get_obstacle_info(
                camera_img,
                ground_baseline,
                &baseline_inited,
                0.05f,   /* oa_color_count_frac */
                5,       /* median_ksize        */
                20,      /* min_width           */
                obstacles,
                boundary_rows,
                &debug);

    for (int i = 0; i < n; i++) {
        printf("Obstacle %d: left=%d width=%d\n",
               i, obstacles[i].left, obstacles[i].width);
    }
}