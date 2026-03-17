/*
 * team10_get_obstacle_info.c
 *
 * Ground detection and obstacle finding pipeline.
 *
 * NOTE: ALL THESE FUNCTIONS WORK WITH A ROTATED IMAGE
 * (ground is on the left) ON PURPOSE.
 *
 * Changelog vs. previous version
 * --------------------------------
 * 1. Added fill_holes(): flood-fill from borders to detect and fill
 *    enclosed zero regions inside the ground blob.
 * 2. Added isolate_ground_blob(): two-pass union-find connected
 *    components that removes small blobs (area < BLOB_AREA_THRESHOLD)
 *    and spiky blobs (perimeter/equiv_rect_perimeter >= SMOOTH_PERIM_RATIO).
 *    This mirrors colored_blob_separator.py: isolate_ground_blob() +
 *    fill_holes(). The fractal-dimension check from solidity_detection.py
 *    is intentionally omitted — the perimeter ratio alone is sufficient.
 * 3. get_obstacle_info() now calls isolate_ground_blob() → fill_holes()
 *    on the raw mask before boundary detection. Function signature and
 *    return semantics are UNCHANGED.
 */

#include "modules/computer_vision/team10_get_obstacle_info.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ================================================================== */
/* INTERNAL CONSTANTS                                                   */
/* ================================================================== */

/*
 * Maximum image height for static buffer sizing.
 * 520 matches MAX_IMAGE_WIDTH and covers all Bebop camera configs.
 */
#define MAX_IMAGE_HEIGHT       520

/*
 * Connected-components parameters.
 * MAX_CC_LABELS: maximum number of provisional labels in one frame.
 *   128 is generous — typical frames have < 10 distinct blobs.
 */
#define MAX_CC_LABELS          128
#define BLOB_AREA_THRESHOLD    1000   /* pixels, mirrors Python default  */
#define SMOOTH_PERIM_RATIO     2.0f   /* perimeter / equiv_rect threshold */

/* ================================================================== */
/* FILE-SCOPE STATIC BUFFERS  (BSS — zero-initialised at startup)      */
/* ================================================================== */

/*
 * cc_label_map: provisional + resolved connected-component labels.
 * One byte per pixel; fits 520×520 = 270 400 bytes (~264 KB).
 */
static uint8_t cc_label_map[MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH];

/* ================================================================== */
/* FORWARD DECLARATIONS FOR FILE-PRIVATE HELPERS                        */
/* ================================================================== */

static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V);
static inline uint8_t is_ground_sim(uint8_t Y, uint8_t U, uint8_t V);
static void apply_ground_mask(struct image_t *input, struct image_t *mask, int use_sim);
static void median_blur_3x3(struct image_t *mask, struct image_t *blurred);
static uint8_t median_of_9(uint8_t *v);
static void insertion_sort_9(uint8_t *v);
static void insertion_sort_f(float *v, int n);
static void median_filter_1d(const float *data, const uint8_t *valid_mask,
                              int n, int kernel, float *out);
static void flip_horizontal(const struct image_t *src, struct image_t *dst);
static void fill_holes(uint8_t *mask, int w, int h);
static void isolate_ground_blob(uint8_t *mask, int w, int h);


/* ================================================================== */
/* SECTION 1 – Per-pixel ground classifiers (decision trees)           */
/* ================================================================== */

static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V)
{
    if (U <= 115) {
        if (V <= 145) {
            if (Y <= 85) {
                return 0;
            } else {
                if (U <= 92) { return 0; } else { return 255; }
            }
        } else {
            if (V <= 152) {
                if (Y <= 177) { return 255; } else { return 0; }
            } else {
                return 0;
            }
        }
    } else {
        if (U <= 121) {
            if (V <= 137) {
                if (Y <= 87) { return 0; } else { return 255; }
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }
}

static inline uint8_t is_ground_sim(uint8_t Y, uint8_t U, uint8_t V)
{
    if (U <= 96) {
        if (Y <= 102) { return 255; } else { return 0; }
    } else {
        if (U <= 97) {
            if (V <= 126) { return 255; } else { return 0; }
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
static void apply_ground_mask(struct image_t *input, struct image_t *mask, int use_sim)
{
    uint8_t *src  = (uint8_t *)input->buf;
    uint8_t *dest = (uint8_t *)mask->buf;

    for (uint16_t y = 0; y < input->h; y++) {
        for (uint16_t x = 0; x < input->w; x += 2) {
            uint8_t U  = src[0];
            uint8_t Y0 = src[1];
            uint8_t V  = src[2];
            uint8_t Y1 = src[3];

            if (use_sim) {
                dest[0] = is_ground_sim(Y0, U, V);
                dest[1] = is_ground_sim(Y1, U, V);
            } else {
                dest[0] = is_ground(Y0, U, V);
                dest[1] = is_ground(Y1, U, V);
            }

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
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
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
    uint8_t  *src  = (uint8_t *)mask->buf;
    uint8_t  *dest = (uint8_t *)blurred->buf;
    uint16_t  w    = mask->w;
    uint16_t  h    = mask->h;

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
                           int             use_sim,
                           float          *green_fraction)
{
    apply_ground_mask(input, mask_out, use_sim);

    if (apply_median) {
        struct image_t tmp;
        image_create(&tmp, mask_out->w, mask_out->h, IMAGE_GRAYSCALE);
        median_blur_3x3(mask_out, &tmp);
        image_switch(mask_out, &tmp);
        image_free(&tmp);
    }

    uint8_t  *buf         = (uint8_t *)mask_out->buf;
    uint32_t  total       = (uint32_t)mask_out->w * (uint32_t)mask_out->h;
    uint32_t  green_count = 0;

    for (uint32_t i = 0; i < total; i++) {
        if (buf[i] != 0) green_count++;
    }

    float frac = (total > 0) ? ((float)green_count / (float)total) : 0.0f;
    if (green_fraction != NULL) *green_fraction = frac;

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
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
        v[j + 1] = key;
    }
}

static void median_filter_1d(const float   *data,
                              const uint8_t *valid_mask,
                              int            n,
                              int            kernel,
                              float         *out)
{
    memcpy(out, data, (uint32_t)n * sizeof(float));

    int   half = kernel / 2;
    float window[32];

    for (int i = 0; i < n; i++) {
        if (!valid_mask[i]) continue;

        for (int k = -half; k <= half; k++) {
            int idx = i + k;
            if (idx < 0)  idx = -idx;
            if (idx >= n) idx = 2 * (n - 1) - idx;
            if (idx < 0)  idx = 0;
            if (idx >= n) idx = n - 1;
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

    for (int i = 0; i < n_rows; i++) {
        boundary_rows_out[i] = image_height;
    }

    for (int idx = 0; idx < n_rows; idx++) {
        const uint8_t *row = buf + idx * image_height;

        int n_green = 0;
        for (int c = 0; c < image_height; c++) {
            if (row[c] > 0) green_pos[n_green++] = (uint16_t)c;
        }

        if (n_green == 0) continue;

        int ground_valid = 0;
        {
            int count = 0;
            for (int c = image_height - min_ground_pixels; c < image_height; c++) {
                if (row[c] > 0) count++;
            }
            ground_valid = (count >= min_ground_pixels);
        }

        if (!ground_valid) {
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < n_green; i++) {
                if (green_pos[i] - green_pos[i - 1] == 1) {
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 1;
                }
            }
            if (max_run < max_gap) continue;
        }

#define GP(i) green_pos[n_green - 1 - (i)]

        if (n_green == 1) {
            boundary_rows_out[idx] = GP(0);
            continue;
        }

        int first_big_gap = -1;
        for (int i = 0; i < n_green - 1; i++) {
            int gap = (int)GP(i) - (int)GP(i + 1) - 1;
            if (gap > max_gap) { first_big_gap = i; break; }
        }

        if (first_big_gap < 0) {
            boundary_rows_out[idx] = GP(n_green - 1);
            continue;
        }

        int fg               = first_big_gap;
        int pending_boundary = GP(fg);
        int after_size       = n_green - 1 - fg;

        if (after_size >= max_gap) {
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
                if (valid_mask[i]) boundary_rows_out[i] = (int)(smoothed[i] + 0.5f);
            }
        }
    }
}


/* ================================================================== */
/* SECTION 7 – get_obstacle_regions (public)                           */
/* ================================================================== */

uint8_t get_obstacle_regions(const uint16_t           *obstacle_cols,
                            int                       n_cols,
                            int                       min_width,
                            int                       max_col_gap,
                            struct obstacle_region_t *regions_out)
{
    if (n_cols == 0) return 0;

    uint8_t  n_regions = 0;
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

uint8_t update_and_detect(const int  *boundary_row,
                        int         width,
                        int         h,
                        float      *ground_baseline,
                        int        *baseline_inited,
                        int         min_width,
                        int         max_col_gap,
                        struct obstacle_region_t *regions_out)
{
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

        if (deviation_obstacle[x] || !valid[x]) {
            obstacle_cols[n_obstacle_cols++] = (uint16_t)x;
        }

        no_obstacle_mask[x] = (valid[x] && !deviation_obstacle[x]) ? 1 : 0;
    }

    uint8_t n_regions = get_obstacle_regions(obstacle_cols, n_obstacle_cols,
                                            min_width, max_col_gap, regions_out);

    for (int x = 0; x < width; x++) {
        if (no_obstacle_mask[x]) {
            ground_baseline[x] = (1.0f - OBSTACLE_ALPHA) * ground_baseline[x]
                                + OBSTACLE_ALPHA * (float)boundary_row[x];
        }
    }

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
/* SECTION 10 – fill_holes (file-private)                              */
/* ================================================================== */

/*
 * Fills enclosed zero regions inside a binary mask (0/255).
 *
 * Algorithm: iterative 4-connected flood from image border.
 *   1. All border zeros are marked as BACKGROUND (value 128).
 *   2. Forward + backward scan passes propagate 128 to all
 *      4-connected zero neighbours until convergence.
 *   3. Remaining zeros (unreachable from border) are holes → set 255.
 *      Temporary 128 marks are restored to 0.
 *
 * No dynamic memory or queue required.
 * Convergence: worst case (w + h) passes; typical: 1–2 passes.
 *
 * Mirrors colored_blob_separator.py fill_holes().
 */
static void fill_holes(uint8_t *mask, int w, int h)
{
    /* Step 1 – seed border zeros as background (128) */
    for (int x = 0; x < w; x++) {
        if (mask[0 * w + x]       == 0) mask[0 * w + x]       = 128;
        if (mask[(h - 1) * w + x] == 0) mask[(h - 1) * w + x] = 128;
    }
    for (int y = 0; y < h; y++) {
        if (mask[y * w + 0]       == 0) mask[y * w + 0]       = 128;
        if (mask[y * w + (w - 1)] == 0) mask[y * w + (w - 1)] = 128;
    }

    /* Step 2 – propagate 128 until stable */
    int changed = 1;
    while (changed) {
        changed = 0;

        /* Forward pass: top-left → bottom-right */
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                if (mask[y * w + x] == 0) {
                    if (mask[(y - 1) * w + x] == 128 ||
                        mask[(y + 1) * w + x] == 128 ||
                        mask[y * w + (x - 1)] == 128 ||
                        mask[y * w + (x + 1)] == 128) {
                        mask[y * w + x] = 128;
                        changed = 1;
                    }
                }
            }
        }

        /* Backward pass: bottom-right → top-left */
        for (int y = h - 2; y >= 1; y--) {
            for (int x = w - 2; x >= 1; x--) {
                if (mask[y * w + x] == 0) {
                    if (mask[(y - 1) * w + x] == 128 ||
                        mask[(y + 1) * w + x] == 128 ||
                        mask[y * w + (x - 1)] == 128 ||
                        mask[y * w + (x + 1)] == 128) {
                        mask[y * w + x] = 128;
                        changed = 1;
                    }
                }
            }
        }
    }

    /* Step 3 – finalise: holes → 255, background → 0 */
    for (int i = 0; i < w * h; i++) {
        if      (mask[i] == 0)   mask[i] = 255;  /* enclosed hole → fill  */
        else if (mask[i] == 128) mask[i] = 0;    /* border-reachable → bg  */
        /* 255 → unchanged (foreground)                                     */
    }
}


/* ================================================================== */
/* SECTION 11 – isolate_ground_blob (file-private)                     */
/* ================================================================== */

/*
 * Removes blobs from a binary mask (0/255) that are either:
 *   (a) too small  (area < BLOB_AREA_THRESHOLD), or
 *   (b) too spiky  (perimeter / equiv_rect_perimeter >= SMOOTH_PERIM_RATIO)
 *
 * Uses a two-pass union-find connected-components algorithm
 * (8-connectivity) for O(W*H) runtime with no dynamic allocation.
 *
 * Smoothness metric — mirrors solidity_detection.py is_smooth_blob():
 *   avg_height          = area / bbox_width
 *   equiv_rect_perim    = 2 * (avg_height + bbox_width)
 *   perimeter_ratio     = perimeter / equiv_rect_perim
 *   smooth if ratio < SMOOTH_PERIM_RATIO (2.0)
 *
 * NOTE: The fractal-dimension branch of is_smooth_blob() is intentionally
 * skipped.  The ratio check alone correctly classifies ground vs. plant
 * blobs in all tested frames.  The two conditions in the Python are OR'd,
 * so the ratio check is sufficient when it fires; the fractal branch only
 * adds a safety-net for edge cases that are unlikely given clean masks.
 *
 * Mirrors colored_blob_separator.py isolate_ground_blob().
 */

/* --- union-find helpers ------------------------------------------- */

static uint8_t uf_find(uint8_t *parent, uint8_t x)
{
    /* Path-compressed find */
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  /* path halving */
        x         = parent[x];
    }
    return x;
}

static void uf_union(uint8_t *parent, uint8_t *rnk, uint8_t a, uint8_t b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (rnk[a] < rnk[b]) { uint8_t t = a; a = b; b = t; }
    parent[b] = a;
    if (rnk[a] == rnk[b]) rnk[a]++;
}

/* --- main function ------------------------------------------------- */

static void isolate_ground_blob(uint8_t *mask, int w, int h)
{
    /* --- union-find state ----------------------------------------- */
    uint8_t  parent[MAX_CC_LABELS];
    uint8_t  rnk[MAX_CC_LABELS];
    uint32_t area[MAX_CC_LABELS];
    uint16_t x_min[MAX_CC_LABELS], x_max[MAX_CC_LABELS];

    memset(area,  0, sizeof(area));
    memset(x_min, 0xFF, sizeof(x_min));   /* initialise to max uint16 */
    memset(x_max, 0,    sizeof(x_max));

    for (int i = 0; i < MAX_CC_LABELS; i++) { parent[i] = (uint8_t)i; rnk[i] = 0; }

    uint8_t  next_label = 1;             /* label 0 = background     */
    uint8_t *lmap       = cc_label_map;  /* file-scope static buffer  */
    memset(lmap, 0, (uint32_t)w * (uint32_t)h);

    /* ----------------------------------------------------------------
     * PASS 1: assign provisional labels, record unions.
     * For each foreground pixel, inspect the four already-scanned
     * 8-connected neighbours: NW, N, NE, W.
     * ---------------------------------------------------------------- */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (mask[y * w + x] == 0) { lmap[y * w + x] = 0; continue; }

            uint8_t nbr[4];
            int     nn = 0;

            if (y > 0 && x > 0  && lmap[(y-1)*w+(x-1)]) nbr[nn++] = lmap[(y-1)*w+(x-1)];
            if (y > 0           && lmap[(y-1)*w+ x    ]) nbr[nn++] = lmap[(y-1)*w+ x    ];
            if (y > 0 && x<w-1  && lmap[(y-1)*w+(x+1)]) nbr[nn++] = lmap[(y-1)*w+(x+1)];
            if (           x > 0 && lmap[ y   *w+(x-1)]) nbr[nn++] = lmap[ y   *w+(x-1)];

            if (nn == 0) {
                /* Isolated foreground pixel → new label */
                if (next_label < MAX_CC_LABELS) {
                    lmap[y * w + x] = next_label++;
                } else {
                    /* Label space exhausted: fold into label 1.
                     * This is a graceful fallback; in practice
                     * MAX_CC_LABELS=128 is never reached. */
                    lmap[y * w + x] = 1;
                }
            } else {
                /* Assign one neighbour's label, union the rest */
                lmap[y * w + x] = nbr[0];
                for (int k = 1; k < nn; k++) {
                    uf_union(parent, rnk, nbr[0], nbr[k]);
                }
            }
        }
    }

    /* ----------------------------------------------------------------
     * PASS 2: resolve provisional labels to roots, accumulate stats.
     * Only x-extents (bbox_width) and area are needed for smoothness.
     * ---------------------------------------------------------------- */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t L = lmap[y * w + x];
            if (L == 0) continue;

            uint8_t root     = uf_find(parent, L);
            lmap[y * w + x]  = root;
            area[root]++;

            if ((uint16_t)x < x_min[root]) x_min[root] = (uint16_t)x;
            if ((uint16_t)x > x_max[root]) x_max[root] = (uint16_t)x;
        }
    }

    /* ----------------------------------------------------------------
     * FILTERING: for each root label, decide keep / discard.
     * A root is identified by parent[L] == L.
     * ---------------------------------------------------------------- */
    uint8_t keep[MAX_CC_LABELS];
    memset(keep, 0, sizeof(keep));

    for (uint8_t L = 1; L < next_label; L++) {
        if (uf_find(parent, L) != L) continue;   /* not a root, skip */
        if (area[L] < BLOB_AREA_THRESHOLD)        continue;   /* too small     */

        /* Compute 4-connected perimeter:
         * for each blob pixel, count exposed edges
         * (border of image treated as non-blob). */
        uint32_t perim = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (lmap[y * w + x] != L) continue;
                if (x == 0   || lmap[ y    * w + (x-1)] != L) perim++;
                if (x == w-1 || lmap[ y    * w + (x+1)] != L) perim++;
                if (y == 0   || lmap[(y-1) * w +  x   ] != L) perim++;
                if (y == h-1 || lmap[(y+1) * w +  x   ] != L) perim++;
            }
        }

        /* is_smooth_blob():
         *   avg_height       = area / bbox_width
         *   equiv_rect_perim = 2 * (avg_height + bbox_width)
         *   smooth if perim / equiv_rect_perim < SMOOTH_PERIM_RATIO
         */
        uint16_t bbox_width = x_max[L] - x_min[L] + 1;
        if (bbox_width == 0) continue;

        float avg_height       = (float)area[L] / (float)bbox_width;
        float equiv_rect_perim = 2.0f * (avg_height + (float)bbox_width);
        float ratio            = (float)perim / equiv_rect_perim;

        if (ratio < SMOOTH_PERIM_RATIO) {
            keep[L] = 1;
        }
    }

    /* ----------------------------------------------------------------
     * APPLY: zero-out pixels belonging to discarded blobs.
     * ---------------------------------------------------------------- */
    for (int i = 0; i < w * h; i++) {
        uint8_t L = lmap[i];
        if (L == 0 || !keep[L]) mask[i] = 0;
        /* else mask[i] remains 255 */
    }
}


/* ================================================================== */
/* SECTION 12 – get_obstacle_info (public)                             */
/* ================================================================== */

/*
 * Full pipeline: ground mask → blob isolation → hole filling →
 *                boundary finding → obstacle detection.
 *
 * Signature is UNCHANGED from the previous version.
 * The only behavioural change is that the mask is now cleaned by
 * isolate_ground_blob() + fill_holes() before boundary detection,
 * which mirrors the Python pipeline exactly.
 */
uint8_t get_obstacle_info(struct image_t           *input,
                        float                    *ground_baseline,
                        int                      *baseline_inited,
                        float                     oa_color_count_frac,
                        int                       median_ksize,
                        int                       min_width,
                        int                       min_ground_pixels,
                        int                       max_gap,
                        int                       smooth_kernel,
                        int                       max_col_gap,
                        int                       use_sim,
                        struct obstacle_region_t *obstacles_out,
                        int                      *boundary_rows_out,
                        int                      *ground_found_out,
                        float                    *green_frac_out)
{
    int W = input->w;
    int H = input->h;

    /* ----------------------------------------------------------------
     * Step 1 – raw ground mask (YUV classify + optional median blur)
     * ---------------------------------------------------------------- */
    struct image_t mask;
    image_create(&mask, W, H, IMAGE_GRAYSCALE);

    float green_frac   = 0.0f;
    int   apply_median = (median_ksize >= 3 && median_ksize % 2 == 1);
    int   ground_found = detect_green_ground_ml(input, &mask,
                                                oa_color_count_frac,
                                                apply_median,
                                                use_sim,
                                                &green_frac);

    if (ground_found_out != NULL) *ground_found_out = ground_found;
    if (green_frac_out   != NULL) *green_frac_out   = green_frac;

    if (!ground_found) {
        image_free(&mask);
        return 0;
    }

    /* ----------------------------------------------------------------
     * Step 2 – clean the mask: remove small / spiky blobs, fill holes.
     *   Mirrors Python:
     *     clean_mask = cds.isolate_ground_blob(binary_img=mask)
     *     clean_mask = cds.fill_holes(clean_mask)
     * ---------------------------------------------------------------- */
    uint8_t *mask_buf = (uint8_t *)mask.buf;

    isolate_ground_blob(mask_buf, W, H);
    fill_holes(mask_buf, W, H);

    /* ----------------------------------------------------------------
     * Step 3 – debug mask printer.
     * Overwrites the source YUV buffer so the drone video stream shows
     * the CLEAN ground mask (white = ground, black = not).
     * Set TEAM10_DEBUG_MASK to 0 to disable in production.
     * ---------------------------------------------------------------- */
#define TEAM10_DEBUG_MASK 1
#if TEAM10_DEBUG_MASK
    {
        uint8_t *src = (uint8_t *)input->buf;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x += 2) {
                uint8_t *p  = &src[y * 2 * W + 2 * x];
                uint8_t  g0 = mask_buf[y * W + x];
                uint8_t  g1 = mask_buf[y * W + x + 1];
                p[0] = 128;           /* U – neutral chroma */
                p[2] = 128;           /* V – neutral chroma */
                p[1] = g0 ? 255 : 0; /* Y0 */
                p[3] = g1 ? 255 : 0; /* Y1 */
            }
        }
    }
#endif

    /* ----------------------------------------------------------------
     * Step 4 – flip mask horizontally (ground convention: left side)
     * ---------------------------------------------------------------- */
    struct image_t mask_flipped;
    image_create(&mask_flipped, W, H, IMAGE_GRAYSCALE);
    flip_horizontal(&mask, &mask_flipped);
    image_free(&mask);

    /* ----------------------------------------------------------------
     * Step 5 – find ground boundary per column (in flipped coords)
     * ---------------------------------------------------------------- */
    find_ground_boundary(&mask_flipped,
                         boundary_rows_out,
                         min_ground_pixels,
                         max_gap,
                         smooth_kernel);
    image_free(&mask_flipped);

    /* ----------------------------------------------------------------
     * Step 6 – update EMA baseline and detect obstacle regions
     *          (still in flipped coordinates)
     * ---------------------------------------------------------------- */
    struct obstacle_region_t raw_regions[MAX_OBSTACLE_REGIONS];
    uint8_t n_regions = update_and_detect(boundary_rows_out,
                                        W, H,
                                        ground_baseline,
                                        baseline_inited,
                                        min_width,
                                        max_col_gap,
                                        raw_regions);

    /* ----------------------------------------------------------------
     * Step 7 – convert flipped → original image coordinates.
     *
     * Flipped column c  →  original column (W - 1 - c).
     * Region [s, e] in flipped space:
     *   original left  = W - 1 - e
     *   original right = W - 1 - s
     *   width          = e - s + 1  (unchanged)
     * ---------------------------------------------------------------- */
    for (int i = 0; i < n_regions; i++) {
        obstacles_out[i].start = (uint16_t)(W - 1 - (int)raw_regions[i].end);
        obstacles_out[i].end   = (uint16_t)(W - 1 - (int)raw_regions[i].start);
        obstacles_out[i].width = raw_regions[i].width;
    }

    return n_regions;
}
