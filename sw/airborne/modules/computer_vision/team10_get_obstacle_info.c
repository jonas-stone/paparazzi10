/*
 * team10_get_obstacle_info.c
 *
 * This file contains the full obstacle and plant detection pipeline.
 * It takes a YUV422 camera image and outputs:
 *   - A list of obstacle regions (column ranges where something blocks the path)
 *   - A list of plant regions (potted plants, detected separately from floor obstacles)
 *   - Gate detection (two blue parallel pillars with a checker pattern)
 *
 * High-level pipeline (obstacles):
 *   1. Classify each pixel as ground or not-ground using a decision tree
 *   2. Remove noise from the binary mask with a median blur
 *   3. Keep only the largest smooth blobs (discard walls, thin noise, etc.)
 *   4. Fill holes inside the ground blob
 *   5. Find the upper boundary of the ground blob per column
 *   6. Compare the boundary to a running baseline; columns that deviate too much
 *      are flagged as containing an obstacle
 *
 * Plants are detected separately: a looser green threshold finds anything greenish,
 * then the confirmed ground mask is subtracted, leaving only above-ground green
 * objects (potted plants).
 *
 * Gate detection runs independently on the same frame using a blue-colour mask
 * and geometric checks.
 *
 * NOTE: The camera is mounted 90° CW, so "columns" in this file correspond to
 * rows in the drone's actual field of view. Left in the image = floor.
 */

#include "team10_get_obstacle_info.h"
#include "modules/computer_vision/lib/vision/image.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <team10_rtp_utilities.h>

/* ══════════════════════════════════════════════════════════════════════════════
 *  STATIC WORK BUFFERS
 *
 *  These are module-level static arrays used as scratch space throughout the
 *  pipeline. Declaring them static (not stack) avoids blowing the stack on an
 *  embedded processor — 520×520 byte arrays are too large for typical stack
 *  limits.
 * ══════════════════════════════════════════════════════════════════════════════ */
#define MAX_PLANT_PIXELS (MAX_PLANT_WIDTH * MAX_PLANT_HEIGHT)

static uint8_t  work_mask[MAX_PIXELS];      /* primary ground binary mask           */
static uint8_t  work_mask2[MAX_PIXELS];     /* secondary mask for blurring          */
static int16_t  work_labels[MAX_PIXELS];    /* connected-component label per pixel  */
static uint8_t  work_visited[MAX_PIXELS];   /* BFS visited flags / padded blob mask */
static uint8_t  work_flipped[MAX_PIXELS];   /* horizontally flipped mask / edge img */
static int32_t  ff_queue[MAX_PIXELS];       /* BFS queue for flood-fill operations  */
static int16_t  uf_parent[MAX_CC_LABELS];   /* union-find parent array for CC       */
static int32_t  cc_area[MAX_CC_LABELS];     /* pixel count per connected component  */
static int16_t  cc_bb_x0[MAX_CC_LABELS];   /* bounding box: left column per label  */
static int16_t  cc_bb_x1[MAX_CC_LABELS];   /* bounding box: right column per label */
static int16_t  cc_bb_y0[MAX_CC_LABELS];   /* bounding box: top row per label      */
static int16_t  cc_bb_y1[MAX_CC_LABELS];   /* bounding box: bottom row per label   */

/* Scratch buffers exclusively for plant detection (kept small via downscaling) */
static uint8_t  plant_green_small[MAX_PLANT_PIXELS];  /* raw lax-green mask at small scale */
static uint8_t  plant_blur_buf[MAX_PLANT_PIXELS];     /* blurred version of the above      */
static uint8_t  plant_clean_small[MAX_PLANT_PIXELS];  /* ground mask downscaled to match   */
static uint8_t  plant_mask_small[MAX_PLANT_PIXELS];   /* plant-only mask at small scale    */
static uint8_t  plant_mask_full[MAX_PIXELS];          /* plant mask upscaled to full res   */

/* ══════════════════════════════════════════════════════════════════════════════
 *  YUV422 (UYVY) PIXEL ACCESS HELPERS
 *
 *  The camera outputs frames in YUV422 UYVY format:
 *    byte layout per macro-pixel (2 pixels): [U, Y0, V, Y1]
 *
 *  Y (luma) is unique per pixel, but U and V (chroma) are shared between
 *  every pair of adjacent pixels. That's why U and V must be read from the
 *  even-column address (x & ~1).
 *
 *  These helpers abstract that byte-packing so the rest of the code can just
 *  call yuv422_Y(buf, w, x, y) without worrying about the layout.
 * ══════════════════════════════════════════════════════════════════════════════ */
static inline uint8_t yuv422_Y(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + x * 2 + 1]; }
static inline uint8_t yuv422_U(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + (x & ~1) * 2]; }
static inline uint8_t yuv422_V(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + (x & ~1) * 2 + 2]; }

/* ══════════════════════════════════════════════════════════════════════════════
 *  1. GROUND PIXEL CLASSIFIER (DECISION TREE)
 *
 *  Returns 255 if a YUV pixel looks like ground, 0 if it does not.
 *
 *  Two variants are compiled in:
 *    GROUND_TREE_SIM  — tuned for the Gazebo simulator environment
 *    GROUND_TREE_REAL — tuned for the real cyberzoo grass floor
 *
 *  Both are small hand-crafted decision trees fitted to labelled training
 *  images. The thresholds on U, V, and Y were found offline and are
 *  hard-coded here for speed (no floating-point, no lookup table).
 * ══════════════════════════════════════════════════════════════════════════════ */
// #define GROUND_TREE_REAL
#define GROUND_TREE_SIM

uint8_t is_ground_pixel(uint8_t Y, uint8_t U, uint8_t V)
{
#ifdef GROUND_TREE_SIM
    /* Simulator ground is a fairly uniform light colour.
       The tree is very shallow because the sim colours are clean. */
    if (U <= 96) {
        return (Y <= 102) ? 255 : 0;
    } else if (U <= 97) {
        return (V <= 126) ? 255 : 0;
    } else {
        return 0;
    }
#else
    /* Real-world ground tree — handles more lighting variation so the tree
       is slightly deeper, using all three channels. */
    if (U <= 115) {
        if (V <= 145) {
            if (Y <= 85) return 0;
            else         return (U <= 92) ? 0 : 255;
        } else {
            if (V <= 152) return (Y <= 177) ? 255 : 0;
            else          return 0;
        }
    } else {
        if (U <= 121) {
            if (V <= 137) return (Y <= 87) ? 0 : 255;
            else          return 0;
        } else {
            return 0;
        }
    }
#endif
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  2. BINARY MEDIAN BLUR (noise reduction)
 *
 *  A standard 2D median filter applied to a binary (0/255) mask.
 *  For each output pixel we count how many of the ksize×ksize neighbours
 *  are set (255). If more than half are set, the output is 255, otherwise 0.
 *
 *  This removes isolated salt-and-pepper noise from the ground mask while
 *  keeping the main ground region intact.
 * ══════════════════════════════════════════════════════════════════════════════ */
static void median_blur_binary(const uint8_t *src, uint8_t *dst, int w, int h, int ksize)
{
    int half = ksize / 2, threshold = (ksize * ksize) / 2;
    for (int y = 0; y < h; y++) {
        int y0 = (y - half < 0) ? 0 : y - half;
        int y1 = (y + half >= h) ? h - 1 : y + half;
        for (int x = 0; x < w; x++) {
            int count = 0;
            int x0 = (x - half < 0) ? 0 : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;
            for (int ky = y0; ky <= y1; ky++)
                for (int kx = x0; kx <= x1; kx++)
                    if (src[ky * w + kx]) count++;
            dst[y * w + x] = (count > threshold) ? 255 : 0;
        }
    }
}

/**
 * @brief Build a binary ground mask from a YUV422 image and estimate how much
 *        of the frame is ground.
 *
 * Only the LEFT half of the image columns is analysed (x < w/2). This is
 * intentional: the camera is mounted sideways and the floor is always on the
 * left side of the raw frame. Classifying the right half would just be
 * wasted work and could introduce false positives from the sky.
 *
 * After classification, a median blur (ksize × ksize majority vote) is applied
 * to reduce salt-and-pepper noise before writing the final mask.
 *
 * @param img             Input YUV422 image
 * @param mask_out        OUTPUT: binary mask, 255 = ground, 0 = not-ground
 * @param median_ksize    Blur kernel size (must be odd and >= 3; pass 0 to skip)
 * @param green_frac_out  OUTPUT: fraction of total pixels classified as ground [0, 1]
 */
void detect_green_ground_ml(struct image_t *img, uint8_t mask_out[],
                            int median_ksize, float *green_frac_out)
{
    int w = img->w, h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;

    // Only classify pixels in the left half of the frame (the floor side)
    int max_x = w / 2;
    memset(work_mask2, 0, w * h);   // start with everything classified as not-ground
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < max_x; x++) {
            work_mask2[y * w + x] = is_ground_pixel(
                yuv422_Y(buf, w, x, y), yuv422_U(buf, w, x, y), yuv422_V(buf, w, x, y));
        }
    }
    // Apply a majority-vote blur if the kernel size is valid (odd and >= 3)
    if (median_ksize >= 3 && (median_ksize & 1))
        median_blur_binary(work_mask2, mask_out, w, h, median_ksize);
    else
        memcpy(mask_out, work_mask2, w * h);

    // Count ground pixels to compute the green fraction
    int gc = 0, total = w * h;
    for (int i = 0; i < total; i++) if (mask_out[i]) gc++;
    if (green_frac_out) *green_frac_out = total > 0 ? (float)gc / total : 0.0f;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  3. CONNECTED COMPONENTS (union-find with bounding boxes)
 *
 *  Labels every white pixel in the mask with an integer region ID.
 *  Pixels that touch each other (8-connected, including diagonals) get the
 *  same ID. This lets us count, size, and locate distinct blobs.
 *
 *  Implementation uses the union-find ("disjoint set union") data structure
 *  for efficiency — we only need two passes over the image:
 *    Pass 1 (top-left to bottom-right): assign provisional labels and record
 *            which labels should be merged (they belong to the same blob).
 *    Pass 2: "flatten" all labels so that every pixel in a blob has the same
 *            canonical root label.
 *
 *  After the two passes, bounding boxes and pixel counts are computed for
 *  every label so we can filter blobs by size and shape.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Union-find helpers ─────────────────────────────────────────────────────── */
static void uf_init(int n) { for (int i = 0; i < n; i++) uf_parent[i] = (int16_t)i; }

/* Path-compressed find: walk up the parent chain, shortcutting by two at each
   step so the tree stays flat and future finds are faster. */
static int16_t uf_find(int16_t x) {
    while (uf_parent[x] != x) { uf_parent[x] = uf_parent[uf_parent[x]]; x = uf_parent[x]; } return x;
}

/* Merge two labels into one: always keep the lower-numbered root as the
   canonical label so renaming is consistent. */
static void uf_union(int16_t a, int16_t b) {
    a = uf_find(a); b = uf_find(b);
    if (a != b) { if (a < b) uf_parent[b] = a; else uf_parent[a] = b; }
}

/**
 * @brief Label connected regions in a binary mask.
 *
 * @param mask    Binary input mask (255 = foreground, 0 = background)
 * @param labels  OUTPUT: label array, same size as mask. 0 = background.
 * @param w       Image width
 * @param h       Image height
 * @return        Number of distinct regions found (highest label value used)
 */
static int connected_components(const uint8_t *mask, int16_t *labels, int w, int h)
{
    int16_t next_label = 1;
    uf_init(MAX_CC_LABELS);
    memset(labels, 0, w * h * sizeof(int16_t));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (!mask[idx]) continue;  // background pixel, skip

            // Check the four 8-connected neighbours that have already been visited
            // (upper-left, above, upper-right, left)
            int16_t nb[4]; int nn = 0;
            if (y > 0 && x > 0     && labels[(y-1)*w+(x-1)]) nb[nn++] = labels[(y-1)*w+(x-1)];
            if (y > 0               && labels[(y-1)*w+x])     nb[nn++] = labels[(y-1)*w+x];
            if (y > 0 && x < w - 1 && labels[(y-1)*w+(x+1)]) nb[nn++] = labels[(y-1)*w+(x+1)];
            if (x > 0               && labels[y*w+(x-1)])     nb[nn++] = labels[y*w+(x-1)];

            if (nn == 0) {
                // No labelled neighbours — start a new region
                if (next_label < MAX_CC_LABELS) labels[idx] = next_label++;
            } else {
                // One or more labelled neighbours — merge them all into the lowest one
                int16_t m = uf_find(nb[0]);
                for (int i = 1; i < nn; i++) { int16_t r = uf_find(nb[i]); if (r < m) m = r; }
                labels[idx] = m;
                for (int i = 0; i < nn; i++) uf_union(m, nb[i]);
            }
        }
    }

    /* Flatten: replace every provisional label with its canonical root label */
    for (int i = 0; i < w * h; i++) if (labels[i]) labels[i] = uf_find(labels[i]);

    /* Compute per-label pixel count and bounding box for later filtering */
    int max_label = 0;
    for (int i = 0; i < w * h; i++) if (labels[i] > max_label) max_label = labels[i];
    if (max_label >= MAX_CC_LABELS) max_label = MAX_CC_LABELS - 1;

    for (int l = 0; l <= max_label; l++) {
        cc_area[l] = 0;
        cc_bb_x0[l] = (int16_t)(w - 1); cc_bb_x1[l] = 0;
        cc_bb_y0[l] = (int16_t)(h - 1); cc_bb_y1[l] = 0;
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int16_t l = labels[y * w + x];
            if (l > 0 && l <= max_label) {
                cc_area[l]++;
                if (x < cc_bb_x0[l]) cc_bb_x0[l] = (int16_t)x;
                if (x > cc_bb_x1[l]) cc_bb_x1[l] = (int16_t)x;
                if (y < cc_bb_y0[l]) cc_bb_y0[l] = (int16_t)y;
                if (y > cc_bb_y1[l]) cc_bb_y1[l] = (int16_t)y;
            }
        }
    }

    return (int)(next_label - 1);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  BLOB PERIMETER CALCULATOR
 *
 *  Counts how many edges the blob has. An edge is any horizontal or vertical
 *  transition between a blob pixel and a non-blob pixel (including the implicit
 *  zero border around the image).
 *
 *  A smooth, compact blob (like the ground) has a small perimeter for its
 *  size. A spiky or irregular blob (like a tree or plant) has a much larger
 *  perimeter relative to its area. We use this ratio to reject non-ground
 *  blobs before they cause false positives.
 * ══════════════════════════════════════════════════════════════════════════════ */
int compute_blob_perimeter(const int16_t *labels, int w, int h, int16_t lbl)
{
    int perim = 0;

    /* Count horizontal transitions row by row, including the imaginary borders */
    for (int y = 0; y < h; y++) {
        if (labels[y * w] == lbl) perim++;               /* left image border */
        for (int x = 0; x < w - 1; x++) {
            int a = (labels[y * w + x] == lbl) ? 1 : 0;
            int b = (labels[y * w + x + 1] == lbl) ? 1 : 0;
            if (a != b) perim++;                          /* interior transition */
        }
        if (labels[y * w + w - 1] == lbl) perim++;       /* right image border */
    }

    /* Count vertical transitions column by column */
    for (int x = 0; x < w; x++) {
        if (labels[x] == lbl) perim++;                   /* top image border */
        for (int y = 0; y < h - 1; y++) {
            int a = (labels[y * w + x] == lbl) ? 1 : 0;
            int b = (labels[(y + 1) * w + x] == lbl) ? 1 : 0;
            if (a != b) perim++;                          /* interior transition */
        }
        if (labels[(h - 1) * w + x] == lbl) perim++;     /* bottom image border */
    }

    return perim;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  FRACTAL DIMENSION (box-counting)
 *
 *  NOTE: This function is defined but its call inside is_smooth_blob() is
 *  currently commented out. It was disabled before the competition because the
 *  computation is too expensive for real-time use on the drone's processor.
 *  It is left here for reference in case a lighter implementation is needed.
 *
 *  The fractal dimension of a blob's edge measures how "rough" or "jagged" it
 *  is. A smooth circle has fractal dimension ~1.0. A very jagged outline
 *  approaches 2.0. Ground blobs should be smooth; plants are jagged.
 *
 *  Method: box-counting at scales 2, 4, 8, 16... Count how many boxes of each
 *  size contain at least one edge pixel. Fractal dimension ≈ –slope of
 *  log(count) vs log(box_size).
 * ══════════════════════════════════════════════════════════════════════════════ */
float compute_fractal_dimension(const int16_t *labels, int w, int h, int16_t lbl)
{
    int pw = w + 2, ph = h + 2;

    /* Step 1: create a zero-padded version of just this blob in work_visited */
    memset(work_visited, 0, pw * ph);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (labels[y * w + x] == lbl)
                work_visited[(y + 1) * pw + (x + 1)] = 1;

    /* Step 2: compute the edge image — a pixel is an "edge" if it differs from
       its right neighbour OR its bottom neighbour */
    memset(work_flipped, 0, pw * ph);
    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            int cur = work_visited[y * pw + x];
            int e1 = 0, e2 = 0;
            if (x < pw - 1) e1 = (work_visited[y * pw + x + 1] != cur) ? 1 : 0;
            if (y < ph - 1) e2 = (work_visited[(y + 1) * pw + x] != cur) ? 1 : 0;
            work_flipped[y * pw + x] = (e1 || e2) ? 1 : 0;
        }
    }

    /* Step 3: box-count at powers of 2 */
    float log_eps[16], log_boxes[16];
    int n_scales = 0;
    int min_dim = (pw < ph) ? pw : ph;

    for (int p = 1; (1 << p) < min_dim && n_scales < 16; p++) {
        int s = 1 << p;
        int trimmed_h = (ph / s) * s;
        int trimmed_w = (pw / s) * s;
        int count = 0;

        for (int by = 0; by < trimmed_h; by += s) {
            for (int bx = 0; bx < trimmed_w; bx += s) {
                int found = 0;
                for (int dy = 0; dy < s && !found; dy++)
                    for (int dx = 0; dx < s && !found; dx++)
                        if (work_flipped[(by + dy) * pw + (bx + dx)])
                            found = 1;
                if (found) count++;
            }
        }

        if (count > 0) {
            log_eps[n_scales] = logf((float)s);
            log_boxes[n_scales] = logf((float)count);
            n_scales++;
        }
    }

    /* Step 4: linear regression to find the slope */
    if (n_scales < 2) return 1.0f;   /* not enough data — assume smooth */

    float sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (int i = 0; i < n_scales; i++) {
        sx  += log_eps[i];
        sy  += log_boxes[i];
        sxy += log_eps[i] * log_boxes[i];
        sxx += log_eps[i] * log_eps[i];
    }
    float denom = n_scales * sxx - sx * sx;
    if (fabsf(denom) < 1e-9f) return 1.0f;

    float slope = (n_scales * sxy - sx * sy) / denom;
    return -slope;   /* slope is negative; fractal dimension is its negation */
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  IS_SMOOTH_BLOB — decide if a blob looks like ground or like an obstacle/plant
 *
 *  A "smooth" blob (ground) has a perimeter that is not much larger than the
 *  perimeter of a plain rectangle with the same average dimensions. We compare:
 *
 *    perimeter_ratio = actual_perimeter / equivalent_rectangle_perimeter
 *
 *  If this ratio is below SMOOTH_PERIMETER_RATIO_THRESH (2.0), the blob is
 *  smooth and we keep it as ground. If it is above the threshold, the blob
 *  is jagged (plant or obstacle) and we discard it from the ground mask.
 *
 *  The fractal dimension check is commented out — it was too slow for the
 *  drone's processor during the competition and caused a crash when left in.
 *
 *  Returns 1 = smooth (keep as ground), 0 = spiky (discard).
 * ══════════════════════════════════════════════════════════════════════════════ */
int is_smooth_blob(const int16_t *labels, int w, int h,
                   int16_t lbl, int blob_area, int bb_height)
{
    /* bb_height is the bounding-box height of the blob in pixels.
       (Called "width" in the Python version because the camera is sideways.) */
    if (bb_height <= 0) return 1;   /* degenerate case — keep it */

    int perim = compute_blob_perimeter(labels, w, h, lbl);

    /* Estimate average blob "thickness" from area and bounding-box height */
    int avg_height = blob_area / bb_height;

    /* What would the perimeter be if this blob were a perfect rectangle? */
    int equiv_rect_perim = 2 * (avg_height + bb_height);

    float perim_ratio = (equiv_rect_perim > 0) ?
        (float)perim / (float)equiv_rect_perim : 0.0f;

    if (perim_ratio < SMOOTH_PERIMETER_RATIO_THRESH)
        return 1;   /* smooth enough to be ground */

    // NOTE: Fractal dimension check was disabled before the competition.
    // It was computationally too expensive and caused a real-time failure.
    // Left here in case it is re-enabled with a lighter implementation:
    // float fd = compute_fractal_dimension(labels, w, h, lbl);
    // if (fd < SMOOTH_FRACTAL_DIM_THRESH) return 1;

    return 0;   /* perimeter is too ragged — this is not ground */
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ISOLATE GROUND BLOB
 *
 *  After the initial green-pixel classification, the binary mask may contain
 *  multiple disconnected blobs: the actual floor, some reflections, parts of
 *  obstacles that happen to look greenish, etc. This function keeps only the
 *  blobs that:
 *    1. Are large enough (>= blob_area_threshold pixels), and
 *    2. Have a smooth perimeter (pass is_smooth_blob())
 *
 *  Everything else is erased from the mask. The result is (usually) just the
 *  ground blob.
 * ══════════════════════════════════════════════════════════════════════════════ */
void isolate_ground_blob(uint8_t mask[], int w, int h, int blob_area_threshold)
{
    int n_labels = connected_components(mask, work_labels, w, h);
    if (n_labels == 0) return;

    int max_label = 0;
    for (int i = 0; i < w * h; i++) if (work_labels[i] > max_label) max_label = work_labels[i];
    if (max_label >= MAX_CC_LABELS) max_label = MAX_CC_LABELS - 1;

    /* Decide once per label whether to keep it — much faster than deciding per-pixel */
    static uint8_t label_keep[MAX_CC_LABELS];
    memset(label_keep, 0, max_label + 1);

    for (int l = 1; l <= max_label; l++) {
        if (cc_area[l] < blob_area_threshold) continue;   /* too small — noise */

        int bb_height = cc_bb_y1[l] - cc_bb_y0[l] + 1;
        if (!is_smooth_blob(work_labels, w, h, (int16_t)l, cc_area[l], bb_height))
            continue;   /* too jagged — not ground */

        label_keep[l] = 1;
    }

    /* Apply the keep/discard decision: erase all non-ground pixels */
    for (int i = 0; i < w * h; i++) {
        int16_t l = work_labels[i];
        mask[i] = (l > 0 && l <= max_label && label_keep[l]) ? 255 : 0;
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  4. HOLE FILLING (border flood-fill BFS)
 *
 *  After isolating the ground blob, there can be small black holes inside it
 *  (shadow patches, grid lines on the floor, etc.). These holes would
 *  confuse the edge finder and the obstacle detector.
 *
 *  The fix: flood-fill from the image border. Any black pixel reachable from
 *  the border is truly background. Any black pixel SURROUNDED by white (ground)
 *  and not reachable from the border is a hole — we fill those white.
 *
 *  This is the classic "fill interior holes" algorithm used in binary
 *  image processing. It does NOT require knowing the shape of the holes in
 *  advance.
 * ══════════════════════════════════════════════════════════════════════════════ */
void fill_holes_mask(uint8_t mask[], int w, int h)
{
    int total = w * h;
    memset(work_visited, 0, total);
    int qh = 0, qt = 0;   /* queue head and tail */

    /* Seed the BFS with all black pixels on the image border */
    for (int x = 0; x < w; x++) {
        if (!mask[x] && !work_visited[x]) { work_visited[x] = 1; ff_queue[qt++] = x; }
        int b = (h-1)*w+x;
        if (!mask[b] && !work_visited[b]) { work_visited[b] = 1; ff_queue[qt++] = b; }
    }
    for (int y = 0; y < h; y++) {
        int l = y*w;
        if (!mask[l] && !work_visited[l]) { work_visited[l] = 1; ff_queue[qt++] = l; }
        int r = y*w+(w-1);
        if (!mask[r] && !work_visited[r]) { work_visited[r] = 1; ff_queue[qt++] = r; }
    }

    /* BFS: spread out from each border-reachable black pixel to its black neighbours */
    while (qh < qt) {
        int i = ff_queue[qh++]; int cy = i / w, cx = i % w;
        if (cy > 0     && !work_visited[i-w] && !mask[i-w]) { work_visited[i-w] = 1; ff_queue[qt++] = i-w; }
        if (cy < h - 1 && !work_visited[i+w] && !mask[i+w]) { work_visited[i+w] = 1; ff_queue[qt++] = i+w; }
        if (cx > 0     && !work_visited[i-1] && !mask[i-1]) { work_visited[i-1] = 1; ff_queue[qt++] = i-1; }
        if (cx < w - 1 && !work_visited[i+1] && !mask[i+1]) { work_visited[i+1] = 1; ff_queue[qt++] = i+1; }
    }

    /* Any black pixel that was NOT reached from the border is a hole — fill it */
    for (int i = 0; i < total; i++)
        if (!mask[i] && !work_visited[i]) mask[i] = 255;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  5. GROUND BOUNDARY EXTRACTION + 1-D MEDIAN SMOOTHING
 *
 *  For each row of the (flipped) ground mask, this function finds the column
 *  where ground ends — i.e. the rightmost ground pixel before a large gap of
 *  non-ground. This gives us one boundary value per row, forming a profile of
 *  where the floor ends across the image.
 *
 *  Challenges handled:
 *    - Rows with no ground pixels at all are skipped.
 *    - Small gaps in the ground (shadows, grid lines) are tolerated if the
 *      run on the other side is long enough.
 *    - A 1-D median filter smooths out single-row noise in the boundary.
 * ══════════════════════════════════════════════════════════════════════════════ */
static void insertion_sort_f(float *a, int n) {
    for (int i = 1; i < n; i++) { float k = a[i]; int j = i-1; while (j >= 0 && a[j] > k) { a[j+1] = a[j]; j--; } a[j+1] = k; }
}
static void median_filter_1d(int *data, int n, int ksize) {
    if (ksize < 3 || n <= ksize) return;
    int half = ksize / 2; float win[25];
    static int mc[MAX_IMAGE_HEIGHT]; if (n > MAX_IMAGE_HEIGHT) n = MAX_IMAGE_HEIGHT;
    memcpy(mc, data, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int wn = 0, lo = (i-half < 0) ? 0 : i-half, hi = (i+half >= n) ? n-1 : i+half;
        for (int j = lo; j <= hi; j++) win[wn++] = (float)mc[j];
        insertion_sort_f(win, wn);
        data[i] = (int)win[wn / 2];
    }
}

/**
 * @brief Find where the ground boundary is in each row of the ground mask.
 *
 * The mask is expected to already be HORIZONTALLY FLIPPED so that ground
 * pixels appear on the RIGHT side (high column values). The boundary is the
 * column of the rightmost contiguous ground region per row.
 *
 * Short runs (< max_gap pixels) across a gap are tolerated — this prevents
 * a single shadow pixel from cutting the ground boundary short.
 *
 * @param mask_flipped  Horizontally flipped binary ground mask
 * @param w             Image width
 * @param h             Image height
 * @param boundary_out  OUTPUT: one boundary column index per row (0..w-1);
 *                      set to w (= "no boundary") if no ground found in that row
 * @param mgp           min_ground_pixels: how many ground pixels the last strip
 *                      must have to count as "ground touches the right edge"
 * @param mg            max_gap: maximum acceptable gap length inside the ground
 * @param sk            smooth_kernel: 1-D median kernel size for the final boundary
 */
void find_ground_boundary(const uint8_t mask_flipped[], int w, int h,
                          int boundary_out[], int mgp, int mg, int sk)
{
    int nr = h, rl = w;
    for (int r = 0; r < nr; r++) boundary_out[r] = rl;  /* default: no ground */
    for (int r = 0; r < nr; r++) {
        const uint8_t *row = &mask_flipped[r * rl];
        int any = 0;
        for (int c = 0; c < rl; c++) if (row[c]) { any = 1; break; }
        if (!any) continue;   /* row is entirely background */
        int gv = 1;
        for (int c = rl - mgp; c < rl; c++) { if (c < 0) continue; if (!row[c]) { gv = 0; break; } }
        if (!gv) {
            int mr = 0, cr = 0;
            for (int c = 0; c < rl; c++) { if (row[c]) { cr++; if (cr > mr) mr = cr; } else cr = 0; }
            if (mr < mg) continue;
        }
        int *gp = (int *)ff_queue; int gn = 0;
        for (int c = rl - 1; c >= 0; c--) if (row[c]) gp[gn++] = c;
        if (gn <= 1) { if (gn == 1) boundary_out[r] = gp[0]; continue; }
        int fg = -1;
        for (int i = 0; i < gn - 1; i++) if (gp[i] - gp[i+1] - 1 > mg) { fg = i; break; }
        if (fg < 0) { boundary_out[r] = gp[gn - 1]; continue; }
        int pending = gp[fg], ac = gn - (fg + 1);
        if (ac >= mg) {
            int *af = &gp[fg+1]; int mr2 = 1, cr2 = 1;
            for (int i = 1; i < ac; i++) { if (af[i-1]-af[i]-1==0) { cr2++; if(cr2>mr2) mr2=cr2; } else cr2=1; }
            if (mr2 >= mg) { boundary_out[r] = gp[gn - 1]; continue; }
        }
        boundary_out[r] = pending;
    }
    int vc = 0;
    for (int r = 0; r < nr; r++) if (boundary_out[r] < rl) vc++;
    if (vc > sk) median_filter_1d(boundary_out, nr, sk);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  6-8. OBSTACLE COLUMN MERGE + BASELINE UPDATE + OBSTACLE DETECTION
 *
 *  The ground boundary profile is compared against a running "baseline" —
 *  a smoothed estimate of where the boundary normally is. A column whose
 *  boundary deviates more than obstacle_threshold pixels ABOVE the baseline
 *  (ground appearing higher = something pushing it up = obstacle) is flagged.
 *
 *  Flagged columns that form contiguous runs of sufficient width become
 *  obstacle regions.
 *
 *  The baseline itself is updated only for columns that are NOT flagged as
 *  obstacles, so obstacles don't corrupt the reference.
 * ══════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Merge a binary per-column obstacle map into contiguous regions.
 *
 * Scans through the set columns in om[], groups them into runs, and discards
 * runs shorter than min_width. Output is stored in out[] as obstacle_region_t
 * structs (start column + width).
 *
 * @param om   Binary array: 1 = this column is an obstacle, 0 = clear
 * @param nc   Number of columns to scan
 * @param mw   Minimum run width to keep
 * @param mcg  Maximum column gap to bridge (columns ≤ mcg apart are merged)
 * @param out  OUTPUT: array of obstacle_region_t structs
 * @param mx   Maximum number of regions to write
 * @return     Number of regions written to out[]
 */
static int merge_obstacle_cols(const uint8_t *om, int nc, int mw, int mcg,
                               struct obstacle_region_t out[], int mx)
{
    static int ocb[MAX_IMAGE_HEIGHT]; int no = 0;
    for (int i = 0; i < nc && no < MAX_IMAGE_HEIGHT; i++) if (om[i]) ocb[no++] = i;
    if (no == 0) return 0;
    int cnt = 0, s = ocb[0], e = ocb[0];
    for (int i = 1; i < no; i++) {
        if (ocb[i] - e <= mcg) e = ocb[i];
        else { int w = e-s+1; if (w >= mw && cnt < mx) { out[cnt].start=s; out[cnt].width=w; cnt++; } s=ocb[i]; e=ocb[i]; }
    }
    { int w = e-s+1; if (w >= mw && cnt < mx) { out[cnt].start=s; out[cnt].width=w; cnt++; } }
    return cnt;
}

/**
 * @brief Update the ground baseline and detect obstacle columns.
 *
 * On the first call (baseline not yet initialised), the boundary profile is
 * stored directly as the baseline. On subsequent calls, columns where the
 * boundary is significantly higher than the baseline are marked as obstacles.
 * The baseline is then updated only for the non-obstacle columns using
 * exponential smoothing, so it adapts slowly to genuine scene changes.
 *
 * After baseline update, a two-pass propagation fills in obstacle columns in
 * the baseline (left-to-right then right-to-left, keeping the minimum) so
 * that the baseline under an obstacle doesn't drift to nonsense.
 *
 * @param br    Boundary row values per column (from find_ground_boundary)
 * @param h     Image height (= number of rows = value when "no boundary")
 * @param w     Number of columns
 * @param gb    Ground baseline array to update IN-PLACE (one float per column)
 * @param bi    Pointer to baseline-initialised flag (set to 1 after first call)
 * @param mw    Minimum obstacle region width
 * @param ot    Obstacle threshold in pixels (deviation above baseline)
 * @param ngb   No-ground baseline: value used for columns with no detected ground
 * @param mcg   Max column gap for merging obstacle runs
 * @param oo    OUTPUT: detected obstacle regions
 * @return      Number of obstacle regions detected
 */
uint8_t update_and_detect(const int br[], int h, int w, float gb[], int *bi,
                          int mw, int ot, int ngb, int mcg,
                          struct obstacle_region_t oo[])
{
    float alpha = DEFAULT_BASELINE_ALPHA;
    if (!(*bi)) {
        /* First frame: initialise the baseline directly from the boundary */
        for (int i = 0; i < w; i++) gb[i] = (br[i] < h) ? (float)br[i] : (float)ngb;
        *bi = 1; return 0;
    }
    static uint8_t om[MAX_IMAGE_HEIGHT], nom[MAX_IMAGE_HEIGHT];
    memset(om, 0, w); memset(nom, 0, w);
    for (int i = 0; i < w; i++) {
        int v = br[i] < h;
        if (v) {
            float d = (float)br[i] - gb[i];
            if (d > (float)ot) om[i] = 1;   /* boundary well above baseline = obstacle */
            else nom[i] = 1;                  /* normal — use for baseline update */
        }
        else om[i] = 1;   /* no ground detected at all = obstacle */
    }
    struct obstacle_region_t fl[MAX_OBSTACLE_REGIONS];
    int nf = merge_obstacle_cols(om, w, mw, mcg, fl, MAX_OBSTACLE_REGIONS);

    /* Update baseline with exponential smoothing for non-obstacle columns only */
    for (int i = 0; i < w; i++)
        if (nom[i]) gb[i] = (1.0f - alpha) * gb[i] + alpha * (float)br[i];

    /* Propagate baseline values into obstacle columns so they don't sit at
       stale values: left-to-right pass using the last valid ground seen,
       then right-to-left pass keeping the MINIMUM (closer = safer reference). */
    float lg = (float)ngb;
    for (int i = 0; i < w; i++) {
        if (nom[i]) lg = gb[i];
        else gb[i] = lg;
    }
    float rg = (float)ngb;
    for (int i = w - 1; i >= 0; i--) {
        if (nom[i]) {
            rg = gb[i];
        } else {
            if (rg == (float)ngb) {
                gb[i] = (float)ngb;   /* no valid ground seen from the right yet */
            } else {
                if (rg < gb[i]) gb[i] = rg;   /* keep the smaller (higher) boundary */
            }
        }
    }

    /* Convert flagged column runs into obstacle_region_t structs */
    int no = 0;
    for (int i = 0; i < nf && no < MAX_OBSTACLE_REGIONS; i++) {
        if (fl[i].width < 5) continue;
        oo[no].start = (uint16_t)fl[i].start;
        oo[no].width = (uint16_t)fl[i].width;

        int s = fl[i].start;
        int e = s + fl[i].width - 1;

        /* Determine how tall the obstacle is: if ground disappears anywhere
           in this column range, treat it as touching the full image bottom. */
        int max_br = 0;
        int touches_bottom = 0;

        for (int r = s; r <= e; r++) {
            if (r < w) {
                if (br[r] >= h) {
                    touches_bottom = 1;
                } else {
                    if (br[r] > max_br) max_br = br[r];
                }
            }
        }

        if (touches_bottom || max_br == 0) max_br = ngb;   /* use safe default */

        oo[no].baseline_height = (uint16_t)max_br;
        no++;
    }
    return (uint8_t)no;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  9-15. PLANT DETECTION
 *
 *  Plants (potted plants in the cyberzoo) are detected separately from floor
 *  obstacles because they are green — the same colour as the ground — and so
 *  would not appear as obstacles in the ground-boundary comparison above.
 *
 *  Strategy:
 *    1. Detect all greenish pixels using LOOSER thresholds than the ground
 *       classifier. This catches both floor AND plants.
 *    2. Subtract the confirmed floor mask. What remains is above-floor green:
 *       the plant pots.
 *    3. Find contiguous column runs of these remaining green pixels.
 *
 *  Processing is done at a downscaled resolution (DEFAULT_PLANT_SCALE_NUM /
 *  DEFAULT_PLANT_SCALE_DEN, typically 3/25 = 12%) to save compute time.
 * ══════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Detect all green-ish pixels at reduced resolution.
 *
 * Uses intentionally loose YUV thresholds (LAX_GREEN_*) to catch plants even
 * when they are partially in shadow or at odd angles. The leftmost third of
 * each row is blanked (those columns are the floor region in the rotated image
 * and would create false positives).
 *
 * @param img   Input YUV422 image
 * @param sn    Scale numerator (e.g. 3 for 3/25 scale)
 * @param sd    Scale denominator (e.g. 25)
 * @param bk    Box-blur kernel size for post-classification smoothing
 * @param mo    OUTPUT: plant green mask at DOWNSCALED resolution
 * @param pwo   OUTPUT: downscaled width
 * @param pho   OUTPUT: downscaled height
 */
void detect_all_green_lax(struct image_t *img, int sn, int sd, int bk,
                          uint8_t mo[], int *pwo, int *pho)
{
    int w = img->w, h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;
    int pw = (w * sn) / sd, ph = (h * sn) / sd;
    if (pw < 1) pw = 1; if (ph < 1) ph = 1;
    if (pw > MAX_PLANT_WIDTH) pw = MAX_PLANT_WIDTH;
    if (ph > MAX_PLANT_HEIGHT) ph = MAX_PLANT_HEIGHT;
    *pwo = pw; *pho = ph;

    /* Classify each downscaled pixel with the loose green thresholds */
    for (int py = 0; py < ph; py++) {
        int sy = (py * sd) / sn; if (sy >= h) sy = h - 1;
        for (int px = 0; px < pw; px++) {
            int sx = (px * sd) / sn; if (sx >= w) sx = w - 1;
            uint8_t Y = yuv422_Y(buf, w, sx, sy);
            uint8_t U = yuv422_U(buf, w, sx, sy);
            uint8_t V = yuv422_V(buf, w, sx, sy);
            plant_green_small[py * pw + px] = (U <= LAX_GREEN_U_MAX && V <= LAX_GREEN_V_MAX &&
                                                Y >= LAX_GREEN_Y_MIN && Y <= LAX_GREEN_Y_MAX) ? 255 : 0;
        }
    }

    /* Blank the leftmost third of each row — that region is always floor and
       would otherwise contaminate the plant detection */
    int lt = pw / 3;
    for (int py = 0; py < ph; py++) for (int px = 0; px < lt; px++) plant_green_small[py * pw + px] = 0;

    /* Optional box blur to reduce noise in the plant mask */
    if (bk >= 3) {
        int half = bk / 2;
        for (int py = 0; py < ph; py++)
            for (int px = 0; px < pw; px++) {
                int sum = 0, cnt = 0;
                int y0 = (py-half < 0) ? 0 : py-half, y1 = (py+half >= ph) ? ph-1 : py+half;
                int x0 = (px-half < 0) ? 0 : px-half, x1 = (px+half >= pw) ? pw-1 : px+half;
                for (int ky = y0; ky <= y1; ky++) for (int kx = x0; kx <= x1; kx++) { sum += plant_green_small[ky*pw+kx]; cnt++; }
                plant_blur_buf[py * pw + px] = (sum / cnt > 127) ? 255 : 0;
            }
        memcpy(mo, plant_blur_buf, pw * ph);
    } else memcpy(mo, plant_green_small, pw * ph);
}

/* Nearest-neighbour downsampler for binary masks */
static void downsample_nn(const uint8_t *s, int sw, int sh, uint8_t *d, int dw, int dh)
{ for (int y = 0; y < dh; y++) { int sy = y*sh/dh; if(sy>=sh)sy=sh-1; for (int x = 0; x < dw; x++) { int sx = x*sw/dw; if(sx>=sw)sx=sw-1; d[y*dw+x] = s[sy*sw+sx]; } } }

/* Nearest-neighbour upsampler for binary masks */
static void upsample_nn(const uint8_t *s, int sw, int sh, uint8_t *d, int dw, int dh)
{ for (int y = 0; y < dh; y++) { int sy = y*sh/dh; if(sy>=sh)sy=sh-1; for (int x = 0; x < dw; x++) { int sx = x*sw/dw; if(sx>=sw)sx=sw-1; d[y*dw+x] = s[sy*sw+sx]; } } }

/**
 * @brief Find column runs of plant pixels in the plant mask.
 *
 * For each row in the downscaled mask, counts how many consecutive green pixels
 * appear scanning RIGHT-TO-LEFT (that side is the "above-floor" region in
 * the rotated camera view). Rows with enough consecutive green pixels are
 * marked as containing a plant. Contiguous marked rows are then grouped into
 * obstacle_region_t structs.
 *
 * @param pm   Plant binary mask (downscaled)
 * @param w    Mask width
 * @param h    Mask height
 * @param mw   Minimum region width in rows to report
 * @param mpc  Minimum consecutive green pixels per row to flag that row
 * @param mcg  Maximum row gap for merging adjacent plant regions
 * @param po   OUTPUT: plant region array
 * @return     Number of plant regions found
 */
uint8_t detect_plant_regions(const uint8_t pm[], int w, int h, int mw, int mpc, int mcg,
                             struct obstacle_region_t po[])
{
    static uint8_t am[MAX_IMAGE_HEIGHT]; memset(am, 0, h);
    for (int r = 0; r < h; r++) {
        int mr = 0, cr = 0;
        /* Scan right-to-left: the right side is the above-floor region */
        for (int c = 0; c < w; c++) { int fc = w-1-c; if (pm[r*w+fc]) { cr++; if (cr > mr) mr = cr; } else cr = 0; }
        if (mr > mpc) am[r] = 1;   /* this row has enough green pixels to be a plant */
    }
    return (uint8_t)merge_obstacle_cols(am, h, mw, mcg, po, MAX_PLANT_REGIONS);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  MAIN PIPELINE — get_obstacle_info
 *
 *  Runs the full obstacle + plant detection pipeline on one camera frame.
 *
 *  Summary of steps:
 *    1. Classify ground pixels → binary mask → green fraction
 *    2. If enough ground found: isolate the main blob, fill holes, find boundary
 *    3. Compare boundary to baseline → obstacle regions
 *    4. Detect plant pixels (loose threshold), subtract floor mask → plant regions
 *    5. Return both region lists plus the green fraction and mask pointer
 * ══════════════════════════════════════════════════════════════════════════════ */
uint8_t get_obstacle_info(struct image_t *img, float gb[], int *bi,
                          float oacf, int mk, int mw,
                          struct obstacle_region_t oo[],
                          struct obstacle_region_t po[], uint8_t *pco,
                          int bro[], int *gfo, float *gfro, obstacle_info_result_t *result_out)
{
    int w = img->w, h = img->h;
    if (w > MAX_IMAGE_WIDTH) w = MAX_IMAGE_WIDTH;
    if (h > MAX_IMAGE_HEIGHT) h = MAX_IMAGE_HEIGHT;

    float gf = 0.0f;
    detect_green_ground_ml(img, work_mask, mk, &gf);
    if (gfro) *gfro = gf;
    int ground = (gf > oacf) ? 1 : 0;   /* is there enough ground to bother? */
    if (gfo) *gfo = ground;

    uint8_t no = 0;
    if (ground) {
        isolate_ground_blob(work_mask, w, h, DEFAULT_BLOB_AREA_THRESH);
        fill_holes_mask(work_mask, w, h);
        /* Flip horizontally before boundary scan so ground pixels appear on the right */
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) work_flipped[y*w+x] = work_mask[y*w+(w-1-x)];
        static int bl[MAX_IMAGE_HEIGHT];
        find_ground_boundary(work_flipped, w, h, bl, DEFAULT_MIN_GROUND_PX, DEFAULT_MAX_GAP, DEFAULT_SMOOTH_KERNEL);
        if (bro) memcpy(bro, bl, h * sizeof(int));
        no = update_and_detect(bl, w, h, gb, bi, mw, DEFAULT_OBSTACLE_THRESH, DEFAULT_NO_GROUND_BASE, DEFAULT_MAX_COL_GAP, oo);
    }

    /* Plant detection runs independently of whether ground was found */
    if (po != NULL) {
        int pw = 0, ph = 0;
        detect_all_green_lax(img, DEFAULT_PLANT_SCALE_NUM, DEFAULT_PLANT_SCALE_DEN,
                             DEFAULT_PLANT_BLUR_KSIZE, plant_mask_small, &pw, &ph);
        /* Subtract the floor mask so only above-floor green remains */
        if (ground) downsample_nn(work_mask, w, h, plant_clean_small, pw, ph);
        else memset(plant_clean_small, 0, pw * ph);
        for (int i = 0; i < pw * ph; i++) {
            int v = (int)plant_mask_small[i] - (int)plant_clean_small[i];
            plant_mask_small[i] = (v > 0) ? (uint8_t)v : 0;
        }
        upsample_nn(plant_mask_small, pw, ph, plant_mask_full, w, h);
        uint8_t np = detect_plant_regions(plant_mask_full, w, h,
                                          DEFAULT_PLANT_MIN_WIDTH, DEFAULT_PLANT_MIN_PX_COL,
                                          DEFAULT_PLANT_MAX_COL_GAP, po);
        if (pco) *pco = np;
    } else { if (pco) *pco = 0; }

    if (result_out) {
        result_out->gf     = gf;
        result_out->ground = ground;
        result_out->mask   = work_mask;
    }

    return no;
}

/* ── DEBUG UTILITIES ─────────────────────────────────────────────────────── */

/* Write a raw binary mask to a file for offline inspection */
int dump_mask_to_file(const char *p, const uint8_t m[], int w, int h)
{ FILE *f = fopen(p, "wb"); if (!f) return -1; fwrite(m, 1, w*h, f); fclose(f); return 0; }

/* Write a list of obstacle regions as (start, width) pairs to a text file */
int dump_obstacles_to_file(const char *p, const struct obstacle_region_t o[], int c)
{ FILE *f = fopen(p, "w"); if (!f) return -1; fprintf(f, "%d\n", c); for (int i = 0; i < c; i++) fprintf(f, "%d %d\n", o[i].start, o[i].width); fclose(f); return 0; }

/* ══════════════════════════════════════════════════════════════════════════════
 *  GATE DETECTION
 *
 *  Detects the TU Delft gate in the camera frame. The gate consists of two
 *  vertical blue pillars with a black-and-white checker pattern visible on
 *  their inner faces.
 *
 *  This is a direct C port of gate_detection.py.
 *
 *  Coordinate convention:
 *    The image arrives in portrait YUV422 with the floor on the LEFT side.
 *    Two real-world vertical pillars appear as two blobs separated along the
 *    IMAGE Y-axis (vertically stacked in the rotated frame).
 *    The output x-coordinate is the pixel distance from the LEFT edge of the
 *    raw image — which is the distance from the TOP edge when displayed upright.
 *
 *  Pipeline:
 *    1. Threshold on YUV to produce a blue-only binary mask
 *    2. Morphological open+close to remove noise and fill small gaps
 *    3. Label connected components; keep blobs above a size threshold
 *    4. For every pair of blobs, check geometric constraints (separation axis,
 *       aspect ratio similarity, area similarity, centroid join angle)
 *    5. Confirm each candidate blob with a checker-pattern score on its inner face
 *    6. Return the midpoint of the best (largest) valid pair
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Work buffers used only by gate detection — separate from the obstacle pipeline */
static uint8_t  gate_blue_mask[MAX_PIXELS];     /* binary blue mask for gate pixels  */
static uint8_t  gate_eroded[MAX_PIXELS];        /* scratch buffer for morphology      */
static int16_t  gate_cc_labels[MAX_PIXELS];     /* connected-component labels (gate)  */

/* Per-blob descriptor: stores only what we need for the pair matching checks */
typedef struct {
    int x_min, y_min, x_max, y_max;   /* axis-aligned bounding box              */
    int cx, cy;                        /* centroid in integer pixel coordinates   */
    int area;                          /* total pixel count inside the blob       */
    int aspect_num, aspect_den;        /* aspect ratio stored as bw / bh integers */
} gate_blob_t;

static gate_blob_t gate_blobs[GATE_MAX_BLOBS];

/* Union-find arrays local to gate detection — kept separate from the obstacle
   pipeline's uf_parent[] so the two pipelines don't interfere. */
static int16_t gate_uf[MAX_CC_LABELS];
static int32_t gate_area[MAX_CC_LABELS];
static int16_t gate_x0[MAX_CC_LABELS], gate_x1[MAX_CC_LABELS];
static int16_t gate_y0[MAX_CC_LABELS], gate_y1[MAX_CC_LABELS];
static int32_t gate_cx_sum[MAX_CC_LABELS], gate_cy_sum[MAX_CC_LABELS];

/* Path-compressed find for the gate's own union-find */
static int16_t gate_uf_find(int16_t x) {
    while (gate_uf[x] != x) { gate_uf[x] = gate_uf[gate_uf[x]]; x = gate_uf[x]; }
    return x;
}
/* Merge two gate labels into one canonical root */
static void gate_uf_union(int16_t a, int16_t b) {
    a = gate_uf_find(a); b = gate_uf_find(b);
    if (a != b) { if (a < b) gate_uf[b] = a; else gate_uf[a] = b; }
}

/**
 * @brief Build a blue-only binary mask from the camera image.
 *
 * Thresholds on YUV channels calibrated for TU Delft blue (same values as in
 * the Python gate_detection.py). U (Cb) is high for blue, V (Cr) is low —
 * that combination rarely triggers on other colours in outdoor/indoor scenes.
 *
 * A morphological OPEN (erode → dilate) removes isolated noise pixels, and a
 * morphological CLOSE (dilate → erode) fills small gaps inside the pillars.
 * Both use a 5×5 rectangular structuring element.
 *
 * @param img       Input YUV422 image
 * @param out_mask  OUTPUT: binary mask, 255 = blue pixel, 0 = not blue
 */
static void gate_make_blue_mask(const struct image_t *img, uint8_t *out_mask)
{
    int w = img->w, h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;
    int total = w * h;

    /* Step 1: threshold each pixel — U_MAX and V_MIN are trivially satisfied
       for uint8_t, so they are omitted from the comparison. */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t Y = yuv422_Y(buf, w, x, y);
            uint8_t U = yuv422_U(buf, w, x, y);
            uint8_t V = yuv422_V(buf, w, x, y);
            int idx = y * w + x;
            out_mask[idx] = (Y <= (uint8_t)GATE_BLUE_Y_MAX &&
                             U >= (uint8_t)GATE_BLUE_U_MIN &&
                             V <= (uint8_t)GATE_BLUE_V_MAX) ? 255 : 0;
        }
    }

    /* Step 2: morphological OPEN = erode then dilate.
       Erode: a pixel stays set only if ALL pixels within the 5×5 neighbourhood
       are set. This removes isolated specks. */
    int k = 3, half = 2;

    memset(gate_eroded, 0, total);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!out_mask[y * w + x]) continue;
            int ok = 1;
            int y0 = (y - half < 0) ? 0 : y - half;
            int y1 = (y + half >= h) ? h - 1 : y + half;
            int x0 = (x - half < 0) ? 0 : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;
            for (int ky = y0; ky <= y1 && ok; ky++)
                for (int kx = x0; kx <= x1 && ok; kx++)
                    if (!out_mask[ky * w + kx]) ok = 0;
            gate_eroded[y * w + x] = ok ? 255 : 0;
        }
    }

    /* Dilate: a pixel is set if ANY pixel within the 5×5 neighbourhood is set.
       This restores the edges of features that survived erosion. */
    memset(out_mask, 0, total);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int y0 = (y - half < 0) ? 0 : y - half;
            int y1 = (y + half >= h) ? h - 1 : y + half;
            int x0 = (x - half < 0) ? 0 : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;
            int found = 0;
            for (int ky = y0; ky <= y1 && !found; ky++)
                for (int kx = x0; kx <= x1 && !found; kx++)
                    if (gate_eroded[ky * w + kx]) found = 1;
            out_mask[y * w + x] = found ? 255 : 0;
        }
    }

    /* Step 3: morphological CLOSE = dilate then erode.
       This fills small gaps inside the blue blobs that erosion may have created. */
    memset(gate_eroded, 0, total);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int y0 = (y - half < 0) ? 0 : y - half;
            int y1 = (y + half >= h) ? h - 1 : y + half;
            int x0 = (x - half < 0) ? 0 : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;
            int found = 0;
            for (int ky = y0; ky <= y1 && !found; ky++)
                for (int kx = x0; kx <= x1 && !found; kx++)
                    if (out_mask[ky * w + kx]) found = 1;
            gate_eroded[y * w + x] = found ? 255 : 0;
        }
    }
    memset(out_mask, 0, total);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!gate_eroded[y * w + x]) continue;
            int ok = 1;
            int y0 = (y - half < 0) ? 0 : y - half;
            int y1 = (y + half >= h) ? h - 1 : y + half;
            int x0 = (x - half < 0) ? 0 : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;
            for (int ky = y0; ky <= y1 && ok; ky++)
                for (int kx = x0; kx <= x1 && ok; kx++)
                    if (!gate_eroded[ky * w + kx]) ok = 0;
            out_mask[y * w + x] = ok ? 255 : 0;
        }
    }
    (void)k;
}

/**
 * @brief Find and characterise all significant blue blobs in the blue mask.
 *
 * Runs connected components on the blue mask, then for each label large
 * enough (area >= min_area) and tall/wide enough (>= GATE_BLOB_MIN_DIM),
 * fills in a gate_blob_t descriptor with its bounding box, centroid, area,
 * and aspect ratio.
 *
 * @param mask      Binary blue mask from gate_make_blue_mask()
 * @param w         Image width
 * @param h         Image height
 * @param blobs     OUTPUT: array of gate_blob_t descriptors
 * @param max_blobs Maximum number of blobs to store
 * @return          Number of blobs found
 */
static int gate_extract_blobs(const uint8_t *mask, int w, int h,
                               gate_blob_t blobs[], int max_blobs)
{
    int total    = w * h;
    int min_area = w * h * GATE_BLOB_MIN_AREA_NUM / GATE_BLOB_MIN_AREA_DEN;
    if (min_area < 1) min_area = 1;

    /* Connected components using the gate's own union-find arrays */
    int16_t next_lbl = 1;
    for (int i = 0; i < MAX_CC_LABELS; i++) gate_uf[i] = (int16_t)i;
    memset(gate_cc_labels, 0, total * sizeof(int16_t));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (!mask[idx]) continue;
            int16_t nb[4]; int nn = 0;
            if (y > 0 && x > 0     && gate_cc_labels[(y-1)*w+(x-1)]) nb[nn++] = gate_cc_labels[(y-1)*w+(x-1)];
            if (y > 0               && gate_cc_labels[(y-1)*w+x])     nb[nn++] = gate_cc_labels[(y-1)*w+x];
            if (y > 0 && x < w - 1 && gate_cc_labels[(y-1)*w+(x+1)]) nb[nn++] = gate_cc_labels[(y-1)*w+(x+1)];
            if (x > 0               && gate_cc_labels[y*w+(x-1)])     nb[nn++] = gate_cc_labels[y*w+(x-1)];
            if (nn == 0) {
                if (next_lbl < MAX_CC_LABELS) gate_cc_labels[idx] = next_lbl++;
            } else {
                int16_t m = gate_uf_find(nb[0]);
                for (int i = 1; i < nn; i++) {
                    int16_t r = gate_uf_find(nb[i]);
                    if (r < m) m = r;
                }
                gate_cc_labels[idx] = m;
                for (int i = 0; i < nn; i++) gate_uf_union(m, nb[i]);
            }
        }
    }
    /* Flatten labels to canonical roots */
    for (int i = 0; i < total; i++)
        if (gate_cc_labels[i]) gate_cc_labels[i] = gate_uf_find(gate_cc_labels[i]);

    /* Accumulate per-label statistics */
    int max_lbl = 0;
    for (int i = 0; i < total; i++) if (gate_cc_labels[i] > max_lbl) max_lbl = gate_cc_labels[i];
    if (max_lbl >= MAX_CC_LABELS) max_lbl = MAX_CC_LABELS - 1;

    for (int l = 0; l <= max_lbl; l++) {
        gate_area[l] = 0;
        gate_cx_sum[l] = 0; gate_cy_sum[l] = 0;
        gate_x0[l] = (int16_t)(w - 1); gate_x1[l] = 0;
        gate_y0[l] = (int16_t)(h - 1); gate_y1[l] = 0;
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int16_t l = gate_cc_labels[y * w + x];
            if (l <= 0 || l > max_lbl) continue;
            gate_area[l]++;
            gate_cx_sum[l] += x;
            gate_cy_sum[l] += y;
            if (x < gate_x0[l]) gate_x0[l] = (int16_t)x;
            if (x > gate_x1[l]) gate_x1[l] = (int16_t)x;
            if (y < gate_y0[l]) gate_y0[l] = (int16_t)y;
            if (y > gate_y1[l]) gate_y1[l] = (int16_t)y;
        }
    }

    /* Fill the output blobs array, skipping labels that are too small or too thin */
    int nb = 0;
    for (int l = 1; l <= max_lbl && nb < max_blobs; l++) {
        int area = (int)gate_area[l];
        if (area < min_area) continue;

        int bw = (int)(gate_x1[l] - gate_x0[l] + 1);
        int bh = (int)(gate_y1[l] - gate_y0[l] + 1);
        if (bw < GATE_BLOB_MIN_DIM || bh < GATE_BLOB_MIN_DIM) continue;

        blobs[nb].x_min      = gate_x0[l];
        blobs[nb].y_min      = gate_y0[l];
        blobs[nb].x_max      = gate_x1[l] + 1;   /* exclusive upper bound */
        blobs[nb].y_max      = gate_y1[l] + 1;
        blobs[nb].cx         = (area > 0) ? (int)(gate_cx_sum[l] / area) : (gate_x0[l] + gate_x1[l]) / 2;
        blobs[nb].cy         = (area > 0) ? (int)(gate_cy_sum[l] / area) : (gate_y0[l] + gate_y1[l]) / 2;
        blobs[nb].area       = area;
        blobs[nb].aspect_num = bw;   /* aspect = bw/bh stored without division */
        blobs[nb].aspect_den = bh;
        nb++;
    }
    return nb;
}

/**
 * @brief Score how well a blob's left face matches a checker pattern.
 *
 * The gate pillars have a black-and-white checker pattern on their inner
 * (floor-side) face. We check for this by examining the luma (Y) values
 * in a band of columns to the LEFT of the blob.
 *
 * For each column in the band, we record the minimum and maximum Y value
 * across all rows of the blob. If the range (max - min) exceeds
 * GATE_CHECKER_CONTRAST, the column passes — it contains both dark and
 * bright pixels, consistent with a checker pattern alternating between them.
 *
 * The score is the fraction of passing columns (0-100%). A solid blue wall
 * scores 0; a checker pattern on a blue pillar scores high.
 *
 * @param img  Input YUV422 image
 * @param b    Blob to score
 * @return     Score in percent (0 = no checker, 100 = perfect checker)
 */
static int gate_checker_score(const struct image_t *img, const gate_blob_t *b)
{
    int img_w = img->w, img_h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;

    int bw     = b->x_max - b->x_min;
    int bh     = b->y_max - b->y_min;
    /* Search band width is a fraction of the blob width */
    int band_w = bw * GATE_CHECKER_BAND_NUM / GATE_CHECKER_BAND_DEN;
    if (band_w < 4) band_w = 4;

    /* The band sits immediately to the LEFT of the blob (the floor-side face) */
    int rx0 = b->x_min - band_w;
    int rx1 = b->x_min;
    int ry0 = b->y_min;
    int ry1 = b->y_max;

    if (rx0 < 0) rx0 = 0;
    if (rx1 > img_w) rx1 = img_w;
    if (ry0 < 0) ry0 = 0;
    if (ry1 > img_h) ry1 = img_h;

    int n_cols = rx1 - rx0;
    int n_rows = ry1 - ry0;
    if (n_cols <= 0 || n_rows <= 0) return 0;

    int high_contrast_cols = 0;
    for (int x = rx0; x < rx1; x++) {
        uint8_t col_min = 255, col_max = 0;
        for (int y = ry0; y < ry1; y++) {
            uint8_t Y = yuv422_Y(buf, img_w, x, y);
            if (Y < col_min) col_min = Y;
            if (Y > col_max) col_max = Y;
        }
        if ((int)(col_max - col_min) > GATE_CHECKER_CONTRAST)
            high_contrast_cols++;
    }

    return high_contrast_cols * 100 / n_cols;   /* percent */
}

/**
 * @brief Check whether two blobs could be the left and right pillars of a gate.
 *
 * Returns 1 if all four geometric conditions are satisfied, 0 otherwise.
 *
 * Condition 0 — Vertical separation in image:
 *   The camera is rotated 90° CW, so the two real-world side-by-side pillars
 *   appear VERTICALLY separated in the image (|dy| > |dx|). Pairs that are
 *   horizontally separated are rejected immediately.
 *
 * Condition 1 — Similar aspect ratio:
 *   Both pillars are the same physical object at the same depth, so their
 *   bounding boxes should have the same shape. Checked via cross-multiplication
 *   to avoid floating-point: |bw1*bh2 - bw2*bh1| * 100 <= max * pct.
 *
 * Condition 2 — Similar area:
 *   Same reasoning as aspect ratio. Up to GATE_MAX_AREA_DIFF_PCT% difference
 *   is tolerated to handle partial occlusion.
 *
 * Condition 3 — Centroid join perpendicular to the blobs' long axis:
 *   For two parallel bars, the line connecting their centres should run at
 *   roughly 90° to the direction the bars point. We check this with integer
 *   fixed-point arithmetic to avoid sqrt and atan2:
 *     |dot(join, long_axis)|^2 / |join|^2 <= sin^2(max_skew_deg)
 *   Cross-multiplied: dot^2 * 1_000_000 <= join^2 * sin2_scaled
 *   where sin2_scaled = sin²(10°) × 1_000_000 ≈ 30154.
 *
 * @param b1  First blob descriptor
 * @param b2  Second blob descriptor
 * @return    1 if the pair passes all checks, 0 otherwise
 */
static int gate_blobs_parallel(const gate_blob_t *b1, const gate_blob_t *b2)
{
    /* Condition 0: vertical separation */
    int dx = b2->cx - b1->cx;
    int dy = b2->cy - b1->cy;
    int adx = dx < 0 ? -dx : dx;
    int ady = dy < 0 ? -dy : dy;
    if (ady <= adx) return 0;

    /* Condition 1: similar aspect ratio (cross-multiplication avoids division) */
    int bw1 = b1->aspect_num, bh1 = b1->aspect_den;
    int bw2 = b2->aspect_num, bh2 = b2->aspect_den;
    if (bh1 <= 0 || bh2 <= 0) return 0;
    int cross1 = bw1 * bh2;
    int cross2 = bw2 * bh1;
    int diff   = cross1 - cross2; if (diff < 0) diff = -diff;
    int mx     = cross1 > cross2 ? cross1 : cross2;
    if (diff * 100 > mx * GATE_MAX_ASPECT_DIFF_PCT) return 0;

    /* Condition 2: similar area */
    int a1 = b1->area, a2 = b2->area;
    int adiff = a1 - a2; if (adiff < 0) adiff = -adiff;
    int amax  = a1 > a2 ? a1 : a2;
    if (amax <= 0) return 0;
    if (adiff * 100 > amax * GATE_MAX_AREA_DIFF_PCT) return 0;

    /* Condition 3: centroid join perpendicular to long axis.
       avg_w/avg_h determines whether the bars point horizontally or vertically.
       The component of the join along the long axis should be small.
       We use sin²(10°) × 1_000_000 ≈ 30154 as the squared sine threshold. */
    int avg_w = ((b1->x_max - b1->x_min) + (b2->x_max - b2->x_min)) / 2;
    int avg_h = ((b1->y_max - b1->y_min) + (b2->y_max - b2->y_min)) / 2;

    /* Component of join along the long axis */
    int dot_abs = (avg_w >= avg_h) ? adx : ady;

    int join2       = dx * dx + dy * dy;
    int dot2        = dot_abs * dot_abs;
    int sin2_scaled = 30154;   /* sin²(10°) × 1 000 000 */
    if (dot2 * 1000000 > join2 * sin2_scaled) return 0;

    return 1;
}

/**
 * @brief Detect the gate in a single camera frame.
 *
 * Runs the full gate detection pipeline and writes the result into result[2]:
 *   result[0] = 1 if a gate was detected, 0 otherwise
 *   result[1] = X pixel of the gate centre (left-edge-relative in raw image)
 *
 * If multiple valid gate pairs are found, the one with the largest combined
 * blob area is chosen — that corresponds to the closest gate.
 *
 * @param img     Input YUV422 image
 * @param result  OUTPUT: [detected_flag, centre_x_pixel]
 */
void detect_gate(struct image_t *img, int result[2])
{
    result[0] = 0;
    result[1] = 0;

    int w = img->w, h = img->h;
    if (w > MAX_IMAGE_WIDTH)  w = MAX_IMAGE_WIDTH;
    if (h > MAX_IMAGE_HEIGHT) h = MAX_IMAGE_HEIGHT;

    /* Step 1 — produce the blue binary mask */
    gate_make_blue_mask(img, gate_blue_mask);

    /* Step 2 — extract blob descriptors from the mask */
    int n_blobs = gate_extract_blobs(gate_blue_mask, w, h,
                                     gate_blobs, GATE_MAX_BLOBS);
    if (n_blobs < 2) return;   /* need at least two blobs to form a pair */

    /* Steps 3+4 — find the best valid pair */
    int best_area  = -1;
    int best_mid_x = 0;
    int gate_found = 0;

    for (int i = 0; i < n_blobs; i++) {
        for (int j = i + 1; j < n_blobs; j++) {
            const gate_blob_t *b1 = &gate_blobs[i];
            const gate_blob_t *b2 = &gate_blobs[j];

            /* Geometric check: are these two blobs shaped and positioned like gate pillars? */
            if (!gate_blobs_parallel(b1, b2)) continue;

            /* Checker confirmation: does each blob have high-contrast luma on its left face? */
            int score1 = gate_checker_score(img, b1);
            int score2 = gate_checker_score(img, b2);
            if (score1 < GATE_CHECKER_MIN_FRAC_PCT) continue;
            if (score2 < GATE_CHECKER_MIN_FRAC_PCT) continue;

            /* Keep the pair with the largest combined area (= the closest gate) */
            int combined = b1->area + b2->area;
            if (combined > best_area) {
                best_area  = combined;
                best_mid_x = (b1->cx + b2->cx) / 2;   /* midpoint between centroids */
                gate_found = 1;
            }
        }
    }

    if (gate_found) {
        result[0] = 1;
        result[1] = best_mid_x;
    }
}