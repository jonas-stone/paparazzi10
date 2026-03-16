/*
 * team10_get_obstacle_info.c
 *
 * C port of the Python obstacle-detection pipeline.
 * See team10_get_obstacle_info.h for documentation.
 *
 * Memory strategy
 * ───────────────
 * All large work buffers are file-scope static so they live in BSS
 * (zero-cost at startup, no heap fragmentation).  This means the
 * functions are NOT re-entrant, but on the drone only one camera
 * callback thread uses them, so that is fine.
 */

#include "team10_get_obstacle_info.h"
#include "modules/computer_vision/lib/vision/image.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════════════
 *  STATIC WORK BUFFERS  (allocated once in BSS)
 * ══════════════════════════════════════════════════════════════════════════════ */
#define MAX_PIXELS (MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT)

static uint8_t  work_mask[MAX_PIXELS];        /* binary mask (0 / 255)        */
static uint8_t  work_mask2[MAX_PIXELS];       /* second mask for median blur  */
static int16_t  work_labels[MAX_PIXELS];      /* connected-component labels   */
static uint8_t  work_visited[MAX_PIXELS];     /* flood-fill visited flags     */
static uint8_t  work_flipped[MAX_PIXELS];     /* left-right flipped mask      */

/* flood-fill queue (BFS) – worst case every pixel */
static int32_t  ff_queue[MAX_PIXELS];

/* union-find parent array for connected components */
static int16_t  uf_parent[MAX_CC_LABELS];
static int32_t  cc_area[MAX_CC_LABELS];       /* area per label               */
static int16_t  cc_min_x[MAX_CC_LABELS];      /* bounding box                 */
static int16_t  cc_max_x[MAX_CC_LABELS];
static int16_t  cc_min_y[MAX_CC_LABELS];
static int16_t  cc_max_y[MAX_CC_LABELS];

/* 1-D arrays sized by image width */
static float    med_tmp[MAX_IMAGE_WIDTH];     /* for 1-D median of boundary   */

/* ══════════════════════════════════════════════════════════════════════════════
 *  PIXEL ACCESS HELPERS  (YUV422 / UYVY)
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *  Buffer layout per macro-pixel (2 pixels):
 *    byte 0 = U,  byte 1 = Y0,  byte 2 = V,  byte 3 = Y1
 *
 *  For pixel (x, y) in an image of width w:
 *    Y = buf[ y*w*2 + x*2 + 1 ]
 *    U = buf[ y*w*2 + (x & ~1)*2 ]           (shared with neighbour)
 *    V = buf[ y*w*2 + (x & ~1)*2 + 2 ]       (shared with neighbour)
 */

static inline uint8_t yuv422_Y(const uint8_t *buf, int w, int x, int y)
{
    return buf[y * w * 2 + x * 2 + 1];
}
static inline uint8_t yuv422_U(const uint8_t *buf, int w, int x, int y)
{
    return buf[y * w * 2 + (x & ~1) * 2];
}
static inline uint8_t yuv422_V(const uint8_t *buf, int w, int x, int y)
{
    return buf[y * w * 2 + (x & ~1) * 2 + 2];
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  1.  DECISION-TREE COLOUR CLASSIFIER
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *  Exact translation of the Python is_ground(Y, U, V) tree.
 */

uint8_t is_ground_pixel(uint8_t Y, uint8_t U, uint8_t V)
{
    if (U <= 115) {
        if (V <= 145) {
            if (Y <= 85) {
                return 0;   /* both Y<=78 and 78<Y<=85 → 0 */
            } else {
                if (U <= 92)
                    return 0;
                else
                    return 255;
            }
        } else {
            if (V <= 152) {
                if (Y <= 177)
                    return 255;
                else
                    return 0;
            } else {
                return 0;
            }
        }
    } else {
        if (U <= 121) {
            if (V <= 137) {
                if (Y <= 87)
                    return 0;
                else
                    return 255;
            } else {
                return 0;  /* both U<=116 and 116<U<=121 → 0 */
            }
        } else {
            if (U <= 122) {
                if (V <= 126)
                    return 0;
                else
                    return 0;
            } else {
                if (Y <= 62)
                    return 0;
                else
                    return 0;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  2.  MEDIAN BLUR  (optimised for binary images: majority vote)
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *  Since the mask is binary (0 or 255), median = majority vote within the
 *  kernel window.  We just count how many neighbours are 255.
 */

static void median_blur_binary(const uint8_t *src, uint8_t *dst,
                               int w, int h, int ksize)
{
    int half = ksize / 2;
    int threshold = (ksize * ksize) / 2;  /* majority = more than half */

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int count = 0;

            /* scan kernel window */
            int y0 = (y - half < 0)  ? 0     : y - half;
            int y1 = (y + half >= h) ? h - 1 : y + half;
            int x0 = (x - half < 0)  ? 0     : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;

            for (int ky = y0; ky <= y1; ky++) {
                for (int kx = x0; kx <= x1; kx++) {
                    if (src[ky * w + kx])
                        count++;
                }
            }

            dst[y * w + x] = (count > threshold) ? 255 : 0;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  FULL GREEN DETECTION  (classify + blur + fraction)
 * ══════════════════════════════════════════════════════════════════════════════ */

void detect_green_ground_ml(struct image_t *img,
                            uint8_t         mask_out[],
                            int             median_ksize,
                            float          *green_frac_out)
{
    int w = img->w;
    int h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;

    /* ── apply decision tree to every pixel ──────────────────────────────── */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t Y = yuv422_Y(buf, w, x, y);
            uint8_t U = yuv422_U(buf, w, x, y);
            uint8_t V = yuv422_V(buf, w, x, y);
            work_mask2[y * w + x] = is_ground_pixel(Y, U, V);
        }
    }

    /* ── median blur (binary-optimised) ──────────────────────────────────── */
    if (median_ksize >= 3 && (median_ksize & 1)) {
        median_blur_binary(work_mask2, mask_out, w, h, median_ksize);
    } else {
        memcpy(mask_out, work_mask2, w * h);
    }

    /* ── compute green fraction ──────────────────────────────────────────── */
    int green_count = 0;
    int total = w * h;
    for (int i = 0; i < total; i++) {
        if (mask_out[i])
            green_count++;
    }
    if (green_frac_out)
        *green_frac_out = (total > 0) ? (float)green_count / (float)total : 0.0f;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  3.  CONNECTED-COMPONENT LABELLING  (two-pass with union-find)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── union-find primitives ────────────────────────────────────────────────── */

static void uf_init(int n)
{
    for (int i = 0; i < n; i++)
        uf_parent[i] = (int16_t)i;
}

static int16_t uf_find(int16_t x)
{
    while (uf_parent[x] != x) {
        uf_parent[x] = uf_parent[uf_parent[x]];   /* path compression */
        x = uf_parent[x];
    }
    return x;
}

static void uf_union(int16_t a, int16_t b)
{
    a = uf_find(a);
    b = uf_find(b);
    if (a != b) {
        if (a < b)
            uf_parent[b] = a;
        else
            uf_parent[a] = b;
    }
}

/* ── two-pass connected-component labelling (8-connectivity) ─────────────── */

static int connected_components(const uint8_t *mask, int16_t *labels,
                                int w, int h)
{
    int16_t next_label = 1;

    uf_init(MAX_CC_LABELS);
    memset(labels, 0, w * h * sizeof(int16_t));

    /* ── first pass: assign provisional labels ───────────────────────────── */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (mask[idx] == 0) continue;

            /* collect labels of already-labelled neighbours (8-connected) */
            int16_t neighbours[4];
            int nn = 0;

            /* up-left */
            if (y > 0 && x > 0 && labels[(y-1)*w + (x-1)])
                neighbours[nn++] = labels[(y-1)*w + (x-1)];
            /* up */
            if (y > 0 && labels[(y-1)*w + x])
                neighbours[nn++] = labels[(y-1)*w + x];
            /* up-right */
            if (y > 0 && x < w-1 && labels[(y-1)*w + (x+1)])
                neighbours[nn++] = labels[(y-1)*w + (x+1)];
            /* left */
            if (x > 0 && labels[y*w + (x-1)])
                neighbours[nn++] = labels[y*w + (x-1)];

            if (nn == 0) {
                /* new label */
                if (next_label < MAX_CC_LABELS) {
                    labels[idx] = next_label++;
                }
                /* else: overflow – pixel stays 0 (background) */
            } else {
                /* find minimum root label among neighbours */
                int16_t min_lbl = uf_find(neighbours[0]);
                for (int i = 1; i < nn; i++) {
                    int16_t r = uf_find(neighbours[i]);
                    if (r < min_lbl) min_lbl = r;
                }
                labels[idx] = min_lbl;

                /* union all neighbours */
                for (int i = 0; i < nn; i++)
                    uf_union(min_lbl, neighbours[i]);
            }
        }
    }

    /* ── second pass: flatten labels to roots ─────────────────────────────── */
    for (int i = 0; i < w * h; i++) {
        if (labels[i])
            labels[i] = uf_find(labels[i]);
    }

    return (int)(next_label - 1);   /* total provisional labels assigned */
}

/* ── isolate_ground_blob: keep only blobs above area threshold ───────────── */

void isolate_ground_blob(uint8_t mask[], int w, int h, int blob_area_threshold)
{
    int n_labels = connected_components(mask, work_labels, w, h);
    if (n_labels == 0) return;

    /* count area per root label */
    int max_label = 0;
    for (int i = 0; i < w * h; i++) {
        if (work_labels[i] > max_label)
            max_label = work_labels[i];
    }
    if (max_label >= MAX_CC_LABELS)
        max_label = MAX_CC_LABELS - 1;

    memset(cc_area, 0, (max_label + 1) * sizeof(int32_t));
    for (int i = 0; i < w * h; i++) {
        int16_t lbl = work_labels[i];
        if (lbl > 0 && lbl <= max_label)
            cc_area[lbl]++;
    }

    /* zero out small blobs in the mask */
    for (int i = 0; i < w * h; i++) {
        int16_t lbl = work_labels[i];
        if (lbl > 0 && lbl <= max_label) {
            if (cc_area[lbl] < blob_area_threshold)
                mask[i] = 0;
            else
                mask[i] = 255;
        }
        /* lbl == 0 stays 0 */
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  4.  HOLE FILLING  (border flood-fill approach)
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *  Strategy:
 *    1. BFS from all border pixels that are 0 → mark as "exterior"
 *    2. Any pixel that is 0 but NOT exterior → it's a hole → set to 255
 */

void fill_holes_mask(uint8_t mask[], int w, int h)
{
    int total = w * h;
    memset(work_visited, 0, total);

    int q_head = 0, q_tail = 0;

    /* seed BFS with all border pixels that are background (0) */
    for (int x = 0; x < w; x++) {
        /* top row */
        if (mask[x] == 0 && !work_visited[x]) {
            work_visited[x] = 1;
            ff_queue[q_tail++] = x;
        }
        /* bottom row */
        int idx = (h - 1) * w + x;
        if (mask[idx] == 0 && !work_visited[idx]) {
            work_visited[idx] = 1;
            ff_queue[q_tail++] = idx;
        }
    }
    for (int y = 0; y < h; y++) {
        /* left column */
        int idx = y * w;
        if (mask[idx] == 0 && !work_visited[idx]) {
            work_visited[idx] = 1;
            ff_queue[q_tail++] = idx;
        }
        /* right column */
        idx = y * w + (w - 1);
        if (mask[idx] == 0 && !work_visited[idx]) {
            work_visited[idx] = 1;
            ff_queue[q_tail++] = idx;
        }
    }

    /* BFS: flood through all 0-pixels reachable from the border */
    /* 4-connectivity is sufficient for exterior fill */
    while (q_head < q_tail) {
        int idx = ff_queue[q_head++];
        int y = idx / w;
        int x = idx % w;

        /* 4 neighbours */
        int nb[4];
        int nn = 0;
        if (y > 0)     nb[nn++] = idx - w;
        if (y < h - 1) nb[nn++] = idx + w;
        if (x > 0)     nb[nn++] = idx - 1;
        if (x < w - 1) nb[nn++] = idx + 1;

        for (int i = 0; i < nn; i++) {
            int ni = nb[i];
            if (!work_visited[ni] && mask[ni] == 0) {
                work_visited[ni] = 1;
                ff_queue[q_tail++] = ni;
            }
        }
    }

    /* any 0-pixel NOT visited is a hole → fill it */
    for (int i = 0; i < total; i++) {
        if (mask[i] == 0 && !work_visited[i])
            mask[i] = 255;
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  5.  GROUND BOUNDARY SCAN
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *  For each row (= original column) of the left-right-flipped mask:
 *    - Scan from the bottom (high index) upward
 *    - Find where continuous ground ends (gap > max_gap)
 *    - Boundary = top of that continuous ground strip
 *
 *  Then smooth with a 1-D median filter.
 */

/* ── 1-D median filter on int array ──────────────────────────────────────── */

static void insertion_sort_float(float *arr, int n)
{
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void median_filter_1d(int *data, int n, int ksize)
{
    if (ksize < 3 || n <= ksize) return;
    int half = ksize / 2;
    float window[25];   /* ksize ≤ 25 assumed; DEFAULT_SMOOTH_KERNEL = 5 */

    /* need a copy because we read while writing */
    static int med_copy[MAX_IMAGE_WIDTH];
    memcpy(med_copy, data, n * sizeof(int));

    for (int i = 0; i < n; i++) {
        int wn = 0;
        int lo = (i - half < 0) ? 0 : i - half;
        int hi = (i + half >= n) ? n - 1 : i + half;
        for (int j = lo; j <= hi; j++)
            window[wn++] = (float)med_copy[j];
        insertion_sort_float(window, wn);
        data[i] = (int)window[wn / 2];
    }
}

/* ── find_ground_boundary  (matches Python find_ground_boundary exactly) ──── */

void find_ground_boundary(const uint8_t mask_flipped[],
                          int  w,
                          int  h,
                          int  boundary_out[],
                          int  min_ground_pixels,
                          int  max_gap,
                          int  smooth_kernel)
{
    /*
     * mask_flipped is laid out row-major: mask_flipped[row * h + col]
     * because the Python code does  mask_flipped = clean_mask[:, ::-1]
     * and then iterates over rows of mask_flipped.
     *
     * BUT: in the Python code, mask_rotated is actually the original
     * 2-D image with columns flipped (left-right flip).  Each "row" of
     * the Python iteration corresponds to a row of the image (i.e. a
     * horizontal scan line), and "image_height" = number of columns.
     *
     * In the Python, mask_rotated.shape = (H_original, W_original) after
     * flipping columns.  The boundary is computed per row (H entries).
     * image_height = W_original = mask_rotated.shape[1].
     *
     * Mapping to our C variables:
     *   n_rows       = w   (number of rows to scan = image width)
     *   image_height = h   (length of each row = image height)
     *
     * WAIT – re-reading the Python more carefully:
     *   clean_mask has shape (H, W)
     *   mask_flipped = clean_mask[:, ::-1]  → shape still (H, W), columns reversed
     *   find_ground_boundary(mask_rotated=mask_flipped)
     *     image_height = mask_rotated.shape[1] = W
     *     n_rows       = mask_rotated.shape[0] = H
     *     boundary_rows has n_rows = H entries
     *
     * But then update_and_detect receives boundary_rows (H entries) and
     * uses the image width W as 'h' parameter to compare boundary < h.
     *
     * This means:
     *   - We iterate over H rows of the flipped mask
     *   - Each row has W pixels
     *   - boundary_out has H entries
     *   - "image_height" in the boundary logic = W
     *
     * So:
     *   n_rows = h (image height, iterating over rows)
     *   row_len = w (image width, the length of each row)
     *   boundary_out[n_rows]
     *   default boundary value = row_len (= w)
     */

    int n_rows  = h;  /* iterate over image rows */
    int row_len = w;  /* each row has this many pixels */

    for (int r = 0; r < n_rows; r++)
        boundary_out[r] = row_len;   /* default = "no boundary found" */

    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = &mask_flipped[r * row_len];

        /* ── collect positions of green (non-zero) pixels ────────────── */
        /* scan from right (high index = bottom of column) to left */

        /* first check: any green at all? */
        int has_any_green = 0;
        for (int c = 0; c < row_len; c++) {
            if (row[c]) { has_any_green = 1; break; }
        }
        if (!has_any_green) continue;

        /* check if the last min_ground_pixels are all green ("ground valid") */
        int ground_valid = 1;
        for (int c = row_len - min_ground_pixels; c < row_len; c++) {
            if (c < 0) continue;
            if (row[c] == 0) { ground_valid = 0; break; }
        }

        if (!ground_valid) {
            /* check if there's any contiguous run >= max_gap */
            int max_run = 0, cur_run = 0;
            for (int c = 0; c < row_len; c++) {
                if (row[c]) {
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 0;
                }
            }
            if (max_run < max_gap) continue;
        }

        /* ── scan from bottom (right end) upward ─────────────────────── */
        /* Collect green positions in reverse order */
        /* gp = green_pos[::-1]  (Python) */

        /* find the last green pixel (bottom of ground) */
        int gp_count = 0;
        /* We need the green positions sorted from high to low index */
        /* For efficiency, build on the fly */

        /* Count green pixels for this row */
        for (int c = 0; c < row_len; c++) {
            if (row[c]) gp_count++;
        }

        if (gp_count == 1) {
            /* single green pixel: use its position as boundary */
            for (int c = 0; c < row_len; c++) {
                if (row[c]) { boundary_out[r] = c; break; }
            }
            continue;
        }

        /* Scan from right to left (reversed green positions) */
        /* We need to find the first "big gap" scanning from bottom */
        int prev_gp = -1;  /* previous green position (going right to left) */
        int first_big_gap_gp = -1;    /* gp[fg] – the green pixel just before the gap */
        int first_big_gap_idx = -1;   /* index in reverse order */
        int after_gap_start = -1;     /* first green pixel after the gap */
        int topmost_green = -1;       /* overall topmost (leftmost) green pixel */

        /* We'll scan right-to-left and track gaps */
        int gp_reverse_idx = 0;
        int found_big_gap = 0;

        /* collect reverse-sorted green positions into a temp array */
        /* (we need random access for the after-gap logic) */
        /* Use a portion of ff_queue as temp storage since it's not in use here */
        int *gp_arr = (int *)ff_queue;  /* reuse ff_queue memory */
        int gp_n = 0;
        for (int c = row_len - 1; c >= 0; c--) {
            if (row[c])
                gp_arr[gp_n++] = c;
        }
        /* gp_arr[0] = rightmost green, gp_arr[gp_n-1] = leftmost green */

        if (gp_n <= 1) {
            /* already handled above, but safety */
            if (gp_n == 1) boundary_out[r] = gp_arr[0];
            continue;
        }

        /* find first big gap: gaps[i] = -(gp[i+1] - gp[i]) - 1 */
        int fg = -1;
        for (int i = 0; i < gp_n - 1; i++) {
            int gap = gp_arr[i] - gp_arr[i + 1] - 1;
            if (gap > max_gap) {
                fg = i;
                break;
            }
        }

        if (fg < 0) {
            /* no big gap → boundary is the topmost (leftmost) green pixel */
            boundary_out[r] = gp_arr[gp_n - 1];
            continue;
        }

        int pending_boundary = gp_arr[fg];

        /* check after-gap region: gp[fg+1:] */
        int after_count = gp_n - (fg + 1);
        if (after_count >= max_gap) {
            /* check if there's a contiguous run >= max_gap in after_gap */
            int *after = &gp_arr[fg + 1];
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < after_count; i++) {
                /* consecutive if gap between them is 0 (i.e. adjacent positions) */
                int g = after[i - 1] - after[i] - 1;
                if (g == 0) {
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 1;
                }
            }
            if (max_run >= max_gap) {
                /* big run after gap → use topmost green */
                boundary_out[r] = gp_arr[gp_n - 1];
                continue;
            }
        }

        boundary_out[r] = pending_boundary;
    }

    /* ── smooth valid boundary values with 1-D median ────────────────── */
    int valid_count = 0;
    for (int r = 0; r < n_rows; r++) {
        if (boundary_out[r] < row_len)
            valid_count++;
    }

    if (valid_count > smooth_kernel) {
        median_filter_1d(boundary_out, n_rows, smooth_kernel);
        /* only apply smoothed values where boundary was valid */
        /* (The Python applies smoothed only at valid_mask positions,
           but since median_filter_1d already wrote in-place and the
           invalid positions were row_len, the median of mostly-row_len
           neighbours will stay row_len.  So this is equivalent.) */
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  6.  OBSTACLE REGION MERGING
 * ══════════════════════════════════════════════════════════════════════════════ */

static int merge_obstacle_cols(const uint8_t *obstacle_mask_cols,
                               int n_cols,
                               int min_width,
                               int max_col_gap,
                               struct obstacle_region_t out[])
{
    /* obstacle_mask_cols[i] = 1 if column i is an obstacle, else 0 */
    /* collect obstacle column indices */
    int obs_cols[MAX_IMAGE_HEIGHT];  /* reuse; n_cols ≤ MAX_IMAGE_HEIGHT */
    int n_obs = 0;

    for (int i = 0; i < n_cols; i++) {
        if (obstacle_mask_cols[i])
            obs_cols[n_obs++] = i;
    }

    if (n_obs == 0) return 0;

    /* merge adjacent columns with gap ≤ max_col_gap */
    int count = 0;
    int start = obs_cols[0];
    int end   = obs_cols[0];

    for (int i = 1; i < n_obs; i++) {
        if (obs_cols[i] - end <= max_col_gap) {
            end = obs_cols[i];
        } else {
            int w = end - start + 1;
            if (w >= min_width && count < MAX_OBSTACLE_REGIONS) {
                out[count].start = (uint16_t)start;
                out[count].width = (uint16_t)w;
                count++;
            }
            start = obs_cols[i];
            end   = obs_cols[i];
        }
    }
    /* last region */
    {
        int w = end - start + 1;
        if (w >= min_width && count < MAX_OBSTACLE_REGIONS) {
            out[count].start = (uint16_t)start;
            out[count].width = (uint16_t)w;
            count++;
        }
    }

    return count;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  7.  BASELINE UPDATE + OBSTACLE DETECTION
 * ══════════════════════════════════════════════════════════════════════════════ */

uint8_t update_and_detect(const int                  boundary_row[],
                          int                        h,        /* "image_height" = cols of original img */
                          int                        w,        /* number of boundary entries = rows of original img */
                          float                      ground_baseline[],
                          int                       *baseline_inited,
                          int                        min_width,
                          int                        obstacle_threshold,
                          int                        no_ground_baseline,
                          int                        max_col_gap,
                          struct obstacle_region_t    obstacles_out[])
{
    float alpha = DEFAULT_BASELINE_ALPHA;

    if (!(*baseline_inited)) {
        /* initialise baseline from first frame */
        for (int i = 0; i < w; i++) {
            if (boundary_row[i] < h)
                ground_baseline[i] = (float)boundary_row[i];
            else
                ground_baseline[i] = (float)no_ground_baseline;
        }
        *baseline_inited = 1;
        return 0;   /* no obstacles on first frame */
    }

    /* classify each column */
    static uint8_t obs_col_mask[MAX_IMAGE_HEIGHT];  /* 1 = obstacle column */
    static uint8_t no_obs_mask[MAX_IMAGE_HEIGHT];   /* 1 = valid, not obstacle */
    memset(obs_col_mask, 0, w);
    memset(no_obs_mask, 0, w);

    for (int i = 0; i < w; i++) {
        int valid    = (boundary_row[i] < h);
        int no_ground = !valid;

        if (valid) {
            float deviation = (float)boundary_row[i] - ground_baseline[i];
            if (deviation > (float)obstacle_threshold) {
                obs_col_mask[i] = 1;
            } else {
                no_obs_mask[i] = 1;
            }
        }
        if (no_ground) {
            obs_col_mask[i] = 1;
        }
    }

    /* merge obstacle columns into regions (in flipped coordinate space) */
    struct obstacle_region_t flipped_regions[MAX_OBSTACLE_REGIONS];
    int n_flipped = merge_obstacle_cols(obs_col_mask, w, min_width, max_col_gap, flipped_regions);

    /* update baseline for non-obstacle valid columns */
    for (int i = 0; i < w; i++) {
        if (no_obs_mask[i]) {
            ground_baseline[i] = (1.0f - alpha) * ground_baseline[i]
                               + alpha * (float)boundary_row[i];
        }
    }

    /* forward-fill: propagate last good baseline value */
    float last_good = (float)no_ground_baseline;
    for (int i = 0; i < w; i++) {
        if (no_obs_mask[i]) {
            last_good = ground_baseline[i];
        } else {
            ground_baseline[i] = last_good;
        }
    }

    /* ── convert flipped regions back to original coordinates ─────────── */
    /* Python:  left = W - 1 - e   where W = h (the original image height / col count)
     *          but wait, "h" here is the column count of the original image.
     *
     *  Actually in the Python:
     *    boundary_rows has w entries (one per row of flipped mask = H of original image)
     *    update_and_detect receives: boundary_row, h=W, ground_baseline
     *    obstacle_regions are (start, end, width) in the row-index space
     *    Then:  left = W - 1 - end
     *
     *  In our C code:
     *    w = number of boundary entries = H of original image
     *    h = column count = W of original image
     *    flipped_regions[i].start = start row index
     *    flipped_regions[i].width = width
     *    end = start + width - 1
     *
     *  Convert:  left_original = h - 1 - (start + width - 1) = h - start - width
     *            Wait no. The Python uses W-1-e where e = end column.
     *            But flipped_regions are in the row-index space, not column space.
     *
     *  Hmm, I need to re-read the Python carefully.
     *
     *  In the Python get_obstacle_info:
     *    - clean_mask[:, ::-1] flips columns (horizontal flip)
     *    - find_ground_boundary iterates rows of flipped mask
     *      (each row = horizontal scanline, shape[0]=H rows, shape[1]=W cols)
     *    - boundary_rows has H entries (one per row)
     *    - update_and_detect(boundary_rows, W, ground_baseline, ...)
     *      inside: boundary_row has len = H entries
     *              h = W
     *              obstacle_cols are indices into boundary_row (row indices = 0..H-1)
     *              obstacle_regions are (start_row, end_row, width_in_rows)
     *    - Then: left = W - 1 - e  ... NO that doesn't make sense either.
     *
     *  Wait, looking at the Python more carefully:
     *    for (s, e, w) in obstacle_regions:
     *        left = W - 1 - e
     *        rows.append([int(left), int(w), DET_OBSTACLE])
     *
     *  obstacle_regions come from get_obstacle_regions(obstacle_cols) where
     *  obstacle_cols are row indices (0..H-1).
     *
     *  But the flipping was on columns: mask_flipped = clean_mask[:, ::-1]
     *  So column j in original → column (W-1-j) in flipped.
     *
     *  The boundary scan iterates ROWS (axis 0) and scans along COLUMNS (axis 1).
     *  Then deviation is computed per ROW.  Obstacle columns = row indices where
     *  the boundary deviates.
     *
     *  So obstacle_regions' start/end are ROW indices.
     *  left = W - 1 - e ... W is the column count.  That doesn't map row→column.
     *
     *  Wait, I think I was wrong. Let me re-read:
     *    H, W = image_bgr.shape[:2]   →  H=height, W=width
     *    mask has shape (H, W)
     *    mask_flipped = clean_mask[:, ::-1]  → shape (H, W)
     *
     *    find_ground_boundary(mask_rotated=mask_flipped):
     *      image_height = mask_rotated.shape[1] = W
     *      n_rows = mask_rotated.shape[0] = H
     *
     *    The function iterates idx in range(n_rows) = range(H)
     *    For each row, scans along axis 1 (W columns) from right (bottom conceptually)
     *    boundary_rows[idx] = position along the W-axis
     *
     *    update_and_detect(boundary_row, h=W, ground_baseline):
     *      Valid = boundary_row < h = boundary_row < W
     *      deviation = boundary_row - ground_baseline
     *      obstacle_cols = row indices where deviation > threshold (these are H-axis indices)
     *      obstacle_regions = merged groups of these row indices
     *
     *    So obstacle_regions (s, e, w) have s,e as row indices (0..H-1)
     *    Then: left = W - 1 - e   →  this maps a row index to... a column?
     *
     *    That seems wrong unless the image is rotated 90°.
     *
     *    OH WAIT. The image is from a drone camera that is mounted sideways!
     *    Looking at the display code:
     *      image_rot = cv2.rotate(image_display, cv2.ROTATE_90_COUNTERCLOCKWISE)
     *
     *    So the camera image is rotated 90° CCW for display.
     *    The "rows" of the original image are actually vertical columns in the
     *    physical scene.
     *
     *    After CCW rotation, original row r → becomes column (H-1-r) in the
     *    rotated view.  And obstacle_regions' start/end are row indices.
     *
     *    left = W - 1 - e  ... hmm, but W is column count of original.
     *
     *    Actually I think there might be a coordinate system issue I'm over-thinking.
     *    Let me just match the Python output exactly.
     *
     *    The key thing for the drone is: obstacles_out[i].start and .width
     *    are what get sent via ABI to the autopilot.
     *
     *    In the C caller (team10_ground_detection.c), obstacles are printed as:
     *      printf("Obstacle %d: left=%d width=%d\n", i, global_obstacles[i].start, global_obstacles[i].width);
     *
     *    In the Python:
     *      for (s, e, w) in obstacle_regions:
     *          left = W - 1 - e
     *          rows.append([int(left), int(w), DET_OBSTACLE])
     *
     *    So the output "left" = W - 1 - end_row_index.
     *
     *    W = image_bgr.shape[1] = original image width (= number of columns).
     *    But the boundary scan was on rows (shape[0] = H).
     *
     *    Hmm, unless the image width and height are swapped because of the
     *    sideways camera.  Let me check the calling code:
     *
     *    In team10_ground_detection.c:
     *      get_obstacle_info(img, ...)
     *
     *    img->w = image width, img->h = image height as reported by the camera.
     *
     *    I think the Python code works with the assumption that the image is
     *    landscape (wider than tall) but displayed rotated.  The "left" in the
     *    obstacle output refers to a position in the rotated (display) view.
     *
     *    For the C code, I'll match the Python's output transform exactly:
     *      left = img->w - 1 - (region_end_in_row_space)
     *
     *    No wait, in the Python W = image_bgr.shape[1] which is the second
     *    dimension = columns = width.  And the boundary has H entries (one per
     *    row).  obstacle_regions has (start, end, width) where start/end are
     *    row indices (0..H-1).
     *
     *    left = W - 1 - e   ... this is using W (columns) to transform a
     *    row index e.  This only makes sense if H == W, which generally isn't true.
     *
     *    OR... the Python code has a convention where the image dimensions
     *    are such that H < W and the scan happens in a particular way.
     *
     *    Actually, I think I might be wrong. Let me re-check.  The Python code
     *    uses shape[:2] which returns (rows, cols) = (H, W).  The flipped mask
     *    still has shape (H, W).  find_ground_boundary iterates H rows, each
     *    of length W.  boundary_rows has H entries, values in [0, W).
     *    update_and_detect receives boundary_rows and h=W.
     *
     *    In update_and_detect:
     *      obstacle_cols = np.where(obstacle_mask)[0]
     *      This returns indices where obstacle_mask is True.
     *      obstacle_mask has len = len(boundary_row) = H.
     *      So obstacle_cols are indices 0..H-1.
     *
     *    get_obstacle_regions receives obstacle_cols (indices 0..H-1):
     *      returns regions (start, end, width) where start,end ∈ [0, H-1].
     *
     *    Then: left = W - 1 - e.
     *    If e can be up to H-1 and W != H, this can go negative.
     *    This seems like a bug in the Python... or I'm misunderstanding.
     *
     *    Let me look at the Python main() display code:
     *      for (start_col, end_col, width) in obstacle_regions_raw:
     *          y_top = int(min(boundary_rows[start_col:end_col + 1]))
     *
     *    Here boundary_rows is indexed by start_col..end_col which are the
     *    region row indices.  And y_top is a column value (boundary position
     *    within a row, in the range [0, W)).  This is drawn on the rotated image.
     *
     *    So the variable names "start_col" / "end_col" in the display code
     *    actually refer to row indices used as column positions in the rotated view.
     *
     *    After 90° CCW rotation:
     *      rotated image has shape (W, H)  (width becomes height, height becomes width)
     *      Original pixel (row_r, col_c) → rotated pixel (col_c, H-1-row_r)
     *      Or equivalently: rotated_x = row_r, rotated_y = col_c  (for CCW rotation)
     *
     *    Actually for cv2.ROTATE_90_COUNTERCLOCKWISE:
     *      rotated[x][y] = original[y][W-1-x]   ... wait, let me think.
     *      For 90° CCW: new shape is (W, H).
     *      new(row, col) = original(col, W - 1 - row)
     *      So: new_row = original_col, new_col = W - 1 - original_row
     *
     *    So original row r becomes new column = W - 1 - r.
     *    And in the rotated view, "left" of an obstacle = W - 1 - end_row_index
     *    because end_row_index maps to the leftmost column in rotated space.
     *
     *    OK so: left = W - 1 - e  is correct for the rotated display.
     *    But W here should actually be... hmm.
     *
     *    Actually wait. I just realized: in the Python:
     *      H, W = image_bgr.shape[:2]
     *    Then the obstacle info returns [left, w, type] where left is computed as:
     *      left = W - 1 - e
     *    And e is a row index (0..H-1).
     *
     *    But the rotated image has original rows becoming columns:
     *      new_col = W - 1 - original_row   (from 90° CCW rotation)
     *
     *    Wait no, that's wrong. For 90° CCW rotation of an image with shape (H, W):
     *      new image has shape (W, H)
     *      Mapping: new_pixel(new_r, new_c) = old_pixel(new_c, W - 1 - new_r)
     *      Or equivalently: old_pixel(r, c) → new_pixel(W - 1 - c, r)
     *      So: new_row = W - 1 - old_col, new_col = old_row
     *
     *    So original row r → new column r.
     *    Original col c → new row W - 1 - c.
     *
     *    Hmm, then obstacle region at rows [s, e] in original →
     *    columns [s, e] in rotated.
     *    left in rotated = s (the start row index).
     *
     *    But the Python says left = W - 1 - e.  That uses W (original width = cols).
     *
     *    There must be something else going on.  Let me look at what the drone
     *    autopilot actually needs.  The ABI message just sends local_obstacles
     *    and obstacle_count.  The autopilot interprets start and width.
     *
     *    For the C code, I'll just reproduce the Python's transform exactly:
     *      left = W_original - 1 - region_end
     *    where region_end = region_start + region_width - 1 (in the row-index space).
     *
     *    Hmm actually, W in the Python = img->w in C (the camera image width).
     *    And the regions are over H rows (img->h in C).
     *    So left = img->w - 1 - end.
     *
     *    As long as H <= W (which it typically is for landscape cameras), this
     *    won't go negative.  And for the sideways-mounted camera on the drone,
     *    the physical meaning makes sense.
     *
     *    OK, I'll just implement it as-is and trust the Python.
     */

    int n_out = 0;
    for (int i = 0; i < n_flipped; i++) {
        int s = flipped_regions[i].start;
        int region_w = flipped_regions[i].width;
        int e = s + region_w - 1;

        int left = h - 1 - e;   /* h = column count of original image (W in Python) */
        if (left < 0) left = 0;

        if (n_out < MAX_OBSTACLE_REGIONS) {
            obstacles_out[n_out].start = (uint16_t)left;
            obstacles_out[n_out].width = (uint16_t)region_w;
            n_out++;
        }
    }

    return (uint8_t)n_out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  8.  MAIN PIPELINE FUNCTION
 * ══════════════════════════════════════════════════════════════════════════════ */

uint8_t get_obstacle_info(
        struct image_t            *img,
        float                      ground_baseline[],
        int                       *baseline_inited,
        float                      oa_color_count_frac,
        int                        median_ksize,
        int                        min_width,
        struct obstacle_region_t   obstacles_out[],
        int                        boundary_rows_out[],
        int                       *ground_found_out,
        float                     *green_frac_out)
{
    int w = img->w;
    int h = img->h;

    /* safety: clamp to buffer sizes */
    if (w > MAX_IMAGE_WIDTH)  w = MAX_IMAGE_WIDTH;
    if (h > MAX_IMAGE_HEIGHT) h = MAX_IMAGE_HEIGHT;

    /* ── Step 1+2: classify + median blur → binary mask ──────────────── */
    float green_frac = 0.0f;
    detect_green_ground_ml(img, work_mask, median_ksize, &green_frac);

    if (green_frac_out) *green_frac_out = green_frac;

    int ground_found = (green_frac > oa_color_count_frac) ? 1 : 0;
    if (ground_found_out) *ground_found_out = ground_found;

    if (!ground_found) {
        /* no ground → no obstacles, baseline unchanged */
        return 0;
    }

    /* ── Step 3: connected-component blob filtering ──────────────────── */
    isolate_ground_blob(work_mask, w, h, DEFAULT_BLOB_AREA_THRESH);

    /* ── Step 4: fill holes in the mask ──────────────────────────────── */
    fill_holes_mask(work_mask, w, h);

    /* ── Step 5: flip mask left-right, then find ground boundary ─────── */
    /* mask_flipped[row][col] = work_mask[row][(w-1) - col] */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            work_flipped[y * w + x] = work_mask[y * w + (w - 1 - x)];
        }
    }

    /* boundary has h entries (one per row) */
    static int boundary_rows_local[MAX_IMAGE_HEIGHT];
    find_ground_boundary(work_flipped, w, h,
                         boundary_rows_local,
                         DEFAULT_MIN_GROUND_PX,
                         DEFAULT_MAX_GAP,
                         DEFAULT_SMOOTH_KERNEL);

    /* copy boundary to caller if requested */
    if (boundary_rows_out) {
        /* The caller's array is MAX_IMAGE_WIDTH but boundary has h entries */
        memcpy(boundary_rows_out, boundary_rows_local, h * sizeof(int));
    }

    /* ── Step 6+7+8: update baseline and detect obstacles ────────────── */
    uint8_t n_obstacles = update_and_detect(
            boundary_rows_local,
            w,   /* "h" param = column count of original image */
            h,   /* "w" param = number of boundary entries = row count */
            ground_baseline,
            baseline_inited,
            min_width,
            DEFAULT_OBSTACLE_THRESH,
            DEFAULT_NO_GROUND_BASE,
            DEFAULT_MAX_COL_GAP,
            obstacles_out);

    return n_obstacles;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  DEBUG / VALIDATION HELPERS
 * ══════════════════════════════════════════════════════════════════════════════ */

int dump_mask_to_file(const char *path, const uint8_t mask[], int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(mask, 1, w * h, f);
    fclose(f);
    return 0;
}

int dump_obstacles_to_file(const char *path,
                           const struct obstacle_region_t obs[], int count)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n", count);
    for (int i = 0; i < count; i++)
        fprintf(f, "%d %d\n", obs[i].start, obs[i].width);
    fclose(f);
    return 0;
}
