/*
 * team10_get_obstacle_info_standalone.c
 *
 * Standalone compilable wrapper.
 * Provides minimal stubs for the Paparazzi types so the real
 * implementation can compile and run on any Linux/macOS machine
 * for testing and validation.
 *
 * BUILD:
 *   gcc -O2 -Wall -c team10_get_obstacle_info_standalone.c -o team10_get_obstacle_info_standalone.o
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

/* ── Minimal type stubs matching Paparazzi's std.h / state.h / image.h ──── */

#ifndef _CV_LIB_VISION_IMAGE_H
#define _CV_LIB_VISION_IMAGE_H

struct FloatEulers { float phi, theta, psi; };

enum image_type {
    IMAGE_YUV422,
    IMAGE_GRAYSCALE,
    IMAGE_JPEG,
    IMAGE_GRADIENT,
    IMAGE_INT16
};

struct image_t {
    enum image_type type;
    uint16_t w;
    uint16_t h;
    struct timeval ts;
    struct FloatEulers eulers;
    uint32_t pprz_ts;
    uint8_t  buf_idx;
    uint32_t buf_size;
    void    *buf;
};

#endif /* _CV_LIB_VISION_IMAGE_H */

/* ── Include the REAL implementation verbatim ─────────────────────────────── */
/* We temporarily redefine the #include for image.h so it doesn't try to
   find the Paparazzi header.  Since we already defined everything above,
   the include guard will prevent double-definition. */

/* Override the Paparazzi-specific include path */
#define modules_computer_vision_lib_vision_image_h_ALREADY_DEFINED

/*
 * We inline the implementation directly.
 * The real .c file does:
 *   #include "team10_get_obstacle_info.h"
 *   #include "modules/computer_vision/lib/vision/image.h"
 *
 * Since image.h is already defined above, we just need to handle
 * the include.  We'll copy the implementation with a sed-like approach
 * in the build script, OR we can use the preprocessor trick below.
 */

/* Rather than fighting includes, we just paste the core implementation.
   This file IS the standalone build unit. */

#include "team10_get_obstacle_info.h"

/* ══════════════════════════════════════════════════════════════════════════════
 *  All code below is identical to team10_get_obstacle_info.c
 *  (copy-paste for standalone compilation).
 * ══════════════════════════════════════════════════════════════════════════════ */

/* --- STATIC WORK BUFFERS --- */
#define MAX_PIXELS (MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT)

static uint8_t  work_mask[MAX_PIXELS];
static uint8_t  work_mask2[MAX_PIXELS];
static int16_t  work_labels[MAX_PIXELS];
static uint8_t  work_visited[MAX_PIXELS];
static uint8_t  work_flipped[MAX_PIXELS];
static int32_t  ff_queue[MAX_PIXELS];
static int16_t  uf_parent[MAX_CC_LABELS];
static int32_t  cc_area[MAX_CC_LABELS];
/* (med_tmp not needed in standalone – 1-D median uses local window array) */

/* --- YUV422 PIXEL ACCESS --- */
static inline uint8_t yuv422_Y(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + x * 2 + 1]; }
static inline uint8_t yuv422_U(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + (x & ~1) * 2]; }
static inline uint8_t yuv422_V(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + (x & ~1) * 2 + 2]; }

/* --- DECISION TREE --- */
uint8_t is_ground_pixel(uint8_t Y, uint8_t U, uint8_t V)
{
    if (U <= 115) {
        if (V <= 145) {
            if (Y <= 85) {
                return 0;
            } else {
                return (U <= 92) ? 0 : 255;
            }
        } else {
            if (V <= 152) {
                return (Y <= 177) ? 255 : 0;
            } else {
                return 0;
            }
        }
    } else {
        if (U <= 121) {
            if (V <= 137) {
                return (Y <= 87) ? 0 : 255;
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }
}

/* --- MEDIAN BLUR (binary-optimised) --- */
static void median_blur_binary(const uint8_t *src, uint8_t *dst,
                               int w, int h, int ksize)
{
    int half = ksize / 2;
    int threshold = (ksize * ksize) / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int count = 0;
            int y0 = (y - half < 0) ? 0 : y - half;
            int y1 = (y + half >= h) ? h - 1 : y + half;
            int x0 = (x - half < 0) ? 0 : x - half;
            int x1 = (x + half >= w) ? w - 1 : x + half;
            for (int ky = y0; ky <= y1; ky++)
                for (int kx = x0; kx <= x1; kx++)
                    if (src[ky * w + kx]) count++;
            dst[y * w + x] = (count > threshold) ? 255 : 0;
        }
    }
}

/* --- GREEN DETECTION --- */
void detect_green_ground_ml(struct image_t *img, uint8_t mask_out[],
                            int median_ksize, float *green_frac_out)
{
    int w = img->w, h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            work_mask2[y * w + x] = is_ground_pixel(
                yuv422_Y(buf, w, x, y),
                yuv422_U(buf, w, x, y),
                yuv422_V(buf, w, x, y));

    if (median_ksize >= 3 && (median_ksize & 1))
        median_blur_binary(work_mask2, mask_out, w, h, median_ksize);
    else
        memcpy(mask_out, work_mask2, w * h);

    int gc = 0, total = w * h;
    for (int i = 0; i < total; i++) if (mask_out[i]) gc++;
    if (green_frac_out) *green_frac_out = total > 0 ? (float)gc / total : 0.0f;
}

/* --- UNION-FIND --- */
static void uf_init(int n)
{ for (int i = 0; i < n; i++) uf_parent[i] = (int16_t)i; }

static int16_t uf_find(int16_t x)
{
    while (uf_parent[x] != x) { uf_parent[x] = uf_parent[uf_parent[x]]; x = uf_parent[x]; }
    return x;
}

static void uf_union(int16_t a, int16_t b)
{
    a = uf_find(a); b = uf_find(b);
    if (a != b) { if (a < b) uf_parent[b] = a; else uf_parent[a] = b; }
}

/* --- CONNECTED COMPONENTS --- */
static int connected_components(const uint8_t *mask, int16_t *labels, int w, int h)
{
    int16_t next_label = 1;
    uf_init(MAX_CC_LABELS);
    memset(labels, 0, w * h * sizeof(int16_t));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (mask[idx] == 0) continue;
            int16_t nb[4]; int nn = 0;
            if (y > 0 && x > 0 && labels[(y-1)*w+(x-1)]) nb[nn++] = labels[(y-1)*w+(x-1)];
            if (y > 0 && labels[(y-1)*w+x])               nb[nn++] = labels[(y-1)*w+x];
            if (y > 0 && x < w-1 && labels[(y-1)*w+(x+1)]) nb[nn++] = labels[(y-1)*w+(x+1)];
            if (x > 0 && labels[y*w+(x-1)])                nb[nn++] = labels[y*w+(x-1)];

            if (nn == 0) {
                if (next_label < MAX_CC_LABELS) labels[idx] = next_label++;
            } else {
                int16_t min_lbl = uf_find(nb[0]);
                for (int i = 1; i < nn; i++) { int16_t r = uf_find(nb[i]); if (r < min_lbl) min_lbl = r; }
                labels[idx] = min_lbl;
                for (int i = 0; i < nn; i++) uf_union(min_lbl, nb[i]);
            }
        }
    }
    for (int i = 0; i < w * h; i++) if (labels[i]) labels[i] = uf_find(labels[i]);
    return (int)(next_label - 1);
}

/* --- ISOLATE GROUND BLOB --- */
void isolate_ground_blob(uint8_t mask[], int w, int h, int blob_area_threshold)
{
    int n_labels = connected_components(mask, work_labels, w, h);
    if (n_labels == 0) return;
    int max_label = 0;
    for (int i = 0; i < w * h; i++) if (work_labels[i] > max_label) max_label = work_labels[i];
    if (max_label >= MAX_CC_LABELS) max_label = MAX_CC_LABELS - 1;
    memset(cc_area, 0, (max_label + 1) * sizeof(int32_t));
    for (int i = 0; i < w * h; i++) { int16_t l = work_labels[i]; if (l > 0 && l <= max_label) cc_area[l]++; }
    for (int i = 0; i < w * h; i++) {
        int16_t l = work_labels[i];
        if (l > 0 && l <= max_label) mask[i] = (cc_area[l] >= blob_area_threshold) ? 255 : 0;
    }
}

/* --- FILL HOLES --- */
void fill_holes_mask(uint8_t mask[], int w, int h)
{
    int total = w * h;
    memset(work_visited, 0, total);
    int q_head = 0, q_tail = 0;

    for (int x = 0; x < w; x++) {
        if (!mask[x] && !work_visited[x])                       { work_visited[x] = 1; ff_queue[q_tail++] = x; }
        int idx = (h-1)*w+x;
        if (!mask[idx] && !work_visited[idx])                   { work_visited[idx] = 1; ff_queue[q_tail++] = idx; }
    }
    for (int y = 0; y < h; y++) {
        int idx = y*w;
        if (!mask[idx] && !work_visited[idx])                   { work_visited[idx] = 1; ff_queue[q_tail++] = idx; }
        idx = y*w + (w-1);
        if (!mask[idx] && !work_visited[idx])                   { work_visited[idx] = 1; ff_queue[q_tail++] = idx; }
    }
    while (q_head < q_tail) {
        int idx = ff_queue[q_head++];
        int cy = idx / w, cx = idx % w;
        int nb[4]; int nn = 0;
        if (cy > 0)     nb[nn++] = idx - w;
        if (cy < h - 1) nb[nn++] = idx + w;
        if (cx > 0)     nb[nn++] = idx - 1;
        if (cx < w - 1) nb[nn++] = idx + 1;
        for (int i = 0; i < nn; i++) {
            int ni = nb[i];
            if (!work_visited[ni] && !mask[ni]) { work_visited[ni] = 1; ff_queue[q_tail++] = ni; }
        }
    }
    for (int i = 0; i < total; i++)
        if (!mask[i] && !work_visited[i]) mask[i] = 255;
}

/* --- 1-D MEDIAN FILTER --- */
static void insertion_sort_float(float *arr, int n)
{
    for (int i = 1; i < n; i++) {
        float key = arr[i]; int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

static void median_filter_1d(int *data, int n, int ksize)
{
    if (ksize < 3 || n <= ksize) return;
    int half = ksize / 2;
    float window[25];
    static int med_copy[MAX_IMAGE_WIDTH];
    memcpy(med_copy, data, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int wn = 0;
        int lo = (i - half < 0) ? 0 : i - half;
        int hi = (i + half >= n) ? n - 1 : i + half;
        for (int j = lo; j <= hi; j++) window[wn++] = (float)med_copy[j];
        insertion_sort_float(window, wn);
        data[i] = (int)window[wn / 2];
    }
}

/* --- FIND GROUND BOUNDARY --- */
void find_ground_boundary(const uint8_t mask_flipped[], int w, int h,
                          int boundary_out[], int min_ground_pixels,
                          int max_gap, int smooth_kernel)
{
    int n_rows = h, row_len = w;
    for (int r = 0; r < n_rows; r++) boundary_out[r] = row_len;

    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = &mask_flipped[r * row_len];

        int has_any = 0;
        for (int c = 0; c < row_len; c++) if (row[c]) { has_any = 1; break; }
        if (!has_any) continue;

        int ground_valid = 1;
        for (int c = row_len - min_ground_pixels; c < row_len; c++) {
            if (c < 0) continue;
            if (!row[c]) { ground_valid = 0; break; }
        }
        if (!ground_valid) {
            int max_run = 0, cur_run = 0;
            for (int c = 0; c < row_len; c++) {
                if (row[c]) { cur_run++; if (cur_run > max_run) max_run = cur_run; } else cur_run = 0;
            }
            if (max_run < max_gap) continue;
        }

        /* collect reverse-sorted green positions */
        int *gp_arr = (int *)ff_queue;
        int gp_n = 0;
        for (int c = row_len - 1; c >= 0; c--) if (row[c]) gp_arr[gp_n++] = c;

        if (gp_n <= 1) { if (gp_n == 1) boundary_out[r] = gp_arr[0]; continue; }

        int fg = -1;
        for (int i = 0; i < gp_n - 1; i++) {
            if (gp_arr[i] - gp_arr[i+1] - 1 > max_gap) { fg = i; break; }
        }
        if (fg < 0) { boundary_out[r] = gp_arr[gp_n - 1]; continue; }

        int pending = gp_arr[fg];
        int after_count = gp_n - (fg + 1);
        if (after_count >= max_gap) {
            int *after = &gp_arr[fg + 1];
            int mr = 1, cr = 1;
            for (int i = 1; i < after_count; i++) {
                if (after[i-1] - after[i] - 1 == 0) { cr++; if (cr > mr) mr = cr; } else cr = 1;
            }
            if (mr >= max_gap) { boundary_out[r] = gp_arr[gp_n - 1]; continue; }
        }
        boundary_out[r] = pending;
    }

    int valid_count = 0;
    for (int r = 0; r < n_rows; r++) if (boundary_out[r] < row_len) valid_count++;
    if (valid_count > smooth_kernel) median_filter_1d(boundary_out, n_rows, smooth_kernel);
}

/* --- OBSTACLE REGION MERGING --- */
static int merge_obstacle_cols(const uint8_t *obs_mask, int n_cols,
                               int min_width, int max_col_gap,
                               struct obstacle_region_t out[])
{
    int obs_cols[MAX_IMAGE_HEIGHT]; int n_obs = 0;
    for (int i = 0; i < n_cols; i++) if (obs_mask[i]) obs_cols[n_obs++] = i;
    if (n_obs == 0) return 0;

    int count = 0, start = obs_cols[0], end = obs_cols[0];
    for (int i = 1; i < n_obs; i++) {
        if (obs_cols[i] - end <= max_col_gap) { end = obs_cols[i]; }
        else {
            int w = end - start + 1;
            if (w >= min_width && count < MAX_OBSTACLE_REGIONS) { out[count].start = start; out[count].width = w; count++; }
            start = obs_cols[i]; end = obs_cols[i];
        }
    }
    { int w = end - start + 1; if (w >= min_width && count < MAX_OBSTACLE_REGIONS) { out[count].start = start; out[count].width = w; count++; } }
    return count;
}

/* --- UPDATE AND DETECT --- */
uint8_t update_and_detect(const int boundary_row[], int h, int w,
                          float ground_baseline[], int *baseline_inited,
                          int min_width, int obstacle_threshold,
                          int no_ground_baseline, int max_col_gap,
                          struct obstacle_region_t obstacles_out[])
{
    float alpha = DEFAULT_BASELINE_ALPHA;
    if (!(*baseline_inited)) {
        for (int i = 0; i < w; i++)
            ground_baseline[i] = (boundary_row[i] < h) ? (float)boundary_row[i] : (float)no_ground_baseline;
        *baseline_inited = 1;
        return 0;
    }

    static uint8_t obs_mask[MAX_IMAGE_HEIGHT];
    static uint8_t no_obs_mask[MAX_IMAGE_HEIGHT];
    memset(obs_mask, 0, w); memset(no_obs_mask, 0, w);

    for (int i = 0; i < w; i++) {
        int valid = boundary_row[i] < h;
        if (valid) {
            float dev = (float)boundary_row[i] - ground_baseline[i];
            if (dev > (float)obstacle_threshold) obs_mask[i] = 1;
            else no_obs_mask[i] = 1;
        } else {
            obs_mask[i] = 1;
        }
    }

    struct obstacle_region_t flipped[MAX_OBSTACLE_REGIONS];
    int n_flipped = merge_obstacle_cols(obs_mask, w, min_width, max_col_gap, flipped);

    for (int i = 0; i < w; i++)
        if (no_obs_mask[i])
            ground_baseline[i] = (1.0f - alpha) * ground_baseline[i] + alpha * (float)boundary_row[i];

    float last_good = (float)no_ground_baseline;
    for (int i = 0; i < w; i++) {
        if (no_obs_mask[i]) last_good = ground_baseline[i];
        else ground_baseline[i] = last_good;
    }

    int n_out = 0;
    for (int i = 0; i < n_flipped; i++) {
        int s = flipped[i].start, rw = flipped[i].width, e = s + rw - 1;
        int left = h - 1 - e;
        if (left < 0) left = 0;
        if (n_out < MAX_OBSTACLE_REGIONS) { obstacles_out[n_out].start = left; obstacles_out[n_out].width = rw; n_out++; }
    }
    return (uint8_t)n_out;
}

/* --- MAIN PIPELINE --- */
uint8_t get_obstacle_info(struct image_t *img, float ground_baseline[],
                          int *baseline_inited, float oa_color_count_frac,
                          int median_ksize, int min_width,
                          struct obstacle_region_t obstacles_out[],
                          int boundary_rows_out[], int *ground_found_out,
                          float *green_frac_out)
{
    int w = img->w, h = img->h;
    if (w > MAX_IMAGE_WIDTH) w = MAX_IMAGE_WIDTH;
    if (h > MAX_IMAGE_HEIGHT) h = MAX_IMAGE_HEIGHT;

    float green_frac = 0.0f;
    detect_green_ground_ml(img, work_mask, median_ksize, &green_frac);
    if (green_frac_out) *green_frac_out = green_frac;

    int gf = (green_frac > oa_color_count_frac) ? 1 : 0;
    if (ground_found_out) *ground_found_out = gf;
    if (!gf) return 0;

    isolate_ground_blob(work_mask, w, h, DEFAULT_BLOB_AREA_THRESH);
    fill_holes_mask(work_mask, w, h);

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            work_flipped[y * w + x] = work_mask[y * w + (w - 1 - x)];

    static int boundary_local[MAX_IMAGE_HEIGHT];
    find_ground_boundary(work_flipped, w, h, boundary_local,
                         DEFAULT_MIN_GROUND_PX, DEFAULT_MAX_GAP, DEFAULT_SMOOTH_KERNEL);

    if (boundary_rows_out) memcpy(boundary_rows_out, boundary_local, h * sizeof(int));

    return update_and_detect(boundary_local, w, h, ground_baseline, baseline_inited,
                             min_width, DEFAULT_OBSTACLE_THRESH, DEFAULT_NO_GROUND_BASE,
                             DEFAULT_MAX_COL_GAP, obstacles_out);
}

/* --- DEBUG HELPERS --- */
int dump_mask_to_file(const char *path, const uint8_t mask[], int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(mask, 1, w * h, f);
    fclose(f);
    return 0;
}

int dump_obstacles_to_file(const char *path, const struct obstacle_region_t obs[], int count)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n", count);
    for (int i = 0; i < count; i++) fprintf(f, "%d %d\n", obs[i].start, obs[i].width);
    fclose(f);
    return 0;
}
