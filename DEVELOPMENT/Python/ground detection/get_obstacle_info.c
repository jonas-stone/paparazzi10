/*
 * obstacle_detection.c
 *
 * Pure-C implementation of the unified obstacle + plant detection pipeline.
 * No external dependencies beyond libc and libm.
 *
 * Compile as shared library:
 *   gcc -shared -fPIC -O2 -o libobstacle.so obstacle_detection.c -lm
 *
 * Compile standalone test:
 *   gcc -O2 -o oa_test obstacle_detection.c -lm -DSTANDALONE_TEST
 */

#include "obstacle_detection.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 1 — Low-level image-processing primitives
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── BGR → YUV conversion (matches OpenCV COLOR_BGR2YUV for uint8) ──────── */

static void bgr_to_yuv(const uint8_t *bgr, int w, int h,
                        uint8_t *Y, uint8_t *U, uint8_t *V)
{
    int n = w * h;
    for (int i = 0; i < n; i++) {
        float b = bgr[i * 3 + 0];
        float g = bgr[i * 3 + 1];
        float r = bgr[i * 3 + 2];

        float yf =  0.299f  * r + 0.587f  * g + 0.114f  * b;
        float uf = -0.169f  * r - 0.331f  * g + 0.500f  * b + 128.0f;
        float vf =  0.500f  * r - 0.419f  * g - 0.081f  * b + 128.0f;

        Y[i] = (uint8_t)(yf < 0 ? 0 : (yf > 255 ? 255 : (int)(yf + 0.5f)));
        U[i] = (uint8_t)(uf < 0 ? 0 : (uf > 255 ? 255 : (int)(uf + 0.5f)));
        V[i] = (uint8_t)(vf < 0 ? 0 : (vf > 255 ? 255 : (int)(vf + 0.5f)));
    }
}

/* ── Median blur on uint8 image (square kernel, must be odd) ────────────── */

static int cmp_uint8(const void *a, const void *b) {
    return (int)(*(const uint8_t *)a) - (int)(*(const uint8_t *)b);
}

static void median_blur_2d(const uint8_t *src, uint8_t *dst,
                           int w, int h, int ksize)
{
    if (ksize < 3 || (ksize & 1) == 0) { memcpy(dst, src, w * h); return; }

    int half = ksize / 2;
    int ksq  = ksize * ksize;
    uint8_t *buf = (uint8_t *)malloc(ksq);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int cnt = 0;
            for (int ky = -half; ky <= half; ky++) {
                int yy = y + ky;
                if (yy < 0)  yy = 0;
                if (yy >= h) yy = h - 1;
                for (int kx = -half; kx <= half; kx++) {
                    int xx = x + kx;
                    if (xx < 0)  xx = 0;
                    if (xx >= w) xx = w - 1;
                    buf[cnt++] = src[yy * w + xx];
                }
            }
            qsort(buf, cnt, 1, cmp_uint8);
            dst[y * w + x] = buf[cnt / 2];
        }
    }
    free(buf);
}

/* ── 1-D median filter on int array (matches scipy.ndimage.median_filter) ─ */

static int cmp_int(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static void median_filter_1d(const int *src, int *dst, int n, int ksize)
{
    if (ksize < 3 || (ksize & 1) == 0 || n <= ksize) {
        memcpy(dst, src, n * sizeof(int));
        return;
    }
    int half = ksize / 2;
    int *buf = (int *)malloc(ksize * sizeof(int));

    for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int k = -half; k <= half; k++) {
            int idx = i + k;
            /* reflect boundary (matches scipy default mode='reflect') */
            if (idx < 0)  idx = -idx;
            if (idx >= n) idx = 2 * (n - 1) - idx;
            if (idx < 0)  idx = 0;
            if (idx >= n) idx = n - 1;
            buf[cnt++] = src[idx];
        }
        qsort(buf, cnt, sizeof(int), cmp_int);
        dst[i] = buf[cnt / 2];
    }
    free(buf);
}

/* ── Gaussian blur on uint8 image ─────────────────────────────────────────
 *  sigma=0 uses OpenCV formula: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8    */

static void gaussian_blur_2d(const uint8_t *src, uint8_t *dst, uint8_t *tmp,
                             int w, int h, int ksize)
{
    if (ksize < 3 || (ksize & 1) == 0) { memcpy(dst, src, w * h); return; }

    int half = ksize / 2;
    float sigma = 0.3f * ((ksize - 1) * 0.5f - 1.0f) + 0.8f;

    /* build 1-D kernel */
    float *kernel = (float *)malloc(ksize * sizeof(float));
    float sum = 0;
    for (int i = 0; i < ksize; i++) {
        float x = (float)(i - half);
        kernel[i] = expf(-x * x / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; i++) kernel[i] /= sum;

    /* horizontal pass → tmp */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc = 0;
            for (int k = -half; k <= half; k++) {
                int xx = x + k;
                if (xx < 0)  xx = 0;
                if (xx >= w) xx = w - 1;
                acc += src[y * w + xx] * kernel[k + half];
            }
            tmp[y * w + x] = (uint8_t)(acc + 0.5f);
        }
    }

    /* vertical pass → dst */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc = 0;
            for (int k = -half; k <= half; k++) {
                int yy = y + k;
                if (yy < 0)  yy = 0;
                if (yy >= h) yy = h - 1;
                acc += tmp[yy * w + x] * kernel[k + half];
            }
            dst[y * w + x] = (uint8_t)(acc + 0.5f);
        }
    }
    free(kernel);
}

/* ── Resize: INTER_AREA (box averaging for downscale) ─────────────────── */

static void __attribute__((unused)) resize_area_u8(const uint8_t *src, int sw, int sh,
                           uint8_t       *dst, int dw, int dh)
{
    float sx = (float)sw / dw;
    float sy = (float)sh / dh;

    for (int dy = 0; dy < dh; dy++) {
        float y0f = dy * sy;
        float y1f = (dy + 1) * sy;
        int y0 = (int)y0f;
        int y1 = (int)ceilf(y1f);
        if (y1 > sh) y1 = sh;
        if (y0 == y1) y1 = y0 + 1;

        for (int dx = 0; dx < dw; dx++) {
            float x0f = dx * sx;
            float x1f = (dx + 1) * sx;
            int x0 = (int)x0f;
            int x1 = (int)ceilf(x1f);
            if (x1 > sw) x1 = sw;
            if (x0 == x1) x1 = x0 + 1;

            float acc = 0;
            int   cnt = 0;
            for (int yy = y0; yy < y1; yy++) {
                for (int xx = x0; xx < x1; xx++) {
                    acc += src[yy * sw + xx];
                    cnt++;
                }
            }
            dst[dy * dw + dx] = (uint8_t)(acc / cnt + 0.5f);
        }
    }
}

/* Same but for 3-channel BGR */
static void resize_area_bgr(const uint8_t *src, int sw, int sh,
                             uint8_t       *dst, int dw, int dh)
{
    float sx = (float)sw / dw;
    float sy = (float)sh / dh;

    for (int dy = 0; dy < dh; dy++) {
        float y0f = dy * sy;
        float y1f = (dy + 1) * sy;
        int y0 = (int)y0f;
        int y1 = (int)ceilf(y1f);
        if (y1 > sh) y1 = sh;
        if (y0 == y1) y1 = y0 + 1;

        for (int dx = 0; dx < dw; dx++) {
            float x0f = dx * sx;
            float x1f = (dx + 1) * sx;
            int x0 = (int)x0f;
            int x1 = (int)ceilf(x1f);
            if (x1 > sw) x1 = sw;
            if (x0 == x1) x1 = x0 + 1;

            float b = 0, g = 0, r = 0;
            int   cnt = 0;
            for (int yy = y0; yy < y1; yy++) {
                for (int xx = x0; xx < x1; xx++) {
                    int idx = (yy * sw + xx) * 3;
                    b += src[idx + 0];
                    g += src[idx + 1];
                    r += src[idx + 2];
                    cnt++;
                }
            }
            int oidx = (dy * dw + dx) * 3;
            dst[oidx + 0] = (uint8_t)(b / cnt + 0.5f);
            dst[oidx + 1] = (uint8_t)(g / cnt + 0.5f);
            dst[oidx + 2] = (uint8_t)(r / cnt + 0.5f);
        }
    }
}

/* ── Resize: INTER_NEAREST ────────────────────────────────────────────── */

static void resize_nearest_u8(const uint8_t *src, int sw, int sh,
                              uint8_t       *dst, int dw, int dh)
{
    for (int dy = 0; dy < dh; dy++) {
        int sy = dy * sh / dh;
        if (sy >= sh) sy = sh - 1;
        for (int dx = 0; dx < dw; dx++) {
            int sx = dx * sw / dw;
            if (sx >= sw) sx = sw - 1;
            dst[dy * dw + dx] = src[sy * sw + sx];
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 2 — Connected-component labelling & morphological helpers
 *              (replaces colored_blob_separator.isolate_ground_blob
 *               and colored_blob_separator.fill_holes)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Union-Find ────────────────────────────────────────────────────────── */

static int uf_find(int *parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static void uf_union(int *parent, int *rank, int a, int b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (rank[a] < rank[b]) { int t = a; a = b; b = t; }
    parent[b] = a;
    if (rank[a] == rank[b]) rank[a]++;
}

/* ── isolate_ground_blob: keep only the largest connected component ────── */

static void isolate_largest_blob(const uint8_t *src, uint8_t *dst,
                                 int w, int h,
                                 int *labels, int *parent, int *rank_buf,
                                 int *area_buf)
{
    int n = w * h;
    int next_label = 0;

    /* init union-find */
    for (int i = 0; i < n; i++) {
        labels[i]   = -1;
        parent[i]   = i;
        rank_buf[i] = 0;
    }

    /* first pass: assign labels, record equivalences (4-connected) */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (src[idx] == 0) continue;

            int up   = (y > 0 && src[(y - 1) * w + x] > 0) ? labels[(y - 1) * w + x] : -1;
            int left = (x > 0 && src[y * w + x - 1]   > 0) ? labels[y * w + x - 1]   : -1;

            if (up < 0 && left < 0) {
                labels[idx] = next_label++;
            } else if (up >= 0 && left < 0) {
                labels[idx] = up;
            } else if (up < 0 && left >= 0) {
                labels[idx] = left;
            } else {
                labels[idx] = up;
                if (up != left) uf_union(parent, rank_buf, up, left);
            }
        }
    }

    if (next_label == 0) { memset(dst, 0, n); return; }

    /* flatten labels and count area */
    memset(area_buf, 0, next_label * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (labels[i] >= 0) {
            labels[i] = uf_find(parent, labels[i]);
            area_buf[labels[i]]++;
        }
    }

    /* find largest */
    int best_label = 0, best_area = 0;
    for (int l = 0; l < next_label; l++) {
        if (area_buf[l] > best_area) {
            best_area  = area_buf[l];
            best_label = l;
        }
    }

    /* write output */
    for (int i = 0; i < n; i++) {
        dst[i] = (labels[i] == best_label) ? 255 : 0;
    }
}

/* ── fill_holes: flood-fill background from border, invert remaining 0s ── */

static void fill_holes(uint8_t *mask, int w, int h, int *stack)
{
    int n = w * h;

    /* Create visited array (reuse first n ints of stack as temp) */
    /* Actually we need a separate visited buffer. We'll use the mask itself:
       - 0 = background (to be flood-filled)
       - 255 = foreground
       - 128 = visited background (temp marker)
       After flood fill, any remaining 0 → set to 255 (it's a hole).
    */
    int sp = 0;

    /* seed all border background pixels */
    for (int x = 0; x < w; x++) {
        if (mask[x] == 0)               { mask[x] = 128;               stack[sp++] = x; }
        if (mask[(h-1)*w + x] == 0)     { mask[(h-1)*w + x] = 128;     stack[sp++] = (h-1)*w + x; }
    }
    for (int y = 1; y < h - 1; y++) {
        if (mask[y*w] == 0)             { mask[y*w] = 128;             stack[sp++] = y*w; }
        if (mask[y*w + w - 1] == 0)     { mask[y*w + w - 1] = 128;     stack[sp++] = y*w + w - 1; }
    }

    /* BFS flood fill */
    while (sp > 0) {
        int idx = stack[--sp];
        int x = idx % w;
        int y = idx / w;

        int neighbors[4] = {
            y > 0     ? (y-1)*w + x : -1,
            y < h - 1 ? (y+1)*w + x : -1,
            x > 0     ? y*w + x - 1 : -1,
            x < w - 1 ? y*w + x + 1 : -1,
        };

        for (int i = 0; i < 4; i++) {
            int ni = neighbors[i];
            if (ni >= 0 && mask[ni] == 0) {
                mask[ni] = 128;
                stack[sp++] = ni;
            }
        }
    }

    /* finalize: 128 → 0 (real background), remaining 0 → 255 (hole filled) */
    for (int i = 0; i < n; i++) {
        if (mask[i] == 128)     mask[i] = 0;
        else if (mask[i] == 0)  mask[i] = 255;
        /* else 255 stays 255 */
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 3 — Colour classifiers (decision trees)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Strict ground classifier — trained decision tree */
static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V)
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
            if (U <= 122) {
                return 0;
            } else {
                return 0;
            }
        }
    }
}

/*
 * NOTE on decision-tree thresholds:
 *
 * Python uses float comparisons like  U <= 115.50  which for integer U
 * values (0-255) is equivalent to  U <= 115.  The C version uses integer
 * comparisons directly.  The mapping is:
 *
 *   Python  U <= 115.50   →   C  U <= 115   (since U is uint8)
 *   Python  V <= 145.00   →   C  V <= 145
 *   Python  Y <= 85.50    →   C  Y <= 85
 *   Python  U <= 92.50    →   C  U <= 92
 *   Python  V <= 152.50   →   C  V <= 152
 *   Python  Y <= 177.00   →   C  Y <= 177   (boundary: Y=177 → true)
 *   Python  U <= 121.50   →   C  U <= 121
 *   Python  V <= 137.50   →   C  V <= 137
 *   Python  Y <= 87.50    →   C  Y <= 87
 *   Python  U <= 116.50   →   C  U <= 116
 *   Python  U <= 122.50   →   C  U <= 122
 *   Python  V <= 126.00   →   C  V <= 126   (boundary)
 *   Python  Y <= 62.50    →   C  Y <= 62
 *   Python  Y <= 78.50    →   C  Y <= 78
 */


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 4 — Core pipeline functions
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── detect_green_ground_ml ──────────────────────────────────────────────
 *  Applies decision-tree classifier per pixel, then median-blurs the mask.
 *  Returns green_fraction; writes binary mask into state->mask.            */

static float detect_green_ground_ml(const uint8_t *bgr, int w, int h,
                                    int median_ksize,
                                    uint8_t *yuv_y, uint8_t *yuv_u,
                                    uint8_t *yuv_v, uint8_t *mask_out)
{
    int n = w * h;

    bgr_to_yuv(bgr, w, h, yuv_y, yuv_u, yuv_v);

    for (int i = 0; i < n; i++) {
        mask_out[i] = is_ground(yuv_y[i], yuv_u[i], yuv_v[i]);
    }

    /* in-place median blur (need temp copy) */
    if (median_ksize >= 3 && (median_ksize & 1)) {
        uint8_t *tmp = (uint8_t *)malloc(n);
        memcpy(tmp, mask_out, n);
        median_blur_2d(tmp, mask_out, w, h, median_ksize);
        free(tmp);
    }

    int green_count = 0;
    for (int i = 0; i < n; i++) {
        if (mask_out[i] > 0) green_count++;
    }
    return (n > 0) ? (float)green_count / n : 0.0f;
}

/* ── flip_horizontal (uint8 mask) ─────────────────────────────────────── */

static void flip_horizontal(const uint8_t *src, uint8_t *dst, int w, int h)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            dst[y * w + x] = src[y * w + (w - 1 - x)];
        }
    }
}

/* ── find_ground_boundary ────────────────────────────────────────────────
 *  mask_flipped: shape (h, w) — already horizontally flipped.
 *  Writes boundary[0..h-1].  Each value ∈ [0, w] where w = "no ground".  */

static void find_ground_boundary(const uint8_t *mask_flipped, int w, int h,
                                 int min_ground_pixels, int max_gap,
                                 int smooth_kernel,
                                 int *boundary)
{
    /* w plays the role of "image_height" in the Python code */
    for (int idx = 0; idx < h; idx++) {
        boundary[idx] = w;   /* default: no ground */

        const uint8_t *row = mask_flipped + idx * w;

        /* collect green positions */
        int gp_buf[4096];   /* stack buffer — should be enough for typical widths */
        int *green_pos = gp_buf;
        int gp_n = 0;
        int gp_cap = 4096;

        /* if image is very wide, heap-allocate */
        if (w > gp_cap) {
            gp_cap = w;
            green_pos = (int *)malloc(gp_cap * sizeof(int));
        }

        for (int x = 0; x < w; x++) {
            if (row[x] > 0) green_pos[gp_n++] = x;
        }

        if (gp_n == 0) {
            if (green_pos != gp_buf) free(green_pos);
            continue;
        }

        /* ground_valid: last min_ground_pixels columns all green? */
        int ground_valid = 1;
        if (min_ground_pixels > 0) {
            int cnt = 0;
            int start = w - min_ground_pixels;
            if (start < 0) start = 0;
            for (int x = start; x < w; x++) {
                if (row[x] > 0) cnt++;
            }
            ground_valid = (cnt >= min_ground_pixels) ? 1 : 0;
        }

        if (!ground_valid) {
            /* check if any run is long enough */
            int max_run = 1, cur_run = 1;
            for (int i = 1; i < gp_n; i++) {
                if (green_pos[i] == green_pos[i - 1] + 1) {
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 1;
                }
            }
            if (max_run < max_gap) {
                if (green_pos != gp_buf) free(green_pos);
                continue;
            }
        }

        /* gp = green_pos reversed (scan from right to left) */
        /* We just index green_pos backwards: gp[k] = green_pos[gp_n - 1 - k] */
        #define GP(k) green_pos[gp_n - 1 - (k)]

        if (gp_n == 1) {
            boundary[idx] = GP(0);
            if (green_pos != gp_buf) free(green_pos);
            continue;
        }

        /* find first big gap scanning right-to-left */
        int fg = -1;
        for (int k = 0; k < gp_n - 1; k++) {
            int gap = GP(k) - GP(k + 1) - 1;   /* equivalent to -np.diff(gp) - 1 */
            if (gap > max_gap) { fg = k; break; }
        }

        if (fg < 0) {
            /* no big gap → boundary is leftmost green pixel */
            boundary[idx] = GP(gp_n - 1);
            if (green_pos != gp_buf) free(green_pos);
            continue;
        }

        int pending_boundary = GP(fg);

        /* check after_gap: gp[fg+1 :] */
        int after_n = gp_n - (fg + 1);
        if (after_n >= max_gap) {
            /* find max run length in after_gap */
            int max_run = 1, cur_run = 1;
            for (int k = 1; k < after_n; k++) {
                int ag_prev = GP(fg + 1 + k - 1);
                int ag_cur  = GP(fg + 1 + k);
                if (ag_prev - ag_cur == 1) {   /* consecutive decreasing */
                    cur_run++;
                    if (cur_run > max_run) max_run = cur_run;
                } else {
                    cur_run = 1;
                }
            }
            if (max_run >= max_gap) {
                boundary[idx] = GP(gp_n - 1);
                if (green_pos != gp_buf) free(green_pos);
                continue;
            }
        }

        boundary[idx] = pending_boundary;

        #undef GP
        if (green_pos != gp_buf) free(green_pos);
    }

    /* smooth valid entries with 1-D median filter */
    int valid_count = 0;
    for (int i = 0; i < h; i++) {
        if (boundary[i] < w) valid_count++;
    }

    if (valid_count > smooth_kernel) {
        int *smoothed = (int *)malloc(h * sizeof(int));
        median_filter_1d(boundary, smoothed, h, smooth_kernel);
        for (int i = 0; i < h; i++) {
            if (boundary[i] < w) {
                boundary[i] = smoothed[i];
            }
        }
        free(smoothed);
    }
}

/* ── get_obstacle_regions ─────────────────────────────────────────────── */

static int get_obstacle_regions(const int *cols, int n_cols,
                                int min_width, int max_col_gap,
                                Region *out, int max_out)
{
    if (n_cols == 0) return 0;

    int n_regions = 0;
    int start = cols[0];
    int end   = cols[0];

    for (int i = 1; i < n_cols; i++) {
        if (cols[i] - end <= max_col_gap) {
            end = cols[i];
        } else {
            int w = end - start + 1;
            if (w >= min_width && n_regions < max_out) {
                out[n_regions].start = start;
                out[n_regions].end   = end;
                out[n_regions].width = w;
                n_regions++;
            }
            start = cols[i];
            end   = cols[i];
        }
    }
    /* last region */
    {
        int w = end - start + 1;
        if (w >= min_width && n_regions < max_out) {
            out[n_regions].start = start;
            out[n_regions].end   = end;
            out[n_regions].width = w;
            n_regions++;
        }
    }
    return n_regions;
}

/* ── update_and_detect ────────────────────────────────────────────────── */

static int update_and_detect(const int *boundary, int boundary_len,
                             int h,    /* = image width (W) */
                             float *ground_baseline,
                             int *baseline_initialised,
                             int min_width, int obstacle_threshold,
                             int no_ground_baseline, int max_col_gap,
                             Region *regions, int max_regions)
{
    float alpha = 0.6f;

    if (!(*baseline_initialised)) {
        for (int i = 0; i < boundary_len; i++) {
            ground_baseline[i] = (boundary[i] < h)
                ? (float)boundary[i]
                : (float)no_ground_baseline;
        }
        *baseline_initialised = 1;
        return 0;
    }

    /* collect obstacle columns */
    int *obs_cols = (int *)malloc(boundary_len * sizeof(int));
    int  n_obs    = 0;

    /* also track which indices are valid non-obstacle (for baseline update) */
    uint8_t *no_obs_mask = (uint8_t *)calloc(boundary_len, 1);

    for (int i = 0; i < boundary_len; i++) {
        int valid    = (boundary[i] < h);
        int no_gnd   = (boundary[i] >= h);
        float dev    = (float)boundary[i] - ground_baseline[i];
        int dev_obs  = valid && (dev > (float)obstacle_threshold);
        int is_obs   = dev_obs || no_gnd;

        if (is_obs) {
            obs_cols[n_obs++] = i;
        }

        no_obs_mask[i] = (valid && !dev_obs) ? 1 : 0;
    }

    int n_regions = get_obstacle_regions(obs_cols, n_obs,
                                         min_width, max_col_gap,
                                         regions, max_regions);

    /* update baseline for non-obstacle pixels */
    for (int i = 0; i < boundary_len; i++) {
        if (no_obs_mask[i]) {
            ground_baseline[i] = (1.0f - alpha) * ground_baseline[i]
                               + alpha * (float)boundary[i];
        }
    }

    /* forward-fill: propagate last good baseline over obstacle/no-ground */
    float last_good = (float)no_ground_baseline;
    for (int i = 0; i < boundary_len; i++) {
        if (no_obs_mask[i]) {
            last_good = ground_baseline[i];
        } else {
            ground_baseline[i] = last_good;
        }
    }

    free(obs_cols);
    free(no_obs_mask);
    return n_regions;
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 5 — Lax green / plant detection
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── detect_all_green_lax  (at plant scale) ──────────────────────────── */

static void detect_all_green_lax(const uint8_t *bgr, int w, int h,
                                 float scale_factor, int blur_ksize,
                                 OAState *st)
{
    int pw = (int)(w * scale_factor);
    int ph = (int)(h * scale_factor);
    if (pw < 1) pw = 1;
    if (ph < 1) ph = 1;

    st->plant_w = pw;
    st->plant_h = ph;

    /* downscale BGR */
    resize_area_bgr(bgr, w, h, st->plant_bgr, pw, ph);

    /* BGR → YUV at plant scale */
    bgr_to_yuv(st->plant_bgr, pw, ph,
               st->plant_yuv_y, st->plant_yuv_u, st->plant_yuv_v);

    /* apply lax thresholds */
    int pn = pw * ph;
    for (int i = 0; i < pn; i++) {
        uint8_t y = st->plant_yuv_y[i];
        uint8_t u = st->plant_yuv_u[i];
        uint8_t v = st->plant_yuv_v[i];

        st->plant_small[i] = (u <= 116 && v <= 141 && y >= 29 && y <= 140)
                              ? 255 : 0;
    }

    /* zero left third */
    int left_cutoff = pw / 3;
    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < left_cutoff; x++) {
            st->plant_small[y * pw + x] = 0;
        }
    }

    /* Gaussian blur + threshold */
    if (blur_ksize >= 3) {
        gaussian_blur_2d(st->plant_small, st->plant_small, st->blur_tmp,
                         pw, ph, blur_ksize);
        for (int i = 0; i < pn; i++) {
            st->plant_small[i] = (st->plant_small[i] > 127) ? 255 : 0;
        }
    }
}

/* ── detect_plant_regions ──────────────────────────────────────────────
 *  Finds row-ranges where significant lax-green content exists
 *  (after subtracting the strict ground mask).                            */

static int detect_plant_regions(const uint8_t *plant_mask, int w, int h,
                                int min_width, int min_pix_per_col,
                                int max_col_gap,
                                Region *out, int max_out)
{
    /* flip horizontally (match Python convention) */
    /* We'll scan in-place with reversed column indexing instead of allocating */

    int *active_rows = (int *)malloc(h * sizeof(int));
    int  n_active = 0;

    for (int y = 0; y < h; y++) {
        /* row of flipped image: pixel at (y, x) = original (y, w-1-x) */
        /* count green positions and find max run */
        int gp_n = 0;
        int max_run = 0, cur_run = 0;
        int prev_x = -2;

        for (int x = 0; x < w; x++) {
            uint8_t val = plant_mask[y * w + (w - 1 - x)];  /* flipped */
            if (val > 0) {
                gp_n++;
                if (x == prev_x + 1) {
                    cur_run++;
                } else {
                    cur_run = 1;
                }
                if (cur_run > max_run) max_run = cur_run;
                prev_x = x;
            }
        }

        if (gp_n > 0 && max_run > min_pix_per_col) {
            active_rows[n_active++] = y;
        }
    }

    int n = get_obstacle_regions(active_rows, n_active,
                                 min_width, max_col_gap, out, max_out);
    free(active_rows);
    return n;
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 6 — State management
 * ═══════════════════════════════════════════════════════════════════════════ */

OAParams oa_default_params(void)
{
    OAParams p;
    p.oa_color_count_frac     = 0.05f;
    p.median_ksize            = 5;
    p.min_width               = 20;
    p.max_col_gap             = 5;
    p.min_ground_pixels       = 5;
    p.max_gap                 = 10;
    p.smooth_kernel           = 5;
    p.obstacle_threshold      = 50;
    p.no_ground_baseline      = 220;
    p.plant_scale_factor      = 0.15f;
    p.plant_blur_ksize        = 3;
    p.plant_min_width         = 20;
    p.plant_min_pixels_per_col = 2;
    p.plant_max_col_gap       = 5;
    return p;
}

OAState *oa_state_create(int img_w, int img_h, const OAParams *p)
{
    OAState *s = (OAState *)calloc(1, sizeof(OAState));
    s->img_w = img_w;
    s->img_h = img_h;
    s->baseline_initialised = 0;

    int n = img_w * img_h;

    s->baseline     = (float *)  calloc(img_h, sizeof(float));
    s->yuv_y        = (uint8_t *)malloc(n);
    s->yuv_u        = (uint8_t *)malloc(n);
    s->yuv_v        = (uint8_t *)malloc(n);
    s->mask         = (uint8_t *)malloc(n);
    s->clean_mask   = (uint8_t *)malloc(n);
    s->mask_flipped = (uint8_t *)malloc(n);
    s->boundary     = (int *)    malloc(img_h * sizeof(int));

    /* plant buffers — sized for max possible plant scale */
    int pw = (int)(img_w * p->plant_scale_factor) + 2;
    int ph = (int)(img_h * p->plant_scale_factor) + 2;
    int pn = pw * ph;
    s->plant_w     = pw;
    s->plant_h     = ph;
    s->plant_small = (uint8_t *)malloc(pn);
    s->clean_small = (uint8_t *)malloc(pn);
    s->plant_diff  = (uint8_t *)malloc(pn);
    s->plant_full  = (uint8_t *)malloc(n);

    s->plant_bgr   = (uint8_t *)malloc(pn * 3);
    s->plant_yuv_y = (uint8_t *)malloc(pn);
    s->plant_yuv_u = (uint8_t *)malloc(pn);
    s->plant_yuv_v = (uint8_t *)malloc(pn);

    /* connected-component scratch */
    s->cc_labels = (int *)malloc(n * sizeof(int));
    s->cc_parent = (int *)malloc(n * sizeof(int));
    s->cc_rank   = (int *)malloc(n * sizeof(int));
    s->cc_area   = (int *)malloc(n * sizeof(int));

    /* flood-fill stack */
    s->ff_stack  = (int *)malloc(n * sizeof(int));

    /* blur scratch */
    s->blur_tmp  = (uint8_t *)malloc(pn);

    return s;
}

void oa_state_destroy(OAState *s)
{
    if (!s) return;
    free(s->baseline);
    free(s->yuv_y);    free(s->yuv_u);     free(s->yuv_v);
    free(s->mask);      free(s->clean_mask); free(s->mask_flipped);
    free(s->boundary);
    free(s->plant_small); free(s->clean_small); free(s->plant_diff);
    free(s->plant_full);
    free(s->plant_bgr);   free(s->plant_yuv_y);
    free(s->plant_yuv_u); free(s->plant_yuv_v);
    free(s->cc_labels); free(s->cc_parent);
    free(s->cc_rank);   free(s->cc_area);
    free(s->ff_stack);  free(s->blur_tmp);
    free(s);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 7 — get_obstacle_info  (main entry point)
 * ═══════════════════════════════════════════════════════════════════════════ */

void get_obstacle_info(const uint8_t *bgr_data,
                       int            width,
                       int            height,
                       OAState       *st,
                       const OAParams *p,
                       OAResult      *result)
{
    int W = width;
    int H = height;
    int n = W * H;

    result->n_detections = 0;

    /* ── strict ground mask ─────────────────────────────────────────────── */
    float green_frac = detect_green_ground_ml(
        bgr_data, W, H, p->median_ksize,
        st->yuv_y, st->yuv_u, st->yuv_v, st->mask);

    result->green_frac = green_frac;

    int ground_found = (green_frac > p->oa_color_count_frac);
    result->status = ground_found ? STATUS_GROUND_FOUND : STATUS_NO_GROUND;

    Region obs_regions[OA_MAX_REGIONS];
    int    n_obs_regions = 0;

    if (ground_found) {
        /* isolate largest blob */
        isolate_largest_blob(st->mask, st->clean_mask, W, H,
                             st->cc_labels, st->cc_parent,
                             st->cc_rank, st->cc_area);

        /* fill holes */
        fill_holes(st->clean_mask, W, H, st->ff_stack);

        /* flip horizontally */
        flip_horizontal(st->clean_mask, st->mask_flipped, W, H);

        /* find ground boundary */
        find_ground_boundary(st->mask_flipped, W, H,
                             p->min_ground_pixels, p->max_gap,
                             p->smooth_kernel, st->boundary);

        /* update baseline and detect obstacles */
        n_obs_regions = update_and_detect(
            st->boundary, H, W,
            st->baseline, &st->baseline_initialised,
            p->min_width, p->obstacle_threshold,
            p->no_ground_baseline, p->max_col_gap,
            obs_regions, OA_MAX_REGIONS);

        /* convert obstacle regions to detections */
        for (int i = 0; i < n_obs_regions && result->n_detections < OA_MAX_DETECTIONS; i++) {
            int left = W - 1 - obs_regions[i].end;
            result->detections[result->n_detections].left_x   = left;
            result->detections[result->n_detections].width    = obs_regions[i].width;
            result->detections[result->n_detections].det_type = DET_OBSTACLE;
            result->n_detections++;
        }
    } else {
        /* no ground → clean_mask is all zeros */
        memset(st->clean_mask, 0, n);
    }

    /* ── plant detection ────────────────────────────────────────────────── */
    detect_all_green_lax(bgr_data, W, H,
                         p->plant_scale_factor, p->plant_blur_ksize, st);

    int pw = st->plant_w;
    int ph = st->plant_h;

    /* resize clean_mask down to plant scale */
    resize_nearest_u8(st->clean_mask, W, H, st->clean_small, pw, ph);

    /* saturating subtract: plant = lax_green - ground */
    int pn = pw * ph;
    for (int i = 0; i < pn; i++) {
        int diff = (int)st->plant_small[i] - (int)st->clean_small[i];
        st->plant_diff[i] = (diff > 0) ? (uint8_t)diff : 0;
    }

    /* upscale plant mask to full resolution */
    resize_nearest_u8(st->plant_diff, pw, ph, st->plant_full, W, H);

    /* detect plant regions */
    Region plant_regions[OA_MAX_REGIONS];
    int n_plant_regions = detect_plant_regions(
        st->plant_full, W, H,
        p->plant_min_width, p->plant_min_pixels_per_col,
        p->plant_max_col_gap,
        plant_regions, OA_MAX_REGIONS);

    for (int i = 0; i < n_plant_regions && result->n_detections < OA_MAX_DETECTIONS; i++) {
        result->detections[result->n_detections].left_x   = plant_regions[i].start;
        result->detections[result->n_detections].width    = plant_regions[i].width;
        result->detections[result->n_detections].det_type = DET_PLANT;
        result->n_detections++;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SECTION 8 — Standalone test (compile with -DSTANDALONE_TEST)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef STANDALONE_TEST
#include <stdio.h>
#include <time.h>

/* Minimal PPM reader for testing without any image library */
static uint8_t *read_ppm(const char *path, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || (strcmp(magic,"P6") != 0)) {
        fclose(f); return NULL;
    }
    /* skip comments */
    int c;
    while ((c = fgetc(f)) == '#') { while (fgetc(f) != '\n'); }
    ungetc(c, f);
    int maxval;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) { fclose(f); return NULL; }
    fgetc(f); /* skip single whitespace */
    int n = (*w) * (*h) * 3;
    uint8_t *rgb = (uint8_t *)malloc(n);
    if (fread(rgb, 1, n, f) != (size_t)n) { free(rgb); fclose(f); return NULL; }
    fclose(f);
    /* convert RGB → BGR in-place */
    for (int i = 0; i < n; i += 3) {
        uint8_t tmp = rgb[i]; rgb[i] = rgb[i+2]; rgb[i+2] = tmp;
    }
    return rgb;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image.ppm\n", argv[0]);
        return 1;
    }
    int w, h;
    uint8_t *bgr = read_ppm(argv[1], &w, &h);
    if (!bgr) { fprintf(stderr, "Failed to read %s\n", argv[1]); return 1; }

    printf("Image: %d x %d\n", w, h);

    OAParams params = oa_default_params();
    OAState *state  = oa_state_create(w, h, &params);
    OAResult result;

    clock_t t0 = clock();
    get_obstacle_info(bgr, w, h, state, &params, &result);
    clock_t t1 = clock();

    printf("Status:     %s\n", result.status ? "GROUND FOUND" : "NO GROUND");
    printf("Green frac: %.4f\n", result.green_frac);
    printf("Detections: %d\n", result.n_detections);
    for (int i = 0; i < result.n_detections; i++) {
        printf("  [%d] left_x=%d  width=%d  type=%s\n",
               i, result.detections[i].left_x, result.detections[i].width,
               result.detections[i].det_type == DET_PLANT ? "PLANT" : "OBSTACLE");
    }
    printf("Time: %.3f ms\n", (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0);

    oa_state_destroy(state);
    free(bgr);
    return 0;
}
#endif