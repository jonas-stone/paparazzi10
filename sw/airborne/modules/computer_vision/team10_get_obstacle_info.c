/*
 * team10_get_obstacle_info.c — Paparazzi build.
 * Complete pipeline with is_smooth_blob (perimeter + fractal dimension).
 * Same algorithm as _standalone.c, with real Paparazzi includes.
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
 * ══════════════════════════════════════════════════════════════════════════════ */
#define MAX_PLANT_PIXELS (MAX_PLANT_WIDTH * MAX_PLANT_HEIGHT)

static uint8_t  work_mask[MAX_PIXELS];
static uint8_t  work_mask2[MAX_PIXELS];
static int16_t  work_labels[MAX_PIXELS];
static uint8_t  work_visited[MAX_PIXELS];   /* also used as padded blob mask */
static uint8_t  work_flipped[MAX_PIXELS];   /* also used as edge image       */
static int32_t  ff_queue[MAX_PIXELS];
static int16_t  uf_parent[MAX_CC_LABELS];
static int32_t  cc_area[MAX_CC_LABELS];
static int16_t  cc_bb_x0[MAX_CC_LABELS];    /* bounding box per label */
static int16_t  cc_bb_x1[MAX_CC_LABELS];
static int16_t  cc_bb_y0[MAX_CC_LABELS];
static int16_t  cc_bb_y1[MAX_CC_LABELS];

static uint8_t  plant_green_small[MAX_PLANT_PIXELS];
static uint8_t  plant_blur_buf[MAX_PLANT_PIXELS];
static uint8_t  plant_clean_small[MAX_PLANT_PIXELS];
static uint8_t  plant_mask_small[MAX_PLANT_PIXELS];
static uint8_t  plant_mask_full[MAX_PIXELS];

/* ══════════════════════════════════════════════════════════════════════════════
 *  YUV422 (UYVY) PIXEL ACCESS
 * ══════════════════════════════════════════════════════════════════════════════ */
static inline uint8_t yuv422_Y(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + x * 2 + 1]; }
static inline uint8_t yuv422_U(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + (x & ~1) * 2]; }
static inline uint8_t yuv422_V(const uint8_t *buf, int w, int x, int y)
{ return buf[y * w * 2 + (x & ~1) * 2 + 2]; }

/* ══════════════════════════════════════════════════════════════════════════════
 *  1. DECISION TREE (exact match of Python is_ground)
 * ══════════════════════════════════════════════════════════════════════════════ */
<<<<<<< HEAD
/* Uncomment ONE of these: */
=======
>>>>>>> e73ebd5cd7473d5228fcb3f969bef505ed7c43b1
// #define GROUND_TREE_REAL
#define GROUND_TREE_SIM

/* --- DECISION TREE --- */
uint8_t is_ground_pixel(uint8_t Y, uint8_t U, uint8_t V)
{
#ifdef GROUND_TREE_SIM
    if (U <= 96) {
        return (Y <= 102) ? 255 : 0;
    } else if (U <= 97) {
        return (V <= 126) ? 255 : 0;
    } else {
        return 0;
    }
#else
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
 *  2. BINARY MEDIAN BLUR (majority vote)
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

void detect_green_ground_ml(struct image_t *img, uint8_t mask_out[],
                            int median_ksize, float *green_frac_out)
{
    int w = img->w, h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;
    
    // Calculate the starting row (halfway down the screen)
    int max_x = w / 2; 
    // Explicitly paint the entire top half of the working mask black (0)
    memset(work_mask2, 0, w * h);
    // Loop through all rows (y), but ONLY loop halfway across the columns (x)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < max_x; x++) {
            work_mask2[y * w + x] = is_ground_pixel(
                yuv422_Y(buf, w, x, y), yuv422_U(buf, w, x, y), yuv422_V(buf, w, x, y));
        }
    }
    if (median_ksize >= 3 && (median_ksize & 1))
        median_blur_binary(work_mask2, mask_out, w, h, median_ksize);
    else
        memcpy(mask_out, work_mask2, w * h);
    int gc = 0, total = w * h;
    for (int i = 0; i < total; i++) if (mask_out[i]) gc++;
    if (green_frac_out) *green_frac_out = total > 0 ? (float)gc / total : 0.0f;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  3. CONNECTED COMPONENTS (union-find, 8-connectivity, with bounding boxes)
 * ══════════════════════════════════════════════════════════════════════════════ */
static void uf_init(int n) { for (int i = 0; i < n; i++) uf_parent[i] = (int16_t)i; }
static int16_t uf_find(int16_t x) {
    while (uf_parent[x] != x) { uf_parent[x] = uf_parent[uf_parent[x]]; x = uf_parent[x]; } return x;
}
static void uf_union(int16_t a, int16_t b) {
    a = uf_find(a); b = uf_find(b);
    if (a != b) { if (a < b) uf_parent[b] = a; else uf_parent[a] = b; }
}

static int connected_components(const uint8_t *mask, int16_t *labels, int w, int h)
{
    int16_t next_label = 1;
    uf_init(MAX_CC_LABELS);
    memset(labels, 0, w * h * sizeof(int16_t));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (!mask[idx]) continue;
            int16_t nb[4]; int nn = 0;
            if (y > 0 && x > 0     && labels[(y-1)*w+(x-1)]) nb[nn++] = labels[(y-1)*w+(x-1)];
            if (y > 0               && labels[(y-1)*w+x])     nb[nn++] = labels[(y-1)*w+x];
            if (y > 0 && x < w - 1 && labels[(y-1)*w+(x+1)]) nb[nn++] = labels[(y-1)*w+(x+1)];
            if (x > 0               && labels[y*w+(x-1)])     nb[nn++] = labels[y*w+(x-1)];
            if (nn == 0) { if (next_label < MAX_CC_LABELS) labels[idx] = next_label++; }
            else {
                int16_t m = uf_find(nb[0]);
                for (int i = 1; i < nn; i++) { int16_t r = uf_find(nb[i]); if (r < m) m = r; }
                labels[idx] = m;
                for (int i = 0; i < nn; i++) uf_union(m, nb[i]);
            }
        }
    }
    /* flatten labels */
    for (int i = 0; i < w * h; i++) if (labels[i]) labels[i] = uf_find(labels[i]);

    /* compute per-label area and bounding boxes */
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
 *  BLOB PERIMETER (matching Python: transitions in zero-padded image)
 *
 *  Counts horizontal + vertical transitions between blob and non-blob,
 *  including the implicit zero-padding border.
 * ══════════════════════════════════════════════════════════════════════════════ */
int compute_blob_perimeter(const int16_t *labels, int w, int h, int16_t lbl)
{
    int perim = 0;

    /* horizontal transitions */
    for (int y = 0; y < h; y++) {
        /* left border → first pixel */
        if (labels[y * w] == lbl) perim++;
        /* interior */
        for (int x = 0; x < w - 1; x++) {
            int a = (labels[y * w + x] == lbl) ? 1 : 0;
            int b = (labels[y * w + x + 1] == lbl) ? 1 : 0;
            if (a != b) perim++;
        }
        /* last pixel → right border */
        if (labels[y * w + w - 1] == lbl) perim++;
    }

    /* vertical transitions */
    for (int x = 0; x < w; x++) {
        /* top border → first pixel */
        if (labels[x] == lbl) perim++;
        /* interior */
        for (int y = 0; y < h - 1; y++) {
            int a = (labels[y * w + x] == lbl) ? 1 : 0;
            int b = (labels[(y + 1) * w + x] == lbl) ? 1 : 0;
            if (a != b) perim++;
        }
        /* last pixel → bottom border */
        if (labels[(h - 1) * w + x] == lbl) perim++;
    }

    return perim;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  FRACTAL DIMENSION (box-counting on blob edge, matching Python exactly)
 *
 *  Steps:
 *    1. Create zero-padded blob mask (h+2)×(w+2) in work_visited
 *    2. Compute edge image in work_flipped:
 *         edge[y,x] = (padded[y,x+1] != padded[y,x]) || (padded[y+1,x] != padded[y,x])
 *    3. Box-counting at scales 2, 4, 8, 16, ...
 *    4. Linear regression on log(scale) vs log(count)
 *    5. Return -slope
 * ══════════════════════════════════════════════════════════════════════════════ */
float compute_fractal_dimension(const int16_t *labels, int w, int h, int16_t lbl)
{
    int pw = w + 2, ph = h + 2;

    /* step 1: create padded blob mask in work_visited */
    memset(work_visited, 0, pw * ph);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (labels[y * w + x] == lbl)
                work_visited[(y + 1) * pw + (x + 1)] = 1;

    /* step 2: compute edge image in work_flipped */
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

    /* step 3: box-counting at powers of 2 */
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
                /* check if any pixel in this s×s block is edge */
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

    /* step 4: linear regression y = a*x + b */
    if (n_scales < 2) return 1.0f;  /* not enough data → assume smooth */

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
    return -slope;   /* negate: slope is negative, fractal dim is positive */
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  IS_SMOOTH_BLOB (matching Python solidity_detection.is_smooth_blob)
 *
 *  A blob is "smooth" (likely ground) if EITHER:
 *    perimeter / equiv_rect_perimeter < 2.0
 *    OR fractal_dimension < 1.3
 *
 *  Returns 1 = smooth (keep), 0 = spiky (remove).
 * ══════════════════════════════════════════════════════════════════════════════ */
int is_smooth_blob(const int16_t *labels, int w, int h,
                   int16_t lbl, int blob_area, int bb_height)
{
    /* bb_height = bounding box HEIGHT in pixels (Python calls this "width"
       because the camera is sideways — values[label, 3] = CC_STAT_HEIGHT) */
    if (bb_height <= 0) return 1;  /* degenerate → keep */

    int perim = compute_blob_perimeter(labels, w, h, lbl);

    /* average height = area / bb_height  (matches Python compute_average_blob_height) */
    int avg_height = blob_area / bb_height;

    /* equivalent rectangle perimeter */
    int equiv_rect_perim = 2 * (avg_height + bb_height);

    /* perimeter ratio check */
    float perim_ratio = (equiv_rect_perim > 0) ?
        (float)perim / (float)equiv_rect_perim : 0.0f;

    if (perim_ratio < SMOOTH_PERIMETER_RATIO_THRESH)
        return 1;  /* smooth */

    /* fractal dimension check (expensive — only if perimeter ratio failed) */
    float fd = compute_fractal_dimension(labels, w, h, lbl);

    if (fd < SMOOTH_FRACTAL_DIM_THRESH)
        return 1;  /* smooth */

    return 0;  /* both checks failed → spiky, remove this blob */
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ISOLATE GROUND BLOB (CC + area filter + smooth-blob filter)
 * ══════════════════════════════════════════════════════════════════════════════ */
void isolate_ground_blob(uint8_t mask[], int w, int h, int blob_area_threshold)
{
    int n_labels = connected_components(mask, work_labels, w, h);
    if (n_labels == 0) return;

    int max_label = 0;
    for (int i = 0; i < w * h; i++) if (work_labels[i] > max_label) max_label = work_labels[i];
    if (max_label >= MAX_CC_LABELS) max_label = MAX_CC_LABELS - 1;

    /* decide which labels pass ONCE (not per-pixel!) */
    static uint8_t label_keep[MAX_CC_LABELS];
    memset(label_keep, 0, max_label + 1);

    for (int l = 1; l <= max_label; l++) {
        if (cc_area[l] < blob_area_threshold) continue;

        int bb_height = cc_bb_y1[l] - cc_bb_y0[l] + 1;
        if (!is_smooth_blob(work_labels, w, h, (int16_t)l, cc_area[l], bb_height))
            continue;

        label_keep[l] = 1;
    }

    /* apply: keep only passing labels */
    for (int i = 0; i < w * h; i++) {
        int16_t l = work_labels[i];
        mask[i] = (l > 0 && l <= max_label && label_keep[l]) ? 255 : 0;
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  4. HOLE FILLING (border flood-fill BFS)
 * ══════════════════════════════════════════════════════════════════════════════ */
void fill_holes_mask(uint8_t mask[], int w, int h)
{
    int total = w * h;
    memset(work_visited, 0, total);
    int qh = 0, qt = 0;
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
    while (qh < qt) {
        int i = ff_queue[qh++]; int cy = i / w, cx = i % w;
        if (cy > 0     && !work_visited[i-w] && !mask[i-w]) { work_visited[i-w] = 1; ff_queue[qt++] = i-w; }
        if (cy < h - 1 && !work_visited[i+w] && !mask[i+w]) { work_visited[i+w] = 1; ff_queue[qt++] = i+w; }
        if (cx > 0     && !work_visited[i-1] && !mask[i-1]) { work_visited[i-1] = 1; ff_queue[qt++] = i-1; }
        if (cx < w - 1 && !work_visited[i+1] && !mask[i+1]) { work_visited[i+1] = 1; ff_queue[qt++] = i+1; }
    }
    for (int i = 0; i < total; i++)
        if (!mask[i] && !work_visited[i]) mask[i] = 255;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  5. BOUNDARY SCAN + 1-D MEDIAN
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

void find_ground_boundary(const uint8_t mask_flipped[], int w, int h,
                          int boundary_out[], int mgp, int mg, int sk)
{
    int nr = h, rl = w;
    for (int r = 0; r < nr; r++) boundary_out[r] = rl;
    for (int r = 0; r < nr; r++) {
        const uint8_t *row = &mask_flipped[r * rl];
        int any = 0;
        for (int c = 0; c < rl; c++) if (row[c]) { any = 1; break; }
        if (!any) continue;
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
 *  6-8. OBSTACLE REGION MERGE + BASELINE UPDATE
 * ══════════════════════════════════════════════════════════════════════════════ */
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

uint8_t update_and_detect(const int br[], int h, int w, float gb[], int *bi,
                          int mw, int ot, int ngb, int mcg,
                          struct obstacle_region_t oo[])
{
    float alpha = DEFAULT_BASELINE_ALPHA;
    if (!(*bi)) {
        for (int i = 0; i < w; i++) gb[i] = (br[i] < h) ? (float)br[i] : (float)ngb;
        *bi = 1; return 0;
    }
    static uint8_t om[MAX_IMAGE_HEIGHT], nom[MAX_IMAGE_HEIGHT];
    memset(om, 0, w); memset(nom, 0, w);
    for (int i = 0; i < w; i++) {
        int v = br[i] < h;
        if (v) { float d = (float)br[i] - gb[i]; if (d > (float)ot) om[i] = 1; else nom[i] = 1; }
        else om[i] = 1;
    }
    struct obstacle_region_t fl[MAX_OBSTACLE_REGIONS];
    int nf = merge_obstacle_cols(om, w, mw, mcg, fl, MAX_OBSTACLE_REGIONS);
    for (int i = 0; i < w; i++)
        if (nom[i]) gb[i] = (1.0f - alpha) * gb[i] + alpha * (float)br[i];
    //float lg = (float)ngb;
    //for (int i = 0; i < w; i++) { if (nom[i]) lg = gb[i]; else gb[i] = lg; }
    /* Two-pass propagation: left-to-right, then right-to-left, take min */
    float lg = (float)ngb;
    for (int i = 0; i < w; i++) {
        if (nom[i]) lg = gb[i];
        else gb[i] = lg;  /* left neighbor's value */
    }

    /* Right-to-left pass */
    float rg = (float)ngb;
    for (int i = w - 1; i >= 0; i--) {
        if (nom[i]) {
            rg = gb[i];
        } else {
            /* If we haven't seen ANY valid ground to the right yet, 
               force it to the default low baseline, ignoring the left anchor. */
            if (rg == (float)ngb) {
                gb[i] = (float)ngb;
            } else {
                /* Keep the MINIMUM (visually highest) of left-propagated and right-propagated */
                if (rg < gb[i]) {
                    gb[i] = rg;
                }
            }
        }
    }
    int no = 0;
    for (int i = 0; i < nf && no < MAX_OBSTACLE_REGIONS; i++) {
        if (fl[i].width < 5) continue;
        oo[no].start = (uint16_t)fl[i].start;
        oo[no].width = (uint16_t)fl[i].width;

        int s = fl[i].start;
        int e = s + fl[i].width - 1;
        
        // ── SIMPLIFIED HEIGHT LOGIC ──
        int max_br = 0; 
        int touches_bottom = 0; // Flag to track if ground vanishes
        
        for (int r = s; r <= e; r++) {
            if (r < w) {
                if (br[r] >= h) {
                    // No ground found in this specific column
                    touches_bottom = 1; 
                } else {
                    // Track valid points just in case it DOESN'T touch the bottom
                    if (br[r] > max_br) {
                        max_br = br[r];
                    }
                }
            }
        }
        
        // The Simple Rule: If it loses the ground ANYWHERE, or has no valid points, set to default.
        if (touches_bottom || max_br == 0) {
            max_br = ngb;
        }
        
        oo[no].baseline_height = (uint16_t)max_br;
        no++;
    }
    return (uint8_t)no;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  9-15. PLANT DETECTION
 * ══════════════════════════════════════════════════════════════════════════════ */
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
    int lt = pw / 3;
    for (int py = 0; py < ph; py++) for (int px = 0; px < lt; px++) plant_green_small[py * pw + px] = 0;

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

static void downsample_nn(const uint8_t *s, int sw, int sh, uint8_t *d, int dw, int dh)
{ for (int y = 0; y < dh; y++) { int sy = y*sh/dh; if(sy>=sh)sy=sh-1; for (int x = 0; x < dw; x++) { int sx = x*sw/dw; if(sx>=sw)sx=sw-1; d[y*dw+x] = s[sy*sw+sx]; } } }

static void upsample_nn(const uint8_t *s, int sw, int sh, uint8_t *d, int dw, int dh)
{ for (int y = 0; y < dh; y++) { int sy = y*sh/dh; if(sy>=sh)sy=sh-1; for (int x = 0; x < dw; x++) { int sx = x*sw/dw; if(sx>=sw)sx=sw-1; d[y*dw+x] = s[sy*sw+sx]; } } }

uint8_t detect_plant_regions(const uint8_t pm[], int w, int h, int mw, int mpc, int mcg,
                             struct obstacle_region_t po[])
{
    static uint8_t am[MAX_IMAGE_HEIGHT]; memset(am, 0, h);
    for (int r = 0; r < h; r++) {
        int mr = 0, cr = 0;
        for (int c = 0; c < w; c++) { int fc = w-1-c; if (pm[r*w+fc]) { cr++; if (cr > mr) mr = cr; } else cr = 0; }
        if (mr > mpc) am[r] = 1;
    }
    return (uint8_t)merge_obstacle_cols(am, h, mw, mcg, po, MAX_PLANT_REGIONS);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  MAIN PIPELINE
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
    int ground = (gf > oacf) ? 1 : 0;
    if (gfo) *gfo = ground;

    uint8_t no = 0;
    if (ground) {
        isolate_ground_blob(work_mask, w, h, DEFAULT_BLOB_AREA_THRESH);
        fill_holes_mask(work_mask, w, h);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) work_flipped[y*w+x] = work_mask[y*w+(w-1-x)];
        static int bl[MAX_IMAGE_HEIGHT];
        find_ground_boundary(work_flipped, w, h, bl, DEFAULT_MIN_GROUND_PX, DEFAULT_MAX_GAP, DEFAULT_SMOOTH_KERNEL);
        if (bro) memcpy(bro, bl, h * sizeof(int));
        no = update_and_detect(bl, w, h, gb, bi, mw, DEFAULT_OBSTACLE_THRESH, DEFAULT_NO_GROUND_BASE, DEFAULT_MAX_COL_GAP, oo);
    }

    if (po != NULL) {
        int pw = 0, ph = 0;
        detect_all_green_lax(img, DEFAULT_PLANT_SCALE_NUM, DEFAULT_PLANT_SCALE_DEN,
                             DEFAULT_PLANT_BLUR_KSIZE, plant_mask_small, &pw, &ph);
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

/* ── DEBUG ────────────────────────────────────────────────────────────────── */
int dump_mask_to_file(const char *p, const uint8_t m[], int w, int h)
{ FILE *f = fopen(p, "wb"); if (!f) return -1; fwrite(m, 1, w*h, f); fclose(f); return 0; }
int dump_obstacles_to_file(const char *p, const struct obstacle_region_t o[], int c)
{ FILE *f = fopen(p, "w"); if (!f) return -1; fprintf(f, "%d\n", c); for (int i = 0; i < c; i++) fprintf(f, "%d %d\n", o[i].start, o[i].width); fclose(f); return 0; }

/* ══════════════════════════════════════════════════════════════════════════════
 *  GATE DETECTION
 *
 *  Direct C port of gate_detection.py.
 *
 *  Coordinate note
 *  ---------------
 *  The image arrives as portrait YUV422 with the floor on the LEFT.
 *  Two real-world vertical pillars therefore appear as two blobs separated
 *  along the IMAGE Y-axis (|dy| > |dx|).
 *  The output centre coordinate is the X pixel of the midpoint, i.e. the
 *  distance from the left edge of the raw image — which equals the distance
 *  from the top when the image is rotated 90° CCW for display.
 *
 *  Blue mask
 *  ---------
 *  Uses the existing yuv422_Y/U/V accessors and a simple threshold on the
 *  YUV channels.  A 5×5 morphological open+close is applied via a raster
 *  scan (same logic as median_blur_binary) to suppress noise.
 *
 *  Checker confirmation
 *  --------------------
 *  For each blue blob the band of pixels on its LEFT face (floor side, low x)
 *  is extracted.  The score is the fraction of columns whose grayscale
 *  max-min range exceeds GATE_CHECKER_CONTRAST.  A checker pattern scores
 *  high (every column crosses dark and bright squares); a uniform dark wall
 *  scores 0.
 *
 *  Parallelism check
 *  -----------------
 *  1. |dy| > |dx|  — blobs are vertically separated in image coords
 *  2. Similar aspect ratio (w/h within GATE_MAX_ASPECT_DIFF_PCT %)
 *  3. Similar area (within GATE_MAX_AREA_DIFF_PCT %)
 *  4. Centroid-join vector is within GATE_MAX_SKEW_DEG10/10 degrees of
 *     perpendicular to the blobs' average long axis
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── Static work buffers used only by detect_gate ─────────────────────────── */
static uint8_t  gate_blue_mask[MAX_PIXELS];          /* binary blue mask       */
static uint8_t  gate_eroded[MAX_PIXELS];             /* morphological scratch  */
static int16_t  gate_cc_labels[MAX_PIXELS];          /* CC labels for blue     */

/* Per-blob descriptor — keeps only what we need */
typedef struct {
    int x_min, y_min, x_max, y_max;   /* bounding box                */
    int cx, cy;                        /* centroid (integer)          */
    int area;                          /* pixel count                 */
    int aspect_num, aspect_den;        /* aspect = width/height as ratio */
} gate_blob_t;

static gate_blob_t gate_blobs[GATE_MAX_BLOBS];

/* ── Local union-find (separate from the obstacle pipeline's uf_parent) ───── */
static int16_t gate_uf[MAX_CC_LABELS];
static int32_t gate_area[MAX_CC_LABELS];
static int16_t gate_x0[MAX_CC_LABELS], gate_x1[MAX_CC_LABELS];
static int16_t gate_y0[MAX_CC_LABELS], gate_y1[MAX_CC_LABELS];
static int32_t gate_cx_sum[MAX_CC_LABELS], gate_cy_sum[MAX_CC_LABELS];

static int16_t gate_uf_find(int16_t x) {
    while (gate_uf[x] != x) { gate_uf[x] = gate_uf[gate_uf[x]]; x = gate_uf[x]; }
    return x;
}
static void gate_uf_union(int16_t a, int16_t b) {
    a = gate_uf_find(a); b = gate_uf_find(b);
    if (a != b) { if (a < b) gate_uf[b] = a; else gate_uf[a] = b; }
}

/* ── Step 1: build binary blue mask ──────────────────────────────────────── */
static void gate_make_blue_mask(const struct image_t *img, uint8_t *out_mask)
{
    int w = img->w, h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;
    int total = w * h;

    /* threshold */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t Y = yuv422_Y(buf, w, x, y);
            uint8_t U = yuv422_U(buf, w, x, y);
            uint8_t V = yuv422_V(buf, w, x, y);
            int idx = y * w + x;
            /* U_MAX=255 and V_MIN=0 are always true for uint8_t — omitted */
            out_mask[idx] = (Y <= (uint8_t)GATE_BLUE_Y_MAX &&
                             U >= (uint8_t)GATE_BLUE_U_MIN &&
                             V <= (uint8_t)GATE_BLUE_V_MAX) ? 255 : 0;
        }
    }

    /* morphological open (erode then dilate) with 5×5 rect kernel:
       reuse gate_eroded as scratch.  We implement erode/dilate as a
       simple box majority-vote (threshold = kernel_area, i.e. ALL pixels
       must be set for erode, ANY pixel for dilate). */
    int k = 5, half = 2;

    /* erode: all 5×5 neighbours must be set */
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

    /* dilate: any 5×5 neighbour set → set output */
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

    /* morphological close (dilate then erode): reuse same scheme */
    /* dilate first */
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
    /* then erode */
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

/* ── Step 2: extract blobs from blue mask ─────────────────────────────────── */
static int gate_extract_blobs(const uint8_t *mask, int w, int h,
                               gate_blob_t blobs[], int max_blobs)
{
    int total    = w * h;
    int min_area = w * h * GATE_BLOB_MIN_AREA_NUM / GATE_BLOB_MIN_AREA_DEN;
    if (min_area < 1) min_area = 1;

    /* connected components */
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
    for (int i = 0; i < total; i++)
        if (gate_cc_labels[i]) gate_cc_labels[i] = gate_uf_find(gate_cc_labels[i]);

    /* stats per label */
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

    /* fill blobs array */
    int nb = 0;
    for (int l = 1; l <= max_lbl && nb < max_blobs; l++) {
        int area = (int)gate_area[l];
        if (area < min_area) continue;

        int bw = (int)(gate_x1[l] - gate_x0[l] + 1);
        int bh = (int)(gate_y1[l] - gate_y0[l] + 1);
        if (bw < GATE_BLOB_MIN_DIM || bh < GATE_BLOB_MIN_DIM) continue;

        blobs[nb].x_min      = gate_x0[l];
        blobs[nb].y_min      = gate_y0[l];
        blobs[nb].x_max      = gate_x1[l] + 1;  /* exclusive */
        blobs[nb].y_max      = gate_y1[l] + 1;
        blobs[nb].cx         = (area > 0) ? (int)(gate_cx_sum[l] / area) : (gate_x0[l] + gate_x1[l]) / 2;
        blobs[nb].cy         = (area > 0) ? (int)(gate_cy_sum[l] / area) : (gate_y0[l] + gate_y1[l]) / 2;
        blobs[nb].area       = area;
        blobs[nb].aspect_num = bw;    /* aspect = bw / bh — stored as ratio */
        blobs[nb].aspect_den = bh;
        nb++;
    }
    return nb;
}

/* ── Step 3: checker score for the LEFT band of a blob ──────────────────────
 *
 * The checker pattern sits on the LEFT face of each blue bar (lower x values
 * = floor side in the portrait/rotated image).
 *
 * We read Y (luma) values directly from the YUV422 buffer for each column in
 * the search band, track per-column min and max, and count how many columns
 * have range > GATE_CHECKER_CONTRAST.  Score = count * 100 / n_cols.
 * Returns score in percent (0-100).
 * ─────────────────────────────────────────────────────────────────────────── */
static int gate_checker_score(const struct image_t *img, const gate_blob_t *b)
{
    int img_w = img->w, img_h = img->h;
    const uint8_t *buf = (const uint8_t *)img->buf;

    int bw     = b->x_max - b->x_min;
    int bh     = b->y_max - b->y_min;
    int band_w = bw * GATE_CHECKER_BAND_NUM / GATE_CHECKER_BAND_DEN;
    if (band_w < 4) band_w = 4;

    /* band to the LEFT of the blob: cols [x_min - band_w, x_min) */
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

    return high_contrast_cols * 100 / n_cols;  /* 0-100 */
}

/* ── Step 4: parallelism check ──────────────────────────────────────────────
 *
 * Returns 1 if two blobs look like a valid gate pair.
 *
 * Uses only integer arithmetic. The skew angle is approximated via the dot
 * product: dot(join_unit, long_axis) = cos(angle_between).
 * We need angle_between close to 90°, i.e. |dot| close to 0.
 * |dot| = |cos(angle)| = |sin(skew)| ≈ skew (radians) for small skew.
 * Threshold: sin(max_skew) where max_skew = GATE_MAX_SKEW_DEG10/10 degrees.
 * sin(10°) ≈ 0.174 — we use fixed-point: threshold = 174, scale = 1000.
 * ─────────────────────────────────────────────────────────────────────────── */
static int gate_blobs_parallel(const gate_blob_t *b1, const gate_blob_t *b2)
{
    /* 0. Vertical separation: |dy| > |dx| */
    int dx = b2->cx - b1->cx;
    int dy = b2->cy - b1->cy;
    int adx = dx < 0 ? -dx : dx;
    int ady = dy < 0 ? -dy : dy;
    if (ady <= adx) return 0;

    /* 1. Similar aspect ratio (bw/bh)
          |ar1 - ar2| / max(ar1, ar2) <= GATE_MAX_ASPECT_DIFF_PCT / 100
          Using cross-multiplication to avoid division:
          |bw1*bh2 - bw2*bh1| * 100 <= max(bw1*bh2, bw2*bh1) * pct        */
    int bw1 = b1->aspect_num, bh1 = b1->aspect_den;
    int bw2 = b2->aspect_num, bh2 = b2->aspect_den;
    if (bh1 <= 0 || bh2 <= 0) return 0;
    int cross1 = bw1 * bh2;
    int cross2 = bw2 * bh1;
    int diff   = cross1 - cross2; if (diff < 0) diff = -diff;
    int mx     = cross1 > cross2 ? cross1 : cross2;
    if (diff * 100 > mx * GATE_MAX_ASPECT_DIFF_PCT) return 0;

    /* 2. Similar area */
    int a1 = b1->area, a2 = b2->area;
    int adiff = a1 - a2; if (adiff < 0) adiff = -adiff;
    int amax  = a1 > a2 ? a1 : a2;
    if (amax <= 0) return 0;
    if (adiff * 100 > amax * GATE_MAX_AREA_DIFF_PCT) return 0;

    /* 3. Centroid join ≈ perpendicular to long axis.
          avg_w = average blob width, avg_h = average blob height
          if avg_w >= avg_h → bars are horizontal → long axis = (1, 0)
            → join must be mostly vertical → |dx| / |join| should be small
          if avg_h > avg_w  → bars are vertical  → long axis = (0, 1)
            → join must be mostly horizontal → |dy| / |join| should be small

          |dot(join_unit, long_axis)| = sin(skew)
          Threshold: sin(max_skew_deg) * |join|
          sin(10°) * 1000 ≈ 174 → use fixed-point scale 1000               */
    int avg_w = ((b1->x_max - b1->x_min) + (b2->x_max - b2->x_min)) / 2;
    int avg_h = ((b1->y_max - b1->y_min) + (b2->y_max - b2->y_min)) / 2;

    /* |dot(join, long_axis)| — the component of join along the long axis   */
    int dot_abs = (avg_w >= avg_h) ? adx : ady;

    /* |join|^2 = dx^2 + dy^2; we compare dot_abs^2 / join^2 <= sin^2(skew)
       Cross-multiply: dot_abs^2 * 1000000 <= join^2 * sin2_thresh
       sin(10°)^2 ≈ 0.03015 → sin2_thresh = 30150 / 1000000              */
    /* Simpler: compare dot_abs * 1000 / join <= sin_thresh_1000
       where sin_thresh_1000 = sin(GATE_MAX_SKEW_DEG10/10 degrees) * 1000
       For 10°: 174.  Avoid sqrt by squaring both sides:
       dot_abs^2 * 1000000 <= (dx^2+dy^2) * 174^2 = * 30276             */
    int join2   = dx * dx + dy * dy;
    int dot2    = dot_abs * dot_abs;
    /* sin_thresh^2 * 1000000, sin(10°)^2 * 1000000 = 30154 */
    int sin2_scaled = 30154;  /* sin²(10°) × 1 000 000 */
    if (dot2 * 1000000 > join2 * sin2_scaled) return 0;

    return 1;
}

/* ── Main gate detection function ─────────────────────────────────────────── */
void detect_gate(struct image_t *img, int result[2])
{
    result[0] = 0;
    result[1] = 0;

    int w = img->w, h = img->h;
    if (w > MAX_IMAGE_WIDTH)  w = MAX_IMAGE_WIDTH;
    if (h > MAX_IMAGE_HEIGHT) h = MAX_IMAGE_HEIGHT;

    /* Step 1 — blue mask */
    gate_make_blue_mask(img, gate_blue_mask);

    /* Step 2 — blobs */
    int n_blobs = gate_extract_blobs(gate_blue_mask, w, h,
                                     gate_blobs, GATE_MAX_BLOBS);
    if (n_blobs < 2) return;

    /* Step 3 + 4 — find best parallel pair with checker on both blobs */
    int best_area  = -1;
    int best_mid_x = 0;
    int gate_found = 0;

    for (int i = 0; i < n_blobs; i++) {
        for (int j = i + 1; j < n_blobs; j++) {
            const gate_blob_t *b1 = &gate_blobs[i];
            const gate_blob_t *b2 = &gate_blobs[j];

            if (!gate_blobs_parallel(b1, b2)) continue;

            /* Checker confirmation on both blobs */
            int score1 = gate_checker_score(img, b1);
            int score2 = gate_checker_score(img, b2);
            if (score1 < GATE_CHECKER_MIN_FRAC_PCT) continue;
            if (score2 < GATE_CHECKER_MIN_FRAC_PCT) continue;

            /* Best = largest combined area */
            int combined = b1->area + b2->area;
            if (combined > best_area) {
                best_area  = combined;
                best_mid_x = (b1->cx + b2->cx) / 2;
                gate_found = 1;
            }
        }
    }

    if (gate_found) {
        result[0] = 1;
        result[1] = best_mid_x;
    }
}