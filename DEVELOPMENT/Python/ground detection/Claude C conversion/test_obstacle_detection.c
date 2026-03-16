/*
 * test_obstacle_detection.c
 *
 * Standalone test harness for the C obstacle-detection pipeline.
 * Reads raw YUV422 frames from disk, runs get_obstacle_info(),
 * and dumps intermediate + final results so a Python script can
 * compare them against the Python pipeline.
 *
 * BUILD (on any Linux/macOS box with gcc):
 *   gcc -O2 -Wall -o test_obstacle_detection \
 *       test_obstacle_detection.c              \
 *       team10_get_obstacle_info_standalone.c   \
 *       -lm
 *
 * USAGE:
 *   ./test_obstacle_detection  <raw_yuv422_file>  <width>  <height>  <output_dir>
 *
 *   raw_yuv422_file : binary file, width*height*2 bytes, UYVY pixel order
 *   output_dir      : directory where result files are written
 *
 * OUTPUT FILES  (inside output_dir):
 *   mask_after_classify.raw   – binary mask after decision tree (before blur)
 *   mask_after_blur.raw       – mask after median blur
 *   mask_after_cc.raw         – mask after connected-component filtering
 *   mask_after_fill.raw       – mask after hole filling
 *   mask_flipped.raw          – left-right flipped mask
 *   boundary_rows.txt         – one integer per line (boundary per row)
 *   obstacles.txt             – "count\n" then "start width\n" per obstacle
 *   green_frac.txt            – single float
 *   ground_found.txt          – 0 or 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>

/* ─────────────────────────────────────────────────────────────────────────────
 *  MINIMAL STUBS  – just enough to compile without the full Paparazzi tree.
 *  These must match the real definitions in image.h / state.h / std.h.
 * ───────────────────────────────────────────────────────────────────────────── */

#ifndef _CV_LIB_VISION_IMAGE_H   /* guard so we don't clash if image.h is present */
#define _CV_LIB_VISION_IMAGE_H

/* types that std.h normally provides */
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

/* minimal FloatEulers stub */
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

/* image_create / image_free stubs (only used by the test) */
static void image_create(struct image_t *img, uint16_t width, uint16_t height, enum image_type type)
{
    img->type = type;
    img->w = width;
    img->h = height;
    if (type == IMAGE_YUV422)
        img->buf_size = 2 * width * height;
    else
        img->buf_size = width * height;
    img->buf = malloc(img->buf_size);
}

static void image_free(struct image_t *img)
{
    free(img->buf);
    img->buf = NULL;
}

#endif /* _CV_LIB_VISION_IMAGE_H */

/* ── now include the obstacle detection header ────────────────────────────── */
#include "team10_get_obstacle_info.h"

/* ─────────────────────────────────────────────────────────────────────────────
 *  FILE I/O HELPERS
 * ───────────────────────────────────────────────────────────────────────────── */

static int write_raw(const char *dir, const char *name, const uint8_t *data, int nbytes)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return -1; }
    fwrite(data, 1, nbytes, f);
    fclose(f);
    return 0;
}

static int write_int_array(const char *dir, const char *name, const int *data, int n)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return -1; }
    for (int i = 0; i < n; i++)
        fprintf(f, "%d\n", data[i]);
    fclose(f);
    return 0;
}

static int write_float(const char *dir, const char *name, float val)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%.8f\n", val);
    fclose(f);
    return 0;
}

static int write_int(const char *dir, const char *name, int val)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n", val);
    fclose(f);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  Expose internal work buffers for dumping intermediate results.
 *  These are declared in team10_get_obstacle_info.c as static, so we
 *  duplicate a lightweight version of the pipeline here for step-by-step
 *  dumping.  The REAL pipeline function is still called at the end for
 *  the final obstacle output.
 * ───────────────────────────────────────────────────────────────────────────── */

/* We need access to internal buffers. Since they're static in the .c file,
   we re-declare temporary buffers here for step-by-step dumping only. */
static uint8_t  dbg_mask[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];
static uint8_t  dbg_mask2[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];
static uint8_t  dbg_flipped[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];

/* ─────────────────────────────────────────────────────────────────────────────
 *  MAIN
 * ───────────────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <raw_yuv422_file> <width> <height> <output_dir>\n"
            "\n"
            "  raw_yuv422_file:  binary, width*height*2 bytes, UYVY layout\n"
            "  output_dir:       where result files are written\n",
            argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    int width  = atoi(argv[2]);
    int height = atoi(argv[3]);
    const char *outdir = argv[4];

    if (width <= 0 || height <= 0 || width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_HEIGHT) {
        fprintf(stderr, "Invalid dimensions %dx%d (max %dx%d)\n",
                width, height, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
        return 1;
    }

    /* create output directory (best-effort) */
    mkdir(outdir, 0755);

    /* ── read raw YUV422 file ────────────────────────────────────────────── */
    int expected_size = width * height * 2;
    FILE *f = fopen(input_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", input_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < expected_size) {
        fprintf(stderr, "File too small: %ld bytes, expected %d\n", fsize, expected_size);
        fclose(f);
        return 1;
    }

    struct image_t img;
    image_create(&img, width, height, IMAGE_YUV422);
    if (fread(img.buf, 1, expected_size, f) != (size_t)expected_size) {
        fprintf(stderr, "Short read\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    printf("Loaded %dx%d YUV422 image from %s (%d bytes)\n",
           width, height, input_path, expected_size);

    /* ── Step-by-step pipeline for dumping intermediates ──────────────────── */

    /* Step 1: classify only (no blur) */
    {
        const uint8_t *buf = (const uint8_t *)img.buf;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uint8_t Y = buf[y * width * 2 + x * 2 + 1];
                uint8_t U = buf[y * width * 2 + (x & ~1) * 2];
                uint8_t V = buf[y * width * 2 + (x & ~1) * 2 + 2];
                dbg_mask[y * width + x] = is_ground_pixel(Y, U, V);
            }
        }
        write_raw(outdir, "mask_after_classify.raw", dbg_mask, width * height);
        printf("  [1] Decision tree classification done\n");
    }

    /* Step 2: median blur */
    {
        /* do a binary median blur of the classified mask */
        int ksize = DEFAULT_MEDIAN_KSIZE;
        int half = ksize / 2;
        int threshold = (ksize * ksize) / 2;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int count = 0;
                int y0 = (y - half < 0) ? 0 : y - half;
                int y1 = (y + half >= height) ? height - 1 : y + half;
                int x0 = (x - half < 0) ? 0 : x - half;
                int x1 = (x + half >= width) ? width - 1 : x + half;
                for (int ky = y0; ky <= y1; ky++)
                    for (int kx = x0; kx <= x1; kx++)
                        if (dbg_mask[ky * width + kx]) count++;
                dbg_mask2[y * width + x] = (count > threshold) ? 255 : 0;
            }
        }
        memcpy(dbg_mask, dbg_mask2, width * height);
        write_raw(outdir, "mask_after_blur.raw", dbg_mask, width * height);
        printf("  [2] Median blur done (ksize=%d)\n", DEFAULT_MEDIAN_KSIZE);
    }

    /* check green fraction */
    int green_count = 0;
    for (int i = 0; i < width * height; i++)
        if (dbg_mask[i]) green_count++;
    float green_frac = (float)green_count / (float)(width * height);
    printf("  Green fraction: %.4f (threshold: %.4f)\n", green_frac, DEFAULT_OA_COLOR_FRAC);

    if (green_frac <= DEFAULT_OA_COLOR_FRAC) {
        printf("  NO GROUND FOUND – skipping remaining steps\n");
        write_float(outdir, "green_frac.txt", green_frac);
        write_int(outdir, "ground_found.txt", 0);

        char path[512];
        snprintf(path, sizeof(path), "%s/obstacles.txt", outdir);
        FILE *of = fopen(path, "w");
        fprintf(of, "0\n");
        fclose(of);

        image_free(&img);
        return 0;
    }

    /* Step 3: connected-component filtering */
    isolate_ground_blob(dbg_mask, width, height, DEFAULT_BLOB_AREA_THRESH);
    write_raw(outdir, "mask_after_cc.raw", dbg_mask, width * height);
    printf("  [3] Connected-component filtering done (threshold=%d)\n", DEFAULT_BLOB_AREA_THRESH);

    /* Step 4: fill holes */
    fill_holes_mask(dbg_mask, width, height);
    write_raw(outdir, "mask_after_fill.raw", dbg_mask, width * height);
    printf("  [4] Hole filling done\n");

    /* Step 5: flip + boundary */
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            dbg_flipped[y * width + x] = dbg_mask[y * width + (width - 1 - x)];
    write_raw(outdir, "mask_flipped.raw", dbg_flipped, width * height);

    int boundary_rows[MAX_IMAGE_HEIGHT];
    find_ground_boundary(dbg_flipped, width, height,
                         boundary_rows,
                         DEFAULT_MIN_GROUND_PX,
                         DEFAULT_MAX_GAP,
                         DEFAULT_SMOOTH_KERNEL);
    write_int_array(outdir, "boundary_rows.txt", boundary_rows, height);
    printf("  [5] Ground boundary scan done\n");

    /* ── Now run the REAL pipeline for the final obstacle output ──────── */
    float baseline[MAX_IMAGE_WIDTH];
    int   baseline_inited = 0;
    struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
    int   boundary_full[MAX_IMAGE_WIDTH];
    int   ground_found = 0;
    float gf = 0.0f;

    uint8_t n_obs = get_obstacle_info(
            &img,
            baseline,
            &baseline_inited,
            DEFAULT_OA_COLOR_FRAC,
            DEFAULT_MEDIAN_KSIZE,
            DEFAULT_MIN_WIDTH,
            obstacles,
            boundary_full,
            &ground_found,
            &gf);

    printf("  [FINAL] %d obstacle(s) detected  (ground_found=%d, green_frac=%.4f)\n",
           n_obs, ground_found, gf);
    for (int i = 0; i < n_obs; i++)
        printf("    obstacle %d: left=%d  width=%d\n", i, obstacles[i].start, obstacles[i].width);

    /* write final outputs */
    write_float(outdir, "green_frac.txt", gf);
    write_int(outdir, "ground_found.txt", ground_found);
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/obstacles.txt", outdir);
        FILE *of = fopen(path, "w");
        fprintf(of, "%d\n", n_obs);
        for (int i = 0; i < n_obs; i++)
            fprintf(of, "%d %d\n", obstacles[i].start, obstacles[i].width);
        fclose(of);
    }

    /* also dump the baseline for multi-frame testing */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/baseline.txt", outdir);
        FILE *of = fopen(path, "w");
        for (int i = 0; i < MAX_IMAGE_WIDTH; i++)
            fprintf(of, "%.4f\n", baseline[i]);
        fclose(of);
    }

    image_free(&img);
    printf("All outputs written to %s/\n", outdir);
    return 0;
}
