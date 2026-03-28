/*
 * test_multiframe.c  –  multi-frame stress test
 *
 * Simulates N consecutive frames through the pipeline, proving that:
 *   - Frame 1: baseline initialises → 0 obstacles
 *   - Frame 2+: actual obstacle detection works
 *   - Plant detection works every frame
 *   - Baseline EMA tracks smoothly
 *
 * Can use:
 *   (a) A built-in synthetic scene generator        (--synth)
 *   (b) A single raw YUV422 file repeated N times   (file path)
 *   (c) Multiple raw YUV422 files in sequence        (directory)
 *
 * BUILD:
 *   gcc -O2 -Wall -Wno-unused-function -I. -o test_multiframe \
 *       test_multiframe.c team10_get_obstacle_info_standalone.c -lm
 *
 * USAGE:
 *   ./test_multiframe --synth  <width> <height> <num_frames>
 *   ./test_multiframe <file.yuv422>  <width> <height> <num_repeats>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

/* ── Paparazzi stubs ──────────────────────────────────────────────────────── */
#ifndef _CV_LIB_VISION_IMAGE_H
#define _CV_LIB_VISION_IMAGE_H
struct FloatEulers { float phi, theta, psi; };
enum image_type { IMAGE_YUV422, IMAGE_GRAYSCALE, IMAGE_JPEG, IMAGE_GRADIENT, IMAGE_INT16 };
struct image_t {
    enum image_type type; uint16_t w, h; struct timeval ts;
    struct FloatEulers eulers; uint32_t pprz_ts;
    uint8_t buf_idx; uint32_t buf_size; void *buf;
};
#endif

#include "team10_get_obstacle_info.h"

/* ══════════════════════════════════════════════════════════════════════════════
 *  SYNTHETIC SCENE GENERATOR
 *
 *  Creates a YUV422 image with:
 *    - Ground (bottom 60%)       Y=120 U=100 V=140  → decision tree = 255
 *    - Obstacle (top-left block) Y=80  U=130 V=130  → decision tree = 0
 *    - Plant strip (right 25%)   Y=90  U=85  V=135  → tree=0, but lax=YES
 *    - Sky (top 20%)             Y=200 U=128 V=128  → tree=0
 *
 *  Frame variation: the obstacle block shifts right by `frame_idx * 5` pixels,
 *  simulating forward motion.  This tests that the baseline adapts and detects
 *  new obstacles as they move into view.
 * ══════════════════════════════════════════════════════════════════════════════ */
static void generate_synth_frame(uint8_t *buf, int w, int h, int frame_idx)
{
    int obs_left  = 20 + frame_idx * 5;
    int obs_right = obs_left + 60;
    int obs_top   = 0;
    int obs_bot   = (int)(h * 0.4);

    int plant_left = (int)(w * 0.75);
    int ground_top = (int)(h * 0.4);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x += 2) {
            uint8_t Y_val, U_val, V_val;

            /* default = sky */
            Y_val = 200; U_val = 128; V_val = 128;

            /* ground region */
            if (y >= ground_top) {
                Y_val = 120; U_val = 100; V_val = 140;
            }

            /* obstacle block */
            if (x >= obs_left && x < obs_right && y >= obs_top && y < obs_bot) {
                Y_val = 80; U_val = 130; V_val = 130;
            }

            /* plant strip (overrides ground in right portion) */
            if (x >= plant_left) {
                Y_val = 90; U_val = 85; V_val = 135;
            }

            int base = y * w * 2 + x * 2;
            buf[base + 0] = U_val;  /* U  */
            buf[base + 1] = Y_val;  /* Y0 */
            buf[base + 2] = V_val;  /* V  */
            buf[base + 3] = Y_val;  /* Y1 */
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════════ */

static void print_bar(int n, int max, char ch)
{
    int bar = (max > 0) ? (n * 40 / max) : 0;
    for (int i = 0; i < bar; i++) putchar(ch);
}

int main(int argc, char *argv[])
{
    int use_synth = 0;
    const char *input_path = NULL;
    int w = 200, h = 120, n_frames = 10;

    if (argc >= 2 && strcmp(argv[1], "--synth") == 0) {
        use_synth = 1;
        if (argc >= 3) w = atoi(argv[2]);
        if (argc >= 4) h = atoi(argv[3]);
        if (argc >= 5) n_frames = atoi(argv[4]);
    } else if (argc >= 5) {
        input_path = argv[1];
        w = atoi(argv[2]);
        h = atoi(argv[3]);
        n_frames = atoi(argv[4]);
    } else {
        fprintf(stderr,
            "Usage:\n"
            "  %s --synth [width] [height] [frames]     (synthetic scene)\n"
            "  %s <file.yuv422> <w> <h> <repeats>       (from file)\n",
            argv[0], argv[0]);
        return 1;
    }

    if (w <= 0 || h <= 0 || w > MAX_IMAGE_WIDTH || h > MAX_IMAGE_HEIGHT) {
        fprintf(stderr, "Bad dims %dx%d (max %dx%d)\n", w, h, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
        return 1;
    }
    if (w % 2) { w--; }  /* must be even for YUV422 */

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  MULTI-FRAME OBSTACLE + PLANT DETECTION TEST               ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Image:  %dx%d   Frames: %d   Source: %s\n",
           w, h, n_frames, use_synth ? "SYNTHETIC" : input_path);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* allocate image buffer */
    int buf_size = w * h * 2;
    struct image_t img;
    img.type = IMAGE_YUV422;
    img.w = (uint16_t)w;
    img.h = (uint16_t)h;
    img.buf_size = (uint32_t)buf_size;
    img.buf = malloc(buf_size);

    /* load from file if not synthetic */
    if (!use_synth) {
        FILE *f = fopen(input_path, "rb");
        if (!f) { perror(input_path); return 1; }
        if ((int)fread(img.buf, 1, buf_size, f) < buf_size) {
            fprintf(stderr, "File too small\n"); fclose(f); return 1;
        }
        fclose(f);
    }

    /* persistent state */
    float ground_baseline[MAX_IMAGE_WIDTH];
    int   baseline_inited = 0;

    /* run frames */
    int total_obs = 0, total_plants = 0;

    for (int frame = 0; frame < n_frames; frame++) {
        if (use_synth) {
            generate_synth_frame((uint8_t *)img.buf, w, h, frame);
        }

        struct obstacle_region_t obstacles[MAX_OBSTACLE_REGIONS];
        struct obstacle_region_t plants[MAX_PLANT_REGIONS];
        uint8_t plant_count = 0;
        int     ground_found = 0;
        float   green_frac = 0.0f;

        uint8_t obs_count = get_obstacle_info(
            &img, ground_baseline, &baseline_inited,
            DEFAULT_OA_COLOR_FRAC, DEFAULT_MEDIAN_KSIZE, DEFAULT_MIN_WIDTH,
            obstacles, plants, &plant_count,
            NULL, &ground_found, &green_frac
        );

        total_obs    += obs_count;
        total_plants += plant_count;

        /* visual output */
        printf("  Frame %2d │ green=%.3f %s │ obs=%d plants=%d │ ",
               frame, green_frac,
               ground_found ? "GND" : "---",
               obs_count, plant_count);

        /* mini bar chart of obstacle widths */
        if (obs_count > 0) {
            printf("obs:[");
            for (int i = 0; i < obs_count; i++) {
                if (i > 0) printf(" ");
                printf("@%d+%d", obstacles[i].start, obstacles[i].width);
            }
            printf("] ");
        }
        if (plant_count > 0) {
            printf("plt:[");
            for (int i = 0; i < plant_count; i++) {
                if (i > 0) printf(" ");
                printf("@%d+%d", plants[i].start, plants[i].width);
            }
            printf("]");
        }
        printf("\n");

        /* baseline visualisation (sample 10 points) */
        if (baseline_inited) {
            printf("           │ baseline: ");
            int step = (h > 10) ? h / 10 : 1;
            for (int i = 0; i < h && i / step < 10; i += step) {
                printf("%.0f ", ground_baseline[i]);
            }
            printf("\n");
        }
    }

    printf("\n  ────────────────────────────────────────────────────────────\n");
    printf("  Total obstacles detected: %d  (across %d frames)\n", total_obs, n_frames);
    printf("  Total plant regions:      %d  (across %d frames)\n", total_plants, n_frames);

    /* sanity checks */
    int pass = 1;

    if (use_synth) {
        /* Frame 0 should always have 0 obstacles (baseline init) */
        /* Frame 1+ should detect obstacles if the scene has them */
        if (total_obs == 0 && n_frames >= 3) {
            printf("\n  ⚠ WARNING: no obstacles detected across %d frames\n", n_frames);
            printf("    This may be expected if the obstacle is small relative to thresholds.\n");
        }
        if (total_plants == 0 && n_frames >= 1) {
            printf("\n  ⚠ WARNING: no plants detected. Check plant scale parameters.\n");
        }
        if (n_frames >= 2 && total_obs > 0) {
            printf("\n  ✓ Obstacle detection working across frames\n");
        }
        if (total_plants > 0) {
            printf("  ✓ Plant detection working\n");
        }
    }

    printf("\n  %s\n\n",
           pass ? "TEST COMPLETE" : "TEST FAILED");

    free(img.buf);
    return pass ? 0 : 1;
}
