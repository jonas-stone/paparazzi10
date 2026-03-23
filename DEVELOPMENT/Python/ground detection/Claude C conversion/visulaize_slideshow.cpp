/*
 * visualize_slideshow.cpp
 *
 * Pure C/C++ replacement for visualize_slideshow_Win32.py.
 * No Python, no ctypes, no struct-alignment issues.
 *
 * Reads every .jpg in a folder, feeds each one through the
 * team10 obstacle+plant detection pipeline, and displays results
 * in an OpenCV window — identical layout to the Python slideshow.
 *
 * BUILD (Windows, MinGW64 + OpenCV):
 *   g++ -O2 -I. -I<opencv_include> -o visualize_slideshow.exe ^
 *       visualize_slideshow.cpp team10_get_obstacle_info_standalone.c ^
 *       -lm -lopencv_world4xx -L<opencv_lib>
 *
 * BUILD (Linux/macOS):
 *   g++ -O2 -I. -o visualize_slideshow \
 *       visualize_slideshow.cpp team10_get_obstacle_info_standalone.c \
 *       -lm $(pkg-config --cflags --libs opencv4)
 *
 * Or just run:   make slideshow
 *   (add the target from the comment at the bottom of this file)
 *
 * KEYBOARD CONTROLS:
 *   q / ESC   — quit
 *   SPACE     — pause / resume
 *   d         — next frame
 *   a         — previous frame
 *
 * EDIT THE SECTION BELOW to set your image folder and start frame.
 */

/* ═══════════════════════════════════════════════════════════════════════════
 *  ▶  USER SETTINGS — edit these two lines
 * ═══════════════════════════════════════════════════════════════════════════ */
#define IMAGE_FOLDER  "../paparazzi10/DEVELOPMENT/downloads from drone/20260320/"
#define START_INDEX   1          /* skip this many images at the beginning   */
#define DELAY_MS      50         /* ms between frames when playing           */
/* ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

/* ── Paparazzi type stubs (same as in standalone .c) ──────────────────────── */
#ifndef _CV_LIB_VISION_IMAGE_H
#define _CV_LIB_VISION_IMAGE_H
struct FloatEulers { float phi, theta, psi; };
enum image_type { IMAGE_YUV422, IMAGE_GRAYSCALE, IMAGE_JPEG,
                  IMAGE_GRADIENT, IMAGE_INT16 };
struct image_t {
    enum image_type   type;
    uint16_t          w, h;
    struct timeval    ts;
    struct FloatEulers eulers;
    uint32_t          pprz_ts;
    uint8_t           buf_idx;
    uint32_t          buf_size;
    void             *buf;       /* ← plain C pointer, no alignment games */
};
#endif

#include "team10_get_obstacle_info.h"

/* ── OpenCV (C++ header, links against the C++ runtime) ───────────────────── */
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>

/* ═══════════════════════════════════════════════════════════════════════════
 *  BGR → UYVY conversion  (mirrors bgr_to_uyvy_numpy in Python)
 *
 *  Output layout per pixel pair (x even, x+1 odd):
 *    byte 0: U   (from even pixel)
 *    byte 1: Y0  (even pixel)
 *    byte 2: V   (from even pixel)
 *    byte 3: Y1  (odd  pixel)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bgr_to_uyvy(const cv::Mat &bgr, uint8_t *uyvy_out)
{
    cv::Mat yuv;
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);

    int h = bgr.rows, w = bgr.cols;
    for (int y = 0; y < h; y++) {
        const uint8_t *src = yuv.ptr<uint8_t>(y);
        uint8_t       *dst = uyvy_out + y * w * 2;
        for (int x = 0; x < w; x += 2) {
            uint8_t Y0 = src[x * 3 + 0];
            uint8_t U0 = src[x * 3 + 1];
            uint8_t V0 = src[x * 3 + 2];
            uint8_t Y1 = src[(x + 1) * 3 + 0];
            dst[x * 2 + 0] = U0;
            dst[x * 2 + 1] = Y0;
            dst[x * 2 + 2] = V0;
            dst[x * 2 + 3] = Y1;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  COLLECT + SORT JPEG PATHS
 *  Sorted numerically by the first run of digits in the filename,
 *  matching Python's   int(re.search(r'\d+', basename).group())
 * ═══════════════════════════════════════════════════════════════════════════ */
static int first_number_in_name(const std::string &path)
{
    /* find the filename part */
    size_t sep = path.find_last_of("/\\");
    std::string name = (sep == std::string::npos) ? path : path.substr(sep + 1);
    for (size_t i = 0; i < name.size(); i++)
        if (isdigit((unsigned char)name[i]))
            return atoi(name.c_str() + i);
    return 0;
}

static std::vector<std::string> collect_jpegs(const char *folder)
{
    std::vector<std::string> out;
    std::vector<cv::String>  cv_paths;
    std::string pattern = std::string(folder) + "*.jpg";
    cv::glob(pattern, cv_paths, false);
    /* also try uppercase */
    std::vector<cv::String> cv_paths2;
    std::string pattern2 = std::string(folder) + "*.JPG";
    cv::glob(pattern2, cv_paths2, false);
    for (auto &p : cv_paths)  out.push_back(std::string(p));
    for (auto &p : cv_paths2) out.push_back(std::string(p));
    std::sort(out.begin(), out.end(),
              [](const std::string &a, const std::string &b){
                  return first_number_in_name(a) < first_number_in_name(b);
              });
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  DRAW DETECTIONS  (matches draw_original_style in Python exactly)
 *
 *  img_rot   — image already rotated 90° CCW
 *  obs[]     — obstacle regions in rotated coords
 *  n_obs
 *  plants[]  — plant regions in rotated coords
 *  n_plants
 *  boundary  — boundary_arr[0..h_orig-1] in original (pre-rotation) coords
 *  baseline  — ground_baseline[0..h_orig-1], NULL if not yet inited
 *  h_orig    — height of the original (pre-rotation) image = width of img_rot
 *  label     — text label drawn at the bottom
 * ═══════════════════════════════════════════════════════════════════════════ */
static cv::Mat draw_detections(const cv::Mat                   &img_rot,
                               const struct obstacle_region_t  *obs,    int n_obs,
                               const struct obstacle_region_t  *plants, int n_plants,
                               const int                       *boundary,
                               const float                     *baseline,
                               int                              h_orig,
                               const char                      *label)
{
    cv::Mat vis = img_rot.clone();
    int h_rot = vis.rows;   /* = original image width  */
    int w_rot = vis.cols;   /* = original image height */

    /* ── Ground baseline curve (blue) ────────────────────────────────────── */
    if (baseline != NULL) {
        std::vector<cv::Point> pts;
        for (int r = 0; r < h_orig; r++) {
            int col = (int)baseline[r];
            if (col < h_rot)
                pts.push_back(cv::Point(r, col));
        }
        if (pts.size() > 1) {
            const cv::Point *arr = pts.data();
            int n = (int)pts.size();
            cv::polylines(vis, &arr, &n, 1, false, cv::Scalar(255, 0, 0), 1);
        }
    }

    /* ── Free-space boundary curve (cyan) ────────────────────────────────── */
    if (boundary != NULL) {
        std::vector<cv::Point> pts;
        for (int r = 0; r < h_orig; r++) {
            int col = boundary[r];
            if (col < h_rot)
                pts.push_back(cv::Point(r, col));
        }
        if (pts.size() > 1) {
            const cv::Point *arr = pts.data();
            int n = (int)pts.size();
            cv::polylines(vis, &arr, &n, 1, false, cv::Scalar(255, 255, 0), 1);
        }
    }

    /* ── Obstacle regions (red) ──────────────────────────────────────────── */
    for (int i = 0; i < n_obs; i++) {
        int s  = obs[i].start;
        int e  = s + obs[i].width - 1;
        int w  = obs[i].width;
        int bh = obs[i].baseline_height;

        /* find top-y from boundary within [s,e] */
        int yt = 0;
        if (boundary != NULL) {
            int best = h_rot;
            for (int r = s; r <= e && r < h_orig; r++)
                if (boundary[r] < best) best = boundary[r];
            yt = (best < h_rot) ? best : 0;
        }

        cv::rectangle(vis, cv::Point(s, 0), cv::Point(e, h_rot - 1),
                      cv::Scalar(0, 0, 255), 2);

        char buf[32];
        snprintf(buf, sizeof(buf), "w=%d", w);
        cv::putText(vis, buf, cv::Point(s, std::max(15, yt - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

        snprintf(buf, sizeof(buf), "h=%d", bh);
        cv::putText(vis, buf, cv::Point(s, std::max(30, yt - 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 150, 255), 1);

        int y_line = std::min(bh, h_rot - 1);
        cv::line(vis, cv::Point(s, y_line), cv::Point(e, y_line),
                 cv::Scalar(255, 255, 255), 1);
    }

    /* ── Plant regions (green) ───────────────────────────────────────────── */
    for (int i = 0; i < n_plants; i++) {
        int s = plants[i].start;
        int e = s + plants[i].width - 1;
        int w = plants[i].width;
        cv::rectangle(vis, cv::Point(s, 0), cv::Point(e, h_rot - 1),
                      cv::Scalar(0, 255, 0), 2);
        char buf[32];
        snprintf(buf, sizeof(buf), "p=%d", w);
        cv::putText(vis, buf, cv::Point(s, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    }

    /* ── Frame label (white outline + black fill for readability) ─────────── */
    cv::putText(vis, label, cv::Point(5, h_rot - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis, label, cv::Point(5, h_rot - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1);

    return vis;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  BUILD THE FULL COMPOSITE FRAME
 *
 *  TOP ROW:  original | work-mask | plant-overlay   (3 panels side by side)
 *  BOT ROW:  detection view | info panel
 *  HDR:      one-line status bar
 * ═══════════════════════════════════════════════════════════════════════════ */
static cv::Mat build_combined_frame(
        const cv::Mat                  &bgr_s,        /* resized original       */
        const uint8_t                  *work_mask_ptr,/* from get_work_mask()   */
        const uint8_t                  *plant_mask_ptr,/* from get_plant_mask() */
        const struct obstacle_region_t *obs,    int n_obs,
        const struct obstacle_region_t *plants, int n_plants,
        const int                      *boundary,
        const float                    *baseline,      /* NULL if not inited    */
        float                           green_frac,
        bool                            is_ground,
        bool                            paused,
        int                             frame_idx,
        int                             total_frames,
        const char                     *name,
        int                             delay_ms)
{
    int tH = bgr_s.rows, tW = bgr_s.cols;
    const char *st = is_ground ? "GROUND FOUND" : "NO GROUND";

    /* ── Rotate 90° CCW (same as Python cv2.ROTATE_90_COUNTERCLOCKWISE) ───── */
    cv::Mat img_rot, mask_bgr_rot, plant_rot_mat;
    cv::rotate(bgr_s, img_rot, cv::ROTATE_90_COUNTERCLOCKWISE);

    /* work mask → grayscale → BGR */
    cv::Mat mask_gray(tH, tW, CV_8UC1, (void *)work_mask_ptr);
    cv::Mat mask_rot_gray;
    cv::rotate(mask_gray, mask_rot_gray, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::cvtColor(mask_rot_gray, mask_bgr_rot, cv::COLOR_GRAY2BGR);

    /* plant overlay: copy of rotated original, green pixels painted */
    cv::Mat plant_gray(tH, tW, CV_8UC1, (void *)plant_mask_ptr);
    cv::Mat plant_rot_gray;
    cv::rotate(plant_gray, plant_rot_gray, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::Mat plant_overlay = img_rot.clone();
    plant_overlay.setTo(cv::Scalar(0, 200, 0), plant_rot_gray > 0);

    /* ── TOP ROW ─────────────────────────────────────────────────────────── */
    cv::Mat top;
    cv::hconcat(std::vector<cv::Mat>{img_rot, mask_bgr_rot, plant_overlay}, top);

    /* ── DETECTION VIEW ─────────────────────────────────────────────────── */
    char det_label[64];
    snprintf(det_label, sizeof(det_label), "%d obs, %d plants", n_obs, n_plants);
    cv::Mat det_vis = draw_detections(img_rot,
                                      obs, n_obs, plants, n_plants,
                                      boundary, baseline, tW,
                                      det_label);

    /* ── INFO PANEL ──────────────────────────────────────────────────────── */
    int target_w = top.cols;
    int det_w    = det_vis.cols;
    int extra_w  = target_w - det_w;
    cv::Mat bot;

    if (extra_w > 0) {
        cv::Mat info(det_vis.rows, extra_w, CV_8UC3, cv::Scalar(0, 0, 0));

        /* build text lines */
        char lines[32][80];
        int  nl = 0;

        snprintf(lines[nl++], 80, "Frame %d/%d", frame_idx, total_frames);
        snprintf(lines[nl++], 80, "%s", name);
        snprintf(lines[nl++], 80, " ");
        snprintf(lines[nl++], 80, "Status: %s", st);
        snprintf(lines[nl++], 80, "Green:  %.3f", green_frac);
        snprintf(lines[nl++], 80, " ");
        snprintf(lines[nl++], 80, "Obstacles: %d", n_obs);
        for (int i = 0; i < n_obs && nl < 28; i++)
            snprintf(lines[nl++], 80, "  [%d] row=%d-%d w=%d h=%d",
                     i, obs[i].start, obs[i].start + obs[i].width - 1,
                     obs[i].width, obs[i].baseline_height);
        snprintf(lines[nl++], 80, " ");
        snprintf(lines[nl++], 80, "Plants: %d", n_plants);
        for (int i = 0; i < n_plants && nl < 30; i++)
            snprintf(lines[nl++], 80, "  [%d] row=%d-%d w=%d",
                     i, plants[i].start, plants[i].start + plants[i].width - 1,
                     plants[i].width);
        snprintf(lines[nl++], 80, " ");
        snprintf(lines[nl++], 80, "%s  delay=%dms",
                 paused ? "PAUSED" : "PLAYING", delay_ms);
        snprintf(lines[nl++], 80, "q=quit  SPACE=pause");
        snprintf(lines[nl++], 80, "a=prev  d=next");

        int y = 20;
        for (int i = 0; i < nl; i++, y += 17)
            cv::putText(info, lines[i], cv::Point(10, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.38,
                        cv::Scalar(200, 200, 200), 1);

        cv::hconcat(det_vis, info, bot);
    } else {
        bot = det_vis(cv::Rect(0, 0, target_w, det_vis.rows));
    }

    /* ── HEADER BAR ──────────────────────────────────────────────────────── */
    cv::Mat hdr(30, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    char hdr_text[256];
    snprintf(hdr_text, sizeof(hdr_text),
             "C PIPELINE | %s (%dx%d)  %s  green=%.3f  obs=%d plt=%d",
             name, tW, tH, st, green_frac, n_obs, n_plants);
    cv::putText(hdr, hdr_text, cv::Point(8, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    /* ── STACK VERTICALLY ────────────────────────────────────────────────── */
    cv::Mat combined;
    cv::vconcat(std::vector<cv::Mat>{hdr, top, bot}, combined);
    return combined;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  UYVY IMAGE BUFFER  (static, matches C MAX_IMAGE_WIDTH × MAX_IMAGE_HEIGHT)
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint8_t uyvy_buf[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 2];

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* allow overriding folder from command line */
    const char *folder    = (argc >= 2) ? argv[1] : IMAGE_FOLDER;
    int         start_idx = (argc >= 3) ? atoi(argv[2]) : START_INDEX;
    int         delay_ms  = (argc >= 4) ? atoi(argv[3]) : DELAY_MS;

    /* ── Collect images ───────────────────────────────────────────────────── */
    std::vector<std::string> paths = collect_jpegs(folder);
    if (paths.empty()) {
        fprintf(stderr, "No .jpg files found in: %s\n", folder);
        return 1;
    }
    if (start_idx > 0 && start_idx < (int)paths.size())
        paths.erase(paths.begin(), paths.begin() + start_idx);

    printf("Found %d images (starting at index %d)\n",
           (int)paths.size(), start_idx);

    /* ── Persistent pipeline state ───────────────────────────────────────── */
    static float ground_baseline[MAX_IMAGE_HEIGHT];
    int          baseline_inited = 0;

    struct obstacle_region_t obs_out[MAX_OBSTACLE_REGIONS];
    struct obstacle_region_t plants_out[MAX_PLANT_REGIONS];
    uint8_t  plant_count   = 0;
    int      ground_found  = 0;
    float    green_frac    = 0.0f;
    static int boundary_arr[MAX_IMAGE_HEIGHT];

    /* ── OpenCV window ────────────────────────────────────────────────────── */
    const char *WIN = "C Pipeline Slideshow  |  q=quit  SPACE=pause  a=prev  d=next";
    cv::namedWindow(WIN, cv::WINDOW_NORMAL);

    int      idx              = 0;
    bool     paused           = false;
    bool     needs_processing = true;
    cv::Mat  last_frame;

    while (idx < (int)paths.size()) {
        const std::string &path = paths[idx];
        const char *name = path.c_str() + path.find_last_of("/\\") + 1;

        /* ── Process frame ────────────────────────────────────────────────── */
        if (needs_processing) {
            cv::Mat bgr = cv::imread(path);
            if (bgr.empty()) {
                fprintf(stderr, "Cannot read: %s\n", path.c_str());
                idx++;
                continue;
            }

            /* clamp to C buffer limits */
            int tW = std::min(bgr.cols, MAX_IMAGE_WIDTH);
            int tH = std::min(bgr.rows, MAX_IMAGE_HEIGHT);
            if (tW % 2) tW--;   /* UYVY needs even width */
            if (tW != bgr.cols || tH != bgr.rows)
                printf("   Clamped %dx%d -> %dx%d\n", bgr.cols, bgr.rows, tW, tH);

            cv::Mat bgr_s;
            cv::resize(bgr, bgr_s, cv::Size(tW, tH), 0, 0, cv::INTER_AREA);

            /* BGR → UYVY directly into static buffer → no heap, no alignment issue */
            bgr_to_uyvy(bgr_s, uyvy_buf);

            /* fill image_t — plain C struct, pointer set directly */
            struct image_t img;
            memset(&img, 0, sizeof(img));
            img.type     = IMAGE_YUV422;
            img.w        = (uint16_t)tW;
            img.h        = (uint16_t)tH;
            img.buf_size = (uint32_t)(tW * tH * 2);
            img.buf      = uyvy_buf;   /* ← direct C pointer assignment */

            printf("[%4d/%d] %-30s %dx%d  ",
                   idx + start_idx, (int)paths.size() + start_idx - 1,
                   name, tW, tH);
            fflush(stdout);

            /* run pipeline */
            memset(obs_out,    0, sizeof(obs_out));
            memset(plants_out, 0, sizeof(plants_out));
            plant_count  = 0;
            ground_found = 0;
            green_frac   = 0.0f;

            uint8_t n_obs = get_obstacle_info(
                &img,
                ground_baseline,
                &baseline_inited,
                0.05f,   /* oa_color_count_frac */
                5,       /* median_ksize        */
                20,      /* min_width           */
                obs_out,
                plants_out,
                &plant_count,
                boundary_arr,
                &ground_found,
                &green_frac
            );

            printf("green=%.3f %s  %d obs  %d plt\n",
                   green_frac,
                   ground_found ? "GROUND FOUND" : "NO GROUND",
                   (int)n_obs, (int)plant_count);

            /* get mask pointers from the compiled library
               (work_mask and plant_mask_full are static inside the .c) */
            extern uint8_t work_mask[];
            extern uint8_t plant_mask_full[];

            last_frame = build_combined_frame(
                bgr_s,
                work_mask,
                plant_mask_full,
                obs_out,    n_obs,
                plants_out, plant_count,
                boundary_arr,
                baseline_inited ? ground_baseline : NULL,
                green_frac,
                ground_found != 0,
                paused,
                idx + start_idx,
                (int)paths.size() + start_idx - 1,
                name,
                delay_ms
            );

            needs_processing = false;
        }

        /* ── Display ──────────────────────────────────────────────────────── */
        if (!last_frame.empty())
            cv::imshow(WIN, last_frame);

        int wait = paused ? 30 : delay_ms;
        int key  = cv::waitKey(wait) & 0xFF;

        if (key == 'q' || key == 27 /* ESC */) {
            break;
        } else if (key == ' ') {
            paused = !paused;
            if (!paused) { idx++; needs_processing = true; }
        } else if (key == 'd') {
            idx++; needs_processing = true;
        } else if (key == 'a') {
            if (idx > 0) { idx--; needs_processing = true; }
        } else if (!paused) {
            idx++; needs_processing = true;
        }
    }

    cv::destroyAllWindows();
    printf("\nDone. Processed %d frames.\n", idx);
    return 0;
}

/*
 * ─────────────────────────────────────────────────────────────────────────
 *  ADD THIS TARGET TO YOUR Makefile:
 *
 *  slideshow: visualize_slideshow.cpp team10_get_obstacle_info_standalone.c
 *      g++ -O2 -Wall -I. -o visualize_slideshow \
 *          visualize_slideshow.cpp team10_get_obstacle_info_standalone.c \
 *          -lm $$(pkg-config --cflags --libs opencv4)
 *
 *  On Windows with MinGW64 + OpenCV (adjust paths to match your install):
 *  OPENCV_DIR = C:/opencv/build
 *  slideshow: visualize_slideshow.cpp team10_get_obstacle_info_standalone.c
 *      g++ -O2 -Wall -I. -I$(OPENCV_DIR)/include \
 *          -o visualize_slideshow.exe \
 *          visualize_slideshow.cpp team10_get_obstacle_info_standalone.c \
 *          -lm -L$(OPENCV_DIR)/x64/mingw/lib -lopencv_world490 \
 *          -static-libgcc -static-libstdc++
 * ─────────────────────────────────────────────────────────────────────────
 */