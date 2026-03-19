#include "team10_rtp_utilities.h"
#include <string.h>

// ── 1. MASK PRINTER ───────────────────────────────────────────────────────────
void draw_mask_printer(struct image_t *img, uint8_t *work_mask, int w, int h)
{
    uint8_t *src = (uint8_t *)img->buf;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x += 2) {
            uint8_t *p  = &src[y * 2 * w + 2 * x];
            uint8_t  g0 = work_mask[y * w + x];
            uint8_t  g1 = work_mask[y * w + x + 1];
            p[0] = 128;
            p[2] = 128;
            p[1] = g0 ? 255 : 0;
            p[3] = g1 ? 255 : 0;
        }
    }
}

// ── 2. TOOLBAR VERTICAL (text overlay) ───────────────────────────────────────
static const uint8_t font3x5[20][5] = {
    {0x7,0x5,0x5,0x5,0x7}, // 0
    {0x2,0x2,0x2,0x2,0x2}, // 1
    {0x7,0x1,0x7,0x4,0x7}, // 2
    {0x7,0x1,0x7,0x1,0x7}, // 3
    {0x5,0x5,0x7,0x1,0x1}, // 4
    {0x7,0x4,0x7,0x1,0x7}, // 5
    {0x7,0x4,0x7,0x5,0x7}, // 6
    {0x7,0x1,0x1,0x1,0x1}, // 7
    {0x7,0x5,0x7,0x5,0x7}, // 8
    {0x7,0x5,0x7,0x1,0x7}, // 9
    {0x7,0x4,0x6,0x5,0x7}, // G [10]
    {0x5,0x7,0x7,0x5,0x5}, // N [11]
    {0x6,0x5,0x5,0x5,0x6}, // D [12]
    {0x7,0x5,0x5,0x5,0x7}, // O [13]
    {0x6,0x5,0x6,0x5,0x6}, // B [14]
    {0x7,0x4,0x7,0x1,0x7}, // S [15]
    {0x5,0x5,0x2,0x5,0x5}, // % [16]
    {0x0,0x0,0x0,0x0,0x0}, // space [17]
    {0x0,0x0,0x7,0x0,0x0}, // - [18]
    {0x0,0x2,0x0,0x2,0x0}, // : [19]
};

#define DRAW_CHAR(cx, cy, ci) \
do { \
    for (int _r = 0; _r < 5; _r++) { \
        uint8_t _bits = font3x5[(ci)][_r]; \
        for (int _c = 0; _c < 3; _c++) { \
            if (_bits & (0x4 >> _c)) { \
                int _px = (cx) + _c; \
                int _py = (cy) + _r; \
                if (_px < w && _py < h) { \
                    uint8_t *_p = &((uint8_t *)img->buf)[_py * 2 * w + (_px & ~1) * 2]; \
                    _p[1 + (_px & 1) * 2] = 255; \
                } \
            } \
        } \
    } \
} while(0)

#define DRAW_NUM(cx, cy, num) \
do { \
    int _n = (num); \
    if (_n >= 100) { DRAW_CHAR((cx), (cy), _n / 100);       (cx) += 4; } \
    if (_n >= 10)  { DRAW_CHAR((cx), (cy), (_n / 10) % 10); (cx) += 4; } \
    DRAW_CHAR((cx), (cy), _n % 10); (cx) += 4; \
} while(0)

void draw_toolbar_vertical(struct image_t *img, int w, int h,
                           float gf, uint8_t no,
                           struct obstacle_region_t oo[])
{
    // Blacken top 10 rows
    {
        uint8_t *src = (uint8_t *)img->buf;
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < w; x += 2) {
                uint8_t *p = &src[y * 2 * w + 2 * x];
                p[0] = 128; p[1] = 0; p[2] = 128; p[3] = 0;
            }
        }
    }

    // Draw "GND:XX%"
    {
        int cx = 2, cy = 2;
        int pct = (int)(gf * 100.0f);
        DRAW_CHAR(cx, cy, 10); cx += 4;  // G
        DRAW_CHAR(cx, cy, 11); cx += 4;  // N
        DRAW_CHAR(cx, cy, 12); cx += 4;  // D
        DRAW_CHAR(cx, cy, 19); cx += 4;  // :
        DRAW_NUM(cx, cy, pct);
        DRAW_CHAR(cx, cy, 16); cx += 4;  // %
    }

    // Draw "OBS:XX"
    {
        int cx = 64, cy = 2;
        DRAW_CHAR(cx, cy, 13); cx += 4;  // O
        DRAW_CHAR(cx, cy, 14); cx += 4;  // B
        DRAW_CHAR(cx, cy, 15); cx += 4;  // S
        DRAW_CHAR(cx, cy, 19); cx += 4;  // :

        if (no == 0) {
            DRAW_CHAR(cx, cy, 18);            // -
        } else {
            for (int i = 0; i < no; i++) {
                DRAW_NUM(cx, cy, oo[i].width);
                cx += 6;
            }
        }
    }
}

#undef DRAW_CHAR
#undef DRAW_NUM

// ── 3. OBSTACLE DETECTION BAR ────────────────────────────────────────────────
void draw_obstacle_detection_bar(struct image_t *img, int w, int h,
                                 uint8_t no, struct obstacle_region_t oo[])
{
    uint8_t *src = (uint8_t *)img->buf;

    static uint8_t is_obs_y[MAX_IMAGE_HEIGHT];
    memset(is_obs_y, 0, w);
    for (int i = 0; i < no; i++) {
        int s = (int)oo[i].start;
        int e = s + (int)oo[i].width - 1;
        if (s < 0) s = 0;
        if (e >= w) e = w - 1;
        for (int r = s; r <= e; r++) is_obs_y[r] = 1;
    }

    for (int y = 0; y < h; y++) {
        for (int x = w - 4; x < w - 2; x += 2) {
            uint8_t *p = &src[y * 2 * w + 2 * x];
            if (is_obs_y[h - 1 - y]) {
                p[0] = 85;   // U → red
                p[1] = 76;   // Y0
                p[2] = 255;  // V
                p[3] = 76;   // Y1
            } else {
                p[0] = 128;  // U → black
                p[1] = 0;    // Y0
                p[2] = 128;  // V
                p[3] = 0;    // Y1
            }
        }
    }
}

// ── GET SAFEST COLUMN (0=leftmost, 6=rightmost, -1=all blocked) ──────────────
int get_safest_column(int h, uint8_t no, struct obstacle_region_t oo[])
{
    int col_width = h / 7;
    uint8_t has_obstacle[7] = {0};

    for (int col = 0; col < 7; col++) {
        int y_start = col * col_width;
        int y_end   = (col == 6) ? h : y_start + col_width;

        for (int i = 0; i < no; i++) {
            int s  = (int)oo[i].start;
            int e  = s + (int)oo[i].width - 1;
            int vs = h - 1 - e;
            int ve = h - 1 - s;
            if (vs < 0)  vs = 0;
            if (ve >= h) ve = h - 1;
            if (vs < y_end && ve >= y_start) {
                has_obstacle[col] = 1;
                break;
            }
        }
    }

    // Priority order: center outward → 3, 2, 4, 1, 5, 0, 6
    static const int priority[7] = {3, 2, 4, 1, 5, 0, 6};
    for (int i = 0; i < 7; i++) {
        int col = priority[i];
        if (!has_obstacle[col]) return col;
    }
    return -1; // all blocked
}

// ── 4. SAFE DIRECTION BAR ────────────────────────────────────────────────────
void draw_safe_direction_bar(struct image_t *img, int w, int h,
                             uint8_t no, struct obstacle_region_t oo[])
{
    uint8_t *src = (uint8_t *)img->buf;
    int col_width = h / 7;
    int best = get_safest_column(h, no, oo);

    for (int col = 0; col < 7; col++) {
        int y_start = col * col_width;
        int y_end   = (col == 6) ? h : y_start + col_width;

        int has_obstacle = 0;
        for (int i = 0; i < no; i++) {
            int s  = (int)oo[i].start;
            int e  = s + (int)oo[i].width - 1;
            int vs = h - 1 - e;
            int ve = h - 1 - s;
            if (vs < 0)  vs = 0;
            if (ve >= h) ve = h - 1;
            if (vs < y_end && ve >= y_start) {
                has_obstacle = 1;
                break;
            }
        }

        for (int y = y_start; y < y_end; y++) {
            for (int x = w - 2; x < w; x += 2) {
                uint8_t *p = &src[y * 2 * w + 2 * x];
                if (col == best) {
                    p[0] = 44;  p[1] = 150; p[2] = 21;  p[3] = 150; // green
                } else {
                    p[0] = 128; p[1] = 0;   p[2] = 128; p[3] = 0;   // black
                }
            }
        }
    }
}