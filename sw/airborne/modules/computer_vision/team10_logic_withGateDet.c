#include "team10_logic.h"
#include <string.h>
#include <stdlib.h>   /* abs() */

/* ══════════════════════════════════════════════════════════════════════════════
 *  CORRIDOR CLEAR THRESHOLD
 *  A column is considered "clear" (passable) if its baseline value is less than
 *  CLEAR_FRAC * image_height.  0.75 means the ground must be visible for at
 *  least the bottom 25 % of the image in that column.
 *
 *  IMPROVEMENT #1 — replaces the raw greenest_pixel call with a corridor-aware
 *  picker that prefers wide passable gaps over single deep pixels.
 * ══════════════════════════════════════════════════════════════════════════════ */
#define CLEAR_FRAC 0.75f

/* ══════════════════════════════════════════════════════════════════════════════
 *  WIDEST CORRIDOR CENTRE  [replaces greenest_pixel as the final picker]
 *
 *  Scans gb[] for contiguous runs of columns whose value is < CLEAR_FRAC*h
 *  ("clear" columns).  Returns the centre column of the widest such run.
 *
 *  Tie-break: if two runs share the same width the first (leftmost) one wins —
 *  but because we scan the whole array the returned column is always the true
 *  centre, never the edge of a plateau (fixes improvement #6).
 *
 *  Fallback: if no column clears the threshold (everything blocked) we fall
 *  back to the single column with the smallest baseline value so the drone
 *  still picks a direction to rotate toward.
 * ══════════════════════════════════════════════════════════════════════════════ */
int widest_corridor_centre(const float gb[], int w, int h)
{
    if (!gb || w <= 0) return 0;

    float threshold = CLEAR_FRAC * (float)h;

    int best_centre  = -1;
    int best_width   = 0;
    int run_start    = -1;

    for (int i = 0; i <= w; i++) {
        int clear = (i < w) && (gb[i] < threshold);

        if (clear && run_start < 0) {
            run_start = i;                              /* start new run      */
        } else if (!clear && run_start >= 0) {
            int run_w = i - run_start;
            if (run_w > best_width) {
                best_width  = run_w;
                best_centre = run_start + run_w / 2;   /* true centre        */
            }
            run_start = -1;
        }
    }

    if (best_centre >= 0)
        return best_centre;

    /* ── Fallback: no clear corridor — return centre of minimum-value plateau ── */
    /* Find global minimum */
    float min_val = gb[0];
    for (int i = 1; i < w; i++)
        if (gb[i] < min_val) min_val = gb[i];

    /* Find centre of the widest run at that minimum (improvement #6) */
    best_centre = 0;
    best_width  = 0;
    run_start   = -1;
    for (int i = 0; i <= w; i++) {
        int at_min = (i < w) && (gb[i] <= min_val + 0.5f);
        if (at_min && run_start < 0) {
            run_start = i;
        } else if (!at_min && run_start >= 0) {
            int run_w = i - run_start;
            if (run_w > best_width) {
                best_width  = run_w;
                best_centre = run_start + run_w / 2;
            }
            run_start = -1;
        }
    }
    return best_centre;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  GREENEST PIXEL  [kept for API compatibility — now delegates to corridor picker]
 *
 *  Previously this returned the first column with the minimum baseline value,
 *  which had a left-bias tie-breaking problem (improvement #6) and ignored
 *  corridor width (improvement #1).
 *
 *  It now delegates to widest_corridor_centre so all callers get the improved
 *  behaviour without needing their signatures changed.
 * ══════════════════════════════════════════════════════════════════════════════ */
int greenest_pixel(const float gb[], int w)
{
    /* h is not available here; pass MAX_IMAGE_HEIGHT as the height reference.
     * widest_corridor_centre only uses h to compute the clear threshold, so
     * using the compile-time constant is correct. */
    return widest_corridor_centre(gb, w, MAX_IMAGE_HEIGHT);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  GREENEST SECTION -- Column version  [unchanged]
 * ══════════════════════════════════════════════════════════════════════════════ */
int greenest_section_column(const float gb[], int w, int ns)
{
    int best_sec = 0;
    if (!gb || w <= 0 || ns <= 0) return best_sec;

    int section_w = w / ns;
    float best_avg = 1e9f;

    for (int s = 0; s < ns; s++) {
        int start = s * section_w;
        int end   = (s == ns - 1) ? w : start + section_w;

        float sum = 0.0f;
        for (int i = start; i < end; i++)
            sum += gb[i];

        float avg = sum / (float)(end - start);
        if (avg < best_avg) {
            best_avg = avg;
            best_sec = s + 1;
        }
    }

    return best_sec;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  OBSTACLE TOUCHES THE IMAGE BOTTOM  [unchanged]
 * ══════════════════════════════════════════════════════════════════════════════ */
uint8_t obstacle_touches_ground(const struct obstacle_region_t oo[], uint8_t no,
                                int h, uint8_t touches_out[], uint8_t *touch)
{
    *touch = 0;
    for (int i = 0; i < no; i++) {
        touches_out[i] = (oo[i].baseline_height >= (uint16_t)h) ? 1 : 0;
        if (touches_out[i]) *touch = 1;
    }
    return *touch;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  NORMALISED BIAS  [unchanged]
 * ══════════════════════════════════════════════════════════════════════════════ */
int normalised_bias(const struct obstacle_region_t *o, float frac)
{
    int bias = (int)(o->width * frac);
    if (bias < 1) bias = 1;
    return bias;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  BIAS LOGIC  [improvement #1: final pick uses widest_corridor_centre]
 * ══════════════════════════════════════════════════════════════════════════════ */
int bias_logic(const struct obstacle_region_t oo[], uint8_t no,
               const struct obstacle_region_t po[], uint8_t np,
               const float gb[], int w, int h,
               const uint8_t touches_out[],
               int obs_bias, int plant_bias)
{
    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    for (int i = 0; i < no; i++) {
        if (!touches_out[i]) continue;
        int low_bound  = (int)oo[i].start - obs_bias;
        int high_bound = (int)oo[i].start + (int)oo[i].width - 1 + obs_bias;
        if (low_bound  < 0) low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    for (int i = 0; i < np; i++) {
        int low_bound  = (int)po[i].start - plant_bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + plant_bias;
        if (low_bound  < 0) low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* IMPROVEMENT #1: widest clear corridor instead of single deepest pixel */
    return widest_corridor_centre(gb_lg, w, h);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  MOTION LOGIC  [unchanged logic, improvement #1 inherited via bias_logic]
 * ══════════════════════════════════════════════════════════════════════════════ */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                 const struct obstacle_region_t po[], uint8_t np,
                 const float gb[], int w, int h,
                 int obs_bias, int plant_bias)
{
    uint8_t touches_out[MAX_OBSTACLE_REGIONS];
    uint8_t touch = 0;
    obstacle_touches_ground(oo, no, h, touches_out, &touch);

    if (!no && !np)
        return widest_corridor_centre(gb, w, h);   /* #1 */

    if (!np && !touch)
        return widest_corridor_centre(gb, w, h);   /* #1 */

    return bias_logic(oo, no, po, np, gb, w, h, touches_out, obs_bias, plant_bias);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  MOTION LOGIC NORMALISED
 *
 *  IMPROVEMENT #1  — final pick via widest_corridor_centre.
 *  IMPROVEMENT #5  — plants use their own bias fraction (plant_bias_frac),
 *                    which should be set larger than obs_bias_frac in the
 *                    caller (e.g. 0.75 vs 0.5) because plant pots are narrow
 *                    and require proportionally more clearance.
 * ══════════════════════════════════════════════════════════════════════════════ */
int motion_logic_normalised(const struct obstacle_region_t oo[], uint8_t no,
                            const struct obstacle_region_t po[], uint8_t np,
                            const float gb[], int w, int h,
                            float obs_bias_frac, float plant_bias_frac)
{
    uint8_t touches_out[MAX_OBSTACLE_REGIONS];
    uint8_t touch = 0;
    obstacle_touches_ground(oo, no, h, touches_out, &touch);

    if (!no && !np)
        return widest_corridor_centre(gb, w, h);   /* #1 */

    if (!np && !touch)
        return widest_corridor_centre(gb, w, h);   /* #1 */

    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    for (int i = 0; i < no; i++) {
        if (!touches_out[i]) continue;
        int bias       = normalised_bias(&oo[i], obs_bias_frac);
        int low_bound  = (int)oo[i].start - bias;
        int high_bound = (int)oo[i].start + (int)oo[i].width - 1 + bias;
        if (low_bound  < 0)  low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* IMPROVEMENT #5: plant_bias_frac is intentionally a separate parameter
     * so the caller can set it higher than obs_bias_frac.                   */
    for (int i = 0; i < np; i++) {
        int bias       = normalised_bias(&po[i], plant_bias_frac);
        int low_bound  = (int)po[i].start - bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + bias;
        if (low_bound  < 0)  low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* IMPROVEMENT #1 */
    return widest_corridor_centre(gb_lg, w, h);
}