/*
 * team10_logic_withGateDet.c
 *
 * Waypoint selection logic for the obstacle avoidance autopilot.
 *
 * Takes the ground baseline array (one float per image column representing
 * how high the ground boundary is in that column), plus lists of detected
 * obstacle and plant regions, and returns a single target column for the
 * drone to steer toward.
 *
 * The two runtime globals clear_frac and min_corridor_width_px control what
 * counts as a "passable" column and how wide a gap must be before the drone
 * will commit to flying through it. Both can be adjusted via Paparazzi GCS
 * sliders without recompiling.
 */

#include "team10_logic.h"
#include <string.h>
#include <stdlib.h>

/* ── Runtime-tunable corridor parameters ──────────────────────────────────── */

/* A column is considered "clear" if its baseline value is below this fraction
   of the image height. 0.90 means the ground boundary must be visible in the
   top 90% of the image for that column to qualify as usable. */
float clear_frac = 0.90f;

/* Any clear column run shorter than this number of pixels is discarded.
   Should be at least the drone's body width in pixels at a typical flying
   distance so we never steer toward a gap the drone cannot fit through. */
int min_corridor_width_px = 60;

/**
 * @brief Find the centre column of the widest contiguous run of clear columns.
 *
 * Scans gb[] left to right, tracking runs of columns whose value is below
 * clear_frac * h ("clear"). The widest run that is also at least
 * min_corridor_width_px columns wide has its centre returned.
 *
 * If no run meets the minimum width, -1 is returned so the caller knows
 * there is no usable corridor and the drone should keep rotating.
 *
 * @param gb  Ground baseline array (one float per column, length w)
 * @param w   Number of columns
 * @param h   Image height (used to compute the clear threshold)
 * @return    Centre column of the widest clear corridor, or -1 if none found
 */
int widest_corridor_centre(const float gb[], int w, int h)
{
    if (!gb || w <= 0) return 0;

    float threshold = clear_frac * (float)h;

    int best_centre = -1;
    int best_width  = 0;
    int run_start   = -1;

    /* Iterate one past the end so the last run gets closed by the loop itself
       rather than needing a duplicate check after the loop. */
    for (int i = 0; i <= w; i++) {
        int clear = (i < w) && (gb[i] < threshold);

        if (clear && run_start < 0) {
            run_start = i;   /* start of a new clear run */
        } else if (!clear && run_start >= 0) {
            /* The run that started at run_start just ended at column i-1 */
            int run_w = i - run_start;
            /* Only accept this run if it is wide enough to be physically passable */
            if (run_w >= min_corridor_width_px && run_w > best_width) {
                best_width  = run_w;
                best_centre = run_start + run_w / 2;   /* centre of the run */
            }
            run_start = -1;
        }
    }

    /* -1 signals "no usable corridor found" — caller should keep rotating */
    return best_centre;
}

/**
 * @brief Return the column index with the most visible ground.
 *
 * Delegates to widest_corridor_centre() using the full image height.
 * Kept so existing call sites do not need to be updated.
 *
 * @param gb  Ground baseline array
 * @param w   Number of columns
 * @return    Widest corridor centre column, or -1 if no corridor qualifies
 */
int greenest_pixel(const float gb[], int w)
{
    return widest_corridor_centre(gb, w, MAX_IMAGE_HEIGHT);
}

/**
 * @brief Find which of ns equal sections of the baseline has the lowest average.
 *
 * Divides the baseline into ns equal-width sections and returns the 1-based
 * index of the section whose average baseline value is smallest — meaning the
 * most ground is visible in that section.
 *
 * @param gb  Ground baseline array (one float per column, length w)
 * @param w   Number of columns
 * @param ns  Number of sections to split into
 * @return    1-based index of the greenest section
 */
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
            best_sec = s + 1;   /* 1-based */
        }
    }

    return best_sec;
}

/**
 * @brief Check whether each obstacle has no detectable ground beneath it.
 *
 * An obstacle whose baseline_height >= h had no ground boundary found inside
 * its column range — meaning it blocks the full floor width and the drone
 * cannot fly under or beside it. These obstacles must be avoided; ones that
 * do not touch the bottom may be small enough to ignore.
 *
 * @param oo          Array of detected obstacle regions
 * @param no          Number of obstacles
 * @param h           Image height
 * @param touches_out OUTPUT: [i] = 1 if obstacle i touches the bottom, else 0
 * @param touch       OUTPUT: 1 if any obstacle touches, 0 if none do
 * @return            Same value as *touch
 */
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

/**
 * @brief Compute a pixel safety margin proportional to a region's width.
 *
 * Returns frac * region_width, with a minimum of 1 pixel so we never return
 * zero even for very narrow regions.
 *
 * @param o    Pointer to the obstacle or plant region
 * @param frac Fraction of the width to use as a margin (e.g. 0.5 = 50%)
 * @return     Margin in pixels, minimum 1
 */
int normalised_bias(const struct obstacle_region_t *o, float frac)
{
    int bias = (int)(o->width * frac);
    if (bias < 1) bias = 1;
    return bias;
}

/**
 * @brief Mark dangerous columns as impassable and return the centre of the
 *        widest remaining clear corridor.
 *
 * Works on a local copy of gb[] so the caller's array is never modified.
 * Columns occupied by ground-touching obstacles (plus obs_bias padding on
 * each side) and all plant columns (plus plant_bias padding) are set to h,
 * making them look like the worst possible ground. The corridor finder then
 * naturally avoids them.
 *
 * @param oo          Obstacle region array
 * @param no          Number of obstacles
 * @param po          Plant region array
 * @param np          Number of plants
 * @param gb          Original baseline array (not modified)
 * @param w           Number of columns
 * @param h           Image height (used as the "blocked" sentinel value)
 * @param touches_out Which obstacles touch the bottom (from obstacle_touches_ground)
 * @param obs_bias    Fixed pixel padding around each obstacle
 * @param plant_bias  Fixed pixel padding around each plant
 * @return            Centre column of the widest remaining clear corridor
 */
int bias_logic(const struct obstacle_region_t oo[], uint8_t no,
               const struct obstacle_region_t po[], uint8_t np,
               const float gb[], int w, int h,
               const uint8_t touches_out[],
               int obs_bias, int plant_bias)
{
    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    /* Erase columns belonging to ground-touching obstacles */
    for (int i = 0; i < no; i++) {
        if (!touches_out[i]) continue;   /* obstacle does not reach the floor — skip */
        int low_bound  = (int)oo[i].start - obs_bias;
        int high_bound = (int)oo[i].start + (int)oo[i].width - 1 + obs_bias;
        if (low_bound  < 0) low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* Erase columns belonging to plants */
    for (int i = 0; i < np; i++) {
        int low_bound  = (int)po[i].start - plant_bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + plant_bias;
        if (low_bound  < 0) low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    return widest_corridor_centre(gb_lg, w, h);
}

/**
 * @brief Choose a target column using fixed pixel bias values.
 *
 * Short-circuits to the corridor finder directly if there is nothing to avoid.
 * Otherwise delegates to bias_logic() to erase dangerous columns first.
 *
 * @param oo          Obstacle region array
 * @param no          Number of obstacles
 * @param po          Plant region array
 * @param np          Number of plants
 * @param gb          Ground baseline array
 * @param w           Number of columns
 * @param h           Image height
 * @param obs_bias    Fixed pixel margin around obstacles
 * @param plant_bias  Fixed pixel margin around plants
 * @return            Target column index, or -1 if no passable corridor exists
 */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                 const struct obstacle_region_t po[], uint8_t np,
                 const float gb[], int w, int h,
                 int obs_bias, int plant_bias)
{
    uint8_t touches_out[MAX_OBSTACLE_REGIONS];
    uint8_t touch = 0;
    obstacle_touches_ground(oo, no, h, touches_out, &touch);

    /* Nothing to avoid — find the widest open corridor directly */
    if (!no && !np)
        return widest_corridor_centre(gb, w, h);

    /* Obstacles present but none reach the floor, and no plants — still clear */
    if (!np && !touch)
        return widest_corridor_centre(gb, w, h);

    return bias_logic(oo, no, po, np, gb, w, h, touches_out, obs_bias, plant_bias);
}

/**
 * @brief Choose a target column using width-proportional bias values.
 *
 * Same decision tree as motion_logic() but each obstacle and plant gets its
 * own margin scaled to its width rather than a single fixed pixel value.
 * plant_bias_frac can be set higher than obs_bias_frac independently.
 *
 * @param oo              Obstacle region array
 * @param no              Number of obstacles
 * @param po              Plant region array
 * @param np              Number of plants
 * @param gb              Ground baseline array
 * @param w               Number of columns
 * @param h               Image height
 * @param obs_bias_frac   Fraction of each obstacle's width to use as its margin
 * @param plant_bias_frac Fraction of each plant's width to use as its margin
 * @return                Target column index, or -1 if no passable corridor exists
 */
int motion_logic_normalised(const struct obstacle_region_t oo[], uint8_t no,
                            const struct obstacle_region_t po[], uint8_t np,
                            const float gb[], int w, int h,
                            float obs_bias_frac, float plant_bias_frac)
{
    uint8_t touches_out[MAX_OBSTACLE_REGIONS];
    uint8_t touch = 0;
    obstacle_touches_ground(oo, no, h, touches_out, &touch);

    if (!no && !np)
        return widest_corridor_centre(gb, w, h);

    if (!np && !touch)
        return widest_corridor_centre(gb, w, h);

    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    /* Erase each obstacle column range with its own proportional margin */
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

    /* Erase each plant column range with its own proportional margin */
    for (int i = 0; i < np; i++) {
        int bias       = normalised_bias(&po[i], plant_bias_frac);
        int low_bound  = (int)po[i].start - bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + bias;
        if (low_bound  < 0)  low_bound  = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    return widest_corridor_centre(gb_lg, w, h);
}