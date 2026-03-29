/*
 * team10_logic.c
 *
 * Waypoint selection logic for the obstacle avoidance autopilot.
 *
 * The ground detection pipeline produces a "ground baseline" array: one float
 * per image column that says how high (in pixels) the ground boundary is in
 * that column. A lower value means more ground is visible — the column is
 * "greener" and therefore safer to fly toward.
 *
 * This file converts that baseline, plus lists of detected obstacle and plant
 * regions, into a single target column that the drone should steer toward.
 *
 * The key decision is:
 *   - If the path is clear, just pick the column with the most visible ground.
 *   - If there are obstacles or plants blocking some columns, "erase" those
 *     columns by setting their baseline value to the image height (worst
 *     possible) before picking the best remaining column.
 *   - The erased zone is padded with a bias on each side so the drone does
 *     not try to squeeze through a gap right at the obstacle edge.
 *
 * NOTE: The camera is rotated 90° CW on the drone. "Columns" in the ground
 * baseline array correspond to horizontal strips in the drone's field of view
 * when displayed upright. Left column = right side of drone view, etc.
 */

#include "team10_logic.h"
#include <string.h>

/**
 * @brief Find which section of the image has the most visible ground.
 *
 * Divides the baseline array into ns equal sections and returns the 1-based
 * index of the section whose average baseline value is lowest (= most ground
 * visible). Useful when the autopilot needs a coarse left/centre/right
 * decision rather than an exact pixel column.
 *
 * @param gb  Ground baseline array (one float per column, length w)
 * @param w   Number of columns (image width)
 * @param ns  Number of sections to divide into
 * @return    1-based section index of the greenest section (1 = leftmost)
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
            best_sec = s + 1;   /* 1-based index */
        }
    }

    return best_sec;
}

/**
 * @brief Find the single column with the lowest (most visible) ground.
 *
 * Scans the baseline and returns the column index where gb[i] is smallest.
 * Ties are broken by taking the first (leftmost) occurrence.
 *
 * This is the fine-grained version of greenest_section_column — it gives
 * the autopilot a precise pixel column to aim for rather than a section.
 *
 * @param gb  Ground baseline array (one float per column, length w)
 * @param w   Number of columns
 * @return    Column index (0 to w-1) with the lowest baseline value
 */
int greenest_pixel(const float gb[], int w)
{
    if (!gb || w <= 0) return 0;

    float min_val = gb[0];
    int   min_col = 0;

    for (int i = 1; i < w; i++) {
        if (gb[i] < min_val) {
            min_val = gb[i];
            min_col = i;
        }
    }

    return min_col;
}

/**
 * @brief Check whether each obstacle extends all the way to the image bottom.
 *
 * An obstacle that "touches the bottom" has no ground visible below it —
 * the ground boundary disappears inside the obstacle column range. This means
 * the obstacle blocks the path to the floor and the drone cannot simply fly
 * under or beside it safely.
 *
 * We check this by comparing baseline_height against image height h.
 * If baseline_height >= h it means find_ground_boundary() never found ground
 * in those columns — the obstacle extends to the far side of the floor.
 *
 * Obstacles that do NOT touch the bottom are small raised objects; the floor
 * is still visible around them and they may not block the drone's path.
 *
 * @param oo          Array of detected obstacle regions
 * @param no          Number of obstacles
 * @param h           Image height (= number of boundary rows)
 * @param touches_out OUTPUT: array of length no; [i]=1 if obstacle i touches bottom
 * @param touch       OUTPUT: 1 if ANY obstacle touches, 0 if none do
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
 * @brief Mark obstacle and plant columns as impassable, then pick the best
 *        remaining column.
 *
 * Works on a LOCAL COPY of the baseline so the original is never modified.
 * Each obstacle or plant region (plus a safety bias on each side) is "erased"
 * by setting those columns to the maximum possible value (h). The greenest_pixel
 * call then naturally avoids those columns.
 *
 * Only obstacles that TOUCH the bottom are erased — obstacles that float above
 * the ground may not actually block the drone's path.
 * ALL plants are erased regardless, because potted plants always block the
 * full floor width.
 *
 * @param oo           Obstacle region array
 * @param no           Number of obstacles
 * @param po           Plant region array
 * @param np           Number of plants
 * @param gb           Original baseline array (NOT modified)
 * @param w            Number of columns
 * @param h            Image height (used as the "blocked" sentinel value)
 * @param touches_out  From obstacle_touches_ground(): which obstacles touch bottom
 * @param obs_bias     Pixel padding to add around each obstacle region
 * @param plant_bias   Pixel padding to add around each plant region
 * @return             Chosen safe column index (0 to w-1)
 */
int bias_logic(const struct obstacle_region_t oo[], uint8_t no,
               const struct obstacle_region_t po[], uint8_t np,
               const float gb[], int w, int h,
               const uint8_t touches_out[],
               int obs_bias, int plant_bias)
{
    /* Work on a local copy so the caller's baseline is untouched */
    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    /* Erase columns occupied by ground-touching obstacles (plus their safety margin) */
    for (int i = 0; i < no; i++) {
        if (!touches_out[i]) continue;   /* obstacle not touching ground — skip */

        int low_bound  = (int)oo[i].start - obs_bias;
        int high_bound = (int)oo[i].start + (int)oo[i].width - 1 + obs_bias;
        if (low_bound < 0) low_bound = 0;
        if (high_bound >= w) high_bound = w - 1;

        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;   /* mark as worst possible — drone will avoid this */
    }

    /* Erase columns occupied by plants (plus their safety margin) */
    for (int i = 0; i < np; i++) {
        int low_bound  = (int)po[i].start - plant_bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + plant_bias;
        if (low_bound < 0) low_bound = 0;
        if (high_bound >= w) high_bound = w - 1;

        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    return greenest_pixel(gb_lg, w);
}

/**
 * @brief Compute a safety margin in pixels proportional to the obstacle width.
 *
 * Instead of a fixed pixel bias, this scales with the obstacle's actual size.
 * A wide obstacle gets a bigger margin so the drone steers well clear; a
 * narrow obstacle gets a smaller margin. The minimum is always 1 pixel so
 * we never return zero.
 *
 * @param o    Pointer to the obstacle or plant region
 * @param frac Fraction of the region width to use as bias (e.g. 0.5 = 50%)
 * @return     Bias in pixels (minimum 1)
 */
int normalised_bias(const struct obstacle_region_t *o, float frac)
{
    int bias = (int)(o->width * frac);
    if (bias < 1) bias = 1;
    return bias;
}

/**
 * @brief Main waypoint selection function with fixed pixel bias.
 *
 * Decides which column the drone should aim for by:
 *   1. If no obstacles and no plants: return the greenest (most visible) column.
 *   2. If only non-ground-touching obstacles and no plants: same — no erasing needed.
 *   3. Otherwise: erase dangerous zones with fixed obs_bias/plant_bias margins
 *      and return the greenest remaining column.
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
 * @return            Chosen safe column index (0 to w-1)
 */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                          const struct obstacle_region_t po[], uint8_t np,
                          const float gb[], int w, int h,
                          int obs_bias, int plant_bias)
{
    uint8_t touches_out[MAX_OBSTACLE_REGIONS];
    uint8_t touch = 0;
    obstacle_touches_ground(oo, no, h, touches_out, &touch);

    /* No detections at all — just find the clearest column */
    if (!no && !np)
        return greenest_pixel(gb, w);

    /* Obstacles present but none touching the ground, and no plants — safe to proceed */
    if (!np && !touch)
        return greenest_pixel(gb, w);

    /* Some obstacles or plants need to be avoided */
    return bias_logic(oo, no, po, np, gb, w, h, touches_out, obs_bias, plant_bias);
}

/**
 * @brief Waypoint selection with width-proportional bias per obstacle/plant.
 *
 * Same logic as motion_logic() but instead of a single global pixel bias, each
 * obstacle or plant gets its own bias proportional to its width. This makes the
 * drone give wider clearance to large obstacles and can be tuned more naturally
 * with fractions rather than absolute pixel values.
 *
 * @param oo              Obstacle region array
 * @param no              Number of obstacles
 * @param po              Plant region array
 * @param np              Number of plants
 * @param gb              Ground baseline array
 * @param w               Number of columns
 * @param h               Image height
 * @param obs_bias_frac   Fraction of each obstacle's width to use as bias (e.g. 0.5)
 * @param plant_bias_frac Fraction of each plant's width to use as bias (e.g. 0.7)
 * @return                Chosen safe column index (0 to w-1)
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
        return greenest_pixel(gb, w);

    if (!np && !touch)
        return greenest_pixel(gb, w);

    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    /* Erase each obstacle with its own proportional margin */
    for (int i = 0; i < no; i++) {
        if (!touches_out[i]) continue;
        int bias       = normalised_bias(&oo[i], obs_bias_frac);
        int low_bound  = (int)oo[i].start - bias;
        int high_bound = (int)oo[i].start + (int)oo[i].width - 1 + bias;
        if (low_bound < 0)   low_bound = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* Erase each plant with its own proportional margin */
    for (int i = 0; i < np; i++) {
        int bias       = normalised_bias(&po[i], plant_bias_frac);
        int low_bound  = (int)po[i].start - bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + bias;
        if (low_bound < 0)   low_bound = 0;
        if (high_bound >= w) high_bound = w - 1;
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    return greenest_pixel(gb_lg, w);
}