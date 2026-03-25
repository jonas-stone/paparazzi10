#include "team10_logic.h"
#include <string.h>

/* ══════════════════════════════════════════════════════════════════════════════
 *  GREENEST SECTION -- Column version
 *  Finds the column region where the baseline gb[] is at its highest
 *  (i.e. smallest row value -- it starts counting from the top).
 *
 *  Inputs:
 *    gb[]   - current baseline array (one float per column, length w)
 *    ns     - number of sections
 *
 *  Output:
 *    n      - int from 0 to ns indicating which section is greenest (1-based index)
 * ══════════════════════════════════════════════════════════════════════════════ */
int greenest_section_column(const float gb[], int w, int ns)
{
    int best_sec = 0; // Default to the first section
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
 *  GREENEST SECTION -- Single pixel version
 *  Finds the pixel where the baseline gb[] is at its highest
 *  (i.e. smallest row value -- it starts counting from the top).
 *
 *  Inputs:
 *    gb[]   - current baseline array (one float per column, length w)
 *    w      - width of the image (length of gb[])
 *
 *  Output:
 *    min_col - int from 0 to w-1 indicating which column is greenest
 * ══════════════════════════════════════════════════════════════════════════════ */
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

/* ══════════════════════════════════════════════════════════════════════════════
 *  OBSTACLE TOUCHES THE IMAGE BOTTOM
 *  Checks whether each obstacle extends all the way to the bottom of the image,
 *  meaning no ground was detected beneath it.
 *
 *  Inputs:
 *    oo[]   - array of obstacle regions
 *    no     - number of obstacles
 *    h      - image height (pixels)
 *
 *  Output:
 *    touches_out[] - caller-provided array of size no;
 *                    touches_out[i] = 1 if obstacle i touches the ground, 0 otherwise
 * ══════════════════════════════════════════════════════════════════════════════ */
uint8_t obstacle_touches_ground(const struct obstacle_region_t oo[], uint8_t no,
                                int h, uint8_t touches_out[], uint8_t *touch)
{
    *touch = 0;
    for (int i = 0; i < no; i++) {
      // baseline height >= to the image height --> the obstacle extends to the bottom
        touches_out[i] = (oo[i].baseline_height >= (uint16_t)h) ? 1 : 0;
        if (touches_out[i]) *touch = 1;
    }
    // Any obstacle touching the ground --> touch = 1
    return *touch;
}


/* ══════════════════════════════════════════════════════════════════════════════
 *  BIAS LOGIC TO FIND THE WAYPOINT
  *  Modifies the baseline array by applying a bias around detected obstacles and plants,
  *  then finds the greenest column in the modified baseline.

  *  Inputs:
  *    oo[]       - array of obstacle regions
  *    no         - number of obstacles
  *    po[]       - array of plant regions
  *    np         - number of plants
  *    gb[]       - current baseline array (one float per column, length w)
  *    w          - width of the image (length of gb[])
  *    h          - height of the image
  *    touches_out- array indicating which obstacles touch the ground (1) vs. not
  *    obs_bias    - number of pixels to expand the obstacle region for biasing
  *    plant_bias  - number of pixels to expand the plant region for biasing

 * ══════════════════════════════════════════════════════════════════════════════ */
int bias_logic(const struct obstacle_region_t oo[], uint8_t no,
               const struct obstacle_region_t po[], uint8_t np,
               const float gb[], int w, int h,
               const uint8_t touches_out[],
               int obs_bias, int plant_bias)
{
    /* Copy gb into a local modifiable version */
    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    /* ── OBSTACLES ─────────────────────────────────────────────────────────── */
    for (int i = 0; i < no; i++) {
        /* Skip obstacles that don't touch the ground */
        if (!touches_out[i]) continue;

        int low_bound = (int)oo[i].start - obs_bias;
        int high_bound = (int)oo[i].start + (int)oo[i].width - 1 + obs_bias;
        if (low_bound < 0) low_bound = 0;
        if (high_bound >= w) high_bound = w - 1;

        /* Erase the dangerous zone by setting h as maximum value */
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* ── PLANTS ────────────────────────────────────────────────────────────── */
    for (int i = 0; i < np; i++) {
        int low_bound = (int)po[i].start - plant_bias;
        int high_bound = (int)po[i].start + (int)po[i].width - 1 + plant_bias;
        if (low_bound < 0) low_bound = 0;
        if (high_bound >= w) high_bound = w - 1;

        /* Erase the dangerous zone by setting h as maximum value */
        for (int c = low_bound; c <= high_bound; c++)
            gb_lg[c] = (float)h;
    }

    /* ── FIND SAFE WAYPOINT ─────────────────────────────────────────────────── */
    return greenest_pixel(gb_lg, w);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  NORMALISED BIAS
    *  Computes a bias in pixels as a fraction of the obstacle/plant width, with a minimum of 1 pixel.
    *  Inputs:
    *    o    - pointer to the obstacle/plant region
    *    frac - fraction of the width to use as bias (e.g. 0.5 = 50%)
    *  Output:
    *    int  - bias in pixels (minimum 1)
 * ══════════════════════════════════════════════════════════════════════════════ */
int normalised_bias(const struct obstacle_region_t *o, float frac)
{
    int bias = (int)(o->width * frac);
    if (bias < 1) bias = 1;
    return bias;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * LOGIC THAT FINDS THE WAYPOINT,
  * 1. If no obstacles or plants, return the greenest pixel.
  * 2. If there are obstacles but no plants, and none of the obstacles touch the ground, return the greenest pixel.
  * 3. Otherwise, apply bias logic to find the safest column.
  *
  *  Inputs:
  *    oo[]   - array of obstacle regions
  *    no     - number of obstacles
  *    po[]   - array of plant regions
  *    np     - number of plants
  *    gb[]   - current baseline array (one float per column, length w)
  *    w      - width of the image (length of gb[])
  *    h      - height of the image

  *  Output:
  *    int from 0 to w-1 indicating which column is the chosen waypoint
 * ══════════════════════════════════════════════════════════════════════════════ */
int motion_logic(const struct obstacle_region_t oo[], uint8_t no,
                          const struct obstacle_region_t po[], uint8_t np,
                          const float gb[], int w, int h,
                          int obs_bias, int plant_bias)
{
    uint8_t touches_out[MAX_OBSTACLE_REGIONS];
    uint8_t touch = 0;
    obstacle_touches_ground(oo, no, h, touches_out, &touch);

    /* No obstacles or plants detected, just return the greenest column */
    if (!no && !np)
        return greenest_pixel(gb, w);

    /* No plants and no obstacles touching the ground, return the greenest column */
    if (!np && !touch)
        return greenest_pixel(gb, w);

    /* Otherwise apply bias logic to find the safest column */
    return bias_logic(oo, no, po, np, gb, w, h, touches_out, obs_bias, plant_bias);
}

/* ══════════════════════════════════════════════════════════════════════════════
 * LOGIC THAT FINDS THE WAYPOINT -- BIAS NORMALISED
    * Same as motion_logic but uses normalised bias per obstacle/plant width
    * instead of fixed pixel values.
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
        return greenest_pixel(gb, w);

    if (!np && !touch)
        return greenest_pixel(gb, w);

    static float gb_lg[MAX_IMAGE_WIDTH];
    memcpy(gb_lg, gb, w * sizeof(float));

    /* Erase each obstacle with its own normalised bias */
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

    /* Erase each plant with its own normalised bias */
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