#include "modules/computer_vision/team10_gemini_implementation.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <string.h>

// ----------------------------------------------------------------------------
// 1. Simplified Color Classifier
// ----------------------------------------------------------------------------
// This condenses the massive nested if/else tree from the Python `is_ground`
// function into its mathematical equivalent.
static inline bool is_ground(uint8_t Y, uint8_t U, uint8_t V) {
    if (U <= 115) {
        if (V <= 145) {
            if (Y > 85 && U > 92) return true;
        } else if (V <= 152) {
            if (Y <= 177) return true;
        }
    } else if (U <= 121) {
        if (V <= 137 && Y > 87) return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// 2. Fast 1D Median Filter for Boundary Smoothing
// ----------------------------------------------------------------------------
static void median_filter_1d(int* input, int* output, int width, int ksize) {
    int half_k = ksize / 2;
    int window[21]; // Supports up to ksize=21
    
    for (int i = 0; i < width; i++) {
        int count = 0;
        for (int j = -half_k; j <= half_k; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < width) {
                window[count++] = input[idx];
            }
        }
        // Simple insertion sort for tiny arrays
        for (int m = 1; m < count; m++) {
            int key = window[m];
            int n = m - 1;
            while (n >= 0 && window[n] > key) {
                window[n + 1] = window[n];
                n--;
            }
            window[n + 1] = key;
        }
        output[i] = window[count / 2];
    }
}

// ----------------------------------------------------------------------------
// 3. Main Processing Pipeline
// ----------------------------------------------------------------------------
uint8_t get_obstacle_info(
    struct image_t *img,
    float *ground_baseline,
    int *baseline_inited,
    float oa_color_count_frac,
    int median_ksize,
    int min_width,
    struct obstacle_region_t *obstacles,
    int *boundary_rows_out,
    int *ground_found_out,
    float *green_frac_out) 
{
    int w = img->w;
    int h = img->h;
    uint8_t *buf = (uint8_t*)img->buf;
    
    int raw_boundary[MAX_IMAGE_WIDTH];
    bool obstacle_mask[MAX_IMAGE_WIDTH];
    int green_pixels = 0;

    // Hardcoded parameters from Python pipeline
    int min_ground_pixels = 5;
    int max_gap = 10;
    int no_ground_baseline = 220; // Ensure this scales with image size if needed
    int obstacle_threshold = 50;
    float alpha = 0.6f;
    int max_col_gap = 5;

    // --- STEP 1: Bottom-up column scan to find the ground boundary ---
    for (int x = 0; x < w; x++) {
        int bound = h;
        int gap = 0;
        int current_run = 0;
        
        // Scan from bottom to top
        for (int y = h - 1; y >= 0; y--) {
            // YUV422 (UYVY) extraction logic:
            // Every 4 bytes contains 2 pixels: U, Y1, V, Y2
            int byte_idx = 2 * (y * w + (x & ~1)); 
            uint8_t U = buf[byte_idx];
            uint8_t V = buf[byte_idx + 2];
            uint8_t Y = (x % 2 == 0) ? buf[byte_idx + 1] : buf[byte_idx + 3];

            if (is_ground(Y, U, V)) {
                green_pixels++;
                current_run++;
                gap = 0;
                // If we hit our minimum solid blob, set pending boundary
                if (current_run >= min_ground_pixels && bound == h) {
                    bound = y; 
                }
            } else {
                gap++;
                // If we found the ground, and now hit a large gap (like empty space or obstacle), stop.
                if (gap > max_gap && bound != h) {
                    break; 
                }
            }
        }
        raw_boundary[x] = bound;
    }

    // --- STEP 2: Assess overall ground status ---
    float green_frac = (float)green_pixels / (w * h);
    if (green_frac_out) *green_frac_out = green_frac;
    
    bool ground_found = (green_frac > oa_color_count_frac);
    if (ground_found_out) *ground_found_out = ground_found;

    if (!ground_found) {
        // Reset or maintain state if no ground found
        for (int i = 0; i < w; i++) boundary_rows_out[i] = h;
        return 0; // Return 0 obstacles
    }

    // --- STEP 3: Smooth the boundary & update baseline ---
    median_filter_1d(raw_boundary, boundary_rows_out, w, median_ksize);

    // Initialize baseline if first frame
    if (!*baseline_inited) {
        for (int i = 0; i < w; i++) {
            ground_baseline[i] = (boundary_rows_out[i] >= h) ? (float)no_ground_baseline : (float)boundary_rows_out[i];
        }
        *baseline_inited = 1;
    }

    // Calculate deviations and update the exponential moving average
    for (int i = 0; i < w; i++) {
        int b = boundary_rows_out[i];
        bool valid = (b < h);
        bool no_ground = (b >= h);
        
        float dev = (float)b - ground_baseline[i];
        bool deviation_obstacle = valid && (dev > obstacle_threshold);
        obstacle_mask[i] = deviation_obstacle || no_ground;

        if (!obstacle_mask[i]) {
            ground_baseline[i] = (1.0f - alpha) * ground_baseline[i] + alpha * (float)b;
        }
    }

    // Forward fill the baseline for obstacle areas (replicates Python's "last_good" sweep)
    float last_good = (float)no_ground_baseline;
    for (int i = 0; i < w; i++) {
        if (!obstacle_mask[i]) {
            last_good = ground_baseline[i];
        } else {
            ground_baseline[i] = last_good;
        }
    }

    // --- STEP 4: Extract Obstacle Regions ---
    uint8_t obs_count = 0;
    int start = -1;
    int end = -1;

    for (int i = 0; i < w; i++) {
        if (obstacle_mask[i]) {
            if (start == -1) {
                start = i; 
                end = i;
            } else if (i - end <= max_col_gap) {
                end = i;
            } else {
                if (end - start + 1 >= min_width) {
                    // Match Python's coordinate flip: left = W - 1 - e
                    obstacles[obs_count].start = w - 1 - end; 
                    obstacles[obs_count].width = end - start + 1;
                    obstacles[obs_count].type  = DET_OBSTACLE;
                    obs_count++;
                    if (obs_count >= MAX_OBSTACLE_REGIONS) break;
                }
                start = i; 
                end = i;
            }
        }
    }
    
    // Process final hanging region
    if (start != -1 && (end - start + 1) >= min_width && obs_count < MAX_OBSTACLE_REGIONS) {
        obstacles[obs_count].start = w - 1 - end;
        obstacles[obs_count].width = end - start + 1;
        obstacles[obs_count].type  = DET_OBSTACLE;
        obs_count++;
    }

    return obs_count;
}