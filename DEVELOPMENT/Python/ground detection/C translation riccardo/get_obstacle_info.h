#include "modules/computer_vision/image.h"
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Forward declarations for helpers defined below                       */
/* ------------------------------------------------------------------ */
static inline uint8_t is_ground(uint8_t Y, uint8_t U, uint8_t V);
static void apply_ground_mask(struct image_t *input, struct image_t *mask);
static void median_blur_3x3(struct image_t *mask, struct image_t *blurred);
static uint8_t median_of_9(uint8_t *v);
static void insertion_sort_9(uint8_t *v);