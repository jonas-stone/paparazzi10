/* c_wrapper.c */
/* Includes the standalone code to access its static variables */
#include "team10_get_obstacle_info_standalone.c"

/* Export pointers to the static masks so Python can display them */
uint8_t* get_work_mask() {
    return work_mask;
}

uint8_t* get_plant_mask() {
    return plant_mask_full;
}