#ifndef CONFIG_H
#define CONFIG_H

// Minimum green pixel fraction to consider ground visible
#define OA_COLOR_COUNT_FRAC 0.05f

// Blob filtering parameters
#define BLOB_AREA_THRESHOLD 1000

// Obstacle detection parameters
#define OBSTACLE_THRESHOLD    50
#define MIN_OBSTACLE_WIDTH    20
#define MAX_COL_GAP            5
#define MIN_GROUND_PIXELS      5
#define MAX_GAP               10
#define SMOOTH_KERNEL          5
#define NO_GROUND_BASELINE   220
#define ALPHA                0.6f

// Camera resolution (Bebop 1 default)
#define IMAGE_WIDTH  640
#define IMAGE_HEIGHT 480

// FPS (0 = run at camera fps)
#define GROUND_DETECTION_FPS 0

#endif