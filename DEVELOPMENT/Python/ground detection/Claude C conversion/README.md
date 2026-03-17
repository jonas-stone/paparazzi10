# Team 10 – Obstacle + Plant Detection: C Implementation

## Overview

Complete C port of the Python pipeline (`get_obstacle_info_Imp.py` + `colored_blob_separator.py`), including **both obstacle and plant detection**. Operates directly on Paparazzi `image_t` (YUV422/UYVY) with zero heap allocation.

## Pipeline

```
Camera (YUV422)
  │
  ├── OBSTACLE PATH ──────────────────────────────────────────
  │  [1] Per-pixel decision tree  →  binary mask
  │  [2] Binary median blur       →  de-noised mask
  │  [3] Connected components      →  blob filtering (area ≥ 1000)
  │  [4] Hole fill (BFS)          →  clean ground mask
  │  [5] Boundary scan (bottom-up) + 1-D median smooth
  │  [6] EMA baseline update
  │  [7] Deviation → obstacle columns
  │  [8] Region merge (min_width, max_col_gap)
  │  └─→  obstacles_out[]
  │
  └── PLANT PATH ─────────────────────────────────────────────
     [9]  Downsample YUV422 to ~12% resolution
     [10] Lax green filter (Y∈[29,140], U≤116, V≤141)
     [11] Zero left third + box blur + threshold
     [12] Downsample clean_mask to same small resolution
     [13] Subtract ground from all-green → plant mask
     [14] Upsample plant mask to full resolution
     [15] Detect plant regions (flip, active rows, merge)
     └─→  plants_out[]
```

## Files

### Drone (Paparazzi)
| File | What |
|------|------|
| `team10_get_obstacle_info.h` | Header: all constants, structs, API |
| `team10_get_obstacle_info.c` | Full implementation (Paparazzi includes) |
| `team10_ground_detection.c` | Master module — **updated** with plant arrays |
| `team10_ground_detection.h` | Master header (unchanged) |

### Standalone testing
| File | What |
|------|------|
| `team10_get_obstacle_info_standalone.c` | Same code, minimal type stubs |
| `test_obstacle_detection.c` | Reads raw YUV422, dumps all intermediates |
| `validate_c_vs_python.py` | Runs both pipelines, compares pixel-by-pixel |
| `Makefile` | `make test`, `make validate IMAGE=...` |

## Build

```bash
# Standalone test (any Linux/macOS + gcc):
make test

# Run on a raw YUV422 file:
./test_obstacle_detection  input.yuv422  520  240  output_dir/

# Validate against Python (needs opencv-python + numpy):
make validate IMAGE=/path/to/image.jpg
```

## How to validate with your drone images

```bash
# On your machine, from the directory with the C files:
python3 validate_c_vs_python.py \
    "/home/jonas/paparazzi/DEVELOPMENT/downloads from drone/20260306-095826/1352900896.jpg" \
    --src-dir .

# Or all images in the folder:
python3 validate_c_vs_python.py \
    "/home/jonas/paparazzi/DEVELOPMENT/downloads from drone/20260306-095826/" \
    --all --src-dir .
```

The script converts each JPEG → raw YUV422, runs the C test binary + a Python reference, and compares:
- Decision-tree classification mask (pixel-perfect)
- Median-blurred mask (pixel-perfect)
- Green fraction and ground-found flag
- Obstacle and plant region counts

## API Changes from Previous Version

The `get_obstacle_info()` signature now includes plant output:

```c
uint8_t get_obstacle_info(
    struct image_t            *img,
    float                      ground_baseline[],
    int                       *baseline_inited,
    float                      oa_color_count_frac,  // 0.05f
    int                        median_ksize,          // 5
    int                        min_width,             // 20
    struct obstacle_region_t   obstacles_out[],       // obstacle results
    struct obstacle_region_t   plants_out[],          // plant results (NEW)
    uint8_t                   *plant_count_out,       // plant count (NEW)
    int                        boundary_rows_out[],   // may be NULL
    int                       *ground_found_out,      // may be NULL
    float                     *green_frac_out         // may be NULL
);
// Returns: obstacle count
```

**In `team10_ground_detection.c`**, the callback now looks like:

```c
struct obstacle_region_t local_obstacles[MAX_OBSTACLE_REGIONS];
struct obstacle_region_t local_plants[MAX_PLANT_REGIONS];
uint8_t plant_count = 0;

uint8_t obstacle_count = get_obstacle_info(
    img, ground_baseline, &baseline_inited,
    0.05f, 5, 20,
    local_obstacles,
    local_plants, &plant_count,    // plant output
    boundary_rows, NULL, NULL
);
```

Plant regions are printed in the callback. To send them to the autopilot, add a new ABI message (see the TODO in `team10_ground_detection.c`).

## Memory (~1.25 MB static BSS, zero heap)

| Buffer | Size | Purpose |
|--------|------|---------|
| work_mask | 125 KB | Binary masks |
| work_mask2 | 125 KB | Blur scratch |
| work_labels (int16) | 250 KB | CC labels |
| work_visited | 125 KB | Flood-fill |
| work_flipped | 125 KB | L-R flip |
| ff_queue (int32) | 500 KB | BFS queue |
| plant_mask_full | 125 KB | Upsampled plant mask |
| plant_*_small | ~4 KB total | Plant-resolution buffers |
| **Total** | **~1.38 MB** | |

Reduce `MAX_IMAGE_WIDTH` / `MAX_IMAGE_HEIGHT` if needed.

## Tuning

Obstacle defaults in `team10_get_obstacle_info.h`:

| Param | Default | Meaning |
|-------|---------|---------|
| `DEFAULT_OA_COLOR_FRAC` | 0.05 | Min green fraction for ground |
| `DEFAULT_MEDIAN_KSIZE` | 5 | Median blur kernel |
| `DEFAULT_BLOB_AREA_THRESH` | 1000 | Min blob area (pixels) |
| `DEFAULT_OBSTACLE_THRESH` | 50 | Min baseline deviation |
| `DEFAULT_MIN_WIDTH` | 20 | Min obstacle width |
| `DEFAULT_BASELINE_ALPHA` | 0.6 | EMA smoothing factor |

Plant defaults:

| Param | Default | Meaning |
|-------|---------|---------|
| `DEFAULT_PLANT_SCALE_NUM/DEN` | 3/25 | ≈0.12× downscale |
| `DEFAULT_PLANT_BLUR_KSIZE` | 3 | Box blur at plant res |
| `DEFAULT_PLANT_MIN_WIDTH` | 20 | Min plant region width |
| `DEFAULT_PLANT_MIN_PX_COL` | 2 | Min contiguous run per row |
| `DEFAULT_PLANT_MAX_COL_GAP` | 5 | Max gap to merge plant rows |
| `LAX_GREEN_*` | Y[29,140] U[0,116] V[0,141] | Lax green YUV thresholds |

## Known differences from Python

1. **`solidity_detection.is_smooth_blob()`** not ported — C uses area-only filtering. Add perimeter/area check in `isolate_ground_blob()` if you see false-positive blobs.

2. **Box blur vs Gaussian blur** in plant path — the C uses a box blur + threshold at 127, which closely approximates the Python's `GaussianBlur + threshold(127)` for small binary images. Minor pixel differences possible at plant resolution.

3. **Nearest-neighbour resampling** for plant downscale/upscale. Python uses `INTER_AREA` for downsample and `INTER_NEAREST` for upsample. The C uses nearest-neighbour for both, which is faster and nearly identical at the ~12% scale.
