# Team 10 – Obstacle Detection: C Implementation

## Overview

Full C port of the Python obstacle-detection pipeline (`get_obstacle_info_Imp.py` + `colored_blob_separator.py`), designed to run on the Paparazzi drone autopilot using `image_t` (YUV422/UYVY) images.

## Pipeline (8 stages)

```
Camera frame (YUV422)
  │
  ▼
[1] Per-pixel YUV decision tree  →  binary mask (0 / 255)
  │
  ▼
[2] Binary median blur            →  de-noised mask
  │
  ▼
[3] Connected-component labelling →  keep blobs ≥ 1000 px
  │  (two-pass union-find, 8-connectivity)
  ▼
[4] Hole filling                  →  border flood-fill
  │
  ▼
[5] Left-right flip + boundary scan (bottom-up per column)
  │  + 1-D median smoothing
  ▼
[6] Exponential-average baseline update
  │
  ▼
[7] Deviation detection  →  obstacle column mask
  │
  ▼
[8] Region merging       →  obstacle_region_t array
```

## File Inventory

### For the drone (Paparazzi integration)

| File | Purpose |
|------|---------|
| `team10_get_obstacle_info.h` | Header: constants, structs, API declarations |
| `team10_get_obstacle_info.c` | Full implementation (uses Paparazzi `image.h`) |
| `team10_ground_detection.c` | **Unchanged** master module – calls `get_obstacle_info()` |
| `team10_ground_detection.h` | **Unchanged** master header |

### For standalone testing (no Paparazzi needed)

| File | Purpose |
|------|---------|
| `team10_get_obstacle_info_standalone.c` | Self-contained compilation unit with minimal stubs |
| `test_obstacle_detection.c` | C test harness: reads raw YUV422, dumps all intermediates |
| `validate_c_vs_python.py` | Python script: runs both pipelines, compares pixel-by-pixel |
| `Makefile` | Build targets: `test`, `validate`, `clean` |

## Building

### Standalone test (any Linux/macOS with gcc)

```bash
make test
# or manually:
gcc -O2 -Wall -I. -o test_obstacle_detection \
    test_obstacle_detection.c \
    team10_get_obstacle_info_standalone.c \
    -lm
```

### Paparazzi integration

Add to your module's `Makefile.am` or XML build configuration:

```xml
<file name="team10_get_obstacle_info.c"/>
```

The `.c` file includes `modules/computer_vision/lib/vision/image.h` via the standard Paparazzi include path. No external libraries needed.

## Running the standalone test

```bash
./test_obstacle_detection  <raw_yuv422>  <width>  <height>  <output_dir>
```

**Arguments:**
- `raw_yuv422`: Binary file, exactly `width × height × 2` bytes, UYVY layout
- `width`, `height`: Image dimensions (max 520 × 240 by default)
- `output_dir`: Directory for result files

**Output files:**
- `mask_after_classify.raw` – binary mask after decision tree
- `mask_after_blur.raw` – mask after median blur
- `mask_after_cc.raw` – mask after connected-component filtering
- `mask_after_fill.raw` – mask after hole filling
- `mask_flipped.raw` – left-right flipped mask
- `boundary_rows.txt` – per-row boundary position (one integer per line)
- `obstacles.txt` – `count` on first line, then `start width` per obstacle
- `green_frac.txt` – green pixel fraction
- `ground_found.txt` – 0 or 1

## Python ↔ C Validation

```bash
# Single image:
python3 validate_c_vs_python.py  path/to/image.jpg  --src-dir .

# All images in a folder:
python3 validate_c_vs_python.py  path/to/folder/  --all  --src-dir .

# Or via Makefile:
make validate IMAGE=path/to/image.jpg
```

**What it validates:**
1. BGR → YUV422 conversion
2. Decision tree classification (pixel-perfect match expected)
3. Binary median blur (pixel-perfect match expected)
4. Green fraction and ground-found flag
5. Final obstacle regions (start, width)

## Integration with `team10_ground_detection.c`

The `detect_obstacles_from_ground()` callback remains unchanged. The key call:

```c
uint8_t obstacle_count = get_obstacle_info(
    img,                    // image_t* (YUV422)
    ground_baseline,        // float[MAX_IMAGE_WIDTH] – persistent across frames
    &baseline_inited,       // int* – set to 1 after first frame
    0.05f,                  // oa_color_count_frac
    5,                      // median_ksize
    20,                     // min_width
    local_obstacles,        // obstacle_region_t[MAX_OBSTACLE_REGIONS]
    boundary_rows,          // int[MAX_IMAGE_WIDTH] (or NULL)
    NULL,                   // ground_found_out (or &local_int)
    NULL                    // green_frac_out (or &local_float)
);
```

**Return value:** number of obstacles written to `local_obstacles`.

## Memory Usage

All work buffers are `static` (BSS segment), so there is zero heap allocation during processing. At the default `MAX_IMAGE_WIDTH=520, MAX_IMAGE_HEIGHT=240`:

| Buffer | Size |
|--------|------|
| `work_mask` | 125 KB |
| `work_mask2` | 125 KB |
| `work_labels` (int16) | 250 KB |
| `work_visited` | 125 KB |
| `work_flipped` | 125 KB |
| `ff_queue` (int32) | 500 KB |
| **Total static** | **~1.25 MB** |

If memory is tight, reduce `MAX_IMAGE_WIDTH` / `MAX_IMAGE_HEIGHT`.

## Tuning Parameters

All defaults are `#define`d in `team10_get_obstacle_info.h`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `DEFAULT_OA_COLOR_FRAC` | 0.05 | Min green fraction to declare ground |
| `DEFAULT_MEDIAN_KSIZE` | 5 | Median blur kernel (odd, ≥3) |
| `DEFAULT_MIN_WIDTH` | 20 | Min obstacle region width (pixels) |
| `DEFAULT_MAX_COL_GAP` | 5 | Max gap between obstacle columns to merge |
| `DEFAULT_MIN_GROUND_PX` | 5 | Min ground pixels at bottom of column |
| `DEFAULT_MAX_GAP` | 10 | Max gap in green pixels before boundary cut |
| `DEFAULT_SMOOTH_KERNEL` | 5 | 1-D median kernel for boundary smoothing |
| `DEFAULT_OBSTACLE_THRESH` | 50 | Min deviation from baseline (pixels) |
| `DEFAULT_NO_GROUND_BASE` | 220 | Baseline value when no ground detected |
| `DEFAULT_BLOB_AREA_THRESH` | 1000 | Min blob area to keep (pixels) |
| `DEFAULT_BASELINE_ALPHA` | 0.6 | EMA smoothing for baseline update |

## Differences from Python

1. **No `solidity_detection.is_smooth_blob()`** – The Python filters blobs by a smoothness metric from `solidity_detection.py`. The C version uses **area-only filtering** (threshold = 1000 px). If you need smoothness filtering, implement a perimeter/area ratio check in `isolate_ground_blob()`.

2. **No plant detection** – The Python has a secondary plant-detection path (`detect_all_green_lax`, `detect_plant_regions`). This is omitted from the C version since the drone autopilot only uses obstacle regions. Add it if needed.

3. **Median blur is binary-optimised** – Since the mask is binary (0 or 255), the median is computed as a majority vote (count pixels > threshold). This is significantly faster than a generic median sort.

4. **Connected components use union-find** – Instead of OpenCV's `cv2.connectedComponentsWithStats`, we use a two-pass algorithm with path-compressed union-find. Results are equivalent for 8-connectivity.
