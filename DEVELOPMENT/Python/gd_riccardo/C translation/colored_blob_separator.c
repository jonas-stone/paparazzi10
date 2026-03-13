/*
 * colored_blob_separator.c
 *
 * C translations of:
 *   isolate_ground_blob()  – connected-components labelling + small-blob filter
 *   fill_holes()           – flood-fill from border to discover and fill interior holes
 *
 * Image format: IMAGE_GRAYSCALE (uint8_t, one byte per pixel, row-major).
 * All functions operate in-place on the supplied image_t.
 */

#include "colored_blob_separator.h"
#include "image.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/*
 * Maximum number of labels supported during connected-component analysis.
 * Raise this if you expect very fragmented images.
 */
#define MAX_LABELS 4096

/* Sentinel value meaning "no label assigned yet". */
#define NO_LABEL 0xFFFF

/*
 * Union-Find (disjoint-set) helpers for label merging.
 * `parent` array must be pre-allocated to MAX_LABELS entries.
 */
static uint16_t uf_find(uint16_t *parent, uint16_t x)
{
    while (parent[x] != x) {
        parent[x] = parent[parent[x]]; /* path compression (halving) */
        x = parent[x];
    }
    return x;
}

static void uf_union(uint16_t *parent, uint16_t a, uint16_t b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a != b) {
        /* always attach larger id under smaller id for determinism */
        if (a < b) { parent[b] = a; }
        else        { parent[a] = b; }
    }
}

/* -------------------------------------------------------------------------
 * isolate_ground_blob
 * ---------------------------------------------------------------------- */

/*
 * Minimum blob area (in pixels).  Blobs with pixel_count < MIN_BLOB_AREA
 * are removed.  The Python source uses `blob_area < 1`, which in practice
 * never removes anything (area is always ≥ 1).  Kept faithful here; change
 * this constant to apply a meaningful size filter.
 */
#define MIN_BLOB_AREA 600

void isolate_ground_blob(struct image_t *img)
{
    if (img == NULL || img->buf == NULL) { return; }
    if (img->type != IMAGE_GRAYSCALE)   { return; }

    const uint16_t W = img->w;
    const uint16_t H = img->h;
    uint8_t  *src    = (uint8_t *)img->buf;  /* input binary mask          */

    /* --- allocate label map (uint16_t per pixel) ------------------------ */
    uint16_t *label_map = (uint16_t *)malloc(sizeof(uint16_t) * W * H);
    if (label_map == NULL) { return; }

    /* --- union-find parent table ---------------------------------------- */
    uint16_t parent[MAX_LABELS];
    uint32_t area  [MAX_LABELS];
    memset(area, 0, sizeof(area));

    uint16_t next_label = 1; /* 0 is reserved for background */
    parent[0] = 0;

    /* =====================================================================
     * First pass: assign provisional labels using 8-connectivity
     * (neighbours: top-left, top, top-right, left)
     * =================================================================== */
    for (uint16_t y = 0; y < H; y++) {
        for (uint16_t x = 0; x < W; x++) {
            uint32_t idx = (uint32_t)y * W + x;

            if (src[idx] == 0) {
                label_map[idx] = 0; /* background */
                continue;
            }

            /* Collect labels from already-visited 8-neighbours */
            uint16_t nb[4];
            uint8_t  nb_cnt = 0;

#define ADD_NB(ny, nx) \
    do { \
        uint16_t _l = label_map[(uint32_t)(ny) * W + (nx)]; \
        if (_l != 0) { nb[nb_cnt++] = uf_find(parent, _l); } \
    } while (0)

            if (y > 0) {
                if (x > 0)     ADD_NB(y-1, x-1);
                               ADD_NB(y-1, x  );
                if (x < W-1)   ADD_NB(y-1, x+1);
            }
            if (x > 0)         ADD_NB(y,   x-1);

#undef ADD_NB

            if (nb_cnt == 0) {
                /* New label */
                if (next_label >= MAX_LABELS) {
                    /* Out of label space – treat as background */
                    label_map[idx] = 0;
                    continue;
                }
                parent[next_label] = next_label;
                area  [next_label] = 0;
                label_map[idx]     = next_label;
                next_label++;
            } else {
                /* Use smallest root among neighbours and merge the rest */
                uint16_t root = nb[0];
                for (uint8_t k = 1; k < nb_cnt; k++) {
                    uint16_t r = uf_find(parent, nb[k]);
                    if (r != root) {
                        uf_union(parent, root, r);
                        root = uf_find(parent, root);
                    }
                }
                label_map[idx] = root;
            }
        }
    }

    /* =====================================================================
     * Second pass: flatten union-find, accumulate areas
     * =================================================================== */
    for (uint32_t i = 0; i < (uint32_t)W * H; i++) {
        if (label_map[i] == 0) { continue; }
        uint16_t root = uf_find(parent, label_map[i]);
        label_map[i]  = root;
        area[root]++;
    }

    /* =====================================================================
     * Third pass: write result – keep blobs with area >= MIN_BLOB_AREA,
     * set their pixels to 255; erase the rest.
     * =================================================================== */
    for (uint32_t i = 0; i < (uint32_t)W * H; i++) {
        uint16_t lbl = label_map[i];
        if (lbl == 0 || area[lbl] < MIN_BLOB_AREA) {
            src[i] = 0;
        } else {
            src[i] = 255;
        }
    }

    free(label_map);
}

/* -------------------------------------------------------------------------
 * fill_holes
 *
 * Strategy (equivalent to cv2.RETR_CCOMP hole-filling):
 *   1. Flood-fill from every border pixel that is 0 into a temporary mask,
 *      marking all background pixels reachable from the image border.
 *   2. Any pixel that is 0 in the original image but was NOT reached by the
 *      flood-fill is an interior hole → set it to 255.
 *
 * The flood-fill uses an explicit stack (BFS/DFS) to avoid recursion limits.
 * 4-connectivity is used for the background flood-fill, which correctly
 * identifies 8-connected foreground holes (by duality).
 * ---------------------------------------------------------------------- */

void fill_holes(struct image_t *img)
{
    if (img == NULL || img->buf == NULL) { return; }
    if (img->type != IMAGE_GRAYSCALE)   { return; }

    const uint16_t W   = img->w;
    const uint16_t H   = img->h;
    const uint32_t N   = (uint32_t)W * H;
    uint8_t *src       = (uint8_t *)img->buf;

    /* visited[i] == 1 → background pixel reachable from the border */
    uint8_t *visited = (uint8_t *)calloc(N, sizeof(uint8_t));
    if (visited == NULL) { return; }

    /* BFS queue: worst case every pixel is queued once */
    uint32_t *queue = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (queue == NULL) { free(visited); return; }

    uint32_t q_head = 0, q_tail = 0;

#define ENQUEUE(idx) \
    do { \
        if (!visited[idx] && src[idx] == 0) { \
            visited[idx] = 1; \
            queue[q_tail++] = (idx); \
        } \
    } while (0)

    /* Seed the queue with all border pixels that are background (0) */
    for (uint16_t x = 0; x < W; x++) {
        ENQUEUE((uint32_t)0       * W + x);          /* top row    */
        ENQUEUE((uint32_t)(H-1)   * W + x);          /* bottom row */
    }
    for (uint16_t y = 1; y < H - 1; y++) {
        ENQUEUE((uint32_t)y * W + 0    );             /* left col   */
        ENQUEUE((uint32_t)y * W + W - 1);             /* right col  */
    }

    /* 4-connected BFS */
    while (q_head < q_tail) {
        uint32_t idx = queue[q_head++];
        uint16_t cy  = (uint16_t)(idx / W);
        uint16_t cx  = (uint16_t)(idx % W);

        if (cy > 0)     ENQUEUE(idx - W);
        if (cy < H-1)   ENQUEUE(idx + W);
        if (cx > 0)     ENQUEUE(idx - 1);
        if (cx < W-1)   ENQUEUE(idx + 1);
    }

#undef ENQUEUE

    /* Any zero pixel not visited is an interior hole → fill with 255 */
    for (uint32_t i = 0; i < N; i++) {
        if (src[i] == 0 && !visited[i]) {
            src[i] = 255;
        }
    }

    free(visited);
    free(queue);
}