/*
 * test_obstacle_detection.c - standalone test harness
 * Runs pipeline N frames, dumps intermediates from last frame.
 * BUILD: gcc -O2 -Wall -Wno-unused-function -I. -o test_obstacle_detection \
 *        test_obstacle_detection.c team10_get_obstacle_info_standalone.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifndef _CV_LIB_VISION_IMAGE_H
#define _CV_LIB_VISION_IMAGE_H
struct FloatEulers { float phi, theta, psi; };
enum image_type { IMAGE_YUV422, IMAGE_GRAYSCALE, IMAGE_JPEG, IMAGE_GRADIENT, IMAGE_INT16 };
struct image_t {
    enum image_type type; uint16_t w, h; struct timeval ts;
    struct FloatEulers eulers; uint32_t pprz_ts;
    uint8_t buf_idx; uint32_t buf_size; void *buf;
};
static void image_create(struct image_t *img, uint16_t w, uint16_t h, enum image_type t) {
    img->type=t; img->w=w; img->h=h;
    img->buf_size=(t==IMAGE_YUV422)?2*w*h:w*h;
    img->buf=malloc(img->buf_size);
}
static void image_free(struct image_t *img) { free(img->buf); img->buf=NULL; }
#endif

#include "team10_get_obstacle_info.h"

static void write_raw(const char *d, const char *n, const uint8_t *dat, int sz) {
    char p[512]; snprintf(p,sizeof(p),"%s/%s",d,n);
    FILE *f=fopen(p,"wb"); if(f){fwrite(dat,1,sz,f);fclose(f);}
}
static void write_regions(const char *d, const char *n, const struct obstacle_region_t *r, int c) {
    char p[512]; snprintf(p,sizeof(p),"%s/%s",d,n);
    FILE *f=fopen(p,"w"); if(!f)return;
    fprintf(f,"%d\n",c); for(int i=0;i<c;i++) fprintf(f,"%d %d\n",r[i].start,r[i].width); fclose(f);
}
static void write_float(const char *d, const char *n, float v) {
    char p[512]; snprintf(p,sizeof(p),"%s/%s",d,n);
    FILE *f=fopen(p,"w"); if(f){fprintf(f,"%.8f\n",v);fclose(f);}
}
static void write_int_val(const char *d, const char *n, int v) {
    char p[512]; snprintf(p,sizeof(p),"%s/%s",d,n);
    FILE *f=fopen(p,"w"); if(f){fprintf(f,"%d\n",v);fclose(f);}
}

static uint8_t dbg_mask[MAX_PIXELS];
static uint8_t dbg_mask2[MAX_PIXELS];

int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <yuv422> <w> <h> <outdir> [frames]\n", argv[0]); return 1;
    }
    const char *input = argv[1];
    int w = atoi(argv[2]), h = atoi(argv[3]);
    const char *outdir = argv[4];
    int nf = (argc >= 6) ? atoi(argv[5]) : 2;
    if (nf < 1) nf = 1;
    if (w <= 0 || h <= 0 || w > MAX_IMAGE_WIDTH || h > MAX_IMAGE_HEIGHT) {
        fprintf(stderr, "Dims %dx%d out of range\n", w, h); return 1;
    }
    mkdir(outdir, 0755);

    int expected = w * h * 2;
    FILE *f = fopen(input, "rb"); if (!f) { perror(input); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < expected) { fprintf(stderr, "Too small\n"); fclose(f); return 1; }
    struct image_t img;
    image_create(&img, w, h, IMAGE_YUV422);
    if (fread(img.buf, 1, expected, f) != (size_t)expected) { fprintf(stderr, "Short read\n"); fclose(f); return 1; }
    fclose(f);
    printf("Loaded %dx%d from %s (%d frames)\n", w, h, input, nf);

    float baseline[MAX_IMAGE_WIDTH];
    int bi = 0;
    struct obstacle_region_t obs[MAX_OBSTACLE_REGIONS], plants[MAX_PLANT_REGIONS];
    uint8_t pc = 0, no = 0;
    int brows[MAX_IMAGE_WIDTH];
    int gfound = 0; float gf = 0.0f;

    for (int frame = 0; frame < nf; frame++) {
        no = get_obstacle_info(&img, baseline, &bi, DEFAULT_OA_COLOR_FRAC,
                               DEFAULT_MEDIAN_KSIZE, DEFAULT_MIN_WIDTH,
                               obs, plants, &pc, brows, &gfound, &gf);
        printf("  Frame %d/%d: ground=%d green=%.4f obs=%d plants=%d%s\n",
               frame+1, nf, gfound, gf, no, pc, frame==0?" (baseline init)":"");
        for (int i = 0; i < no; i++)
            printf("    obs[%d]: left=%d w=%d\n", i, obs[i].start, obs[i].width);
        for (int i = 0; i < pc; i++)
            printf("    plt[%d]: start=%d w=%d\n", i, plants[i].start, plants[i].width);
    }

    /* dump intermediates for last frame */
    const uint8_t *buf = (const uint8_t *)img.buf;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t Y = buf[y*w*2+x*2+1], U = buf[y*w*2+(x&~1)*2], V = buf[y*w*2+(x&~1)*2+2];
            dbg_mask[y*w+x] = is_ground_pixel(Y, U, V);
        }
    write_raw(outdir, "mask_after_classify.raw", dbg_mask, w*h);

    { int ks=DEFAULT_MEDIAN_KSIZE, half=ks/2, thr=(ks*ks)/2;
      for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
          int cnt=0, y0=(y-half<0)?0:y-half, y1=(y+half>=h)?h-1:y+half;
          int x0=(x-half<0)?0:x-half, x1=(x+half>=w)?w-1:x+half;
          for(int ky=y0;ky<=y1;ky++) for(int kx=x0;kx<=x1;kx++) if(dbg_mask[ky*w+kx]) cnt++;
          dbg_mask2[y*w+x]=(cnt>thr)?255:0;
      }
      memcpy(dbg_mask,dbg_mask2,w*h);
    }
    write_raw(outdir, "mask_after_blur.raw", dbg_mask, w*h);

    if (gfound) {
        isolate_ground_blob(dbg_mask, w, h, DEFAULT_BLOB_AREA_THRESH);
        write_raw(outdir, "mask_after_cc.raw", dbg_mask, w*h);
        fill_holes_mask(dbg_mask, w, h);
        write_raw(outdir, "mask_after_fill.raw", dbg_mask, w*h);
    }

    write_float(outdir, "green_frac.txt", gf);
    write_int_val(outdir, "ground_found.txt", gfound);
    write_regions(outdir, "obstacles.txt", obs, no);
    write_regions(outdir, "plants.txt", plants, pc);

    image_free(&img);
    printf("\nResults -> %s/\n", outdir);
    return 0;
}
