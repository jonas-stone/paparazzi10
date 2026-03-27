/* FOR RANSAC COMPILATION

gcc -o sim_viewer     sw/airborne/modules/computer_vision/team10_plot_sim_images.c     sw/airborne/modules/computer_vision/team10_c_test_pipeline.c     sw/airborne/modules/computer_vision/team10_RANSAC_adaptation.c     sw/airborne/modules/computer_vision/team10_get_obstacle_info.c     sw/airborne/modules/computer_vision/team10_rtp_utilities.c     sw/airborne/modules/computer_vision/lib/vision/image.c     sw/airborne/math/pprz_matrix_decomp_float.c     sw/airborne/math/pprz_algebra_float.c     -I/home/riccardogallo/paparazzi10/sw/airborne     -I/home/riccardogallo/paparazzi10/sw/airborne/modules/computer_vision     -I/home/riccardogallo/paparazzi10/sw/airborne/modules/computer_vision/lib/vision     -I/home/riccardogallo/paparazzi10/sw/include     $(sdl2-config --cflags --libs)     -ljpeg -lm

*/

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <dirent.h>
// #include <jpeglib.h>
// #include <SDL2/SDL.h>

// /* Paparazzi Includes */
// #include "std.h"
// #include "lib/vision/image.h"
// #include "team10_plot_sim_images.h"
// #include "team10_ground_detection_RANSAC.h"
// extern float safest_bucket;
// extern float max_safety;

// // Global or static pointers to keep the window alive
// SDL_Window *window = NULL;
// SDL_Renderer *renderer = NULL;
// SDL_Texture *texture = NULL;

// /* 1. The numeric sort function (keep this at the top) */
// int numeric_sort(const struct dirent **a, const struct dirent **b) {
//     return (atoi((*a)->d_name) - atoi((*b)->d_name));
// }

// void wait_for_keypress(void) {
//     SDL_Event event;
//     int pressed = 0;

//     while (!pressed) {
//         // SDL_WaitEvent is better than SDL_PollEvent here because 
//         // it puts the CPU to sleep until something happens.
//         if (SDL_WaitEvent(&event)) {
//             if (event.type == SDL_QUIT) {
//                 exit(0);
//             }
//             if (event.type == SDL_KEYDOWN) {
//                 pressed = 1; // Break the loop and go to the next image
//             }
//         }
//     }
// }

// /* 2. Decodes JPEG and fills a Paparazzi image_t struct */
// int load_jpeg_to_image_t(const char *filename, struct image_t *img) {
//     struct jpeg_decompress_struct cinfo;
//     struct jpeg_error_mgr jerr;
//     FILE *infile = fopen(filename, "rb");
//     if (!infile) return 0;

//     cinfo.err = jpeg_std_error(&jerr);
//     jpeg_create_decompress(&cinfo);
//     jpeg_stdio_src(&cinfo, infile);
//     jpeg_read_header(&cinfo, TRUE);
//     jpeg_start_decompress(&cinfo);

//     image_create(img, cinfo.output_width, cinfo.output_height, IMAGE_YUV422);
    
//     int row_stride = cinfo.output_width * 3;
//     JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
//     uint8_t *dst = (uint8_t *)img->buf;

//     while (cinfo.output_scanline < cinfo.output_height) {
//     jpeg_read_scanlines(&cinfo, buffer, 1);
//     for (int i = 0; i < (int)cinfo.output_width; i += 2) {
//         float r1 = buffer[0][i*3], g1 = buffer[0][i*3+1], b1 = buffer[0][i*3+2];
//         float r2 = buffer[0][(i+1)*3], g2 = buffer[0][(i+1)*3+1], b2 = buffer[0][(i+1)*3+2];
        
//         // Y (Brightness) - Standard coefficients
//         uint8_t y1 = (uint8_t)(0.299f * r1 + 0.587f * g1 + 0.114f * b1);
//         uint8_t y2 = (uint8_t)(0.299f * r2 + 0.587f * g2 + 0.114f * b2);

//         // U (Blue-Luma)
//         uint8_t u  = (uint8_t)(-0.1687f * r1 - 0.3313f * g1 + 0.5f * b1 + 128);
        
//         // V (Red-Luma) <-- This is the one that gives you Red!
//         uint8_t v  = (uint8_t)(0.5f * r1 - 0.4187f * g1 - 0.0813f * b1 + 128);

//         // UYVY Packing: [U, Y1, V, Y2]
//         *dst++ = u;
//         *dst++ = y1;
//         *dst++ = v;
//         *dst++ = y2;
//     }
// }
//     jpeg_finish_decompress(&cinfo);
//     jpeg_destroy_decompress(&cinfo);
//     fclose(infile);
//     return 1;
// }

// /* 3. The display function: Just takes the struct and draws it */
// void run_sim_viewer(struct image_t *img) {
//     if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
//         SDL_Init(SDL_INIT_VIDEO);
//     }

//     if (!window) {
//         window = SDL_CreateWindow("Drone Sim", 100, 100, img->w, img->h, 0);
//         renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
//     }

//     // Update texture (or create if first time)
//     if (texture) SDL_DestroyTexture(texture);
//     texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_UYVY, SDL_TEXTUREACCESS_STATIC, img->w, img->h);

//     SDL_UpdateTexture(texture, NULL, img->buf, img->w * 2);
//     SDL_RenderClear(renderer);
//     SDL_RenderCopy(renderer, texture, NULL, NULL);
//     SDL_RenderPresent(renderer);

//     // Keep the window responsive
//     SDL_Event event;
//     while (SDL_PollEvent(&event)) {
//         if (event.type == SDL_QUIT) exit(0);
//     }
// }

// /* 4. MAIN: Handles the file reading and sorting */
// int main(int argc, char *argv[]) {
//   const char *path = "DEVELOPMENT/downloads from drone/sim_images";
//   struct dirent **namelist;
  
//   int n = scandir(path, &namelist, NULL, numeric_sort);
//   if (n < 0) {
//       perror("scandir");
//       return 1;
//   }

//   while (1) { // Loop the entire sequence forever
//     for (int i = 0; i < n; i++) {
//       if (strstr(namelist[i]->d_name, ".jpg")) {
//         char full_path[1024];
//         snprintf(full_path, sizeof(full_path), "%s/%s", path, namelist[i]->d_name);

//         struct image_t img;
//         if (load_jpeg_to_image_t(full_path, &img)) {
            
//           // This is where the magic happens:
//           // It reads img.w, img.h, and img.buf to find the ground
//           get_obstacles_RANSAC(&img, 0); 

//           // Now you can access the results calculated by RANSAC
//           printf("Image: %s | Best Bucket: %.2f | Confidence: %.2f\n", 
//                   namelist[i]->d_name, safest_bucket, max_safety);

//           run_sim_viewer(&img);
//           wait_for_keypress();
//           image_free(&img); // Crucial: prevent memory leaks!
//         }
//       }
//     }
//     printf("Sequence restarted.\n");
//   }

//   return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <jpeglib.h>
#include <SDL2/SDL.h>

/* Paparazzi Includes */
#include "std.h"
#include "lib/vision/image.h"
#include "team10_plot_sim_images.h"
#include "team10_ground_detection_RANSAC.h"
extern float safest_bucket;
extern float max_safety;

// Global or static pointers to keep the window alive
SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
SDL_Texture *texture = NULL;

/* 1. The numeric sort function */
int numeric_sort(const struct dirent **a, const struct dirent **b) {
    return (atoi((*a)->d_name) - atoi((*b)->d_name));
}

void wait_for_keypress(void) {
    SDL_Event event;
    int pressed = 0;
    while (!pressed) {
        if (SDL_WaitEvent(&event)) {
            if (event.type == SDL_QUIT) exit(0);
            if (event.type == SDL_KEYDOWN) pressed = 1;
        }
    }
}

/*
 * Rotate a YUV422 (UYVY) image 90 degrees counter-clockwise.
 * The result is written into dst, which is created here.
 * dst dimensions: width = src->h, height = src->w
 *
 * CCW mapping: dst pixel (dst_row, dst_col) <- src pixel (dst_col, src_w - 1 - dst_row)
 */
void rotate_yuv422_ccw(struct image_t *src, struct image_t *dst)
{
    int src_w = src->w;
    int src_h = src->h;

    // after CCW rotation: new width = old height, new height = old width
    image_create(dst, src_h, src_w, IMAGE_YUV422);

    int dst_w = dst->w;  // = src_h
    int dst_h = dst->h;  // = src_w

    uint8_t *s = (uint8_t *)src->buf;
    uint8_t *d = (uint8_t *)dst->buf;

    for (int dst_row = 0; dst_row < dst_h; dst_row++) {
        for (int dst_col = 0; dst_col < dst_w; dst_col += 2) {
            // CCW: src row = dst_col, src col = src_w - 1 - dst_row
            int src_row = dst_col;
            int src_col = src_w - 1 - dst_row;

            // align to even column (UYVY pairs are 2 pixels wide)
            int src_col_even = src_col & ~1;

            int src_idx = (src_row * src_w + src_col_even) * 2;
            int dst_idx = (dst_row * dst_w + dst_col) * 2;

            d[dst_idx + 0] = s[src_idx + 0]; // U
            d[dst_idx + 1] = s[src_idx + 1]; // Y1
            d[dst_idx + 2] = s[src_idx + 2]; // V
            d[dst_idx + 3] = s[src_idx + 3]; // Y2
        }
    }
}

/* 2. Decodes JPEG and fills a Paparazzi image_t struct */
int load_jpeg_to_image_t(const char *filename, struct image_t *img) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile = fopen(filename, "rb");
    if (!infile) return 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    image_create(img, cinfo.output_width, cinfo.output_height, IMAGE_YUV422);

    int row_stride = cinfo.output_width * 3;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    uint8_t *dst = (uint8_t *)img->buf;

    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int i = 0; i < (int)cinfo.output_width; i += 2) {
            float r1 = buffer[0][i*3],     g1 = buffer[0][i*3+1],     b1 = buffer[0][i*3+2];
            float r2 = buffer[0][(i+1)*3], g2 = buffer[0][(i+1)*3+1], b2 = buffer[0][(i+1)*3+2];

            uint8_t y1 = (uint8_t)(0.299f * r1 + 0.587f * g1 + 0.114f * b1);
            uint8_t y2 = (uint8_t)(0.299f * r2 + 0.587f * g2 + 0.114f * b2);
            uint8_t u  = (uint8_t)(-0.1687f * r1 - 0.3313f * g1 + 0.5f    * b1 + 128);
            uint8_t v  = (uint8_t)( 0.5f    * r1 - 0.4187f * g1 - 0.0813f * b1 + 128);

            *dst++ = u;
            *dst++ = y1;
            *dst++ = v;
            *dst++ = y2;
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 1;
}

/* 3. The display function */
void run_sim_viewer(struct image_t *img) {
    if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
        SDL_Init(SDL_INIT_VIDEO);
    }

    if (!window) {
        window = SDL_CreateWindow("Drone Sim", 100, 100, img->w, img->h, 0);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    } else {
        // resize window to match rotated image dimensions
        SDL_SetWindowSize(window, img->w, img->h);
    }

    if (texture) SDL_DestroyTexture(texture);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_UYVY, SDL_TEXTUREACCESS_STATIC, img->w, img->h);

    SDL_UpdateTexture(texture, NULL, img->buf, img->w * 2);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) exit(0);
    }
}

/* 4. MAIN */
int main(int argc, char *argv[]) {
    const char *path = "DEVELOPMENT/downloads from drone/sim_images";
    struct dirent **namelist;

    int n = scandir(path, &namelist, NULL, numeric_sort);
    if (n < 0) {
        perror("scandir");
        return 1;
    }

    while (1) {
        for (int i = 0; i < n; i++) {
            if (strstr(namelist[i]->d_name, ".jpg")) {
                char full_path[1024];
                snprintf(full_path, sizeof(full_path), "%s/%s", path, namelist[i]->d_name);

                struct image_t img;
                if (load_jpeg_to_image_t(full_path, &img)) {

                    // 1. Run RANSAC on the raw sideways image
                    get_obstacles_RANSAC(&img, 0);

                    printf("Image: %s | Best Bucket: %.2f | Confidence: %.2f\n",
                           namelist[i]->d_name, safest_bucket, max_safety);

                    // 2. Rotate CCW for display only
                    struct image_t img_rotated;
                    rotate_yuv422_ccw(&img, &img_rotated);

                    // 3. Display the rotated image
                    run_sim_viewer(&img_rotated);
                    wait_for_keypress();

                    // 4. Free both images
                    image_free(&img);
                    image_free(&img_rotated);
                }
            }
        }
        printf("Sequence restarted.\n");
    }

    return 0;
}