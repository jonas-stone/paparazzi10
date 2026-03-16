#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <jpeglib.h>
#include <SDL2/SDL.h>

/* Paparazzi Includes */
#include "std.h"
#include "modules/computer_vision/lib/vision/image.h"

// Global or static pointers to keep the window alive
SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
SDL_Texture *texture = NULL;

/* 1. The numeric sort function (keep this at the top) */
int numeric_sort(const struct dirent **a, const struct dirent **b) {
    return (atoi((*a)->d_name) - atoi((*b)->d_name));
}

void wait_for_keypress() {
    SDL_Event event;
    int pressed = 0;

    while (!pressed) {
        // SDL_WaitEvent is better than SDL_PollEvent here because 
        // it puts the CPU to sleep until something happens.
        if (SDL_WaitEvent(&event)) {
            if (event.type == SDL_QUIT) {
                exit(0);
            }
            if (event.type == SDL_KEYDOWN) {
                pressed = 1; // Break the loop and go to the next image
            }
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
        float r1 = buffer[0][i*3], g1 = buffer[0][i*3+1], b1 = buffer[0][i*3+2];
        float r2 = buffer[0][(i+1)*3], g2 = buffer[0][(i+1)*3+1], b2 = buffer[0][(i+1)*3+2];
        
        // Y (Brightness) - Standard coefficients
        uint8_t y1 = (uint8_t)(0.299f * r1 + 0.587f * g1 + 0.114f * b1);
        uint8_t y2 = (uint8_t)(0.299f * r2 + 0.587f * g2 + 0.114f * b2);

        // U (Blue-Luma)
        uint8_t u  = (uint8_t)(-0.1687f * r1 - 0.3313f * g1 + 0.5f * b1 + 128);
        
        // V (Red-Luma) <-- This is the one that gives you Red!
        uint8_t v  = (uint8_t)(0.5f * r1 - 0.4187f * g1 - 0.0813f * b1 + 128);

        // UYVY Packing: [U, Y1, V, Y2]
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

/* 3. The display function: Just takes the struct and draws it */
void run_sim_viewer(struct image_t *img) {
    if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
        SDL_Init(SDL_INIT_VIDEO);
    }

    if (!window) {
        window = SDL_CreateWindow("Drone Sim", 100, 100, img->w, img->h, 0);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    }

    // Update texture (or create if first time)
    if (texture) SDL_DestroyTexture(texture);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_UYVY, SDL_TEXTUREACCESS_STATIC, img->w, img->h);

    SDL_UpdateTexture(texture, NULL, img->buf, img->w * 2);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    // Keep the window responsive
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) exit(0);
    }
}

/* 4. MAIN: Handles the file reading and sorting */
int main(int argc, char *argv[]) {
    const char *path = "DEVELOPMENT/downloads from drone/sim_images";
    struct dirent **namelist;
    
    int n = scandir(path, &namelist, NULL, numeric_sort);
    if (n < 0) {
        perror("scandir");
        return 1;
    }

    while (1) { // Loop the entire sequence forever
      for (int i = 0; i < n; i++) {
        if (strstr(namelist[i]->d_name, ".jpg")) {
          char full_path[1024];
          snprintf(full_path, sizeof(full_path), "%s/%s", path, namelist[i]->d_name);

          struct image_t img;
          if (load_jpeg_to_image_t(full_path, &img)) {
            printf("Processing: %s\n", namelist[i]->d_name);
            
            // --- CALL THE VIEWER ---
            run_sim_viewer(&img);

            // Delay so we can see it (100ms = 10fps)
            // SDL_Delay(100);
            wait_for_keypress();

            image_free(&img);
          }
        }
      }
      printf("Sequence restarted.\n");
    }

    return 0;
}