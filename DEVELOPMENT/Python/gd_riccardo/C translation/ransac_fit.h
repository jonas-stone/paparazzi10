#ifndef RANSAC_FIT
#define RANSAC_FIT

#include <stdint.h>

static int  solve_3x3(int A[3][3], int b[3], double x[3]);
double     *fit_parabola_3pts(int pts[3][2], double coeffs[3]);
int         eval_parabola(double coeffs[3], int x);
static void random_choice_3(int n, int out[3]);
int         ransac_parabola(const int *x, const int *y, int n, int iterations, double threshold, int min_inliers, double best_coeffs[3], int *inlier_mask);
void        draw_parabola_on_image(struct image_t *img, const double coeffs[3], uint8_t Y, uint8_t U, uint8_t V, int thickness);

#endif