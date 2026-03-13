#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "ransac_fit.h"

/*
 * Solve a 3x3 linear system Ax = b using Cramer's Rule.
 * A is stored row-major: A[row][col]
 * Returns 1 on success, 0 if the matrix is singular.
 * 
 */
static int solve_3x3(double A[3][3], double b[3], double x[3]) {
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
               - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (fabs(det) < 1e-10)
        return 0;

    double inv_det = 1.0 / det;

    double det0 = b[0]    * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (b[1]    * A[2][2] - A[1][2] * b[2]   )
                + A[0][2] * (b[1]    * A[2][1] - A[1][1] * b[2]   );

    double det1 = A[0][0] * (b[1]    * A[2][2] - A[1][2] * b[2]   )
                - b[0]    * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * b[2]    - b[1]    * A[2][0]);

    double det2 = A[0][0] * (A[1][1] * b[2]    - b[1]    * A[2][1])
                - A[0][1] * (A[1][0] * b[2]    - b[1]    * A[2][0])
                + b[0]    * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    x[0] = det0 * inv_det;
    x[1] = det1 * inv_det;
    x[2] = det2 * inv_det;

    return 1;
}


/*
 * Fit y = ax² + bx + c through exactly 3 (x, y) points.
 *
 * pts layout: pts[0] = {x0, y0}, pts[1] = {x1, y1}, pts[2] = {x2, y2}
 * On success, writes [a, b, c] into coeffs and returns coeffs.
 * Returns NULL if the system is degenerate (e.g. two points share the same x).
 * 
 */
double *fit_parabola_3pts(int x[3], int y[3], double coeffs[3]) {
    double x0 = x[0], y0 = y[0];
    double x1 = x[1], y1 = y[1];
    double x2 = x[2], y2 = y[2];

    double A[3][3] = {
        { x0 * x0, x0, 1.0 },
        { x1 * x1, x1, 1.0 },
        { x2 * x2, x2, 1.0 },
    };

    double b[3] = { y0, y1, y2 };

    if (!solve_3x3(A, b, coeffs))
        return NULL; /* mirrors Python's: return None */

    return coeffs;
}


/*
 * Least-squares parabola fit: y = ax² + bx + c
 * Mirrors: np.polyfit(x, y, deg=2)
 *
 * Assumes the following helper is already implemented elsewhere:
 *   int solve_3x3(double A[3][3], double b[3], double x[3]);
 *   returns 1 on success, 0 if singular.
 *
 * Returns 1 on success, 0 on failure.
 * On success, coeffs[0]=a, coeffs[1]=b, coeffs[2]=c.
 * 
 */
int polyfit2(const double *x, const double *y, int n, double coeffs[3])
{
    /* ── accumulate sums ─────────────────────────────────────────────────── */
    double sx1 = 0, sx2 = 0, sx3 = 0, sx4 = 0;
    double sy  = 0, sxy = 0, sx2y = 0;
 
    for (int i = 0; i < n; i++) {
        double xi  = x[i];
        double yi  = y[i];
        double xi2 = xi  * xi;
        double xi3 = xi2 * xi;
        double xi4 = xi3 * xi;
 
        sx1  += xi;
        sx2  += xi2;
        sx3  += xi3;
        sx4  += xi4;
        sy   += yi;
        sxy  += xi  * yi;
        sx2y += xi2 * yi;
    }
 
    /* ── build the normal equations and delegate to solve_3x3 ───────────── */
    double A[3][3] = {
        { sx4, sx3, sx2 },
        { sx3, sx2, sx1 },
        { sx2, sx1, (double)n }
    };
    double b[3] = { sx2y, sxy, sy };
    
    // returns 1 on success, 0 on failure. 
    // The "coeffs" array is modified in-place because it's passed as a pointer.
    return solve_3x3(A, b, coeffs);
}


/*
 * Evaluate the parabola on given x-values.
 *
 * Returns: y, vector the same size as x, rounded to the nearest integer.
 * 
 */
int eval_parabola(const double coeffs[3], int x) {
  double a = coeffs[0];
  double b = coeffs[1];
  double c = coeffs[2];

  // compute result
  double y = a * x*x + b * x + c;
  
  // turn y into an "int" type after rounding
  return (int)round(y);
}


/* 
 * Random sample of 3 distinct indices in [0, n) 
 * Mirrors: np.random.choice(n, size=3, replace=False)
 * Uses Fisher-Yates partial shuffle on a small index buffer.
 *                
 */
static void random_choice_3(int n, int out[3])
{ 
    // create range of numbers [0, n-1]
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 0; i < 3; i++) {
        int j = i + rand() % (n - i);   /* random position in [i, n) */
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        out[i] = idx[i];
    }
    free(idx);
}
 

/* 
 * Fit parabola to a noisy set of points using the RANSAC algorithm.
 *
 */
int ransac_parabola(const int *x, const int *y, int n,
                    int iterations, double threshold, int min_inliers,
                    double best_coeffs[3], int *inlier_mask)
{
	memset(inlier_mask, 0, n * sizeof(int));

	if (n < 3) 
		return 0;

	int  best_inlier_count = 0;
	int *best_mask    = calloc(n, sizeof(int));
	int *current_mask = malloc(n * sizeof(int));

	for (int iter = 0; iter < iterations; iter++) {

		/* ── 3a. RANDOM SAMPLE ──────────────────────────────────────────── */
		int sample[3];
		random_choice_3(n, sample);

		/* ── 3b. FIT MODEL TO SAMPLE ────────────────────────────────────── */
		double  coeffs[3];
		int     x_points[3];
		int     y_points[3];

		// remember, x and y are passed as pointers inside the functionm, 
		// so if I call x[25] it returns the 25th memory slot after the beginning of x.
		// If x and y were passed as values, it wouldn't make sense in C.
		for (int i = 0; i < 3; i++) {
			x_points[i] = x[sample[i]];
			y_points[i] = y[sample[i]];
		}

		if (fit_parabola_3pts(x_points, y_points, coeffs) == NULL)
			continue;   /* degenerate sample, skip */

		/* ── 3c. EVALUATE ALL POINTS AGAINST THIS MODEL ─────────────────── */
		int inlier_count = 0;
		for (int i = 0; i < n; i++) {
				double residual = fabs(eval_parabola(coeffs, x[i]) - y[i]);
				current_mask[i] = (residual < threshold) ? 1 : 0;
				if (current_mask[i]) inlier_count++;
		}

		/* ── 3d. KEEP BEST MODEL ────────────────────────────────────────── */
		// the memcpy function saves its argument in memory.
		if (inlier_count > best_inlier_count) {
				best_inlier_count = inlier_count;
				memcpy(best_coeffs, coeffs, 3 * sizeof(double));
				memcpy(best_mask,   current_mask, n * sizeof(int));
		}
	}

	/* ── 3e. REFINEMENT ─────────────────────────────────────────────────── */
	int success = 0;
	if (best_inlier_count >= min_inliers) {
		double *xi = malloc(best_inlier_count * sizeof(double));
		double *yi = malloc(best_inlier_count * sizeof(double));
		int k = 0;
		for (int i = 0; i < n; i++) {
			// advance k only when an in-lier is copied, so that 
			// xi and yi are tightly packed and not sparse (they're not a mask)
			if (best_mask[i]) { xi[k] = x[i]; yi[k] = y[i]; k++; }
		}

		if (polyfit2(xi, yi, best_inlier_count, best_coeffs)) {
			memcpy(inlier_mask, best_mask, n * sizeof(int));
			success = 1;
		}

		free(xi);
		free(yi);
	} else {
		printf("RANSAC failed: not enough inliers found.\n");
	}

	free(best_mask);
	free(current_mask);
	return success;
}


#include <math.h>
#include "image.h"
/*
 * Draws y = ax² + bx + c onto struct image_t.
 *
 * Assumes the following helpers are already implemented elsewhere:
 *   double eval_parabola(const double coeffs[3], double xval);
 *   void   set_color_yuv422(struct image_t *im, int x, int y, uint8_t Y, uint8_t U, uint8_t V);
 * 
 */
void draw_parabola_on_image(struct image_t *img,
                            const double coeffs[3],
                            uint8_t Y, uint8_t U, uint8_t V,
                            int thickness)
{  
	int thickness = 2;
	int width     = img->h;   /* rows  — as in the original Python */
	int height 		= img->w;   /* cols  — as in the original Python */

	/* Iterate over every row (x in the original Python) */
	for (int x = 0; x < width; x++) {

		/* Compute the column this parabola passes through at this row */
		int y = (int)round(eval_parabola(coeffs, (double)x));

		/* Draw `thickness` pixels horizontally centered on that column */
		for (int dy = -(thickness / 2); dy <= thickness / 2; dy++) {
			int y_draw = y + dy;

			/* Skip if outside image bounds */
			if (y_draw < 0 || y_draw >= height)
					continue;

			// RED in YUV format
			uint8_t Y = 76, U = 84, V = 255;

			set_color_yuv422(img, x, y_draw, Y, U, V);
		}
	}
}