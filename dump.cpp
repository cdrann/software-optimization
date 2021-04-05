#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <malloc.h>
#include <alg.h>


#define ALIGNMENT 32
#define VECTOR_SIZE 8
#define SHIFT1 1
#define SHIFT2 2
#define REAL_Nx (Nx + SHIFT2)


# define M_PI	(double)3.14159265358979323846

/* (a, b, c, d); (e, f, g, h) -> (d, e, f, g) */
#define getLeft(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x07), 0b10010011)


/* (a, b, c, d); (e, f, g, h) -> (b, c, d, e) */
#define getRight(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x01), 0b111001)



// размеры сетки
#define Nx	600
#define Ny	600
//#define GRID_SIZE (Nx * Ny)
#define GRID_SIZE (REAL_Nx * Ny)

// число шагов
#define Nt	1111

// область моделирования
#define Xa 	(double)0.0
#define Xb	(double)4.0
#define Ya	(double)0.0
#define Yb 	(double)4.0

// шаги сетки по пространству
#define hx (double)((Xb - Xa) / ((double)Nx - (double)1))
#define hy (double)((Yb - Ya) / ((double)Ny - (double)1))

// Координаты узла с источником импульса: SX ∈ [0; NX – 1], SY ∈ [0; NY – 1].
#define Sx Nx-2
#define Sy Ny-2

#define f0 (double)1.0
#define t0 (double)1.5
#define gamma (double)4.0

// #define getU(U, i, j)	U[(i) * Nx + (j)]
#define getU(U, i, j)	U[(i) * REAL_Nx + (j)]
// #define getP(f, i, j)	f[(i) * Nx + (j)]
#define getP(f, i, j)	f[(i) * REAL_Nx + (j)]


//  Величина шага по времени (между последовательными моментами времени):
double tau = (Nx <= 1000 && Ny <= 1000) ? (double)0.01 : (double)0.001;

void init_arrays(double *U, double *U_old, double *U_new, double *P) {
	for (int i = 0; i < GRID_SIZE; i++) {
		U[i] = (double)0.0;
		U_new[i] = (double)0.0;
		U_old[i] = (double)0.0;
	}

	for (int i = 0; i < Ny; i++) {
		for (int j = 0; j < Nx; j++) {
			if (j < Nx / 2) {
				getP(P, i, j) = (double)(0.1 * 0.1);
			}
			else {
				getP(P, i, j) = (double)(0.2 * 0.2);
			}
		}
	}
}

void calculate_step(const double *p, const double *U_old, double *curr_U0, double *curr_U1) {
	__m256d vU_max_n = _mm256_setzero_pd();
	__m256d *vU_old = (__m256d *)U_old;

	for (int i = 1; i < Ny - 1; i++) {
		__m256d *v_U_prev = (__m256d *)(curr_U0 + (i - 1) * REAL_Nx); // (i - 1, j - 1)
		__m256d *v_U_prev_shifted = (__m256d *)(curr_U0 + (i - 1) * REAL_Nx + SHIFT2); // (i - 1, j + 1)  
		__m256d *v_U_prev_vertical = (__m256d *)(curr_U0 + (i - 1) * REAL_Nx + SHIFT1); // (i - 1, j)

		__m256d *v_U_curr = (__m256d *)(curr_U0 + i * REAL_Nx); // (i, j - 1)
		__m256d *v_U_curr_shifted = (__m256d *)(curr_U0 + i * REAL_Nx + SHIFT2); // (i, j + 1)

		__m256d *v_U_next = (__m256d *)(curr_U0 + (i + 1) * REAL_Nx); // (i + 1, j - 1)
		__m256d *v_U_next_shifted = (__m256d *)(curr_U0 + (i + 1) * REAL_Nx + SHIFT2); // (i + 1, j + 1)
		__m256d *v_U_next_vertical = (__m256d *)(curr_U0 + (i + 1) * REAL_Nx + SHIFT1); // (i + 1, j)

		__m256d *v_U_curr_vertical = (__m256d *)(curr_U0 + i * REAL_Nx + SHIFT1); // (i, j)

		__m256d *v_U_rez = (__m256d *)(curr_U1 + i * REAL_Nx + SHIFT1); 

		__m256d *v_p_curr = (__m256d *)(p + i * REAL_Nx); // (i, j - 1)
		__m256d *v_p_curr_shifted = (__m256d *)(p + i * REAL_Nx + SHIFT2); // (i, j + 1)

		__m256d *v_p_prev_vertical = (__m256d *)(p + (i - 1) * REAL_Nx + SHIFT1); // (i - 1, j)
		__m256d *v_p_curr_vertical = (__m256d *)(p + i * REAL_Nx + SHIFT1); // (i, j)
		__m256d *v_p_next_vertical = (__m256d *)(p + (i + 1) * REAL_Nx + SHIFT1); // (i + 1, j)

		__m256d *v_p_prev = (__m256d *)(p + (i - 1) * REAL_Nx); // (i - 1, j - 1)

		for (int j = 1; j < Nx / VECTOR_SIZE; j++) {
			__m256d rez = _mm256_add_pd(_mm256_sub_pd(
				_mm256_mul_pd(_mm256_set1_pd(2), v_U_curr_vertical[j]) , vU_old[(i)* REAL_Nx / VECTOR_SIZE + (j)]) ,
				_mm256_mul_pd( _mm256_set1_pd(tau * tau),
				(
					_mm256_add_pd(
					_mm256_mul_pd(_mm256_set1_pd(1 / (2 * hx * hx)),
					_mm256_add_pd(
					_mm256_mul_pd(_mm256_sub_pd(v_U_curr_shifted[j], v_U_curr_vertical[j]),
					_mm256_add_pd(v_p_prev_vertical[j], v_p_curr_vertical[j]))
					,
					_mm256_mul_pd(_mm256_sub_pd(v_U_curr[j], v_U_curr_vertical[j]),
					_mm256_add_pd(v_p_prev[j],  v_p_curr[j]))
					)
					)
					,
					_mm256_mul_pd(_mm256_set1_pd(1 / (2 * hy * hy)),
					_mm256_add_pd(
					_mm256_mul_pd(_mm256_sub_pd(v_U_next_vertical[j], v_U_curr_vertical[j]),
					_mm256_add_pd(v_p_curr[j], v_p_curr_vertical[j]))
					, 						
					_mm256_mul_pd(_mm256_sub_pd(v_U_prev_vertical[j], v_U_curr_vertical[j]), 
					_mm256_add_pd(v_p_prev[j], v_p_prev_vertical[j]))
					)
					)
					)
					)
				));

			// custom abs alternative
			__m256d _vU_max_n = _mm256_max_pd(_mm256_sub_pd(v_U_rez[j], rez), _mm256_sub_pd(rez, v_U_rez[j]));
			vU_max_n = _mm256_max_pd(vU_max_n, _vU_max_n);
			
			v_U_rez[j] = rez;

		}
	}
}


int main() {
	double *U_old = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
	double *U0 = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
	double *U1 = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);

	double *P = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);

	if (!U0 || !U1 || !U_old || !P) {
		perror("error while allocating memory");
		exit(errno);
	}

	init_arrays(U0, U_old, U1, P);

	double *curr_U0;	// current time layer
	double *curr_U1;	// next time layer


	time_t start_time = time(NULL);
	for (int n = 0; n < Nt - 1; n++) {
		if (n % 2 == 0) {
			curr_U0 = U0;
			curr_U1 = U1;
		}
		else {
			curr_U0 = U1;
			curr_U1 = U0;
		}
		
		calculate_step(P, U_old, curr_U0, curr_U1);


		// *************************** getU(v_U0, Sy, Sx) += v_coeff1; *************************** \\
		
		double val = exp(
		(-1)
			* (1 / pow(gamma, 2))
			* pow((2 * M_PI * f0 * (n * tau - t0)), 2))

			* sin(2 * M_PI * f0 * (n * tau - t0));
		
		// Broadcast double-precision (64-bit) floating-point value a to all elements of dst.

		getU(U0, Sy, Sx) += tau * tau * val;

		/*
		const double coeff1 = tau * tau * val;
		__m256d v_coeff1 = _mm256_set1_pd(coeff1);
		__m256d *v_U0 = (__m256d *)U0;
		getU(v_U0, Sy, Sx) = _mm256_add_pd(getU(v_U0, Sy, Sx), v_coeff1);
		*/
		// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
		
		// *************************** getU(v_U0, Sy, Sx) += v_coeff1; *************************** \\

		double *buf = U_old;
		U_old = U0;
		U0 = U1;
		U1 = buf;

		// printf("n = %d, U_max^n = %.8f\n", n, U_max_n);
	}

	time_t end_time = time(NULL);
	printf("Total time: %lld sec.\n", end_time - start_time);

	FILE *fout;
	errno_t err = fopen_s(&fout, "output.dat", "wb");
	if (err != 0) {
		perror("error while opening");
		_aligned_free(U0);
		_aligned_free(U_old);
		_aligned_free(U1);
		_aligned_free(P);
		exit(errno);
	}
	fwrite(U0, sizeof(double), GRID_SIZE, fout);
	fclose(fout);

	_aligned_free(U0);
	_aligned_free(U_old);
	_aligned_free(U1);
	_aligned_free(P);

	return 0;
}


/// -ftree-vectorize -O3 -fopt-info-vec-optimized -fopt-info-vec-missed 
