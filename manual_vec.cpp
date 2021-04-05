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


#define MUL(x, y) _mm256_mul_pd((x), (y))
#define SUB(x, y) _mm256_sub_pd((x), (y))
#define ADD(x, y) _mm256_add_pd((x), (y))






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


	__m256d *v_U_prev = (__m256d *)(curr_U0);
	__m256d *v_U_curr = (__m256d *)(curr_U0 + Nx);
	__m256d *v_U_next = (__m256d *)(curr_U0 + 2 * Nx);
	__m256d *v_U_rez = (__m256d *)(curr_U1 + Nx);

	__m256d *v_p_prev = (__m256d *)(p);
	__m256d *v_p_curr = (__m256d *)(p + Nx);
	__m256d *v_p_next = (__m256d *)(p + 2 * Nx);

	__m256d v_c1 = _mm256_set1_pd(1 / (2 * hx * hx));
	__m256d v_c2 = _mm256_set1_pd(1 / (2 * hy * hy));
	__m256d v_tausq = _mm256_set1_pd(tau * tau);
	__m256d v_2cnst = _mm256_set1_pd(2);

	for (int i = 1; i < Ny - 1; i++) {
		__m256d v_U_prev_before = v_U_prev[0];
		__m256d v_U_prev_central = v_U_prev[1]; // (i - 1, j)
		__m256d v_U_prev_next = v_U_prev[2];

		__m256d v_U_curr_before = v_U_curr[0];
		__m256d v_U_curr_central = v_U_curr[1]; //(i, j)
		__m256d v_U_curr_next = v_U_curr[2];

		__m256d v_U_next_before = v_U_next[0];
		__m256d v_U_next_central = v_U_next[1]; // (i + 1, j)
		__m256d v_U_next_next = v_U_next[2];

		__m256d v_p_curr_before = v_p_curr[0]; // 
		__m256d v_p_curr_central = v_p_curr[1]; // P(i, j)
		__m256d v_p_curr_next = v_p_curr[2]; // 

		for (int j = 1; j < Nx / VECTOR_SIZE; j++) {
			__m256d v_U_prev_left = getLeft(v_U_prev_before, v_U_prev_central); // (i - 1, j - 1)
			__m256d v_U_prev_right = getRight(v_U_prev_next, v_U_prev_central); // (i - 1, j + 1)

			__m256d v_U_curr_left = getLeft(v_U_curr_before, v_U_curr_central);	// (i, j - 1)	
			__m256d v_U_curr_right = getRight(v_U_curr_next, v_U_curr_central); // (i, j + 1)

			__m256d v_U_next_left = getLeft(v_U_next_before, v_U_next_central); // (i + 1, j - 1)
			__m256d v_U_next_right = getLeft(v_U_next_next, v_U_next_central); // (i + 1, j + 1)

			__m256d v_p_curr_left = getLeft(v_p_curr_before, v_p_curr_central); // P(i, j - 1)
			__m256d v_p_curr_right = getRight(v_p_curr_next, v_p_curr_central); // P(i, j + 1)

			__m256d rez = ADD(SUB(MUL(v_2cnst, v_U_curr_central), getU(vU_old, i, j)), 
				MUL(v_tausq,
			   ADD((MUL(v_c1,
				ADD(MUL(SUB(v_U_curr_right, v_U_curr_central), ADD(v_p_prev[j], v_p_curr_central)),
					MUL(SUB(v_U_curr_left, v_U_curr_central), ADD(v_p_prev[j - 1], v_p_curr_left)))),
				
				   MUL(v_c2,
				ADD(MUL(SUB(v_U_next_central, v_U_curr_central), ADD(v_p_curr_left, v_p_curr_central)),
					MUL(SUB(v_U_prev_central, v_U_curr_central), ADD(v_p_prev[j - 1], v_p_prev[j])))
					)))));


			// custom abs alternative
			__m256d _vU_max_n = _mm256_max_pd(SUB(v_U_curr_central, rez), SUB(rez, v_U_curr_central));
			vU_max_n = _mm256_max_pd(vU_max_n, _vU_max_n);
			
			v_U_rez[j] = rez;


			v_U_prev_before = v_U_prev_central;
			v_U_prev_central = v_U_prev_next;

			v_U_curr_before = v_U_curr_central;
			v_U_curr_central = v_U_curr_next;

			v_U_next_before = v_U_next_central;
			v_U_next_central = v_U_next_next;

			v_p_curr_before = v_p_curr_central;
			v_p_curr_central = v_p_curr_next;

			v_U_prev_next = v_U_prev[j + 2];
			v_U_curr_next = v_U_curr[j + 2];
			v_U_next_next = v_U_next[j + 2];
			v_p_curr_next = v_p_curr[j + 2];

		}
		v_U_prev = v_U_curr;
		v_U_curr = v_U_next;
		v_U_next += Nx / VECTOR_SIZE;

		v_U_rez += Nx / VECTOR_SIZE;;

		v_p_prev = v_p_curr;
		v_p_curr = v_p_next;
		v_p_next += Nx / VECTOR_SIZE;;
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