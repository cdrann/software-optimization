 

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <malloc.h>
#include <algorithm>
#include <ostream>

#define MUL(x, y) _mm256_mul_pd((x), (y))
#define SUB(x, y) _mm256_sub_pd((x), (y))
#define ADD(x, y) _mm256_add_pd((x), (y))

#define ALIGNMENT 32
#define STEP 4
# define M_PI	(double)3.14159265358979323846

// (a, b, c, d); (e, f, g, h) -> (d, e, f, g) 
#define getLeft(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x07), 0b10010011)
// (a, b, c, d); (e, f, g, h) -> (b, c, d, e) 
#define getRight(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x01), 0b111001)

// размеры сетки
#define Nx	3000
#define Ny	3000
#define GRID_SIZE (Nx * Ny)

// число шагов
#define Nt	100

// область моделирования
#define Xa 	(double)0.0
#define Xb	(double)4.0
#define Ya	(double)0.0
#define Yb 	(double)4.0

// шаги сетки по пространству
#define hx (double)((Xb - Xa) / ((double)Nx - (double)1))
#define hy (double)((Yb - Ya) / ((double)Ny - (double)1))

// Координаты узла с источником импульса: SX ∈ [0; NX – 1], SY ∈ [0; NY – 1].
#define Sx Nx - 2
#define Sy Ny - 2

#define f0 (double)1.0
#define t0 (double)1.5
#define gamma (double)4.0

#define getU(U, i, j)	U[(i) * Nx + (j)]
#define getP(f, i, j)	f[(i) * Nx + (j)]

//  Величина шага по времени (между последовательными моментами времени):
double tau = (Nx <= 1000 && Ny <= 1000) ? (double)0.01 : (double)0.001;

double c1 = 1 / (2 * hx * hx);
__m256d v_c1 = _mm256_set1_pd(c1);
double c2 = 1 / (2 * hy * hy);
__m256d v_c2 = _mm256_set1_pd(c2);

__m256d v_tausq = _mm256_set1_pd(tau * tau);
__m256d v_2cnst = _mm256_set1_pd(2);

void init_arrays(double *U, double *U_old, double *U_new, double *P) {
	for (int i = 0; i < GRID_SIZE; i++) {
		U[i] = (double)0.0;
		U_old[i] = (double)0.0;
		U_new[i] = (double)0.0;
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

void calculate_step(double *P, double *U, double *U_old, double *U_new) {
	for (int i = 1; i < Ny - 1; ++i) {
		__m256d* prev_vector_pointer_in_row = (__m256d *)(U_old + i * Nx);
		__m256d* calculated_vector_pointer_in_row = (__m256d *)(U_new + i * Nx);

		__m256d* upper_vector_pointer_in_row = (__m256d *)(U + (i - 1) * Nx);
		__m256d* current_vector_pointer_in_row = (__m256d *)(U + i * Nx);
		__m256d* lower_vector_pointer_in_row = (__m256d *)(U + (i + 1) * Nx);

		__m256d U_prev_vector_value = _mm256_setzero_pd();
		__m256d U_curr_vector_value = *current_vector_pointer_in_row;
		__m256d U_next_vector_value = *(current_vector_pointer_in_row + 1);

		__m256d* P_upper_vector_pointer_in_row = (__m256d*)(P + (i - 1) * Nx);
		__m256d* P_lower_vector_pointer_in_row = (__m256d*)(P + i * Nx);
		__m256d P_upper_prev_vector = *P_upper_vector_pointer_in_row;
		__m256d P_lower_prev_vector = *P_lower_vector_pointer_in_row;

		for (int j = 0; j < Nx; j += STEP) {
			__m256d P_upper = *P_upper_vector_pointer_in_row;       
			__m256d P_curr = *P_lower_vector_pointer_in_row;      
			__m256d P_left_upper = getLeft(P_upper_prev_vector, P_upper);   
			__m256d P_left = getLeft(P_lower_prev_vector, P_curr);   

			// Используем функции, создающие невыровненный вектор из двух выровненных 
			double not_first = (j != 0) ? 1.0 : 0.0;
			double not_last = (j + STEP < Nx) ? 1.0 : 0.0;
			__m256d left_m = _mm256_set_pd(1.0, 1.0, 1.0, not_first);
			__m256d U_left_values = MUL(getLeft(U_prev_vector_value, U_curr_vector_value), left_m);
			__m256d right_m = _mm256_set_pd(not_last, 1.0, 1.0, 1.0);
			__m256d U_right_values = MUL(getRight(U_curr_vector_value, U_next_vector_value), right_m);

			__m256d p1 = MUL(SUB(U_right_values, U_curr_vector_value), ADD(P_upper, P_curr));
			__m256d p2 = MUL(SUB(U_left_values, U_curr_vector_value), ADD(P_left_upper, P_left));
			__m256d s1 = MUL(ADD(p1, p2), v_c1);

			__m256d p3 = MUL(SUB(*lower_vector_pointer_in_row, U_curr_vector_value), ADD(P_left, P_curr));
			__m256d p4 = MUL(SUB(*upper_vector_pointer_in_row, U_curr_vector_value), ADD(P_left_upper, P_left));
			__m256d s2 = MUL(ADD(p3, p4), v_c2);

			__m256d s3 = MUL(ADD(s1, s2), v_tausq);

			*calculated_vector_pointer_in_row = SUB(ADD(MUL(v_2cnst, U_curr_vector_value), s3), *prev_vector_pointer_in_row);

			++prev_vector_pointer_in_row;
			++current_vector_pointer_in_row;
			++upper_vector_pointer_in_row;
			++lower_vector_pointer_in_row;
			++calculated_vector_pointer_in_row;

			U_prev_vector_value = *(current_vector_pointer_in_row - 1);
			U_curr_vector_value = *(current_vector_pointer_in_row);
			U_next_vector_value = *(current_vector_pointer_in_row + 1);

			++P_upper_vector_pointer_in_row;
			++P_lower_vector_pointer_in_row;
			P_upper_prev_vector = P_upper;
			P_lower_prev_vector = P_curr;
		}
	}
}


int main() {
	double *U = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
	double *U_old = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
	double *U_new = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);

	double *P = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);

	if (!U || !U_new || !U_old || !P) {
		perror("error while allocating memory");
		exit(errno);
	}

	init_arrays(U, U_old, U_new, P);

	time_t start_time = time(NULL);
	for (int n = 0; n <= Nt; ++n) {
		calculate_step(P, U, U_old, U_new);

		double val = exp(
			(-1)
			* (1.0 / pow(gamma, 2))
			* pow((2 * M_PI * f0 * (n * tau - t0)), 2))
			* sin(2 * M_PI * f0 * (n * tau - t0));
		getU(U_new, Sy, Sx) += tau * tau * val;

		for (int i = 0; i < GRID_SIZE; i++) {
			U_old[i] = U[i];
			U[i] = U_new[i];
		}
	}

	time_t end_time = time(NULL);
	printf("Total time: %lld sec.\n", end_time - start_time);

	FILE *fout;
	errno_t err = fopen_s(&fout, "output.dat", "wb");
	if (err != 0) {
		perror("error while opening");
		_aligned_free(U_old);
		_aligned_free(U);
		_aligned_free(U_new);
		_aligned_free(P);

		exit(errno);
	}
	fwrite(U, sizeof(double), GRID_SIZE, fout);
	fclose(fout);

	_aligned_free(U_old);
	_aligned_free(U);
	_aligned_free(U_new);
	_aligned_free(P);

	return 0;
}

