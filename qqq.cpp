#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <malloc.h>
#include <alg.h>

#define MUL(x, y) _mm256_mul_pd((x), (y))
#define SUB(x, y) _mm256_sub_pd((x), (y))
#define ADD(x, y) _mm256_add_pd((x), (y))

#define ALIGNMENT 32
#define step 4
#define M_PI	(double)3.14159265358979323846

// (a, b, c, d); (e, f, g, h) -> (d, e, f, g) 
#define getLeft(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x07), 0b10010011)
// (a, b, c, d); (e, f, g, h) -> (b, c, d, e) 
#define getRight(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x01), 0b111001)

// размеры сетки
#define Nx	600
#define Ny	600

#define GRID_SIZE (Nx * Ny)

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
#define Sx Nx - 5
#define Sy Ny - 5

#define f0 (double)1.0
#define t0 (double)1.5
#define gamma (double)4.0

// #define getU(U, i, j)	U[(i) * Nx + (j)]
#define getU(U, i, j)	U[(i) * Nx + (j)]
// #define getP(f, i, j)	f[(i) * Nx + (j)]
#define getP(f, i, j)	f[(i) * Nx + (j)]

//  Величина шага по времени (между последовательными моментами времени):
double tau = (Nx <= 1000 && Ny <= 1000) ? (double)0.01 : (double)0.001;

double c1 = 1 / (2 * hx * hx);
__m256d v_c1 = _mm256_set1_pd(c1);
double c2 = 1 / (2 * hy * hy);
__m256d v_c2 = _mm256_set1_pd(c2);


__m256d v_tausq = _mm256_set1_pd(tau * tau);
__m256d v_2cnst = _mm256_set1_pd(2);

void init_arrays(double *U, /* double *U_old, double *U_new,*/ double *P) {
	for (int i = 0; i < 2 * GRID_SIZE; i++) {
		U[i] = (double)0.0;
		//U_new[i] = (double)0.0;
		//U_old[i] = (double)0.0;
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

void calculate_step( double *p, double *prev_U, double *curr_U) {
	//__m256d vU_max_n = _mm256_setzero_pd();
	//__m256d *vU_old = (__m256d *)U_old;

	__m256d max = _mm256_setzero_pd();

	for (int i = 1; i < Ny - 1; ++i) {
		 __m256d* U_current = (__m256d *)(& curr_U[i * Nx]);
		 __m256d* U_new = (__m256d *)(& prev_U[i * Nx]);

		// Представляем строки матрицы U как вектора 
		__m256d* U_upper = (__m256d *)(& curr_U[(i - 1) * Nx]);
		__m256d* U_lower = (__m256d *)(& curr_U[(i + 1) * Nx]);

		// Вектора строк матрицы U
		// Используем кольцо из 3 элементов и только один раз читаем память 
		__m256d U_prev_vector = _mm256_setzero_pd();
		__m256d U_curr_vector = *U_current;
		__m256d U_next_vector = *(U_current + 1);

		//max = _mm256_max_pd(max, _mm256_and_pd(U_curr_vector, abs_mask));

		// Вектора строк матрицы P
		//  кольцо из 3 элементов, только один раз читаем память 
		__m256d* P_upper = (__m256d*)(& p[(i - 1) * Nx]);
		__m256d* P_lower = (__m256d*)(& p[i * Nx]);
		__m256d P_upper_prev_vector = _mm256_setzero_pd();
		__m256d P_lower_prev_vector = _mm256_setzero_pd();

		for (int shift = 0; shift < Nx; shift += step) {
			// Загружаем строки матрицы P 
			__m256d P_01 = *P_upper;                             //upper
			__m256d P_11 = *P_lower;                             //current
			__m256d P_00 = getLeft(P_upper_prev_vector, P_01);   //left upper
			__m256d P_10 = getLeft(P_lower_prev_vector, P_11);   //left

			// Создаем вектора левых и правых элементов матрицы U из двух выровненных 
			double not_first = (shift != 0) ? 1.0 : 0.0;
			double not_last = (shift + step < Nx) ? 1.0 : 0.0;
			__m256d mask = _mm256_set_pd(not_last, 1.0, 1.0, not_first);
			__m256d U_left_values = _mm256_and_pd(getLeft(U_prev_vector, U_curr_vector), mask);
			__m256d U_right_values = _mm256_and_pd(getRight(U_curr_vector, U_next_vector), mask);

			//NEW:
			__m256d current_value = *U_current;
			__m256d prev_value = *U_new;

			__m256d p1 = MUL(SUB(*U_upper, U_curr_vector), ADD(P_00, P_01));
			__m256d p2 = MUL(SUB(*U_lower, U_curr_vector), ADD(P_10, P_11));
			__m256d p3 = MUL(SUB(U_right_values, U_curr_vector), ADD(P_01, P_11));
			__m256d p4 = MUL(SUB(U_left_values, U_curr_vector), ADD(P_00, P_10));
			__m256d s1 = MUL(ADD(p1, p2), v_c1);
			__m256d s2 = MUL(ADD(p3, p4), v_c2);
			__m256d s3 = MUL(ADD(s1, s2), v_tausq);

			*U_new = SUB(ADD(MUL(v_2cnst, U_curr_vector), s3), prev_value);

			++U_current;
			++U_upper;
			++U_lower;
			++U_new;

			// Сдвигаем и читаем следующий 
			U_prev_vector = U_curr_vector;
			U_curr_vector = U_next_vector;
			U_next_vector = *(U_current + 1);

			++P_upper;
			++P_lower;
			P_upper_prev_vector = P_01;
			P_lower_prev_vector = P_11;
		}
	}

}


int main() {
	//double *U_old = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
	double *U0 = (double *)_aligned_malloc(2 * GRID_SIZE * sizeof(double), ALIGNMENT);
	//double *U1 = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);

	double *P = (double *)_aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);

	if (!U0 /* || !U1 || !U_old*/ || !P) {
		perror("error while allocating memory");
		exit(errno);
	}

	init_arrays(U0, /* U_old, U1, */ P);

	//double *curr_U0;	// current time layer
	//double *curr_U1;	// next time layer


	time_t start_time = time(NULL);
	for (int n = 0; n <= Nt; n++) {
		/* if (n % 2 == 0) {
			curr_U0 = U0;
			curr_U1 = U1;
		}
		else {
			curr_U0 = U1;
			curr_U1 = U0;
		} 

		calculate_step(P, U_old, curr_U0, curr_U1);
		*/

		int cur = (n + 1) & 1;
		int prev = n & 1;

		auto& prev_U = U0[prev * GRID_SIZE];
		auto& cur_U = U0[cur * GRID_SIZE];

		calculate_step(P, &prev_U, &cur_U);


		double val = exp(
			(-1)
			* (1 / pow(gamma, 2))
			* pow((2 * M_PI * f0 * (n * tau - t0)), 2))
			* sin(2 * M_PI * f0 * (n * tau - t0));

		getU(U0, Sy, Sx) += tau * tau * val;
		
		/* 
		double *buf = U_old;
		U_old = U0;
		U0 = U1;
		U1 = buf;
		*/

		// printf("n = %d, U_max^n = %.8f\n", n, U_max_n);
	}

	time_t end_time = time(NULL);
	printf("Total time: %lld sec.\n", end_time - start_time);

	FILE *fout;
	errno_t err = fopen_s(&fout, "output.dat", "wb");
	if (err != 0) {
		perror("error while opening");
		_aligned_free(U0);
//		_aligned_free(U_old);
//		_aligned_free(U1);
		_aligned_free(P);
		exit(errno);
	}
	fwrite(U0, sizeof(double), GRID_SIZE, fout);
	fclose(fout);

	_aligned_free(U0);
//	_aligned_free(U_old);
//	_aligned_free(U1);
	_aligned_free(P);

	return 0;
}
