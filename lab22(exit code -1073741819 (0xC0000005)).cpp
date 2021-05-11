#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <malloc.h>
#include <stdint.h>

/* Функции создающие не выровненный вектор из двух выровненных */
/* (a, b, c, d); (e, f, g, h) -> (d, e, f, g) */
#define getLeft(a, b)  _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x07), 0b10010011)

/* (a, b, c, d); (e, f, g, h) -> (b, c, d, e) */
#define getRight(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x01), 0b111001)


#define true_ 1.0


#define M_PI    (double)3.14159265358979323846

#define MUL(x, y) _mm256_mul_pd((x), (y))
#define SUB(x, y) _mm256_sub_pd((x), (y))
#define ADD(x, y) _mm256_add_pd((x), (y))


// размеры сетки
#define Nx    600
#define Ny    600
#define GRID_SIZE (Nx * Ny)

// число шагов
#define Nt    1111

// область моделирования
#define Xa    (double)0.0
#define Xb    (double)4.0
#define Ya    (double)0.0
#define Yb    (double)4.0

// шаги сетки по пространству
#define hx (double)((Xb - Xa) / ((double)Nx - (double)1))
#define hy (double)((Yb - Ya) / ((double)Ny - (double)1))

// Координаты узла с источником импульса: SX ∈ [0; NX – 1], SY ∈ [0; NY – 1].
#define Sx Nx - 2
#define Sy Ny - 2

#define f0 (double)1.0
#define t0 (double)1.5
#define gamma (double)4.0

// #define getU(U, i, j)	U[(i) * Nx + (j)]
#define getU(U, i, j)    U[(i) * Nx + (j)]
// #define getP(f, i, j)	f[(i) * Nx + (j)]
#define getP(f, i, j)    f[(i) * Nx + (j)]

#define VECTOR_SIZE 4
#define ALIGNMENT 32
#define step 32


int cnt = (Nx - 1) % 4;
int auxv = (cnt == 3) ? 0 : 1;

double c1 = 1 / (2 * hx * hx);
__m256d v_c1 = _mm256_set1_pd(c1);
double c2 = 1 / (2 * hy * hy);
__m256d v_c2 = _mm256_set1_pd(c2);


//  Величина шага по времени (между последовательными моментами времени):
double tau = (Nx <= 1000 && Ny <= 1000) ? (double) 0.01 : (double) 0.001;


__m256d v_tausq = _mm256_set1_pd(tau * tau);
__m256d v_2cnst = _mm256_set1_pd(2);


void init_arrays(double **U, /* double *U_old, double *U_new, */ double **P) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            U[i][j] = (double) 0.0;
            // U_new[i] = (double) 0.0;
            // U_old[i] = (double) 0.0;
        }

        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < Ny; i++) {
                for (int j = 0; j < Nx; j++) {
                    if (j < Nx / 2) {
                        // getP(P, i, j) = (double) (0.1 * 0.1);
                        P[k][(i) * Nx + (j)] = (double) (0.1 * 0.1);
                    } else {
                        ///getP(P, i, j) = (double) (0.2 * 0.2);
                        P[k][(i) * Nx + (j)] = (double) (0.1 * 0.1);
                    }
                }
            }
        }
    }
}




int main() {
    double **U = NULL; //(double **) malloc(2 * sizeof(double *));

    for (int i = 0; i < 2; i++){
        U[i] = (double *) _aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
    }
   // double *U_new = (double *) _aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
    double **P;// = NULL; //(double **) malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++){
        P[i] = (double *) _aligned_malloc(GRID_SIZE * sizeof(double), ALIGNMENT);
    }
    if (!*U || !*P) {
        perror("error while allocating memory");
        exit(errno);
    }

    init_arrays(U,  P);


    uint64_t start = clock();
    double maximums[Nt + 1];

    for (size_t n = 0; n <= Nt; ++n) {
        uint8_t cur = (n + 1) & 1;
        uint8_t prev = n & 1;

        auto &prev_U = U[prev];
        auto &cur_U = U[cur];

        const double prod = 2.0 * M_PI * f0 * (n * tau - t0);
        //const double source = exp((-1.0) * sqr(prod) * y) * sin(prod);
        const double source = exp((-1) * (1 / pow(gamma, 2))
                                  * pow((2 * M_PI * f0 * (n * tau - t0)), 2))
                              * sin(2 * M_PI * f0 * (n * tau - t0));


        __m256d max = _mm256_setzero_pd();
        for (size_t i = 1; i < Ny - 1; ++i) {
            __m256d *U_current = (__m256d *) (& getU(cur_U, i, 0)); //(cur_U[i]);
            __m256d *U_dest = (__m256d *) (& getU(prev_U, i, 0)); // (__m256d *) (prev_U[i]);

            /* Представляем строки матрицы U как вектора */
            __m256d *U_upper = (__m256d *) (& getU(cur_U, i - 1, 0)); // (cur_U[i - 1]);
            __m256d *U_lower = (__m256d *) (& getU(cur_U, i + 1, 0)); // (cur_U[i + 1]);

            /* Вектора строк матрицы U
            Используем кольцо из 3 элементов и только один раз читаем память */
            __m256d U_prev_vector = _mm256_setzero_pd();
            __m256d U_curr_vector = *U_current;
            __m256d U_next_vector = *(U_current + 1);

            /* Считаем максимум */
            // max = _mm256_max_pd(max, _mm256_and_pd(U_curr_vector, abs_mask));

            /* Вектора строк матрицы P
             * Используем кольцо из 3 элементов и только один раз читаем память */
            __m256d *P_upper = (__m256d *) P[i - 1];
            __m256d *P_lower = (__m256d *) P[i];
            __m256d P_upper_prev_vector = _mm256_setzero_pd();
            __m256d P_lower_prev_vector = _mm256_setzero_pd();

            for (size_t shift = 0; shift < Nx; shift += step) {
                /* Загружаем строки матрицы P */
                __m256d P_01 = *P_upper;                             //upper
                __m256d P_11 = *P_lower;                             //current
                __m256d P_00 = getLeft(P_upper_prev_vector, P_01);   //left upper
                __m256d P_10 = getLeft(P_lower_prev_vector, P_11);   //left

                /* Создаем вектора левых и правых элементов матрицы U из двух выровненных */
                double not_first = (shift != 0) ? true_ : 0.0;
                double not_last = (shift + step < Nx) ? true_ : 0.0;
                __m256d mask = _mm256_set_pd(not_last, true_, true_, not_first);
                __m256d U_left_values = _mm256_and_pd(getLeft(U_prev_vector, U_curr_vector), mask); // U_{i+1, j}
                __m256d U_right_values = _mm256_and_pd(getRight(U_curr_vector, U_next_vector), mask); // U_{i-1, j}

                /* Считаем */
                __m256d prod_1 = _mm256_mul_pd(_mm256_sub_pd(*U_upper, U_curr_vector), _mm256_add_pd(P_00, P_01));
                __m256d prod_2 = _mm256_mul_pd(_mm256_sub_pd(*U_lower, U_curr_vector), _mm256_add_pd(P_10, P_11));
                __m256d prod_3 = _mm256_mul_pd(_mm256_sub_pd(U_left_values, U_curr_vector), _mm256_add_pd(P_00, P_10));
                __m256d prod_4 = _mm256_mul_pd(_mm256_sub_pd(U_right_values, U_curr_vector), _mm256_add_pd(P_01, P_11));

                __m256d sum1 = _mm256_mul_pd(_mm256_add_pd(prod_1, prod_2), v_c1);
                __m256d sum2 = _mm256_mul_pd(_mm256_add_pd(prod_3, prod_4), v_c2);
                __m256d sum = _mm256_mul_pd(_mm256_add_pd(sum1, sum2), v_tausq);

                *U_dest = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(v_2cnst, U_curr_vector), sum), *U_dest);

                ++U_current;
                ++U_upper;
                ++U_lower;
                ++U_dest;

                /* Сдвигаем и читаем следующий */
                U_prev_vector = U_curr_vector;
                U_curr_vector = U_next_vector;
                U_next_vector = *(U_current + 1);

                /* Сдвигаем и читаем следующий */
                ++P_upper;
                ++P_lower;
                P_upper_prev_vector = P_01;
                P_lower_prev_vector = P_11;
            }
        }
        U[prev][Sy][Sx] += tau * tau * source * source;
        // U[prev][Sy][Sx] += /*t_sqr*/ tau * tau * source;

        /* Сохраняем максимум */
        //_mm256_store_pd(four, max);
        // maximums[n] = *max_element(four, four + 4);
    }
    uint64_t end = clock();

    printf("Total time: %lld sec.\n", end - start);

    FILE *fout;
    errno_t err = fopen_s(&fout, "output.dat", "wb");
    if (err != 0) {
        perror("error while opening");

        _aligned_free(U);
        // _aligned_free(U_old);
        // _aligned_free(U_new);
        _aligned_free(P);
        exit(errno);
    }

    fwrite(U, sizeof(double), GRID_SIZE, fout);
    fclose(fout);

    _aligned_free(U);
    // _aligned_free(U_new);
    // _aligned_free(U_old);
    _aligned_free(P);

    return 0;
}
