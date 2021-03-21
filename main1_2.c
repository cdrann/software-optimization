#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#define comp_type double

# define M_PI	(comp_type)3.14159265358979323846

// размеры сетки
#define Nx	7500 //8000 - 10000
#define Ny	7500 //8000 - 10000
#define GRID_SIZE (Nx * Ny)

// число шагов
#define Nt	100 //100-120

// область моделирования
#define Xa 	(comp_type)0.0
#define Xb	(comp_type)4.0
#define Ya	(comp_type)0.0
#define Yb 	(comp_type)4.0

// шаги сетки по пространству
#define hx (comp_type)((Xb - Xa) / ((comp_type)Nx - (comp_type)1))
#define hy (comp_type)((Yb - Ya) / ((comp_type)Ny - (comp_type)1))

// Координаты узла с источником импульса: SX ∈ [0; NX – 1], SY ∈ [0; NY – 1].
#define Sx Nx-2
#define Sy Ny-2

#define f0 (comp_type)1.0
#define t0 (comp_type)1.5
#define gamma (comp_type)4.0

#define getU(U, i, j)	U[(i) * Nx + (j)]
#define getP(f, i, j)	f[(i) * Nx + (j)]

//  Величина шага по времени (между последовательными моментами времени):
comp_type tau = (Nx <= 1000 && Ny <= 1000) ? (comp_type)0.01 : (comp_type)0.001;

void init_arrays(comp_type *U, comp_type *U_old, comp_type *U_new, comp_type *P) {
    for(int i = 0; i < GRID_SIZE; i++) {
        U[i] = (comp_type)0.0;
        U_new[i] = (comp_type)0.0;
        U_old[i] = (comp_type)0.0;
    }

    for(int i = 0; i < Ny; i++) {
        for(int j = 0; j < Nx; j++) {
            if ( j < Nx / 2) {
                getP(P, i, j) = (comp_type)(0.1 * 0.1);
            } else {
                getP(P, i, j) = (comp_type)(0.2 * 0.2);
            }
        }
    }
}

void calculate_step(const comp_type *U, const comp_type *P, const comp_type *U_old, comp_type *U_new, comp_type *U_max_n, comp_type val) {
    for(int i = 1; i <= Ny - 2; i++) {

        comp_type U_lower = getU(U, i - 1, 1); // (i-1, j)
        comp_type U_lower_right = getU(U, i - 1, 2); // (i-1, j+ 1)

        comp_type U_left = getU(U, i, 0); // (i, j-1)
        comp_type U_curr = getU(U, i, 1); // (i, j)
        comp_type U_right = getU(U, i, 2); // (i, j+1)

        comp_type U_upper = getU(U, i + 1, 1); // (i+1, j)
        comp_type U_upper_right = getU(U, i + 1, 2); // (i+1, j+1)

        comp_type P_left = getP(P, i, 0); // (i, j-1)
        comp_type P_curr = getP(P, i, 1); // (i, j)
        comp_type P_right = getP(P, i, 2); // (i, j+1)

        comp_type P_lower_left = getP(P, i - 1, 0); // (i-1, j-1)
        comp_type P_lower = getP(P, i - 1, 1); // (i-1, j)
        comp_type P_lower_right = getP(P, i - 1, 2); // (i-1, j+1)

        for(int j = 1; j <= Nx - 2; j++) {

            getU(U_new, i, j) = 2 * U_curr - getU(U_old, i, j) +
                                tau * tau *
                                (
                                  (1 / (2 * hx * hx)) *
                                  ((U_right - U_curr) * (P_lower + P_curr)
                                   + (U_left - U_curr) * (P_lower_left + P_left))

                                  +  (1 / (2 * hy * hy)) *
                                     ((U_upper - U_curr) * (P_left + P_curr)
                                      + (U_lower - U_curr) * (P_lower_left + P_lower)  )
                                );

            *U_max_n = max(*U_max_n, fabs(getU(U_new, i, j)));

            U_lower = U_lower_right; // (i-1, j)
            U_lower_right = getU(U, i - 1, j + 2); //(i-1, j+ 1)

            U_left = U_curr; // (i, j-1)
            U_curr = U_right; // (i, j)
            U_right = getU(U, i, j + 2); // (i, j+1)

            U_upper = U_upper_right; // (i+1, j)
            U_upper_right =  getU(U, i + 1, j + 2); // (i+1, j+1)

            P_left = P_curr; // (i, j-1)
            P_curr =  P_right; // (i, j)
            P_right = getP(P, i, j + 2); // (i, j+1)

            P_lower_left = P_lower; // (i-1, j-1)
            P_lower =  P_lower_right; // (i-1, j)
            P_lower_right = getP(P, i - 1, j + 2); // (i-1, j+1)
        }
    }
}

int main() {
    // искомая функция
    comp_type *U = malloc( GRID_SIZE * sizeof(comp_type));
    comp_type *U_old = malloc(GRID_SIZE * sizeof(comp_type));
    comp_type *U_new = malloc(GRID_SIZE * sizeof(comp_type));

    // Фазовая скорость (характеристика пространства, отражает скорость распространения волны)
    comp_type *P = malloc(GRID_SIZE * sizeof(comp_type));

    if(!U || !U_new || !U_old || !P) {
        perror("error while allocating memory");
        exit(errno);
    }

    init_arrays(U, U_old, U_new, P);

     time_t start_time = time(NULL);

    comp_type U_max_n = 0;
    for(int n = 0; n < Nt; n++) {
        U_max_n = 0;
        comp_type val = exp((-1) * (1 / pow(gamma, 2) ) * pow((2 * M_PI * f0 * (n * tau - t0)), 2) )
                        * sin(2 * M_PI * f0 * (n * tau - t0));

        calculate_step(U, P, U_old, U_new, &U_max_n, val);

            getU(U, Sy, Sx) += tau * tau * val;

        comp_type *buf = U_old;
        U_old = U;
        U = U_new;
        U_new = buf;

       // printf("n = %d, U_max^n = %.8f\n", n, U_max_n);
    }

     time_t end_time = time(NULL);

    printf("Total time: %lld sec.\n", end_time - start_time);

    FILE *fout;
    errno_t err = fopen_s(&fout, "output.dat", "wb");

    if(err != 0) {
        perror("error while opening");
        exit(errno);
    }

    fwrite(U , sizeof(comp_type), GRID_SIZE, fout);

    fclose(fout);

    free(U);
    free(U_old);
    free(U_new);
    free(P);

    return 0;
}