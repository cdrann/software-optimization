#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#define comp_type double

# define M_PI	(comp_type)3.14159265358979323846

// размеры сетки
#define Nx	600
#define Ny	600
#define GRID_SIZE (Nx * Ny)

// число шагов
#define Nt	1111

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

void calculate_step2(const comp_type *U, const comp_type *P, const comp_type *U_old, comp_type *U_new) {
    comp_type U_max_n = 0;

    for(int i = 1; i <= Ny - 2; i++) {
        for(int j = 1; j <= Nx - 2; j++) {
            getU(U_new, i, j) = 2 * getU(U, i, j) - getU(U_old, i, j) +
                                tau * tau *
                                ((1 / (2 * hx * hx)) *
                                 ((getU(U, i, j + 1) - getU(U, i, j)) * (getP(P, i - 1, j) + getP(P, i, j))
                                  + (getU(U, i, j - 1) - getU(U, i, j)) * (getP(P, i - 1, j - 1) + getP(P, i, j - 1)))

                                 +  (1 / (2 * hy * hy)) *
                                    ((getU(U, i + 1, j) - getU(U, i, j)) * (getP(P, i, j - 1) + getP(P, i, j))
                                     + (getU(U, i - 1, j) - getU(U, i, j)) * (getP(P, i - 1, j - 1) + getP(P, i - 1, j))  )
                                );

            U_max_n = max(U_max_n, fabs(getU(U_new, i, j)));

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

    //comp_type U_max_n = 0;
    for(int n = 0; n < Nt; n++) {
        //U_max_n = 0;
        comp_type val = exp((-1) * (1 / pow(gamma, 2) ) * pow((2 * M_PI * f0 * (n * tau - t0)), 2) )
                        * sin(2 * M_PI * f0 * (n * tau - t0));
        //calculate_step(U, P, U_old, U_new, &U_max_n, val);

        calculate_step2(U, P, U_old, U_new);

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