#include "Matrix.h"
#include <Windows.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>

// Координаты узла с источником импульса: SX ∈ [0; NX – 1], SY ∈ [0; NY – 1].
#define Sx Nx - 2
#define Sy Ny - 2

#define STEP 4
#define NCORES 4

#define MUL(x, y) _mm256_mul_pd((x), (y))
#define SUB(x, y) _mm256_sub_pd((x), (y))
#define ADD(x, y) _mm256_add_pd((x), (y))

#define sqr(x) ((x) * (x))
#define M_PI	(double)3.14159265358979323846

// (a, b, c, d); (e, f, g, h) -> (d, e, f, g)  
#define getLeft(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x07), 0b10010011)

// (a, b, c, d); (e, f, g, h) -> (b, c, d, e)  
#define getRight(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x01), 0b111001)

struct {
	// область моделирования
	static constexpr double Xa = (double) 0.0;
	static constexpr double Xb = (double) 4.0;
	static constexpr double Ya = (double) 0.0;
	static constexpr double Yb = (double) 4.0;

	static constexpr double f0 = (double) 1.0;
	static constexpr double t0 = (double) 1.5;
	static constexpr double gamma = (double) 4.0;

	__m256d v_c1;
	__m256d v_c2;
	__m256d v_tausq;
	__m256d v_2cnst;

	double tau;

	void init(std::size_t Nx, std::size_t Ny, std::size_t Nt) {
		tau = (Nx <= 1000 && Ny <= 1000) ? (double)0.01 : (double)0.001;
		v_tausq = _mm256_set1_pd(tau * tau);
		v_c1 = _mm256_set1_pd(0.5 * sqr((Nx - 1.0) / (Xb - Xa)));
		v_c2 = _mm256_set1_pd(0.5 * sqr((Ny - 1.0) / (Yb - Ya)));
		v_2cnst = _mm256_set1_pd(2);
	}

	__forceinline double getSource(std::size_t n) {
		double val = exp(
			(-1)
			* (1.0 / pow(gamma, 2))
			* pow((2 * M_PI * f0 * (n * tau - t0)), 2))
			* sin(2 * M_PI * f0 * (n * tau - t0));
		return val;
	}

} InputData;

Matrix initP(std::size_t Nx, std::size_t Ny) {
	Matrix P{ Nx, Ny };
	
	for (std::size_t i = 0; i < Ny; i++) {
		for (std::size_t j = 0; j < Nx; j++) {
			if (j < Nx / 2) {
				P[i][j] = (double)(0.1 * 0.1);
			}
			else {
				P[i][j] = (double)(0.2 * 0.2);
			}
		}
	}

	return P;
}

void bind_thread_to_core() {
	HANDLE process;
	DWORD_PTR processAffinityMask = 0;
	// Windows uses a compact thread topology.  Set mask to every other thread
	for (int i = 0; i < NCORES; i++)
		processAffinityMask |= 1L << (2 * i);

	process = GetCurrentProcess();
	SetProcessAffinityMask(process, processAffinityMask);

	HANDLE thread = GetCurrentThread();
	DWORD_PTR threadAffinityMask = 1L << (2 * omp_get_thread_num());
	SetThreadAffinityMask(thread, threadAffinityMask);
}

void computeLine(Matrix &U_new, const Matrix &U, const Matrix &P, std::size_t line, std::size_t Nx) {
	__m256d *U_cur = (__m256d *)(U[line]);
	__m256d *calculated_vector_pointer_in_row = (__m256d *)(U_new[line]);

	__m256d *U_upper = (__m256d *)(U[line - 1]);
	__m256d *U_lower = (__m256d *)(U[line + 1]);

	__m256d U_prev_vector_value = _mm256_setzero_pd();
	__m256d U_curr_vector_value = *U_cur;
	__m256d U_next_vector_value = *(U_cur + 1);

	__m256d *P_upper_vector_pointer_in_row = (__m256d *)P[line - 1];
	__m256d *P_lower_vector_pointer_in_row = (__m256d *)P[line];
	__m256d P_upper_prev_vector = _mm256_setzero_pd();
	__m256d P_lower_prev_vector = _mm256_setzero_pd();

	for (std::size_t j = 0; j < Nx; j += STEP) {
		__m256d P_upper = *P_upper_vector_pointer_in_row;
		__m256d P_curr = *P_lower_vector_pointer_in_row;
		__m256d P_left_upper = getLeft(P_upper_prev_vector, P_upper); 
		__m256d P_left = getLeft(P_lower_prev_vector, P_curr); 

		// Используем функции, создающие невыровненный вектор из двух выровненных 
		double not_first = (j != 0) ? 1.0 : 0.0;
		double not_last = (j + STEP < Nx) ? 1.0 : 0.0;
		__m256d mask = _mm256_set_pd(1.0, 1.0, 1.0, not_first);
		__m256d U_left_values = _mm256_mul_pd(getLeft(U_prev_vector_value, U_curr_vector_value ), mask);
		__m256d mask2 = _mm256_set_pd(not_last, 1.0, 1.0, 1.0);
		__m256d U_right_values = _mm256_mul_pd(getRight(U_curr_vector_value, U_next_vector_value ), mask2);

		__m256d p1 = MUL(SUB(*U_upper, U_curr_vector_value), ADD(P_left_upper, P_upper));
		__m256d p2 = MUL(SUB(*U_lower, U_curr_vector_value), ADD(P_left, P_curr));
		__m256d s1 = MUL(ADD(p1, p2), InputData.v_c1);

		__m256d p3 = MUL(SUB(U_right_values, U_curr_vector_value), ADD(P_upper, P_curr));
		__m256d p4 = MUL(SUB(U_left_values, U_curr_vector_value),	ADD(P_left_upper, P_left));
		__m256d s2 = MUL(ADD(p3, p4), InputData.v_c2);

		__m256d s3 = MUL(ADD(s1, s2), InputData.v_tausq);

		*calculated_vector_pointer_in_row = SUB(ADD(MUL(InputData.v_2cnst, U_curr_vector_value), s3), *calculated_vector_pointer_in_row);

		++U_cur;
		++U_upper;
		++U_lower;
		++calculated_vector_pointer_in_row;

		U_prev_vector_value = U_curr_vector_value;
		U_curr_vector_value = U_next_vector_value;
		U_next_vector_value = *(U_cur + 1);

		++P_upper_vector_pointer_in_row;
		++P_lower_vector_pointer_in_row;
		P_upper_prev_vector = P_upper;
		P_lower_prev_vector = P_curr;
	}
}

Matrix calculate_matrix(std::size_t Nx, std::size_t Ny, std::size_t Nt, int threads_count) {
	InputData.init(Nx, Ny, Nt);
	Matrix U { Nx, Ny };
	Matrix U_new { Nx, Ny };
	Matrix P = initP(Nx, Ny);

	std::size_t lines_per_thread = Ny / threads_count;
	omp_set_num_threads(threads_count);

	time_t start_time = time(NULL);
	// std::uint64_t start = clock(); // omp_get_wtime();

#pragma omp parallel
	{
		bind_thread_to_core();

		int thread_number = omp_get_thread_num();
		std::size_t beg_line = lines_per_thread * thread_number;
		std::size_t end_line = beg_line + lines_per_thread;

		memset(U.getRaw() + Nx * beg_line, 0, Nx * lines_per_thread * sizeof(double));
		memset(U_new.getRaw() + Nx * beg_line, 0, Nx * lines_per_thread * sizeof(double));

		for (std::size_t n = 0; n <= Nt; ++n) {
			for (std::size_t line = beg_line; line < end_line; ++line) {
				if (line == 0 || line == Ny - 1) {
					continue;
				}
				computeLine(U_new, U, P, line, Nx);
			}

#pragma omp barrier
#pragma omp single
			{
				U_new[Sy][Sx] += sqr(InputData.tau) * InputData.getSource(n);
				std::swap(U_new, U);
			}
#pragma omp barrier
		}
	}
	
	time_t end_time = time(NULL);
	//std::int64_t end = clock(); // omp_get_wtime();
	printf("Total time: %lld sec.\n", end_time - start_time);

	return U;
}


int main(int argc, char **argv) {
	if (argc < 2) {
		std::cerr << "thread_count not stated" << std::endl;
		return EXIT_FAILURE;
	}
	std::stringstream ss{ argv[1] };
	int thread_count;
	ss >> thread_count;
	if (ss.fail()) {
		std::cerr << "thread_count is wrong" << std::endl;
		return EXIT_FAILURE;
	}

	std::size_t Nx;
	std::cout << "Enter Nx:\n";
	std::cin >> Nx;

	std::size_t Ny;
	std::cout << "Enter Ny:\n";
	std::cin >> Ny;

	std::size_t Nt;
	std::cout << "Enter Nt:\n";
	std::cin >> Nt;

	if (Ny % thread_count != 0) {
		std::cerr << "Ny % thread_count != 0" << std::endl;
		return EXIT_FAILURE;
	}

	Matrix U = calculate_matrix(Nx, Ny, Nt, thread_count);

	FILE *fout;
	errno_t err = fopen_s(&fout, "output.dat", "wb");
	if (err != 0) {
		perror("error while opening");
		exit(errno);
	}
	fwrite(U.getRaw(), sizeof(double), Nx * Ny, fout);
	fclose(fout);

	system(".\\plot_script.plt");
	system("output.dat.png");
	
	//_popen("Users\\Admin\\source\\repos\\Lab4_OptSoft\\Lab4_OptSoft]\\plot_script.plt", "rt"); 
	//WinExec("C:\\Users\\Admin\\source\\repos\\Lab4_OptSoft\\Lab4_OptSoft\\plot_script.plt", 1); 
	// WinExec not working
	//swapn, execl
	
	return EXIT_SUCCESS;
}

