
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

#include "thread_worker.h"

#define getTau(Nx, Ny) (Nx <= 1000 && Ny <= 1000) ? 0.01 : 0.001

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

Matrix calculate_matrix(std::size_t Nx, std::size_t Ny, std::size_t Nt, double tau, int threads_count) {

	CounterThread::init(Nx, Ny, Nt, tau, threads_count);

	Matrix U{ Nx, Ny };
	Matrix V{ Nx, Ny };
	Matrix P = initP(Nx, Ny);

	// const double t_sqr = tau * tau;
	const std::size_t Sx = 1;
	const std::size_t Sy = 1;

	std::size_t lines_per_thread = Ny / threads_count;
	omp_set_num_threads(threads_count);

	std::uint64_t start = clock();

#pragma omp parallel
	{
		int thread_number = omp_get_thread_num();
		std::size_t begin = lines_per_thread * thread_number;
		std::size_t end = begin + lines_per_thread;

		CounterThread worker(thread_number);

		memset(U.getRaw() + Nx * begin, 0, Nx * lines_per_thread * sizeof(double));
		memset(V.getRaw() + Nx * begin, 0, Nx * lines_per_thread * sizeof(double));

		for (std::size_t n = 0; n <= Nt; n += ITERS) {
			worker.source_upd(n);

			worker.step_1(V, U, P, Nx);

			worker.step_2(V, U, P, Nx);

#pragma omp barrier
			worker.step_3(V, U, P, Nx); 
#pragma omp barrier
		}
	}
	std::int64_t end = clock();
	std::cout << "Time: " << double(end - start) / CLOCKS_PER_SEC << std::endl;
	return U;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cerr << "Set number of threads" << std::endl;
		return EXIT_FAILURE;
	}
	std::stringstream ss{ argv[1] };
	int thread_count;
	ss >> thread_count;
	if (ss.fail()) {
		std::cerr << "Wrong number" << std::endl;
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

	Matrix U = calculate_matrix(Nx, Ny, Nt, getTau(Nx, Ny), thread_count);

	FILE *fout;
	errno_t err = fopen_s(&fout, "output.dat", "wb");
	if (err != 0) {
		perror("error while opening");
		exit(errno);
	}
	fwrite(U.getRaw(), sizeof(double), Nx * Ny, fout);
	fclose(fout);
	
	return EXIT_SUCCESS;
}

