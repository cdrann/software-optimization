#pragma once

#include "Matrix.h"
#include <vector>
#include <emmintrin.h>
#include <immintrin.h>
#include <array>

#define NCORES 8
#define ITERS 4

#define MUL(x, y) _mm256_mul_pd((x), (y))
#define SUB(x, y) _mm256_sub_pd((x), (y))
#define ADD(x, y) _mm256_add_pd((x), (y))

#define sqr(x) ((x) * (x))
constexpr double M_PI = (double)3.14159265358979323846;

class CounterThread {
	struct CommonData {
		static constexpr double Xa = 0.0;
		static constexpr double Xb = 4.0;
		static constexpr double Ya = 0.0;
		static constexpr double Yb = 4.0;

		static constexpr double f0 = 1.0;
		static constexpr double t0 = 1.5;
		static constexpr double y = 1.0 / (4.0 * 4.0);

		__m256d v_c1;
		__m256d v_c2;
		__m256d v_tausq;
		__m256d v_2cnst;

		std::size_t threads_count;
		std::size_t lines_per_thread;

		std::size_t Sx;
		std::size_t Sy;

		double tau;

		void init(std::size_t Nx, std::size_t Ny, std::size_t Nt, double t_, double threads_count_);

		__forceinline double getSource(std::size_t n) {
			const double prod = 2.0 * M_PI * f0 * (n * tau - t0);
			return tau * tau * std::exp((-1.0) * sqr(prod) * y) * std::sin(prod);
		}
	};

	const int thread_number;

	void bindThreadToCore();

	const std::size_t curr_block_start;
	const std::size_t next_block_start;

	std::array<double, ITERS> to_add = {};

	void computeLine(Matrix& U_new, const Matrix& U, const Matrix& P,
		std::size_t line, std::size_t Nx, double additional);

public:
	static CommonData InputData;

	CounterThread(int thread_number_);

	void source_upd(std::size_t n);

	static void init(std::size_t Nx, std::size_t Ny, std::size_t Nt, double t, double threads_count);

	void step_1(Matrix& U_new, Matrix& U, const Matrix& P, std::size_t Nx);
	void step_2(Matrix& U_new, Matrix& U, const Matrix& P, std::size_t Nx);
	void step_3(Matrix& U_new, Matrix& U, const Matrix& P, std::size_t Nx);
};