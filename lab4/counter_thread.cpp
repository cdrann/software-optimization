#include "counter_thread.h"

#include <Windows.h>

CounterThread::CommonData CounterThread::InputData{};

const __m256d ABS_MASK = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
#define SHIFT (ITERS - 1)
constexpr std::size_t STEP = sizeof(__m256d) / sizeof(double);

// (a, b, c, d); (e, f, g, h) -> (d, e, f, g) 
#define getLeft(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x07), 0b10010011)

// (a, b, c, d); (e, f, g, h) -> (b, c, d, e) 
#define getRight(a, b) _mm256_permute4x64_pd(_mm256_blend_pd((a), (b), 0x01), 0b111001)

void CounterThread::CommonData::init(std::size_t Nx, std::size_t Ny, std::size_t Nt, 
	double t, double threads_num) {
	tau = t;

	std::size_t cycle = Nt;
	std::size_t align = ITERS - Nt % ITERS;

	v_tausq = _mm256_set1_pd(tau * tau);
	v_c1 = _mm256_set1_pd(0.5 * sqr((Nx - 1.0) / (Xb - Xa)));
	v_c2 = _mm256_set1_pd(0.5 * sqr((Ny - 1.0) / (Yb - Ya)));
	v_2cnst = _mm256_set1_pd(2);

	Sx = 1;
	Sy = 1;

	threads_count = threads_num;
	lines_per_thread = Ny / threads_count;
}


void CounterThread::bindThreadToCore() {
	HANDLE process;
	DWORD_PTR processAffinityMask = 0;
	// Windows uses a compact thread topology.  Set mask to every other thread
	for (int i = 0; i < NCORES; i++)
		processAffinityMask |= 1L << (2 * i);

	process = GetCurrentProcess();
	SetProcessAffinityMask(process, processAffinityMask);

	HANDLE thread = GetCurrentThread();
	DWORD_PTR threadAffinityMask = 1L << (2 * thread_number);
	SetThreadAffinityMask(thread, threadAffinityMask);
}

CounterThread::CounterThread(int thread_number_) :
	thread_number(thread_number_), curr_block_start(InputData.lines_per_thread * thread_number), next_block_start(InputData.lines_per_thread * (thread_number + 1)) {
	bindThreadToCore();
}

void CounterThread::computeLine(Matrix &U_new, const Matrix &U, const Matrix &P, 
	std::size_t line, std::size_t Nx, double to_add) {
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
		__m256d U_left_values = _mm256_mul_pd(getLeft(U_prev_vector_value, U_curr_vector_value), mask);
		__m256d mask2 = _mm256_set_pd(not_last, 1.0, 1.0, 1.0);
		__m256d U_right_values = _mm256_mul_pd(getRight(U_curr_vector_value, U_next_vector_value), mask2);
		
		__m256d p1 = MUL(SUB(*U_upper, U_curr_vector_value), ADD(P_left_upper, P_upper));
		__m256d p2 = MUL(SUB(*U_lower, U_curr_vector_value), ADD(P_left, P_curr));
		__m256d s1 = MUL(ADD(p1, p2), InputData.v_c1);

		__m256d p3 = MUL(SUB(U_right_values, U_curr_vector_value), ADD(P_upper, P_curr));
		__m256d p4 = MUL(SUB(U_left_values, U_curr_vector_value), ADD(P_left_upper, P_left));
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

	U_new[line][InputData.Sx] += (InputData.Sy == line) * to_add;
}

void CounterThread::init(std::size_t Nx, std::size_t Ny, std::size_t Nt,
	double t, double threads_count) {
	InputData.init(Nx, Ny, Nt, t, threads_count);
}

void CounterThread::source_upd(std::size_t n) {
	for (int shift = 0; shift < ITERS; ++shift) {
		to_add[shift] = InputData.getSource(n + shift);
	}
}

void CounterThread::step_1(Matrix &U_new, Matrix &U, const Matrix &P, std::size_t Nx) {
	if (thread_number == 0) {
		// (U_new, (const) U, (const) P, line, Nx, additional);
		computeLine(U_new, U, P, 1, Nx, to_add[0]);

		computeLine(U_new, U, P, 2, Nx, to_add[0]);
		computeLine(U, U_new, P, 1, Nx, to_add[1]);

		computeLine(U_new, U, P, 3, Nx, to_add[0]);
		computeLine(U, U_new, P, 2, Nx, to_add[1]);
		computeLine(U_new, U, P, 1, Nx, to_add[2]);
	} else {
		const std::size_t begin = curr_block_start - SHIFT;
		computeLine(U_new, U, P, begin, Nx, to_add[0]);
		computeLine(U_new, U, P, begin + 1, Nx, to_add[0]);
		computeLine(U_new, U, P, begin + 2, Nx, to_add[0]);
		computeLine(U, U_new, P, begin + 1, Nx, to_add[1]);

		computeLine(U_new, U, P, begin + 3, Nx, to_add[0]);
		computeLine(U, U_new, P, begin + 2, Nx, to_add[1]);

		computeLine(U_new, U, P, begin + 4, Nx, to_add[0]);
		computeLine(U, U_new, P, begin + 3, Nx, to_add[1]);
		computeLine(U_new, U, P, begin + 2, Nx, to_add[2]);

		computeLine(U_new, U, P, begin + 5, Nx, to_add[0]);
		computeLine(U, U_new, P, begin + 4, Nx, to_add[1]);
		computeLine(U_new, U, P, begin + 3, Nx, to_add[2]);
	}
}

void CounterThread::step_2(Matrix &U_new, Matrix &U, const Matrix &P, std::size_t Nx) {
	std::size_t start = curr_block_start + SHIFT;
	std::size_t end = next_block_start - SHIFT;

	if (thread_number == 0) 
		++start;
	
	if (thread_number == InputData.threads_count - 1) 
		end = Nx - 1;
	
	for (std::size_t line = start; line < end; ++line) {
		computeLine(U_new, U, P, line, Nx, to_add[0]);
		computeLine(U, U_new, P, line - 1, Nx, to_add[1]);

		computeLine(U_new, U, P, line - 2, Nx, to_add[2]);
		computeLine(U, U_new, P, line - 3, Nx, to_add[3]);
	}
}

// this is after #pragma omp barrier
void CounterThread::step_3(Matrix &U_new, Matrix &U, const Matrix &P, std::size_t Nx) {
	if (thread_number == InputData.threads_count - 1) {
		const std::size_t begin = Nx - 2;

		computeLine(U, U_new, P, begin, Nx, to_add[1]);
		computeLine(U_new, U, P, begin - 1, Nx, to_add[2]);
		computeLine(U, U_new, P, begin - 2, Nx, to_add[3]);

		computeLine(U_new, U, P, begin, Nx, to_add[2]);
		computeLine(U, U_new, P, begin - 1, Nx, to_add[3]);

		computeLine(U, U_new, P, begin, Nx, to_add[3]);
	} else {
		const std::size_t begin = next_block_start - ITERS;

		computeLine(U, U_new, P, begin, Nx, to_add[1]);
		computeLine(U_new, U, P, begin - 1, Nx, to_add[2]);
		computeLine(U, U_new, P, begin - 2, Nx, to_add[3]);

		computeLine(U, U_new, P, begin + 1, Nx, to_add[1]);
		computeLine(U_new, U, P, begin, Nx, to_add[2]);
		computeLine(U, U_new, P, begin - 1, Nx, to_add[3]);

		computeLine(U_new, U, P, begin + 1, Nx, to_add[2]);
		computeLine(U, U_new, P, begin, Nx, to_add[3]);

		computeLine(U_new, U, P, begin + 2, Nx, to_add[2]);
		computeLine(U, U_new, P, begin + 1, Nx, to_add[3]);

		computeLine(U, U_new, P, begin + 2, Nx, to_add[3]);
		computeLine(U, U_new, P, begin + 3, Nx, to_add[3]);
	}
}

