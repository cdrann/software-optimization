#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <tuple>

class Matrix {
	std::size_t width;
	std::size_t height;

	double *matrix = nullptr;

public:
	Matrix(std::size_t w, std::size_t h) : width(w), height(h) {
		std::size_t size = w * h * sizeof(double);

		matrix = (double *)_mm_malloc(size, sizeof(__m256d));
		if (matrix == nullptr)
			throw std::bad_alloc();
		memset(matrix, 0, size);
	}

	Matrix(Matrix &&other) noexcept
		: width(other.width), height(other.height), matrix(other.matrix) {
		other.width = 0;
		other.height = 0;
		other.matrix = nullptr;
	}

	Matrix &operator=(Matrix &&other) noexcept {
		if (matrix)
			delete[] matrix;

		width = other.width;
		height = other.height;
		matrix = other.matrix;

		other.width = 0;
		other.height = 0;
		other.matrix = nullptr;

		return *this;
	}

	int getWidth() const { return width; }
	int getHeight() const { return height; }

	inline const double *operator[](std::size_t idx) const {
		return matrix + idx * width;
	}
	inline double *operator[](std::size_t idx) { return matrix + idx * width; }

	double *getRaw() { return matrix; }
	const double *getRaw() const { return matrix; }

	~Matrix() { _mm_free(matrix); }
};

