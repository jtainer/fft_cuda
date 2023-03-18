/*
  * Complex numbers for use in CUDA
  *
  * 2023, Jonathan Tainer
  */

#include <cuda.h>
#include <math.h>
#include "complex_f32.h"

__host__ __device__ complex_f32::complex_f32() {
	this->real = 0;
	this->imag = 0;
}

__host__ __device__ complex_f32::complex_f32(float r, float i) {
	this->real = r;
	this->imag = i;
}

__host__ __device__ complex_f32::complex_f32(float ang) {
	this->real = cosf(ang);
	this->imag = sinf(ang);
}

__host__ __device__ complex_f32::complex_f32(complex_f32 const& b) {
	this->real = b.real;
	this->imag = b.imag;
}

__host__ __device__ float complex_f32::abs() {
	return sqrtf(real*real + imag*imag);
}

__host__ __device__ float complex_f32::ang() {
	return atan2f(real, imag);
}

__host__ __device__ complex_f32 complex_f32::operator=(complex_f32 const& b) {
	this->real = b.real;
	this->imag = b.imag;
	return *this;
}

__host__ __device__ complex_f32 complex_f32::operator+(complex_f32 const& b) {
	return complex_f32(this->real+b.real, this->imag+b.imag);
}

__host__ __device__ complex_f32 complex_f32::operator-(complex_f32 const& b) {
	return complex_f32(this->real-b.real, this->imag-b.imag);
}

__host__ __device__ complex_f32 complex_f32::operator*(complex_f32 const& b) {
	return complex_f32(this->real*b.real-this->imag*b.imag, this->real*b.imag+this->imag*b.real);
}

__host__ __device__ complex_f32 complex_f32::operator/(complex_f32 const& b) {
	complex_f32 tmp(this->real*b.real+this->imag*b.imag, this->imag*b.real+this->real*b.imag);
	float div = 1.f/(b.real*b.real+b.imag*b.imag);
	tmp.real *= div;
	tmp.imag *= div;
	return tmp;
}

