/*
 * Complex numbers for use in CUDA stuff
 * 
 * 2023, Jonathan Tainer
 */

#ifndef COMPLEX_F32
#define COMPLEX_F32

struct complex_f32 {
	float real;
	float imag;

	__host__ __device__ complex_f32();
	__host__ __device__ complex_f32(float r, float i);
	__host__ __device__ complex_f32(float ang);
	__host__ __device__ complex_f32(complex_f32 const& b);
	__host__ __device__ float abs();
	__host__ __device__ float ang();
	__host__ __device__ complex_f32 operator=(complex_f32 const& b);
	__host__ __device__ complex_f32 operator+(complex_f32 const& b);
	__host__ __device__ complex_f32 operator-(complex_f32 const& b);
	__host__ __device__ complex_f32 operator*(complex_f32 const& b);
	__host__ __device__ complex_f32 operator/(complex_f32 const& b);
};

#endif
