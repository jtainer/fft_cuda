/*
 * CUDA kernel implementation of Cooley-Tukey FFT algorithm
 * 
 * 2023, Jonathan Tainer
 */

#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "complex_f32.h"

// Kernel to calculate values for LUT
__global__
void fft_calculate_lut(complex_f32* lut, unsigned int N) {
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < N/2) {
		lut[tid] = complex_f32(-2*M_PI*tid/N);
	}
}

__host__
complex_f32* fft_gen_device_lut(unsigned int N) {
	complex_f32* lut;
	cudaMalloc((void**)&lut, sizeof(complex_f32) * N/2);
	fft_calculate_lut<<<(N/32) + 1, 32>>>(lut, N);
	return lut;
}


__host__
unsigned int count_bits(unsigned int n) {
	unsigned int bits = 0;
	while ((n >> bits) > 1) bits++;
	return bits;
}

__device__
unsigned int reverse_bits(unsigned int orig, unsigned int bits) {
	unsigned int flip = 0;
	for (unsigned int i = 0; i < bits; i++) {
		flip <<= 1;
		flip |= orig & 1;
		orig >>= 1;
	}
	return flip;
}

// CUDA kernel to rerrange buffer using bit reversal before calculating FFT
__global__
void fft_reorder(complex_f32* buf, unsigned int N, unsigned int bits) {
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < N) {
		unsigned int i0 = tid;
		unsigned int i1 = reverse_bits(i0, bits);
		if (i0 <= i1) {
			complex_f32 tmp = buf[i0];
			buf[i0] = buf[i1];
			buf[i1] = tmp;
		}
	}
}

// Smallest operation that we can easily parallelize
__device__
void fft_combine_single(complex_f32* buf, complex_f32* lut, unsigned int n, unsigned int k, unsigned int s) {
	complex_f32 p(buf[k]);
//	complex_f32 u(-2*M_PI*k/n);
	complex_f32 u = lut[k*s];
	complex_f32 q(u * buf[k+n/2]);
	buf[k] = p + q;
	buf[k+n/2] = p - q;
}

// Complete one full iteration of FFT
// Cant synchronize threads between blocks from inside kernel,
// need to handle synchronization on CPU instead
// n = size of sub-FFT
// N = overall size of FFT
__global__
void fft_iteration(complex_f32* buf, complex_f32* lut, unsigned int n, unsigned int N) {
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < N/2) {
		unsigned int w = tid / (n/2);
		unsigned int k = tid % (n/2);
		unsigned int s = N/n;
		fft_combine_single(buf + w * n, lut, n, k, s);
	}
}


// buf must point to GPU memory
// N = size of FFT
__host__
void fft(complex_f32* buf, complex_f32* lut, unsigned int N) {
	unsigned int bits = count_bits(N);
	fft_reorder<<<(N/32) + 1, 32>>>(buf, N, bits);

	for (unsigned int n = 2; n <= N; n *= 2) {
		fft_iteration<<<(N/32) + 1, 32>>>(buf, lut, n, N);
	}
}

int main() {
	const unsigned int n = 8;
	complex_f32 buf[n] = { {1,0}, {2,0}, {5,0}, {2,0}, {7,0}, {4,0}, {2,0}, {0,0} };
	complex_f32* dbuf;
	cudaMalloc((void**)&dbuf, sizeof(complex_f32) * n);

	cudaMemcpy(dbuf, buf, sizeof(complex_f32) * n, cudaMemcpyHostToDevice);

	complex_f32* lut = fft_gen_device_lut(n);

	fft(dbuf, lut, n);
	
	cudaMemcpy(buf, dbuf, sizeof(complex_f32) * n, cudaMemcpyDeviceToHost);
	cudaFree(dbuf);
	cudaFree(lut);

	for (unsigned int i = 0; i < n; i++) {
		printf("%f\t%f\n", buf[i].real, buf[i].imag);
	}

	return 0;
}
