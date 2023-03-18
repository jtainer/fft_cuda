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

__device__
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

// Smallest operation that we can easily parallelize
__device__
void fft_combine_single(complex_f32* buf, unsigned int n, unsigned int k) {
	complex_f32 p(buf[k]);
	complex_f32 u(-2*M_PI*k/n);
	complex_f32 q(u * buf[k+n/2]);
	buf[k] = p + q;
	buf[k+n/2] = p - q;
}

// Complete one full iteration of FFT
// Cant synchronize threads between blocks from inside kernel,
// need to handle synchronization on CPU instead
__global__
void fft_iteration(complex_f32* buf, unsigned int n) {

}

int main() {

	return 0;
}
