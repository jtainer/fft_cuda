fft_cuda is a work in progress. This is my second attempt at implementing FFT using CUDA.

I am writing this library because I was using cuFFT for image upscaling and it introduced artifacts that didnt appear when I used my own FFT implementation (running on CPU).

Previously I wrote another FT (not FFT) library using CUDA but it is very bad, so I am starting from scratch.
