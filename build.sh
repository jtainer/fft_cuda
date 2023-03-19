nvcc --relocatable-device-code=true -L/usr/local/cuda/lib64 kernel.cu complex_f32.cu -lcudart -O3
