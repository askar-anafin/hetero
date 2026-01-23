
#include <cuda_runtime.h>   // Include CUDA runtime
#include <iostream>         // Include IO stream
#include <vector>           // Include vector
#include <cmath>            // Include math functions
#include <chrono>           // Include chrono for timing
#include <algorithm>        // Include check for std::fill

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// A computationally expensive function to simulate workload
// Can run on both host (CPU) and device (GPU)
__device__ __host__ float heavy_op(float x) {
    // Calculates sin(x)*cos(x) + sqrt(|x|)
    return sinf(x) * cosf(x) + sqrtf(fabsf(x));
}

// CUDA kernel to process data in parallel
// data: pointer to global memory
// offset: helpful if we need absolute index (not used directly here in idx calc)
// n: number of elements to process
__global__ void process_kernel(float* data, int offset, int n) {
    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check bounds
    if (idx < n) {
        // Perform heavy operation and write back
        data[idx] = heavy_op(data[idx]);
    }
}

// CPU implementation of the same processing
void process_cpu(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = heavy_op(data[i]);
    }
}

// Main function
int main() {
    const int N = 10000000; // 10 Million elements to ensure measurable time
    const int BYTES = N * sizeof(float);
    const int BLOCK_SIZE = 256;

    // Allocate host memory
    std::vector<float> h_data(N);
    // Initialize with a value (1.5)
    std::fill(h_data.begin(), h_data.end(), 1.5f);

    // Create a reference copy for verification
    std::vector<float> h_ref = h_data;

    // --- 1. Pure CPU Benchmarking ---
    std::cout << "Starting Pure CPU..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // Process entire array on CPU
    process_cpu(h_ref.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end - start;
    std::cout << "Pure CPU Time: " << cpu_time.count() << " ms" << std::endl;

    // --- 2. Pure GPU Benchmarking ---
    std::cout << "Starting Pure GPU..." << std::endl;
    float *d_data;
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_data, BYTES));
    float *d_data_verify; // Auxiliary pointer if needed (not used here)
    CHECK_CUDA(cudaMalloc(&d_data_verify, BYTES));

    // Reset input data for GPU run
    std::vector<float> h_gpu_in = h_data;
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_gpu_in.data(), BYTES, cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Launch kernel for entire array
    process_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, 0, N);
    // Synchronize to ensure completion
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end - start;
    std::cout << "Pure GPU Time: " << gpu_time.count() << " ms" << std::endl;

    // --- 3. Hybrid (CPU + GPU) Benchmarking ---
    std::cout << "Starting Hybrid..." << std::endl;
    
    // Split strategy: 50% CPU, 50% GPU
    int split_idx = N / 2;
    int cpu_count = split_idx;          // CPU processes first half [0, split_idx)
    int gpu_count = N - split_idx;      // GPU processes second half [split_idx, N)
    
    // Reset data again for hybrid run
    std::vector<float> h_hybrid = h_data;
    
    start = std::chrono::high_resolution_clock::now();
    
    // Create a CUDA stream for asynchronous GPU operations
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // 1. Asynchronously copy the GPU portion of data to device
    // Pointers are offset by split_idx
    CHECK_CUDA(cudaMemcpyAsync(d_data + split_idx, h_hybrid.data() + split_idx, gpu_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // 2. Launch GPU kernel in the stream
    // It will start after the copy completes
    int hybrid_grid = (gpu_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    process_kernel<<<hybrid_grid, BLOCK_SIZE, 0, stream>>>(d_data + split_idx, 0, gpu_count);
    
    // 3. Concurrently, process the CPU portion on the host
    // This runs effectively in parallel with the GPU stream commands
    process_cpu(h_hybrid.data(), cpu_count);
    
    // 4. Asynchronously copy GPU results back to host
    CHECK_CUDA(cudaMemcpyAsync(h_hybrid.data() + split_idx, d_data + split_idx, gpu_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    
    // 5. Wait for the stream (GPU side) to finish
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hybrid_time = end - start;
    std::cout << "Hybrid Time: " << hybrid_time.count() << " ms" << std::endl;

    // --- Verification ---
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        // Compare with CPU reference with a small tolerance
        if (fabs(h_hybrid[i] - h_ref[i]) > 1e-4) {
            std::cout << "Mismatch at " << i << " Ref: " << h_ref[i] << " Hybrid: " << h_hybrid[i] << std::endl;
            passed = false;
            break;
        }
    }
    
    if (passed) std::cout << "Result: PASSED" << std::endl;
    else std::cout << "Result: FAILED" << std::endl;

    // Free resources
    cudaFree(d_data);
    cudaFree(d_data_verify);
    cudaStreamDestroy(stream);
    
    return 0;
}
