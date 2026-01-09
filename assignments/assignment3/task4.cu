
#include <iostream> // For standard output
#include <cuda_runtime.h> // For CUDA runtime functions
#include <vector> // For std::vector container
#include <algorithm> // For std::min_element and std::max_element

// Macro for handling CUDA errors
#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// Optimized Grid-Stride Loop Kernel
// This structure makes the kernel flexible to any grid size, allowing independent optimization of grid configuration
__global__ void optimized_add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate total number of threads in the grid (stride)
    int stride = blockDim.x * gridDim.x;
    
    // Grid-Stride Loop:
    // Instead of assuming one thread per element, we loop.
    // Thread processes element 'idx', then jumps by 'stride' to next element.
    // This allows processing N elements with any number of threads.
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i]; // Perform addition
    }
}

// Main application
int main() {
    // Defines N = 16 million elements
    const int N = 1<<24; 
    // Total memory size in bytes
    const int BYTES = N * sizeof(float);

    // Print Header
    std::cout << "Assignment 3 Task 4: Optimizaton of Grid and Block Config" << std::endl;
    std::cout << "Benchmarking Vector Add for Array Size: " << N << std::endl;

    // Device pointers
    float *d_a, *d_b, *d_c;
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_a, BYTES));
    CHECK_CUDA(cudaMalloc(&d_b, BYTES));
    CHECK_CUDA(cudaMalloc(&d_c, BYTES));

    // Initialize device memory with values directly (faster than copy for this test)
    // Set 'A' elements to something valid (not exactly 1.0 float via memset but non-zero)
    // Note: memset sets bytes. For float benchmarking this is fine as we just need data movement.
    CHECK_CUDA(cudaMemset(d_a, 1, BYTES));
    CHECK_CUDA(cudaMemset(d_b, 1, BYTES));

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Struct to hold benchmark results
    struct Result {
        int blockSize;
        float time;
    };
    std::vector<Result> results; // Vector to store results

    // Generate a list of block sizes to test: 32, 64, ..., 1024
    std::vector<int> test_blocks;
    for(int b=32; b<=1024; b+=32) test_blocks.push_back(b);

    std::cout << "Testing " << test_blocks.size() << " configurations..." << std::endl;
    
    // Warmup run to ensure driver is initialized and GPU is active
    optimized_add_kernel<<<256, 256>>>(d_a, d_b, d_c, N);
    
    // Benchmark Loop
    for (int blockSize : test_blocks) {
        // Calculate Grid Size
        // Heuristic: We map 1-to-1 loosely, but the kernel handles bounds.
        // We use (N + blockSize - 1) / blockSize to ensure we have enough threads 
        // to cover the array at least once without looping too much (though loop handles it).
        int gridSize = (N + blockSize - 1) / blockSize;
        
        // Start Timer
        CHECK_CUDA(cudaEventRecord(start));
        // Launch kernel with current specific configuration
        optimized_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        // Stop Timer
        CHECK_CUDA(cudaEventRecord(stop));
        // Synchronize
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        // Record time
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        // Save result
        results.push_back({blockSize, ms});
    }

    // Find the configuration with the minimum time (Best)
    auto best = std::min_element(results.begin(), results.end(), [](const Result& a, const Result& b){
        return a.time < b.time;
    });
    
    // Find the configuration with the maximum time (Worst)
    auto worst = std::max_element(results.begin(), results.end(), [](const Result& a, const Result& b){
        return a.time < b.time;
    });

    // Print Results Summary
    std::cout << "Optimization Results:" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Optimal Block Size: " << best->blockSize << " threads" << std::endl;
    std::cout << "Best Time: " << best->time << " ms" << std::endl;
    std::cout << "Worst Block Size: " << worst->blockSize << " threads" << std::endl;
    std::cout << "Worst Time: " << worst->time << " ms" << std::endl;
    std::cout << "Speedup (Best/Worst): " << worst->time / best->time << "x" << std::endl;
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Return
    return 0;
}
