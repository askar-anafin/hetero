
#include <iostream> // Standard IO
#include <cuda_runtime.h> // CUDA Runtime
#include <vector> // STL vector

// Macro for error checking
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

// Kernel for Coalesced Memory Access
// This represents the ideal access pattern
__global__ void coalesced_copy_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    // Calculate thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check bounds
    if (idx < n) {
        // Consecutive threads access consecutive memory addresses
        // This allows the GPU memory controller to combine requests into single transactions
        output[idx] = input[idx];
    }
}

// Kernel for Uncoalesced Memory Access
// This purposefully accesses memory in a scattered pattern to degrade performance
__global__ void uncoalesced_copy_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    // Calculate standard linear index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx < n) {
        // Define a stride width to simulate jumping around memory
        // 32 matches the warp size
        int width = 32;
        // Calculate height of the conceptual 2D matrix
        int height = (n + width - 1) / width;
        
        // Calculate row and column if we view linear memory as a 2D matrix of width 32
        int row = idx / width;
        int col = idx % width;
        
        // Calculate an uncoalesced index by effectively transposing the access
        // Consecutive threads (increasing 'col') will access memory locations separated by 'height'
        // This breaks memory coalescing because threads in a warp are accessing widely separated addresses
        long long uncoalesced_idx = (long long)col * height + row;
        
        // Ensure the computed index is within bounds
        if (uncoalesced_idx < n) {
             // Read from scattered location, write to linear location
             output[idx] = input[uncoalesced_idx];
        }
    }
}

// Main Function
int main() {
    // Define problem size, 2^24 approx 16 Million float elements
    const int N = 1<<24; 
    // Size in bytes
    const int BYTES = N * sizeof(float);
    // Standard block size
    const int BLOCK_SIZE = 256;
    // Grid size to cover all elements
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Header output
    std::cout << "Assignment 3 Task 3: Coalesced vs Uncoalesced Memory Access" << std::endl;
    std::cout << "Array Size: " << N << std::endl;

    // Allocate Host Memory
    std::vector<float> h_input(N, 1.0f); // Input array
    std::vector<float> h_output(N);      // Output array (unused but needed for copy)

    // Allocate Device Memory pointers
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, BYTES)); // Input buffer
    CHECK_CUDA(cudaMalloc(&d_output, BYTES)); // Output buffer

    // Initialize device input with host data
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), BYTES, cudaMemcpyHostToDevice));

    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run to initialize GPU state
    coalesced_copy_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for warmup to finish

    // 1. Measure Coalesced Access
    CHECK_CUDA(cudaEventRecord(start)); // Start timer
    coalesced_copy_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N); // Launch kernel
    CHECK_CUDA(cudaEventRecord(stop)); // Stop timer
    CHECK_CUDA(cudaEventSynchronize(stop)); // Sync
    float ms_coalesced = 0;
    cudaEventElapsedTime(&ms_coalesced, start, stop);
    std::cout << "Coalesced Access Time: " << ms_coalesced << " ms" << std::endl;

    // 2. Measure Uncoalesced Access
    CHECK_CUDA(cudaEventRecord(start)); // Start timer
    uncoalesced_copy_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N); // Launch kernel
    CHECK_CUDA(cudaEventRecord(stop)); // Stop timer
    CHECK_CUDA(cudaEventSynchronize(stop)); // Sync
    float ms_uncoalesced = 0;
    cudaEventElapsedTime(&ms_uncoalesced, start, stop);
    std::cout << "Uncoalesced Access Time: " << ms_uncoalesced << " ms" << std::endl;

    // Calculate and print the slowdown factor
    std::cout << "Slowdown factor: " << ms_uncoalesced / ms_coalesced << "x" << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // End program
    return 0;
}
