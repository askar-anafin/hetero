
#include <iostream> // Include the standard I/O library usually for std::cout and std::cerr
#include <cuda_runtime.h> // Include CUDA runtime API header for CUDA functions
#include <vector> // Include vector standard library for dynamic array management on host
#include <numeric> // Include numeric library (unused but consistent with previous version)

// Macro to check CUDA errors easily.
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

// Global Memory Kernel function definition
// uses __global__ to indicate it runs on device and is callable from host
__global__ void global_memory_kernel(const float* __restrict__ input, float* __restrict__ output, float scalar, int n) {
    // Calculate global thread index unique to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the thread index is within bounds of the array
    if (idx < n) {
        // Perform multiplication directly reading/writing to global memory
        output[idx] = input[idx] * scalar; // Read input[idx], multiply, write to output[idx]
    }
}

// Shared Memory Kernel function definition
// Demonstrates loading data into shared memory before processing
__global__ void shared_memory_kernel(const float* __restrict__ input, float* __restrict__ output, float scalar, int n) {
    // Declare dynamic shared memory array unique to each block
    extern __shared__ float shared_data[];

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Get thread index within the block (for shared memory indexing)
    int tid = threadIdx.x;

    // Load data from global memory into shared memory
    if (idx < n) {
        // Each thread loads one element from global 'input' to shared 'shared_data'
        shared_data[tid] = input[idx];
    }

    // Synchronize threads in the block to ensure all data is loaded before proceeding
    __syncthreads();

    // Process data using shared memory and write back to global memory
    if (idx < n) {
        // Read value from fast shared memory
        float val = shared_data[tid];
        // Perform multiplication operation
        val *= scalar;
        // Write the result back to global memory
        output[idx] = val;
    }
}

// Main function entry point
int main() {
    // Define the size of the array (1 million elements)
    const int N = 1000000;
    // Calculate the total size in bytes for memory allocation
    const int BYTES = N * sizeof(float);
    // Define the scalar value to multiply by
    const float SCALAR = 2.0f;
    // Define the number of threads per block
    const int BLOCK_SIZE = 256;
    // Calculate the number of blocks needed to cover all N elements
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Print header info to console
    std::cout << "Assignment 3 Task 1: Global vs Shared Memory" << std::endl;
    std::cout << "Array Size: " << N << std::endl;
    std::cout << "Block Size: " << BLOCK_SIZE << std::endl;

    // Allocate host memory (RAM) for input array
    std::vector<float> h_input(N);
    // Allocate host memory (RAM) for output array
    std::vector<float> h_output(N);

    // Initialize the input array with 1.0f values
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f; // Assign 1.0 to each element
    }

    // Declare pointers for device (GPU) memory
    float *d_input, *d_output;
    // Allocate global device memory for input
    CHECK_CUDA(cudaMalloc(&d_input, BYTES));
    // Allocate global device memory for output
    CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // Copy data from host (CPU) to device (GPU)
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), BYTES, cudaMemcpyHostToDevice));

    // Create CUDA events for timing measurements
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // Create start event
    cudaEventCreate(&stop);  // Create stop event

    // 1. Global Memory Benchmark
    // Record start event for timing
    CHECK_CUDA(cudaEventRecord(start));
    // Launch kernel using global memory
    global_memory_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, SCALAR, N);
    // Record stop event after kernel completes
    CHECK_CUDA(cudaEventRecord(stop));
    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Variable to hold elapsed time
    float milliseconds_global = 0;
    // Calculate elapsed time between start and stop
    cudaEventElapsedTime(&milliseconds_global, start, stop);
    // Print reference timing for global memory
    std::cout << "Global Memory Kernel Time: " << milliseconds_global << " ms" << std::endl;

    // 2. Shared Memory Benchmark
    // Reset output memory to ensure cleanliness (optional but good practice)
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));

    // Record start event for shared memory kernel
    CHECK_CUDA(cudaEventRecord(start));
    // Launch kernel using shared memory
    // Third argument in <<<...>>> specifies shared memory size in bytes per block
    shared_memory_kernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, SCALAR, N);
    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop));
    // Wait for completion
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Variable for shared memory time
    float milliseconds_shared = 0;
    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds_shared, start, stop);
    // Print reference timing for shared memory
    std::cout << "Shared Memory Kernel Time: " << milliseconds_shared << " ms" << std::endl;

    // Output the difference
    std::cout << "Difference using shared memory: " << (milliseconds_shared - milliseconds_global) << " ms" << std::endl;

    // Cleanup device memory
    cudaFree(d_input);
    cudaFree(d_output);
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return 0 indicating success
    return 0;
}
