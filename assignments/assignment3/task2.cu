
#include <iostream> // Include for console IO
#include <cuda_runtime.h> // Include for CUDA API
#include <vector> // Include for std::vector

// Error checking macro to wrap CUDA calls
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

// CUDA Kernel for element-wise addition (C = A + B)
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    // Calculate the global unique index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure we do not access memory out of bounds
    if (idx < n) {
        // Perform addition of corresponding elements
        c[idx] = a[idx] + b[idx];
    }
}

// Main function
int main() {
    // Define array size approx 16 million elements (large enough to hide overhead)
    const int N = 1<<24; 
    // Calculate memory size in bytes
    const int BYTES = N * sizeof(float);

    // Print assignment info
    std::cout << "Assignment 3 Task 2: Element-wise Addition with Variable Block Sizes" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;

    // Host Memory: Create vectors for input A, input B, and result C
    std::vector<float> h_a(N, 1.0f); // Initialize A with 1.0
    std::vector<float> h_b(N, 2.0f); // Initialize B with 2.0
    std::vector<float> h_c(N);       // Output vector

    // Device Memory pointers
    float *d_a, *d_b, *d_c;
    // Allocate memory on GPU for vector A
    CHECK_CUDA(cudaMalloc(&d_a, BYTES));
    // Allocate memory on GPU for vector B
    CHECK_CUDA(cudaMalloc(&d_b, BYTES));
    // Allocate memory on GPU for result C
    CHECK_CUDA(cudaMalloc(&d_c, BYTES));

    // Copy host data to device data for A
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), BYTES, cudaMemcpyHostToDevice));
    // Copy host data to device data for B
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), BYTES, cudaMemcpyHostToDevice));

    // List of different block sizes to benchmark
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    
    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // Init start event
    cudaEventCreate(&stop);  // Init stop event

    // Print table header
    std::cout << "Block Size\tTime (ms)" << std::endl;

    // Loop through each block size configuration
    for (int block_size : block_sizes) {
        // Calculate grid size (number of blocks) needed for current block size
        // Integer ceiling division: (N + block_size - 1) / block_size
        int grid_size = (N + block_size - 1) / block_size;

        // Start timing
        CHECK_CUDA(cudaEventRecord(start));
        // Launch kernel with current configuration
        add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        // Stop timing
        CHECK_CUDA(cudaEventRecord(stop));
        // Sync to ensure completion
        CHECK_CUDA(cudaEventSynchronize(stop));

        // Variable for elapsed time
        float milliseconds = 0;
        // Compute time difference
        cudaEventElapsedTime(&milliseconds, start, stop);
        // Output result row
        std::cout << block_size << "\t\t" << milliseconds << std::endl;
    }

    // Basic verification of result from the last run
    // Copy result back from device to host
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, BYTES, cudaMemcpyDeviceToHost));
    // Check first and last element logic (1 + 2 should be 3)
    if (h_c[0] != 3.0f || h_c[N-1] != 3.0f) {
        std::cerr << "Verification Failed!" << std::endl;
    } else {
        std::cout << "Verification Success." << std::endl;
    }

    // Cleanup device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return success
    return 0;
}
