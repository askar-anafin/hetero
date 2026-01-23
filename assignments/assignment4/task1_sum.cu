
#include <cuda_runtime.h>   // Include CUDA runtime API
#include <iostream>         // Include IO stream for output
#include <vector>           // Include vector container
#include <numeric>          // Include numeric for calculations
#include <chrono>           // Include chrono for CPU timing
#include <random>           // Include random for number generation

// Macro for error checking CUDA calls
// Wraps the call in a do-while loop to catch errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel for global memory reduction (summing array elements)
// __global__ indicates this function runs on the device (GPU) and is called from host
// d_in: pointer to input array on device
// d_out: pointer to output result on device
// n: total number of elements
__global__ void reduce_global_kernel(const int* d_in, int* d_out, int n) {
    // Calculate global thread index
    // blockIdx.x: index of the current block
    // blockDim.x: number of threads per block
    // threadIdx.x: index of the thread within the block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the grid stride (total number of threads in the grid)
    // gridDim.x: number of blocks in the grid
    int stride = blockDim.x * gridDim.x;

    // Initialize local sum for this thread
    int local_sum = 0;
    
    // Loop over the array with stride
    // This allows the kernel to process arrays larger than the total thread count
    for (int i = idx; i < n; i += stride) {
        // Add current element to local sum
        local_sum += d_in[i];
    }
    
    // Atomically add the local sum to the global accumulator
    // atomicAdd ensures no race conditions when multiple threads write to the same address
    atomicAdd(d_out, local_sum);
}

// Function to fill array with random integers
void random_ints(int* a, int n) {
    // Initialize random number generator with seed 1234
    std::mt19937 gen(1234);
    // Define distribution range [0, 10]
    std::uniform_int_distribution<> dist(0, 10);
    // Loop to fill array
    for (int i = 0; i < n; ++i) {
        a[i] = dist(gen);
    }
}

// Function to compute sum on CPU for verification
long long cpu_sum(const int* a, int n) {
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

// Main function
int main() {
    // Define problem size (100,000 elements)
    const int N = 100000;
    // Calculate total bytes
    const int BYTES = N * sizeof(int);
    // Define threads per block
    const int BLOCK_SIZE = 256;
    // Calculate number of blocks needed to cover N (rounding up)
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate host memory (RAM) for input array
    std::vector<int> h_in(N);
    // Variable for host output
    int h_out = 0;

    // Initialize the input array with random data
    random_ints(h_in.data(), N);

    // --- CPU Computation ---
    // Record start time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // Compute sum on CPU
    long long expected_sum = cpu_sum(h_in.data(), N);
    // Record end time
    auto end_cpu = std::chrono::high_resolution_clock::now();
    // Calculate duration in milliseconds
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // Print CPU results
    std::cout << "CPU Sum: " << expected_sum << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- GPU Computation ---
    // Pointers for device memory
    int *d_in, *d_out;
    // Allocate global memory on device for input
    CHECK_CUDA(cudaMalloc(&d_in, BYTES));
    // Allocate global memory on device for output (single integer)
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice));
    // Initialize output value on device to 0
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Record start event
    cudaEventRecord(start_gpu);
    
    // Launch the kernel
    // GRID_SIZE: number of blocks
    // BLOCK_SIZE: threads per block
    reduce_global_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);
    
    // Record stop event
    cudaEventRecord(stop_gpu);

    // Wait for the stop event to complete (synchronize)
    CHECK_CUDA(cudaEventSynchronize(stop_gpu));
    
    // Calculate elapsed time between events
    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);

    // Copy the result from device back to host
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Print GPU results
    std::cout << "GPU Sum: " << h_out << std::endl;
    std::cout << "GPU Time: " << gpu_duration << " ms" << std::endl;

    // Verify result against CPU reference
    if (h_out == expected_sum) {
        std::cout << "Result: PASSED" << std::endl;
    } else {
        std::cout << "Result: FAILED" << std::endl;
    }

    // Free allocated device memory
    cudaFree(d_in);
    cudaFree(d_out);
    // Destroy timing events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
