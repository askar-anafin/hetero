
#include <cuda_runtime.h> // Include CUDA runtime API
#include <iostream>       // Include IO stream
#include <vector>         // Include vector container
#include <chrono>         // Include chrono for timing
#include <random>         // Include random for data generation

// Macro for checking CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Define block size for kernels
#define BLOCK_SIZE 1024

// CUDA kernel for Hillis-Steele scan (inclusive) within a block
// Also writes the total sum of the block to an auxiliary array
// g_odata: Output array
// g_idata: Input array
// g_block_sums: Array to store sum of each block
// n: Number of elements
__global__ void prescan_kernel(int *g_odata, int *g_idata, int *g_block_sums, int n) {
    // Shared memory buffer to hold block data for fast access
    __shared__ int temp[BLOCK_SIZE];
    // Thread index within the block
    int thid = threadIdx.x;
    // Global index in the array
    int idx = blockIdx.x * blockDim.x + thid;

    // Load input into shared memory
    if (idx < n) {
        temp[thid] = g_idata[idx];
    } else {
        // Pad with 0 if outside array bounds
        temp[thid] = 0;
    }
    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Hillis-Steele Scan Algorithm
    // Logarithmic steps: 1, 2, 4, 8...
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        // If thread has a neighbor at 'offset' distance to the left
        if (thid >= offset) {
            val = temp[thid - offset];
        }
        // Barriers needed before reading and after writing to prevent race conditions
        __syncthreads();
        if (thid >= offset) {
            temp[thid] += val; // Accumulate value
        }
        __syncthreads();
    }

    // Write the scanned result from shared memory to global memory
    if (idx < n) {
        g_odata[idx] = temp[thid];
    }

    // Write the total sum of this block to the auxiliary array
    // The last element of the scanned block contains the sum of all elements in the block
    if (thid == blockDim.x - 1) {
        // Only if pointer is provided (not null)
        if (g_block_sums != nullptr) {
            g_block_sums[blockIdx.x] = temp[thid];
        }
    }
}

// CUDA kernel to add the scanned block sums to the prefix sums of the original blocks
// This propagates the sum from previous blocks
__global__ void add_block_sums_kernel(int *g_odata, int *g_block_sums, int n) {
    // Global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_sum_val = 0;
    
    // If this is not the first block, read the scanned sum of the previous block
    if (blockIdx.x > 0) {
        block_sum_val = g_block_sums[blockIdx.x - 1];
    }
    
    // Add that sum to the current element
    if (idx < n) {
        g_odata[idx] += block_sum_val;
    }
}

// Function to generate random integers
void random_ints(int* a, int n) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist(0, 5); // Use small numbers to prevent overflow
    for (int i = 0; i < n; ++i) {
        a[i] = dist(gen);
    }
}

// Sequential CPU Scan implementation for verification
// Computes inclusive prefix sum
void cpu_scan(const int* in, int* out, int n) {
    if (n == 0) return;
    out[0] = in[0];
    for (int i = 1; i < n; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

// Main function
int main() {
    const int N = 1000000; // 1 Million elements
    const int BYTES = N * sizeof(int);
    
    // Allocate host vectors
    std::vector<int> h_in(N);
    std::vector<int> h_out(N);
    std::vector<int> h_cpu_out(N);

    // Fill input with random data
    random_ints(h_in.data(), N);

    // --- CPU Scan ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_scan(h_in.data(), h_cpu_out.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // Print CPU time
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- GPU Scan ---
    // Device pointers
    int *d_in, *d_out, *d_block_sums, *d_block_sums_scanned;
    
    // Allocate memory for input and output
    CHECK_CUDA(cudaMalloc(&d_in, BYTES));
    CHECK_CUDA(cudaMalloc(&d_out, BYTES));

    // Calculate number of blocks required
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate memory for block sums and their scanned version
    CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice));

    // Events for GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Start recording time
    cudaEventRecord(start_gpu);

    // Step 1: Scan each block independently and store partial block sums
    // Output goes to d_out (partial results within blocks)
    // Block sums go to d_block_sums
    prescan_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_in, d_block_sums, N);
    
    // Step 2: Scan the array of block sums
    // Since num_blocks (for 1M elements ~ 1000) fits in a single block (1024), 
    // we can do this with one kernel launch.
    // In a general case, this step would be recursive.
    prescan_kernel<<<1, BLOCK_SIZE>>>(d_block_sums_scanned, d_block_sums, nullptr, num_blocks);
    
    // Step 3: Add the scanned block sums to the partial scan results
    // This adjusts the values in d_out to their correct global prefix sum
    add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, N);

    // Stop recording time
    cudaEventRecord(stop_gpu);
    CHECK_CUDA(cudaEventSynchronize(stop_gpu));
    
    // Calculate elapsed GPU time
    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost));

    // Print GPU time
    std::cout << "GPU Time: " << gpu_duration << " ms" << std::endl;

    // --- Verification ---
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_cpu_out[i]) {
            std::cout << "Mismatch at " << i << " CPU: " << h_cpu_out[i] << " GPU: " << h_out[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Result: PASSED" << std::endl;
    } else {
        std::cout << "Result: FAILED" << std::endl;
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scanned);
    // Destroy events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
