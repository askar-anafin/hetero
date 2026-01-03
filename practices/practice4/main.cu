// Include standard input/output stream library for console operations
#include <iostream>
// Include file stream library for file I/O operations (CSV writing)
#include <fstream>
// Include CUDA runtime API for GPU operations
#include <cuda_runtime.h>
// Include CUDA random number generation library
#include <curand.h>
// Include C math library for mathematical functions
#include <cmath>
// Include I/O manipulators for formatting output
#include <iomanip>

// Forward declaration of kernel function for reduction using global memory
__global__ void reductionGlobal(float* input, float* output, int n);
// Forward declaration of kernel function for reduction using shared memory
__global__ void reductionShared(float* input, float* output, int n);
// Forward declaration of kernel function for bubble sort on chunks
__global__ void bubbleSortKernel(float* data, int n, int chunkSize);
// Forward declaration of kernel function for merge sort
__global__ void mergeSortKernel(float* input, float* output, int n, int width);

// Macro for CUDA error checking - wraps CUDA calls and checks for errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; /* Execute the CUDA call and get error code */ \
        if (err != cudaSuccess) { /* Check if error occurred */ \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; /* Print error details */ \
            exit(EXIT_FAILURE); /* Exit the program with failure code */ \
        } \
    } while (0) /* Do-while(0) allows safe macro usage with semicolon */

// Function to generate random data directly on GPU using cuRAND
void generateRandomData(float* d_data, int n) {
    curandGenerator_t gen; // Declare cuRAND generator object
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); // Create pseudo-random number generator
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Set seed for reproducible results
    curandGenerateUniform(gen, d_data, n); // Generate n uniform random floats in [0, 1)
    curandDestroyGenerator(gen); // Clean up and destroy the generator
}

// CPU reference function to compute sum for verification
float cpuSum(float* data, int n) {
    double sum = 0.0; // Use double for better precision with large sums
    for (int i = 0; i < n; i++) { // Iterate through all elements
        sum += data[i]; // Add each element to the sum
    }
    return (float)sum; // Cast back to float and return
}

// CUDA kernel: Reduction using only global memory (slower approach)
__global__ void reductionGlobal(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread ID
    int stride = blockDim.x * gridDim.x; // Calculate total number of threads (grid-stride)
    
    float sum = 0.0f; // Initialize thread-local sum
    for (int i = tid; i < n; i += stride) { // Grid-stride loop: each thread processes multiple elements
        sum += input[i]; // Accumulate values handled by this thread
    }
    
    // Write to global memory (each thread writes its partial sum)
    if (tid < n) { // Bounds check
        output[tid] = sum; // Store partial sum in global memory
    }
}

// CUDA kernel: Reduction using shared memory (faster approach)
__global__ void reductionShared(float* input, float* output, int n) {
    extern __shared__ float sdata[]; // Declare dynamically allocated shared memory
    
    int tid = threadIdx.x; // Thread index within block
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f; // Load element or 0 if out of bounds
    __syncthreads(); // Wait for all threads in block to finish loading
    
    // Reduction in shared memory using parallel reduction algorithm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { // Halve the active threads each iteration
        if (tid < s) { // Only lower half of threads are active
            sdata[tid] += sdata[tid + s]; // Add corresponding element to current element
        }
        __syncthreads(); // Synchronize before next iteration
    }
    
    // Write result for this block to global memory
    if (tid == 0) { // Only thread 0 in each block writes the result
        output[blockIdx.x] = sdata[0]; // Write block's sum to global memory
    }
}

// CUDA kernel: Bubble sort for small chunks using shared memory
__global__ void bubbleSortKernel(float* data, int n, int chunkSize) {
    extern __shared__ float shared[]; // Declare dynamically allocated shared memory for sorting
    
    int blockStart = blockIdx.x * chunkSize; // Calculate starting index for this block's chunk
    int tid = threadIdx.x; // Thread index within block
    int globalIdx = blockStart + tid; // Global index in the data array
    
    // Load chunk into shared memory
    if (globalIdx < n && tid < chunkSize) { // Check bounds
        shared[tid] = data[globalIdx]; // Load element from global to shared memory
    } else if (tid < chunkSize) { // If out of bounds but within chunk
        shared[tid] = FLT_MAX; // Fill with maximum float value (sentinel)
    }
    __syncthreads(); // Wait for entire chunk to be loaded
    
    // Bubble sort in shared memory
    int actualSize = min(chunkSize, n - blockStart); // Calculate actual valid size of chunk
    for (int i = 0; i < actualSize - 1; i++) { // Outer loop: number of passes
        for (int j = tid; j < actualSize - i - 1; j += blockDim.x) { // Parallel comparison phase
            if (shared[j] > shared[j + 1]) { // If elements are out of order
                float temp = shared[j]; // Swap elements
                shared[j] = shared[j + 1]; // Move smaller element left
                shared[j + 1] = temp; // Move larger element right
            }
        }
        __syncthreads(); // Synchronize after each pass
    }
    
    // Write sorted chunk back to global memory
    if (globalIdx < n && tid < chunkSize) { // Check bounds
        data[globalIdx] = shared[tid]; // Copy sorted data back to global memory
    }
}

// CUDA kernel: Merge two sorted subarrays using shared memory
__global__ void mergeSortKernel(float* input, float* output, int n, int width) {
    extern __shared__ float shared[]; // Declare dynamically allocated shared memory
    
    int tid = threadIdx.x; // Thread index within block
    int blockStart = 2 * blockIdx.x * width; // Starting index of two subarrays to merge
    
    int leftStart = blockStart; // Start of left subarray
    int leftEnd = min(leftStart + width, n); // End of left subarray (exclusive)
    int rightStart = leftEnd; // Start of right subarray
    int rightEnd = min(rightStart + width, n); // End of right subarray (exclusive)
    
    int leftSize = leftEnd - leftStart; // Size of left subarray
    int rightSize = rightEnd - rightStart; // Size of right subarray
    int totalSize = leftSize + rightSize; // Total number of elements to merge
    
    // Load data into shared memory
    if (tid < leftSize) { // If thread handles left subarray element
        shared[tid] = input[leftStart + tid]; // Load left element
    }
    if (tid < rightSize) { // If thread handles right subarray element
        shared[leftSize + tid] = input[rightStart + tid]; // Load right element after left
    }
    __syncthreads(); // Wait for both subarrays to be loaded
    
    // Merge algorithm using parallel approach
    int left = 0, right = leftSize; // Pointers for merge (currently unused - for reference)
    for (int i = tid; i < totalSize; i += blockDim.x) { // Each thread handles multiple output positions
        int leftIdx = 0, rightIdx = 0; // Indices for left and right subarrays
        int pos = i; // Current output position (currently unused)
        
        // Binary search to find position in merged array
        int low = 0, high = leftSize; // Binary search bounds in left subarray
        while (low < high) { // Binary search loop
            int mid = (low + high) / 2; // Calculate middle position
            if (shared[mid] <= (i - mid < rightSize ? shared[leftSize + i - mid] : FLT_MAX)) {
                low = mid + 1; // Search upper half
            } else {
                high = mid; // Search lower half
            }
        }
        leftIdx = low; // Number of elements from left subarray before position i
        rightIdx = i - leftIdx; // Number of elements from right subarray before position i
        
        // Determine which element to place at position i
        if (rightIdx >= rightSize || (leftIdx < leftSize && shared[leftIdx] <= shared[leftSize + rightIdx])) {
            output[blockStart + i] = shared[leftIdx]; // Take from left subarray
        } else {
            output[blockStart + i] = shared[leftSize + rightIdx]; // Take from right subarray
        }
    }
}

// Function to test reduction and measure execution time
float testReduction(float* d_data, int n, bool useShared, float& elapsed) {
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed
    
    float *d_output, *h_output; // Device and host output arrays
    CUDA_CHECK(cudaMalloc(&d_output, blocksPerGrid * sizeof(float))); // Allocate device memory for partial sums
    h_output = new float[blocksPerGrid]; // Allocate host memory for partial sums
    
    cudaEvent_t start, stop; // CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start)); // Create start event
    CUDA_CHECK(cudaEventCreate(&stop)); // Create stop event
    
    CUDA_CHECK(cudaEventRecord(start)); // Record start time
    
    if (useShared) { // If using shared memory reduction
        reductionShared<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_data, d_output, n);
        // Launch kernel with dynamically allocated shared memory
    } else { // If using global memory reduction
        reductionGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output, n);
        // Launch kernel without shared memory
    }
    
    CUDA_CHECK(cudaEventRecord(stop)); // Record stop time
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for kernel to complete
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); // Calculate elapsed time in milliseconds
    
    // Copy results back and finish reduction on CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    // Copy partial sums from device to host
    
    float sum = 0.0f; // Final sum
    for (int i = 0; i < blocksPerGrid; i++) { // Iterate through all partial sums
        sum += h_output[i]; // Add each partial sum to total
    }
    
    delete[] h_output; // Free host memory
    CUDA_CHECK(cudaFree(d_output)); // Free device memory
    CUDA_CHECK(cudaEventDestroy(start)); // Destroy start event
    CUDA_CHECK(cudaEventDestroy(stop)); // Destroy stop event
    
    return sum; // Return final sum
}

// Function to perform sorting and measure execution time
void testSorting(float* d_data, int n, float& elapsed) {
    int chunkSize = 512; // Size of each chunk to sort with bubble sort
    int threadsPerBlock = 256; // Number of threads per block
    int numChunks = (n + chunkSize - 1) / chunkSize; // Calculate number of chunks
    
    float *d_temp; // Temporary buffer for merge operations
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(float))); // Allocate temporary device memory
    
    cudaEvent_t start, stop; // CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start)); // Create start event
    CUDA_CHECK(cudaEventCreate(&stop)); // Create stop event
    
    CUDA_CHECK(cudaEventRecord(start)); // Record start time
    
    // Phase 1: Sort chunks with bubble sort
    bubbleSortKernel<<<numChunks, threadsPerBlock, chunkSize * sizeof(float)>>>(d_data, n, chunkSize);
    // Launch bubble sort kernel with shared memory for each chunk
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all chunks to be sorted
    
    // Phase 2: Merge sorted chunks iteratively
    float *d_input = d_data; // Pointer to input buffer
    float *d_output = d_temp; // Pointer to output buffer
    bool swapped = false; // Track if buffers have been swapped
    
    for (int width = chunkSize; width < n; width *= 2) { // Double the merge width each iteration
        int numMerges = (n + 2 * width - 1) / (2 * width); // Calculate number of merge operations
        int sharedSize = 2 * width * sizeof(float); // Shared memory size for two subarrays
        
        mergeSortKernel<<<numMerges, threadsPerBlock, sharedSize>>>(d_input, d_output, n, width);
        // Launch merge kernel with shared memory
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for merge to complete
        
        // Swap input and output pointers for next iteration
        float* temp = d_input; // Temporary pointer
        d_input = d_output; // Output becomes input
        d_output = temp; // Input becomes output
        swapped = !swapped; // Toggle swap flag
    }
    
    CUDA_CHECK(cudaEventRecord(stop)); // Record stop time
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for kernel to complete
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); // Calculate elapsed time
    
    CUDA_CHECK(cudaEventDestroy(start)); // Destroy start event
    CUDA_CHECK(cudaEventDestroy(stop)); // Destroy stop event
    
    // If the sorted data is in d_temp, copy it back to d_data
    if (swapped) { // If buffers were swapped odd number of times
        CUDA_CHECK(cudaMemcpy(d_data, d_temp, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // Copy sorted data back to original buffer
    }
    
    CUDA_CHECK(cudaFree(d_temp)); // Free temporary buffer
}

// Main function - entry point of the program
int main() {
    std::cout << "=== CUDA Reduction and Sorting Performance Test ===" << std::endl;
    // Print program title
    
    // Test sizes for benchmarking
    int sizes[] = {10000, 100000, 1000000}; // Array of test sizes: 10K, 100K, 1M
    int numSizes = 3; // Number of different sizes to test
    
    // Output file for results
    std::ofstream csvFile("results.csv"); // Create/open CSV file for writing results
    csvFile << "Size,ReductionGlobal,ReductionShared,Sorting\n"; // Write CSV header
    
    for (int s = 0; s < numSizes; s++) { // Loop through each test size
        int n = sizes[s]; // Get current array size
        std::cout << "\n--- Testing with array size: " << n << " ---" << std::endl;
        // Print current test size
        
        // Allocate device memory
        float *d_data, *d_data_copy; // Device pointers for data and copy
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float))); // Allocate memory for main data
        CUDA_CHECK(cudaMalloc(&d_data_copy, n * sizeof(float))); // Allocate memory for backup copy
        
        // Generate random data on GPU
        generateRandomData(d_data, n); // Fill array with random numbers
        CUDA_CHECK(cudaMemcpy(d_data_copy, d_data, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // Make a backup copy for later tests
        
        // Verify with CPU sum
        float *h_data = new float[n]; // Allocate host memory
        CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
        // Copy data from device to host for verification
        float cpuSumResult = cpuSum(h_data, n); // Compute sum on CPU
        std::cout << "CPU Sum: " << cpuSumResult << std::endl; // Print CPU sum for reference
        
        // Test reduction with global memory
        float timeGlobal, timeShared, timeSorting; // Variables to store execution times
        float sumGlobal = testReduction(d_data, n, false, timeGlobal); // Test global memory reduction
        std::cout << "Reduction (Global): " << sumGlobal << " Time: " << timeGlobal << " ms" << std::endl;
        // Print global memory reduction result and time
        
        // Reset data to original random values
        CUDA_CHECK(cudaMemcpy(d_data, d_data_copy, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // Restore data from backup
        
        // Test reduction with shared memory
        float sumShared = testReduction(d_data, n, true, timeShared); // Test shared memory reduction
        std::cout << "Reduction (Shared): " << sumShared << " Time: " << timeShared << " ms" << std::endl;
        // Print shared memory reduction result and time
        
        // Reset data for sorting
        CUDA_CHECK(cudaMemcpy(d_data, d_data_copy, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // Restore data from backup again
        
        // Test sorting
        testSorting(d_data, n, timeSorting); // Perform sorting and measure time
        std::cout << "Sorting Time: " << timeSorting << " ms" << std::endl; // Print sorting time
        
        // Verify sorting (check first few elements)
        CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
        // Copy sorted data back to host
        bool sorted = true; // Flag to track if array is sorted
        for (int i = 0; i < n - 1 && i < 1000; i++) { // Check first 1000 elements (or all if less)
            if (h_data[i] > h_data[i + 1]) { // If elements are out of order
                sorted = false; // Set flag to false
                break; // Stop checking
            }
        }
        std::cout << "Sorting verification: " << (sorted ? "PASSED" : "FAILED") << std::endl;
        // Print verification result
        
        // Write results to CSV file
        csvFile << n << "," << timeGlobal << "," << timeShared << "," << timeSorting << "\n";
        // Write size and all three times to CSV
        
        // Cleanup memory for this iteration
        delete[] h_data; // Free host memory
        CUDA_CHECK(cudaFree(d_data)); // Free device memory for main data
        CUDA_CHECK(cudaFree(d_data_copy)); // Free device memory for backup copy
    }
    
    csvFile.close(); // Close CSV file
    std::cout << "\n=== Results saved to results.csv ===" << std::endl; // Print confirmation
    std::cout << "Speedup (Shared vs Global): " << std::endl; // Print speedup header
    std::cout << "Run 'python plot_results.py' to generate graphs" << std::endl; // Print plotting instructions
    
    return 0; // Exit program successfully
}
