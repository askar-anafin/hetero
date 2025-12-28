#include <iostream> // Include IO stream for console output
#include <vector> // Include vector for dynamic arrays
#include <algorithm> // Include algorithms like sort if needed
#include <chrono> // Include chrono for high-resolution timing
#include <random> // Include random for random number generation
#include <iomanip> // Include iomanip for formatting output
#include "sorters_cpu.h" // Include CPU sorting implementations
#include "sorters_gpu.cuh" // Include GPU sorting implementations

// Error handling macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0) // standard do-while(0) macro trick for error checking

// Helper to fill array
void fillRandom(std::vector<int>& arr) { // Function to fill vector with random numbers
    std::random_device rd; // Create random device
    std::mt19937 gen(rd()); // Initialize Mersenne Twister generator
    std::uniform_int_distribution<> dis(0, 100000); // Define distribution range
    for (auto& x : arr) x = dis(gen); // Iterate and fill vector
}

// Check if sorted
bool checkSorted(const std::vector<int>& arr) { // Function to verify if vector is sorted
    for (size_t i = 1; i < arr.size(); ++i) { // Iterate from second element
        if (arr[i] < arr[i - 1]) return false; // If current element is less than previous, returns false
    }
    return true; // Return true if loop completes
}

// --- GPU Wrappers ---

void gpuMergeSort(int* d_arr, int n) { // Wrapper function for GPU Merge Sort
    int threadsPerBlock = 256; // Define threads per block
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed
    
    // 1. Block Sort
    // We treat blockDim.x as the chunk size? 
    // In sorters_gpu.cuh I used `segment_size = blockDim.x`. 
    // And I handled `segment_idx = blockIdx.x * segment_size`.
    // So if n=10000, 256 threads -> chunks of 256. 
    // block_sort_kernel sorts chunks of 256.
    
    block_sort_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_arr, n); // Launch block sort kernel with shared memory
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    cudaDeviceSynchronize(); // Wait for kernel to finish
    
    // 2. Merge Pairs
    // Initial width = 256 (sorted segments)
    // We merge [0..256) and [256..512) -> [0..512)
    // width increases: 256 -> 512 -> 1024 ...
    
    int* d_temp; // Pointer for temporary buffer
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int))); // Allocate temporary buffer on GPU
    
    for (int width = threadsPerBlock; width < n; width *= 2) { // Loop doubling width at each step
        // Launch one thread per element? Or blocks?
        // merge_rank_kernel uses `tid` to map each element to its destination.
        // It needs `n` threads total.
        int merge_blocks = (n + threadsPerBlock - 1) / threadsPerBlock; // Calculate blocks for merge kernel
        merge_rank_kernel<<<merge_blocks, threadsPerBlock>>>(d_arr, d_temp, n, width); // Launch merge kernel
        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        cudaDeviceSynchronize(); // Wait for kernel to finish
        
        // Swap buffers
        // We want result in d_arr. If this was the last step and result is in d_temp, copy back.
        // Or pointer swap? Pointers are passed by value to wrappers?
        // `d_arr` is a pointer. We can't change the caller's pointer easily unless passed by ref.
        // Copy back is safer.
        CUDA_CHECK(cudaMemcpy(d_arr, d_temp, n * sizeof(int), cudaMemcpyDeviceToDevice)); // Copy merged results back to original buffer
    }
    
    CUDA_CHECK(cudaFree(d_temp)); // Free temporary buffer
}

void gpuQuickSort(int* d_arr, int n) { // Wrapper function for GPU Quick Sort
    // Requires Dynamic Parallelism buffer
    // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16)); 
    // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768));
    
    // Launch initial parent kernel
    // Just 1 thread to manage
    cdp_quicksort_kernel<<<1, 1>>>(d_arr, 0, n - 1, 0); // Launch initial kernel wrapper with 1 thread
    CUDA_CHECK(cudaGetLastError()); // Check for errors
    cudaDeviceSynchronize(); // Wait for completion
}

void gpuHeapSort(int* d_arr, int n) { // Wrapper function for GPU Heap Sort
    // 1. Build Heap (Parallel by levels)
    invoke_gpu_build_heap(d_arr, n); // Helper to build heap in parallel
    
    // 2. Extract (Sequential on Device)
    one_thread_heap_sort_extract<<<1, 1>>>(d_arr, n); // Launch sequential extraction on GPU
    CUDA_CHECK(cudaGetLastError()); // Check for errors
    cudaDeviceSynchronize(); // Wait for completion
}

// --- Benchmarking Driver ---

void runBenchmark(int size) { // Function to run benchmarks for a specific size
    std::cout << "\n--- Array Size: " << size << " ---\n"; // Print array size header
    std::vector<int> h_original(size); // Create vector of given size
    fillRandom(h_original); // Fill with random data
    
    // Keep a copy for reset
    auto reset = [&]() { return h_original; }; // Lambda to get a fresh copy of unsorted data
    
    // CPU Timers
    auto run_cpu = [&](std::string name, void (*func)(std::vector<int>&)) { // Lambda for running CPU tests
        std::vector<int> arr = reset(); // Reset array
        auto start = std::chrono::high_resolution_clock::now(); // Start timer
        func(arr); // Execute CPU sort
        auto end = std::chrono::high_resolution_clock::now(); // Stop timer
        std::chrono::duration<double, std::milli> ms = end - start; // Calculate duration
        std::cout << std::left << std::setw(20) << name << ": " << ms.count() << " ms" // Print result
                  << (checkSorted(arr) ? " [OK]" : " [FAIL]") << std::endl; // Print validity
    };
    
    run_cpu("CPU Merge Sort", cpuMergeSort); // Run CPU Merge Sort
    run_cpu("CPU Quick Sort", cpuQuickSort); // Run CPU Quick Sort
    run_cpu("CPU Heap Sort", cpuHeapSort); // Run CPU Heap Sort
    
    // GPU Timers
    int* d_arr; // Declare device pointer
    CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(int))); // Allocate memory on GPU
    
    auto run_gpu = [&](std::string name, void (*func)(int*, int)) { // Lambda for running GPU tests
        std::vector<int> arr = reset(); // Reset host array
        
        // Measure Transfer + Execute
        auto start = std::chrono::high_resolution_clock::now(); // Start timer
        
        CUDA_CHECK(cudaMemcpy(d_arr, arr.data(), size * sizeof(int), cudaMemcpyHostToDevice)); // Copy data to GPU
        func(d_arr, size); // Execute GPU sort
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for GPU to finish
        CUDA_CHECK(cudaMemcpy(arr.data(), d_arr, size * sizeof(int), cudaMemcpyDeviceToHost)); // Copy results back
        
        auto end = std::chrono::high_resolution_clock::now(); // Stop timer
        std::chrono::duration<double, std::milli> ms = end - start; // Calculate duration
        std::cout << std::left << std::setw(20) << name << ": " << ms.count() << " ms" // Print result
                  << (checkSorted(arr) ? " [OK]" : " [FAIL]") << std::endl; // Print validity
    };
    
    // Increase recursion limit for Quick Sort
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24); // Set max depth for CDP
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768); // Set pending launch limit

    run_gpu("GPU Merge Sort", gpuMergeSort); // Run GPU Merge Sort
    run_gpu("GPU Quick Sort", gpuQuickSort); // Run GPU Quick Sort
    run_gpu("GPU Heap Sort", gpuHeapSort); // Run GPU Heap Sort
    
    CUDA_CHECK(cudaFree(d_arr)); // Free GPU memory
}

int main() { // Main function
    std::cout << "Starting Practice 3 Benchmarks..." << std::endl; // Print start message
    std::cout << "Comparing CPU vs GPU (CUDA) Sorting Algorithms" << std::endl; // Print info
    
    int sizes[] = {10000, 100000, 1000000}; // Define test sizes
    for (int s : sizes) { // Loop through sizes
        runBenchmark(s); // Run benchmark for each size
    }
    
    return 0; // Return success
}
